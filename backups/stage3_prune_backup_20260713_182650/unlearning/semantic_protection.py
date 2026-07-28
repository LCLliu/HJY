"""
Semantic Protection Judgment Module
====================================
Determines which retain items should be protected from pruning because
they share high semantic similarity with forgotten items.

For each (user, forget_item) pair:
1. Computes semantic similarity between forget_item and each retain_item in
   the same user's history, using item title embeddings
2. Identifies "overlapping retain samples": items that are BOTH in the
   collaborative residual neighborhood AND semantically similar to the forget item
3. Outputs a semantic_protection_score for each retain item

The protection score is higher when:
- The retain item has high semantic similarity to the forget item
- The retain item is in the same user's history as the forget item
- The retain item is flagged as a marginal residual by the collaborative diagnosis

Protection means: these items' corresponding parameter directions should
NOT be pruned, even if they overlap with the forget item's collaborative signal.
"""

import os
import json
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm

import torch
import torch.nn.functional as F


class SemanticProtection:
    """
    Compute semantic protection scores for retain items.

    Uses item title text to build semantic embeddings, then computes
    similarity between forget items and retain items to determine which
    retain items are at risk of collateral damage during unlearning.

    Parameters
    ----------
    meta : Dict[int, str]
        Item ID → title string mapping
    retain_train : Dict[int, List[int]]
        User retained training sequences
    forget_train : Dict[int, List[int]]
        User forgotten training sequences
    num_items : int
        Total number of items
    tokenizer : optional, HuggingFace tokenizer for embedding computation
    model : optional, pre-trained model for extracting embeddings
    embedding_cache_path : str, optional, path to save/load cached embeddings
    """

    def __init__(
        self,
        meta: Dict[int, str],
        retain_train: Dict[int, List[int]],
        forget_train: Dict[int, List[int]],
        num_items: int,
        tokenizer=None,
        model=None,
        embedding_cache_path: Optional[str] = None,
    ):
        self.meta = meta
        self.retain_train = retain_train
        self.forget_train = forget_train
        self.num_items = num_items
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_cache_path = embedding_cache_path

        # Item embeddings (built lazily)
        self._item_embeddings: Optional[np.ndarray] = None  # [num_items+1, embed_dim]
        self._embed_dim: int = 0

        # Results
        self.results: Dict = {}
        self.protection_scores: Dict[int, float] = {}  # {iid: protection_score}
        self.overlap_samples: List[Dict] = []  # overlapping retain samples

    # ------------------------------------------------------------------
    # 1. Build Semantic Embeddings
    # ------------------------------------------------------------------

    def build_embeddings(self, method: str = 'auto'):
        """
        Build item embeddings from titles.

        Parameters
        ----------
        method : str
            'auto' - try model embeddings, fall back to TF-IDF
            'model' - use the LLM's embed_tokens layer
            'tfidf' - use TF-IDF over title words
        """
        print('[SemanticProtection] Building item embeddings...')

        if method == 'auto' or method == 'model_embedding':
            if self.tokenizer is not None and self.model is not None:
                method = 'model'
            else:
                method = 'tfidf'

        if method == 'model':
            self._build_model_embeddings()
        elif method == 'tfidf':
            self._build_tfidf_embeddings()
        else:
            raise ValueError(f"Unknown embedding method: {method}")

        print(f'[SemanticProtection] Built embeddings: '
              f'shape={self._item_embeddings.shape}, dim={self._embed_dim}')

    def _build_model_embeddings(self):
        """Build embeddings using the LLM's token embeddings (mean pooling)."""
        self._embed_dim = 4096  # Llama hidden size
        embeddings = np.zeros((self.num_items + 1, self._embed_dim), dtype=np.float32)

        # Get the embedding weight from the model
        embed_weight = None
        try:
            # PeftModel path: model.base_model.model.model.embed_tokens
            #   PeftModel.base_model = LoraModel
            #   LoraModel.model = LlamaForCausalLM
            #   LlamaForCausalLM.model = LlamaModel
            #   LlamaModel.embed_tokens = nn.Embedding(32000, 4096)
            if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
                inner = self.model.base_model.model
                if hasattr(inner, 'model') and hasattr(inner.model, 'embed_tokens'):
                    embed = inner.model.embed_tokens
                elif hasattr(inner, 'embed_tokens'):
                    embed = inner.embed_tokens
                else:
                    raise ValueError(f"Cannot navigate model structure from {type(inner)}")
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                embed = self.model.model.embed_tokens
            elif hasattr(self.model, 'embed_tokens'):
                embed = self.model.embed_tokens
            else:
                raise ValueError(f"Cannot find embed_tokens in model of type {type(self.model)}")

            embed_weight = embed.weight.detach().float().cpu().numpy()
        except Exception as e:
            print(f'  Warning: Could not extract model embeddings ({e}), '
                  f'falling back to TF-IDF')
            self._build_tfidf_embeddings()
            return

        for iid in tqdm(range(1, self.num_items + 1),
                        desc='  Computing model embeddings',
                        total=self.num_items):
            title = self.meta.get(iid, '')
            if not title:
                embeddings[iid] = np.zeros(self._embed_dim, dtype=np.float32)
                continue

            # Tokenize the title
            tokens = self.tokenizer.encode(title, add_special_tokens=False,
                                           truncation=True, max_length=32)
            if not tokens:
                embeddings[iid] = np.zeros(self._embed_dim, dtype=np.float32)
                continue

            # Mean pool the token embeddings
            token_vecs = embed_weight[tokens]  # (n_tokens, 4096)
            embeddings[iid] = token_vecs.mean(axis=0)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self._item_embeddings = embeddings
        self._embed_dim = embeddings.shape[1]

    def _build_tfidf_embeddings(self):
        """Build TF-IDF embeddings over item title words (fallback)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Collect titles
        titles = []
        for iid in range(1, self.num_items + 1):
            titles.append(self.meta.get(iid, ''))

        # Build TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=256,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(titles)  # (num_items, max_features)
        self._embed_dim = tfidf_matrix.shape[1]

        # Convert to dense array (pad index 0)
        embeddings = np.zeros((self.num_items + 1, self._embed_dim), dtype=np.float32)
        embeddings[1:] = tfidf_matrix.toarray()

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self._item_embeddings = embeddings

    # ------------------------------------------------------------------
    # 2. Semantic Similarity Computation
    # ------------------------------------------------------------------

    def compute_similarity(self, iid_a: int, iid_b: int) -> float:
        """Compute cosine similarity between two items based on their title embeddings."""
        if self._item_embeddings is None:
            raise RuntimeError("Embeddings not built. Call build_embeddings() first.")
        vec_a = self._item_embeddings[iid_a]
        vec_b = self._item_embeddings[iid_b]
        sim = np.dot(vec_a, vec_b)
        return float(np.clip(sim, -1.0, 1.0))

    def compute_similarity_matrix(
        self, query_iids: List[int], target_iids: List[int]
    ) -> np.ndarray:
        """Compute pairwise cosine similarity between two lists of items."""
        if self._item_embeddings is None:
            raise RuntimeError("Embeddings not built.")
        q_vecs = self._item_embeddings[query_iids]  # (n_query, dim)
        t_vecs = self._item_embeddings[target_iids]  # (n_target, dim)
        sim = np.dot(q_vecs, t_vecs.T)  # (n_query, n_target)
        return np.clip(sim, -1.0, 1.0)

    # ------------------------------------------------------------------
    # 3. Protection Scoring
    # ------------------------------------------------------------------

    def compute_protection_scores(
        self,
        residual_diagnosis_results: Optional[Dict] = None,
        sim_threshold: float = 0.3,
        semantic_topk_ratio: float = 0.1,
        normalize_scores: bool = True,
    ) -> Dict:
        """
        Compute semantic protection scores for retain items.

        For each forget interaction (u, i_f):
        - Find retain items in the same user's history
        - Compute semantic similarity between i_f and each retain item
        - Only protect the Top-K% most similar retain items (not all)
        - Apply min-max normalization to avoid 1.0 saturation
        - If similarity > sim_threshold AND the retain item is in the
          collaborative residual neighborhood → "overlapping sample" → needs protection

        Parameters
        ----------
        residual_diagnosis_results : dict, optional
        sim_threshold : float
            Minimum cosine similarity (unused when topk_ratio is applied)
        semantic_topk_ratio : float
            Protect only the top K fraction of retain items per forget interaction (e.g., 0.1 = top 10%)
        normalize_scores : bool
            Apply min-max normalization to similarity scores

        Returns
        -------
        Dict with per-interaction and aggregated results
        """
        if self._item_embeddings is None:
            self.build_embeddings()

        # Build residual neighborhood lookup for fast checking
        residual_neighbors = defaultdict(set)
        if residual_diagnosis_results:
            for r in residual_diagnosis_results.get('per_interaction', []):
                uid = r['user_id']
                for miid in r['marginal_residuals']:
                    residual_neighbors[(uid, miid)].add(r['forget_item_id'])

        print(f'\n[SemanticProtection] Computing protection scores '
              f'(method=top{semantic_topk_ratio*100:.0f}%, '
              f'normalize={normalize_scores})...')

        per_interaction = []
        overlap_samples = []
        item_protection_contributions = defaultdict(list)
        all_similarity_values = []  # for global normalization

        total = sum(len(v) for v in self.forget_train.values())

        # Phase 1: compute raw similarities for all forget interactions
        print('  Phase 1: Computing raw similarities...')
        raw_sims_by_interaction = []  # [(uid, forget_iid, retain_list, sims_array)]

        for uid in tqdm(sorted(self.forget_train.keys()), desc='  Phase 1',
                         total=len(self.forget_train)):
            retain_seq = self.retain_train.get(uid, [])
            retain_set = set(retain_seq)
            if not retain_set:
                continue

            retain_list = list(retain_set)
            retain_vecs = self._item_embeddings[retain_list]

            for forget_iid in self.forget_train[uid]:
                forget_vec = self._item_embeddings[forget_iid]
                sims = np.dot(retain_vecs, forget_vec)
                sims = np.clip(sims, -1.0, 1.0)
                raw_sims_by_interaction.append((uid, forget_iid, retain_list, sims))
                all_similarity_values.extend(sims.tolist())

        # Compute global min/max for normalization
        if normalize_scores and all_similarity_values:
            sims_array = np.array(all_similarity_values)
            sim_min = float(sims_array.min())
            sim_max = float(sims_array.max())
            sim_range = sim_max - sim_min if sim_max > sim_min else 1.0
            print(f'  Similarity range: [{sim_min:.4f}, {sim_max:.4f}]')
        else:
            sim_min, sim_max, sim_range = 0.0, 1.0, 1.0

        # Phase 2: per-interaction top-K selection and protection scoring
        print('  Phase 2: Computing protection scores (top-K + normalize)...')
        all_protection_scores = []

        for uid, forget_iid, retain_list, sims in tqdm(
            raw_sims_by_interaction, desc='  Phase 2', total=len(raw_sims_by_interaction)
        ):
            # Normalize similarities
            if normalize_scores:
                sims_norm = (sims - sim_min) / sim_range
            else:
                sims_norm = sims

            # Sort by normalized similarity (descending)
            sorted_idx = np.argsort(-sims_norm)
            n_retain = len(retain_list)

            # Top-K: only protect top semantic_topk_ratio fraction
            top_k = max(1, int(n_retain * semantic_topk_ratio))
            top_k = min(top_k, n_retain)

            # Select top-K
            top_indices = sorted_idx[:top_k]
            top_items = [retain_list[idx] for idx in top_indices]
            top_sims_raw = [float(sims[idx]) for idx in top_indices]
            top_sims_norm = [float(sims_norm[idx]) for idx in top_indices]

            # Determine which are overlapping (in residual neighborhood)
            user_overlaps = []
            for i, idx in enumerate(top_indices):
                riid = retain_list[idx]
                sim_raw = top_sims_raw[i]
                sim_norm = top_sims_norm[i]

                is_in_residual = False
                residual_cooc_score = 0.0
                if residual_diagnosis_results:
                    for r in residual_diagnosis_results.get('per_interaction', []):
                        if r['user_id'] == uid and r['forget_item_id'] == forget_iid:
                            for mi, ms in zip(r['marginal_residuals'],
                                               r['marginal_residual_scores']):
                                if mi == riid:
                                    is_in_residual = True
                                    residual_cooc_score = ms
                                    break
                            break

                # Protection score: use normalized similarity
                protection_score = sim_norm
                if is_in_residual:
                    # Boost for overlapping samples (capped)
                    protection_score = min(1.0, sim_norm * (1.0 + 0.5 * residual_cooc_score))

                item_protection_contributions[riid].append(protection_score)
                all_protection_scores.append(protection_score)

                user_overlaps.append({
                    'retain_item_id': riid,
                    'retain_item_title': self.meta.get(riid, 'N/A'),
                    'semantic_similarity': sim_raw,
                    'semantic_similarity_norm': sim_norm,
                    'is_in_residual_neighborhood': is_in_residual,
                    'residual_cooc_score': residual_cooc_score,
                    'protection_score': protection_score,
                })

            if user_overlaps:
                overlap_samples.extend([
                    {**ov, 'user_id': uid, 'forget_item_id': forget_iid}
                    for ov in user_overlaps
                ])

            n_overlapping = len([o for o in user_overlaps if o['is_in_residual_neighborhood']])
            per_interaction.append({
                'user_id': uid,
                'forget_item_id': forget_iid,
                'forget_item_title': self.meta.get(forget_iid, 'N/A'),
                'n_retain_items': n_retain,
                'n_semantically_similar': len(top_items),
                'similar_retain_items': top_items[:30],
                'similar_retain_scores': top_sims_raw[:30],
                'similar_retain_scores_norm': top_sims_norm[:30],
                'n_overlapping': n_overlapping,
                'overlapping_samples': [o for o in user_overlaps
                                       if o['is_in_residual_neighborhood']][:20],
            })

        # Aggregate global protection scores per item (mean of top 10)
        protection_scores = {}
        for iid, scores in item_protection_contributions.items():
            top_scores = sorted(scores, reverse=True)[:10]
            protection_scores[iid] = float(np.mean(top_scores)) if top_scores else 0.0
        self.protection_scores = protection_scores

        overlap_samples.sort(key=lambda x: x['protection_score'], reverse=True)
        self.overlap_samples = overlap_samples

        n_total_similar = sum(r['n_semantically_similar'] for r in per_interaction)
        n_total_overlap = sum(r['n_overlapping'] for r in per_interaction)

        # Debug: protection score distribution
        prot_vals = list(protection_scores.values()) if protection_scores else [0.0]
        prot_arr = np.array(prot_vals)
        print(f'\n[SemanticProtection] Protection score distribution:')
        print(f'  min={prot_arr.min():.4f}, max={prot_arr.max():.4f}, '
              f'mean={prot_arr.mean():.4f}, median={np.median(prot_arr):.4f}')
        print(f'  q10={np.percentile(prot_arr, 10):.4f}, '
              f'q25={np.percentile(prot_arr, 25):.4f}, '
              f'q75={np.percentile(prot_arr, 75):.4f}, '
              f'q90={np.percentile(prot_arr, 90):.4f}')
        print(f'  Protected items (>0): '
              f'{int(np.count_nonzero(prot_arr))}/{len(prot_arr)}')
        print(f'  Top-20 mean: {np.mean(np.sort(prot_arr)[-20:]):.4f}')

        self.results = {
            'per_interaction': per_interaction,
            'overlap_samples': overlap_samples[:500],
            'protection_scores': {
                str(iid): score for iid, score in protection_scores.items()
            },
            'summary': {
                'total_forget_interactions': total,
                'total_semantically_similar_pairs': n_total_similar,
                'total_overlapping_samples': n_total_overlap,
                'distinct_protected_items': len(protection_scores),
                'protection_score_stats': {
                    'min': float(prot_arr.min()),
                    'max': float(prot_arr.max()),
                    'mean': float(prot_arr.mean()),
                    'median': float(np.median(prot_arr)),
                    'q10': float(np.percentile(prot_arr, 10)),
                    'q25': float(np.percentile(prot_arr, 25)),
                    'q75': float(np.percentile(prot_arr, 75)),
                    'q90': float(np.percentile(prot_arr, 90)),
                },
                'protected_items_top20': [
                    {'item_id': iid, 'title': self.meta.get(iid, 'N/A'),
                     'protection_score': score}
                    for iid, score in sorted(
                        protection_scores.items(), key=lambda x: x[1], reverse=True
                    )[:20]
                ],
                'sim_threshold': sim_threshold,
            },
        }

        print(f'[SemanticProtection] {n_total_similar} similar pairs '
              f'(top {semantic_topk_ratio*100:.0f}%), '
              f'{n_total_overlap} overlapping')
        print(f'[SemanticProtection] {len(protection_scores)} items flagged')

        return self.results

    # ------------------------------------------------------------------
    # 4. Query Interface
    # ------------------------------------------------------------------

    def get_protection_score(self, iid: int) -> float:
        """Get the global protection score for a specific retain item."""
        return self.protection_scores.get(iid, 0.0)

    def is_overlapping(
        self, uid: int, retain_iid: int, forget_iid: int
    ) -> bool:
        """Check if a specific (retain, forget) pair is an overlapping sample."""
        for ov in self.overlap_samples:
            if (ov['user_id'] == uid and
                ov['retain_item_id'] == retain_iid and
                ov['forget_item_id'] == forget_iid):
                return True
        return False

    def get_protected_items_above_threshold(
        self, threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Get retain items with protection score >= threshold."""
        return [(iid, score) for iid, score in self.protection_scores.items()
                if score >= threshold]

    # ------------------------------------------------------------------
    # 5. I/O
    # ------------------------------------------------------------------

    def save_results(self, path: str):
        """Save protection results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f'[SemanticProtection] Results saved to {path}')

    def load_results(self, path: str):
        """Load previously computed protection results."""
        with open(path, 'r') as f:
            self.results = json.load(f)
        # Rebuild protection scores from loaded results
        if 'protection_scores' in self.results:
            self.protection_scores = {
                int(k): v for k, v in self.results['protection_scores'].items()
            }
        print(f'[SemanticProtection] Results loaded from {path}')


class InteractionSemanticProtection:
    """Interaction-level semantic protection for geometry_prune.

    Protection is assigned to retain interactions relative to a forget
    interaction, not globally to an item.
    """

    def __init__(self, meta, split_data, dataset_data, marginal_candidates, args):
        self.meta = meta
        self.split_data = split_data
        self.dataset_data = dataset_data
        self.marginal_candidates = marginal_candidates
        self.args = args
        self.title_tokens = {
            int(iid): self._tokenize(title)
            for iid, title in meta.items()
        }
        self.retain_rows = self._retain_rows()
        self.retain_rows_by_user = self._retain_rows_by_user()
        self.overlap_keys = self._overlap_keys()

    @staticmethod
    def _row_uid(row):
        return int(row.get('uid', row.get('user_id')))

    @staticmethod
    def _row_iid(row):
        return int(row.get('iid', row.get('item_id')))

    @staticmethod
    def _tokenize(title):
        import re
        return set(re.findall(r"[a-z0-9]+", (title or "").lower()))

    def _retain_rows(self):
        rows = []
        for row in self.split_data.get('retain_interactions', []):
            rows.append(row)
        for row in self.split_data.get('overlap_retain_interactions', []):
            rows.append(row)
        return rows

    def _retain_rows_by_user(self):
        by_user = defaultdict(list)
        for row in self.retain_rows:
            by_user[self._row_uid(row)].append(row)
        return by_user

    def _overlap_keys(self):
        keys = set()
        for row in self.split_data.get('overlap_retain_interactions', []):
            keys.add((self._row_uid(row), self._row_iid(row)))
        return keys

    def semantic_similarity(self, a, b):
        ta = self.title_tokens.get(int(a), set())
        tb = self.title_tokens.get(int(b), set())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def user_preference_consistency(self, uid, retain_iid):
        retain_train = self.dataset_data.get('retain_train', {})
        seq = retain_train.get(int(uid), [])
        if not seq:
            return 0.0
        return 1.0 if int(retain_iid) in set(seq) else 0.0

    def _residual_lookup(self):
        lookup = defaultdict(float)
        for cand in self.marginal_candidates:
            key = (int(cand['uid']), int(cand['candidate_iid']))
            lookup[key] = max(lookup[key], float(cand['marginal_residual_score']))
        return lookup

    def compute(self):
        residual_lookup = self._residual_lookup()
        records = []

        for forget in self.split_data.get('forget_interactions', []):
            uid = self._row_uid(forget)
            forget_iid = self._row_iid(forget)
            for retain in self.retain_rows_by_user.get(uid, []):
                retain_iid = self._row_iid(retain)
                sem = self.semantic_similarity(forget_iid, retain_iid)
                consistency = self.user_preference_consistency(uid, retain_iid)
                retain_label = 1.0 if retain_iid in set(self.dataset_data.get('retain_train', {}).get(uid, [])) else 0.0
                overlap = 1.0 if (uid, retain_iid) in self.overlap_keys else 0.0
                residual_risk = residual_lookup.get((uid, retain_iid), 0.0)
                score = float(np.clip(
                    0.35 * sem +
                    0.25 * consistency +
                    0.20 * retain_label +
                    0.15 * overlap +
                    0.05 * residual_risk,
                    0.0, 1.0,
                ))
                if overlap > 0 and retain_label > 0 and score >= 0.45:
                    level = 'strong'
                elif score >= 0.35:
                    level = 'medium'
                else:
                    level = 'weak'
                records.append({
                    'uid': uid,
                    'forget_iid': int(forget_iid),
                    'retain_iid': int(retain_iid),
                    'semantic_similarity': float(sem),
                    'user_preference_consistency': float(consistency),
                    'retain_label': float(retain_label),
                    'overlap_risk': float(overlap),
                    'residual_risk': float(residual_risk),
                    'protection_score': score,
                    'protection_level': level,
                })

        records.sort(key=lambda r: r['protection_score'], reverse=True)
        return {
            'records': records,
            'summary': {
                'num_records': len(records),
                'strong': sum(1 for r in records if r['protection_level'] == 'strong'),
                'medium': sum(1 for r in records if r['protection_level'] == 'medium'),
                'weak': sum(1 for r in records if r['protection_level'] == 'weak'),
                'note': (
                    'Strong protection is limited to retain/overlap retain interactions; '
                    'semantic similarity alone is insufficient.'
                ),
            },
        }
