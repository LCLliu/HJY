"""
Collaborative Residual Diagnosis Module
========================================
For each forgotten interaction (user, forget_item), computes:
1. Co-occurrence-based collaborative residual strength
2. Marginal residual candidates (retain items that co-occur with forget_item)
3. Similar users and similar items in the collaborative neighborhood
4. Score margin impact on ranking boundary (if model is provided)
5. Combined collaborative_residual_score
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

from .counterfactual_boundary_calibration import CounterfactualBoundaryCalibrator


class CollaborativeResidualDiagnosis:
    """
    Diagnose collaborative residual strength for each forgotten interaction.

    For each (user, forget_item) pair in forget_train, this module:
    - Builds item-item co-occurrence graph from retain_train sequences
    - Computes collaborative neighborhood (co-occurring items, similar users)
    - Identifies marginal residual samples (retain items with high co-occurrence
      to the forget item that also appear in the same user's history)
    - Computes score margin impact using the trained ranker model (optional)
    - Outputs a combined collaborative_residual_score

    Parameters
    ----------
    retain_train : Dict[int, List[int]]
        User retained training sequences {uid: [iid, ...]}
    forget_train : Dict[int, List[int]]
        User forgotten training sequences {uid: [iid, ...]}
    val : Dict[int, List[int]]
        Validation items {uid: [iid]}
    test : Dict[int, List[int]]
        Test items {uid: [iid]}
    num_items : int
        Total number of items (1-based indexing, 0 reserved for padding)
    num_users : int
        Total number of users
    meta : Dict[int, str], optional
        Item title mapping {iid: "title string"}
    """

    def __init__(
        self,
        retain_train: Dict[int, List[int]],
        forget_train: Dict[int, List[int]],
        val: Dict[int, List[int]],
        test: Dict[int, List[int]],
        num_items: int,
        num_users: int,
        meta: Optional[Dict[int, str]] = None,
    ):
        self.retain_train = retain_train
        self.forget_train = forget_train
        self.val = val
        self.test = test
        self.num_items = num_items
        self.num_users = num_users
        self.meta = meta or {}

        # Derived structures (built lazily)
        self._item_freq: Optional[np.ndarray] = None
        self._cooc_matrix: Optional[np.ndarray] = None
        self._jaccard_matrix: Optional[np.ndarray] = None
        self._user_item_sets: Optional[Dict[int, Set[int]]] = None
        self._item_user_sets: Optional[Dict[int, Set[int]]] = None

        # Diagnosis results
        self.results: Dict = {}

    # ------------------------------------------------------------------
    # 1. Build Collaborative Structures
    # ------------------------------------------------------------------

    def build_collaborative_structures(self):
        """Build item co-occurrence matrix and user/item set indices."""
        print('[ResidualDiagnosis] Building collaborative structures...')

        self._build_item_frequencies()
        self._build_cooccurrence_matrix()
        self._build_user_item_sets()
        self._build_item_user_sets()

        print(f'[ResidualDiagnosis] Co-occurrence matrix: '
              f'{self._cooc_matrix.shape}, '
              f'{np.count_nonzero(self._cooc_matrix)} non-zero entries')
        print(f'[ResidualDiagnosis] Item frequency range: '
              f'{self._item_freq.min():.0f} - {self._item_freq.max():.0f}')

    def _build_item_frequencies(self):
        """Count how many users interacted with each item (from retain_train)."""
        freq = np.zeros(self.num_items + 1, dtype=np.int32)
        for uid, seq in self.retain_train.items():
            for iid in set(seq):  # count each item once per user
                freq[iid] += 1
        self._item_freq = freq

    def _build_cooccurrence_matrix(self):
        """
        Build item-item co-occurrence matrix from retain_train sequences.
        cooc[i][j] = number of users who interacted with both item i and item j.
        Also compute Jaccard similarity: jaccard(i,j) = |U_i ∩ U_j| / |U_i ∪ U_j|
        """
        n = self.num_items + 1
        cooc = np.zeros((n, n), dtype=np.int32)

        for uid, seq in tqdm(self.retain_train.items(),
                             desc='  Building co-occurrence',
                             total=self.num_users):
            unique_items = list(set(seq))
            m = len(unique_items)
            for a in range(m):
                ia = unique_items[a]
                for b in range(a, m):
                    ib = unique_items[b]
                    cooc[ia][ib] += 1
                    if ia != ib:
                        cooc[ib][ia] += 1

        self._cooc_matrix = cooc

        # Jaccard similarity
        jaccard = np.zeros((n, n), dtype=np.float32)
        for i in range(1, n):
            fi = self._item_freq[i]
            if fi == 0:
                continue
            for j in range(1, n):
                fj = self._item_freq[j]
                if fj == 0:
                    continue
                union = fi + fj - cooc[i][j]
                if union > 0:
                    jaccard[i][j] = cooc[i][j] / union
        self._jaccard_matrix = jaccard

    def _build_user_item_sets(self):
        """Build {uid: set(iid)} for fast lookup."""
        self._user_item_sets = {
            uid: set(seq) for uid, seq in self.retain_train.items()
        }

    def _build_item_user_sets(self):
        """Build {iid: set(uid)} for fast lookup."""
        item_users = defaultdict(set)
        for uid, seq in self.retain_train.items():
            for iid in set(seq):
                item_users[iid].add(uid)
        self._item_user_sets = dict(item_users)

    # ------------------------------------------------------------------
    # 2. Collaborative Neighborhood Metrics
    # ------------------------------------------------------------------

    def compute_user_similarity(
        self, uid: int, top_k: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Find users most similar to `uid` based on Jaccard similarity of
        their retained item sets.
        """
        u_set = self._user_item_sets.get(uid, set())
        if not u_set:
            return [], []

        similarities = []
        for vid in range(1, self.num_users + 1):
            if vid == uid:
                continue
            v_set = self._user_item_sets.get(vid, set())
            if not v_set:
                continue
            inter = len(u_set & v_set)
            union = len(u_set | v_set)
            sim = inter / union if union > 0 else 0.0
            similarities.append((vid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:top_k]
        return [v for v, _ in top], [s for _, s in top]

    def compute_item_similarity(
        self, iid: int, top_k: int = 20
    ) -> Tuple[List[int], List[float]]:
        """
        Find items most similar to `iid` based on Jaccard co-occurrence.
        """
        if self._jaccard_matrix is None:
            return [], []
        if iid >= len(self._jaccard_matrix):
            return [], []

        sims = self._jaccard_matrix[iid]
        # Get top-k (excluding self and padding 0)
        top_indices = np.argsort(-sims)
        result_items = []
        result_sims = []
        for idx in top_indices:
            if idx != iid and idx != 0 and sims[idx] > 0:
                result_items.append(int(idx))
                result_sims.append(float(sims[idx]))
            if len(result_items) >= top_k:
                break
        return result_items, result_sims

    def find_marginal_residuals(
        self, uid: int, forget_iid: int, cooc_threshold: float = 0.05
    ) -> Tuple[List[int], List[float]]:
        """
        Find retain items that are at risk of being marginal residuals.

        A retain item is a marginal residual candidate if:
        1. It is in the same user's retain_train (i.e., the user interacted with it)
        2. It has high co-occurrence (Jaccard) with the forget item

        These items share collaborative structure with the forget item
        and are at risk of being damaged during unlearning.
        """
        if self._jaccard_matrix is None:
            return [], []

        retain_set = self._user_item_sets.get(uid, set())
        if not retain_set or forget_iid >= len(self._jaccard_matrix):
            return [], []

        residuals = []
        for riid in retain_set:
            if riid == 0 or riid == forget_iid:
                continue
            jac = self._jaccard_matrix[forget_iid][riid]
            if jac >= cooc_threshold:
                residuals.append((riid, float(jac)))

        residuals.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in residuals], [s for _, s in residuals]

    # ------------------------------------------------------------------
    # 3. Score Margin Computation (requires trained model)
    # ------------------------------------------------------------------

    def compute_score_margin(
        self,
        uid: int,
        forget_iid: int,
        model,
        tokenizer,
        prompter,
        text_dict: Dict[int, str],
        args,
        candidate_size: int = 20,
    ) -> Dict:
        """
        Compute the impact of a forget item on the ranking boundary.

        Constructs two prompts for the same user:
          (a) With forget item in the candidate pool
          (b) Without forget item in the candidate pool

        Compares the score of the ground-truth (val) item in both cases.
        A large score drop when the forget item is present indicates strong
        collaborative interference.

        Returns dict with margin metrics.
        """
        from dataloader.llm import seq_to_token_ids

        if uid not in self.val or len(self.val[uid]) == 0:
            return {'score_margin': 0.0, 'forget_item_rank': -1, 'error': 'no val item'}

        positive_iid = self.val[uid][0]
        retain_seq = self.retain_train.get(uid, [])[-args.llm_max_history:]

        # Find similar items to serve as negative candidates
        similar_items, _ = self.compute_item_similarity(forget_iid, top_k=50)
        # Also include items from user's history that are not the positive
        history_items = [i for i in retain_seq
                         if i != positive_iid and i != forget_iid]

        # Build candidate pool: positive + forget + diverse negatives
        candidate_pool = [positive_iid]
        if forget_iid not in candidate_pool:
            candidate_pool.append(forget_iid)

        # Add negatives from similar items and history
        for iid in similar_items + history_items:
            if iid not in candidate_pool and iid != 0:
                candidate_pool.append(iid)
            if len(candidate_pool) >= candidate_size:
                break

        # If we don't have enough candidates, pad with random items
        if len(candidate_pool) < candidate_size:
            rng = np.random.RandomState(42)
            while len(candidate_pool) < candidate_size:
                riid = rng.randint(1, self.num_items + 1)
                if riid not in candidate_pool:
                    candidate_pool.append(riid)

        # Shuffle but keep positive at index 0 for tracking
        positive_idx_in_pool = candidate_pool.index(positive_iid)
        candidates_with = candidate_pool[:]

        # Build candidate pool WITHOUT forget item
        candidates_without = [i for i in candidates_with if i != forget_iid]
        # Replace forget item with another random item
        if len(candidates_without) < len(candidates_with):
            rng = np.random.RandomState(123)
            while len(candidates_without) < len(candidates_with):
                riid = rng.randint(1, self.num_items + 1)
                if riid not in candidates_without:
                    candidates_without.append(riid)

        # Compute scores with forget item
        device = next(model.parameters()).device
        model.eval()

        with torch.no_grad():
            # With forget item
            data_with = seq_to_token_ids(
                args, retain_seq, candidates_with, positive_iid,
                text_dict, tokenizer, prompter, eval=True
            )
            input_ids = torch.tensor([data_with['input_ids']]).to(device)
            attn_mask = torch.tensor([data_with['attention_mask']]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits_with = outputs.logits  # (1, vocab_size)

            # Without forget item
            data_without = seq_to_token_ids(
                args, retain_seq, candidates_without, positive_iid,
                text_dict, tokenizer, prompter, eval=True
            )
            input_ids = torch.tensor([data_without['input_ids']]).to(device)
            attn_mask = torch.tensor([data_without['attention_mask']]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits_without = outputs.logits  # (1, vocab_size)

        # Convert to candidate scores via verbalizer-like indexing
        # Letters A-T map to token IDs
        letter_to_idx = {}
        for idx, ch in enumerate([chr(ord('A') + i) for i in range(candidate_size)]):
            tok_id = tokenizer.encode(' ' + ch, add_special_tokens=False)[-1]
            letter_to_idx[idx] = tok_id

        scores_with = []
        scores_without = []
        for idx in range(len(candidates_with)):
            tid = letter_to_idx.get(idx)
            if tid is not None and tid < logits_with.shape[1]:
                scores_with.append(float(logits_with[0, tid].cpu()))
            else:
                scores_with.append(-1e9)

        for idx in range(len(candidates_without)):
            tid = letter_to_idx.get(idx)
            if tid is not None and tid < logits_without.shape[1]:
                scores_without.append(float(logits_without[0, tid].cpu()))
            else:
                scores_without.append(-1e9)

        # Compute metrics
        positive_score_with = scores_with[positive_idx_in_pool]
        forget_idx_with = candidates_with.index(forget_iid) if forget_iid in candidates_with else -1
        forget_score = scores_with[forget_idx_with] if forget_idx_with >= 0 else -1e9

        # Rank of forget item vs positive item
        sorted_with = np.argsort(-np.array(scores_with))
        forget_rank = int(np.where(sorted_with == forget_idx_with)[0][0]) + 1 if forget_idx_with >= 0 else -1
        positive_rank_with = int(np.where(sorted_with == positive_idx_in_pool)[0][0]) + 1

        # Score margin: how much does forget item "compete" with positive item
        score_margin = float(forget_score - positive_score_with)

        return {
            'score_margin': score_margin,
            'forget_item_score': float(forget_score),
            'positive_item_score': float(positive_score_with),
            'forget_item_rank_in_pool': forget_rank,
            'positive_item_rank_with_forget': positive_rank_with,
            'pool_size': len(candidates_with),
        }

    # ------------------------------------------------------------------
    # 4. Main Diagnosis Pipeline
    # ------------------------------------------------------------------

    def diagnose(
        self,
        model=None,
        tokenizer=None,
        prompter=None,
        text_dict=None,
        args=None,
        compute_score_margins: bool = False,
    ) -> Dict:
        """
        Run the full collaborative residual diagnosis.

        For each (user, forget_item) pair, computes:
        - cooc_strength: co-occurrence based residual intensity
        - marginal_residuals: list of retain items at risk
        - user_similarity: top similar users
        - item_similarity: top similar items
        - score_margin: ranking boundary impact (if model provided)
        - collaborative_residual_score: combined score (0-1)

        Parameters
        ----------
        model : optional, the trained LLM Ranker
        tokenizer : optional, tokenizer for the model
        prompter : optional, Prompter instance
        text_dict : optional, {iid: "title"} mapping
        args : optional, config args
        compute_score_margins : bool, whether to run model-based scoring

        Returns
        -------
        Dict with per-interaction results and summary statistics.
        """
        if self._cooc_matrix is None:
            self.build_collaborative_structures()

        print(f'\n[ResidualDiagnosis] Diagnosing {sum(len(v) for v in self.forget_train.values())} '
              f'forgotten interactions across {len(self.forget_train)} users...')

        all_results = []
        total = sum(len(v) for v in self.forget_train.values())

        # Global item statistics for normalization
        max_cooc = self._cooc_matrix.max() if self._cooc_matrix is not None else 1.0
        max_freq = self._item_freq.max() if self._item_freq is not None else 1.0

        pbar = tqdm(total=total, desc='  Diagnosing interactions')
        for uid in sorted(self.forget_train.keys()):
            for forget_iid in self.forget_train[uid]:
                result = self._diagnose_single(
                    uid, forget_iid, max_cooc, max_freq,
                    model, tokenizer, prompter, text_dict, args,
                    compute_score_margins
                )
                all_results.append(result)
                pbar.update(1)
        pbar.close()

        # Aggregate results
        self.results = {
            'per_interaction': all_results,
            'summary': self._compute_summary(all_results),
            'marginal_residual_stats': self._compute_marginal_stats(all_results),
        }

        return self.results

    def _diagnose_single(
        self, uid, forget_iid, max_cooc, max_freq,
        model, tokenizer, prompter, text_dict, args,
        compute_score_margins
    ) -> Dict:
        """Diagnose a single (user, forget_item) interaction."""
        # 1. Co-occurrence strength
        cooc_strength = 0.0
        if self._item_freq is not None and self._item_freq[forget_iid] > 0:
            # Average Jaccard similarity with all co-occurring items
            jac_row = self._jaccard_matrix[forget_iid]
            cooc_count = self._cooc_matrix[forget_iid]
            # Weight: number of co-occurring items normalized
            n_cooc = np.count_nonzero(cooc_count[1:])  # exclude padding
            mean_jac = jac_row[jac_row > 0].mean() if n_cooc > 0 else 0.0
            cooc_strength = float(mean_jac)

        # 2. Item centrality (how "hub-like" is this item)
        item_popularity = float(
            self._item_freq[forget_iid] / max_freq
        ) if max_freq > 0 and self._item_freq is not None else 0.0

        # 3. Similar users
        sim_users, sim_user_scores = self.compute_user_similarity(uid, top_k=10)
        mean_user_sim = float(np.mean(sim_user_scores)) if sim_user_scores else 0.0

        # 4. Similar items
        sim_items, sim_item_scores = self.compute_item_similarity(forget_iid, top_k=20)
        mean_item_sim = float(np.mean(sim_item_scores)) if sim_item_scores else 0.0

        # 5. Marginal residuals (retain items at risk)
        marginal_residuals, marginal_scores = self.find_marginal_residuals(
            uid, forget_iid, cooc_threshold=0.02
        )
        n_marginal = len(marginal_residuals)

        # 6. User history context
        retain_seq = self.retain_train.get(uid, [])
        # Find original position of forget_iid in the full train (if we had it)
        history_len = len(retain_seq)
        history_density = min(1.0, history_len / 200.0)  # normalized by typical max

        # 7. Score margin (if model available)
        score_margin_info = {}
        if compute_score_margins and model is not None:
            try:
                score_margin_info = self.compute_score_margin(
                    uid, forget_iid, model, tokenizer, prompter,
                    text_dict or {}, args
                )
            except Exception as e:
                score_margin_info = {'score_margin': 0.0, 'error': str(e)}

        score_margin = score_margin_info.get('score_margin', 0.0)

        # 8. Combined collaborative residual score
        # Higher = stronger collaborative residual = higher priority for pruning
        # Components: co-occurrence strength, item popularity, marginal residual count,
        #             history density, and score margin
        collaborative_residual_score = (
            0.30 * cooc_strength +
            0.15 * item_popularity +
            0.15 * mean_user_sim +
            0.15 * min(1.0, n_marginal / 10.0) +
            0.10 * history_density +
            0.15 * min(1.0, abs(score_margin) / 10.0)  # normalized score margin
        )
        collaborative_residual_score = float(np.clip(collaborative_residual_score, 0.0, 1.0))

        return {
            'user_id': int(uid),
            'forget_item_id': int(forget_iid),
            'forget_item_title': self.meta.get(forget_iid, 'N/A'),
            # Co-occurrence metrics
            'cooc_strength': cooc_strength,
            'item_popularity': item_popularity,
            'n_cooccurring_items': int(
                np.count_nonzero(self._cooc_matrix[forget_iid][1:])
            ) if self._cooc_matrix is not None else 0,
            # User similarity
            'mean_user_similarity': mean_user_sim,
            'top_similar_users': sim_users[:5],
            # Item similarity
            'mean_item_similarity': mean_item_sim,
            'top_similar_items': sim_items[:10],
            # Marginal residuals
            'n_marginal_residuals': n_marginal,
            'marginal_residuals': marginal_residuals[:20],
            'marginal_residual_scores': marginal_scores[:20],
            # History context
            'history_length': history_len,
            'history_density': history_density,
            # Score margin
            'score_margin': score_margin,
            'score_margin_detail': score_margin_info,
            # Combined score
            'collaborative_residual_score': collaborative_residual_score,
        }

    # ------------------------------------------------------------------
    # 5. Summary and Statistics
    # ------------------------------------------------------------------

    def _compute_summary(self, all_results: List[Dict]) -> Dict:
        """Compute aggregate statistics over all diagnosed interactions."""
        scores = [r['collaborative_residual_score'] for r in all_results]
        cooc_strengths = [r['cooc_strength'] for r in all_results]
        n_marginals = [r['n_marginal_residuals'] for r in all_results]
        score_margins = [r['score_margin'] for r in all_results
                         if r['score_margin_detail'].get('error') is None]

        return {
            'total_interactions': len(all_results),
            'total_users': len(set(r['user_id'] for r in all_results)),
            'collaborative_residual_score': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
            },
            'cooc_strength': {
                'mean': float(np.mean(cooc_strengths)),
                'std': float(np.std(cooc_strengths)),
                'max': float(np.max(cooc_strengths)),
            },
            'marginal_residuals_per_interaction': {
                'mean': float(np.mean(n_marginals)),
                'std': float(np.std(n_marginals)),
                'max': int(np.max(n_marginals)),
                'total_distinct_marginal_items': len(set(
                    i for r in all_results for i in r['marginal_residuals']
                )),
            },
            'score_margin': {
                'mean': float(np.mean(score_margins)) if score_margins else 0.0,
                'std': float(np.std(score_margins)) if score_margins else 0.0,
                'max_positive': float(np.max(score_margins)) if score_margins else 0.0,
                'n_computed': len(score_margins),
            },
        }

    def _compute_marginal_stats(self, all_results: List[Dict]) -> Dict:
        """Aggregate statistics about marginal residual items."""
        # Count how many times each retain item appears as a marginal residual
        item_marginal_count = defaultdict(int)
        item_max_residual_score = defaultdict(float)

        for r in all_results:
            for iid, score in zip(r['marginal_residuals'],
                                   r['marginal_residual_scores']):
                item_marginal_count[iid] += 1
                item_max_residual_score[iid] = max(
                    item_max_residual_score[iid], score
                )

        # Top items most frequently flagged as marginal residuals
        top_marginal = sorted(item_marginal_count.items(),
                              key=lambda x: x[1], reverse=True)[:50]

        return {
            'distinct_marginal_items': len(item_marginal_count),
            'top_marginal_items': [
                {
                    'item_id': iid,
                    'title': self.meta.get(iid, 'N/A'),
                    'times_flagged': count,
                    'max_cooc_with_forget': float(item_max_residual_score[iid]),
                }
                for iid, count in top_marginal
            ],
        }

    # ------------------------------------------------------------------
    # 6. I/O
    # ------------------------------------------------------------------

    def save_results(self, path: str):
        """Save diagnosis results to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Convert any numpy types for JSON serialization
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=_json_serialize)
        print(f'[ResidualDiagnosis] Results saved to {path}')

    def load_results(self, path: str):
        """Load previously computed diagnosis results."""
        with open(path, 'r') as f:
            self.results = json.load(f)
        print(f'[ResidualDiagnosis] Results loaded from {path}')

    def get_high_residual_interactions(
        self, threshold: float = 0.5
    ) -> List[Dict]:
        """Return interactions with collaborative_residual_score >= threshold."""
        if not self.results:
            return []
        return [r for r in self.results['per_interaction']
                if r['collaborative_residual_score'] >= threshold]

    def get_marginal_residual_items(
        self, min_times_flagged: int = 1
    ) -> Dict[int, int]:
        """Return {item_id: times_flagged} for all marginal residual items."""
        counts = defaultdict(int)
        if not self.results:
            return dict(counts)
        for r in self.results['per_interaction']:
            for iid in r['marginal_residuals']:
                counts[iid] += 1
        return {k: v for k, v in counts.items() if v >= min_times_flagged}

class InteractionResidualDiagnosis:
    """Interaction-level marginal residual diagnosis for geometry_prune."""

    def __init__(self, split_data, dataset_data, predictions_before, model, tokenizer, args):
        self.split_data = split_data
        self.dataset_data = dataset_data
        self.predictions_before = predictions_before
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.retain_train = dataset_data.get('retain_train', dataset_data.get('train', {}))
        self.user_sets = {int(u): set(seq) for u, seq in self.retain_train.items()}
        self.item_users = self._build_item_users()
        self.item_freq = {iid: len(users) for iid, users in self.item_users.items()}
        self.max_freq = max(self.item_freq.values()) if self.item_freq else 1
        self.calibrator = CounterfactualBoundaryCalibrator(
            model=model,
            tokenizer=tokenizer,
            dataset_data=dataset_data,
            args=args,
        )
        self.logs = {
            'diagnosis': 'interaction_level_marginal_residual',
            'residual_formula': (
                'alpha*collaborative_proximity + beta*boundary_sensitivity '
                '+ gamma*counterfactual_residual'
            ),
            'counterfactual_logs': self.calibrator.logs,
        }

    def _build_item_users(self):
        item_users = defaultdict(set)
        for uid, seq in self.retain_train.items():
            for iid in set(seq):
                item_users[int(iid)].add(int(uid))
        return dict(item_users)

    @staticmethod
    def _row_uid(row):
        return int(row.get('uid', row.get('user_id')))

    @staticmethod
    def _row_iid(row):
        return int(row.get('iid', row.get('item_id')))

    def _predictions_by_forget(self):
        result = {}
        for rec in self.predictions_before.get('records', []):
            if rec.get('split_tag') == 'forget':
                result[(int(rec['uid']), int(rec['target_iid']))] = rec
        return result

    def _item_jaccard(self, a, b):
        ua = self.item_users.get(int(a), set())
        ub = self.item_users.get(int(b), set())
        if not ua or not ub:
            return 0.0
        return len(ua & ub) / len(ua | ub)

    def _user_user_overlap(self, uid, candidate_iid):
        u_items = self.user_sets.get(int(uid), set())
        candidate_users = self.item_users.get(int(candidate_iid), set())
        if not u_items or not candidate_users:
            return 0.0
        overlaps = []
        for vid in candidate_users:
            if int(vid) == int(uid):
                continue
            v_items = self.user_sets.get(int(vid), set())
            if v_items:
                overlaps.append(len(u_items & v_items) / len(u_items | v_items))
        return float(np.mean(overlaps)) if overlaps else 0.0

    def _popularity_adjusted_cooc(self, forget_iid, candidate_iid):
        jac = self._item_jaccard(forget_iid, candidate_iid)
        pop = self.item_freq.get(int(candidate_iid), 0) / max(self.max_freq, 1)
        return float(jac / np.sqrt(pop + 1e-6))

    def collaborative_proximity(self, uid, forget_iid, candidate_iid):
        same_user = float(candidate_iid in self.user_sets.get(int(uid), set()))
        item_jac = self._item_jaccard(forget_iid, candidate_iid)
        user_overlap = self._user_user_overlap(uid, candidate_iid)
        pop_adj = self._popularity_adjusted_cooc(forget_iid, candidate_iid)
        pop_adj = min(1.0, pop_adj)
        return float(np.clip(
            0.30 * same_user +
            0.35 * item_jac +
            0.20 * user_overlap +
            0.15 * pop_adj,
            0.0, 1.0,
        ))

    def diagnose(self):
        records = []
        pred_by_forget = self._predictions_by_forget()
        cf_failures = 0
        cf_success = 0

        for forget in self.split_data.get('forget_interactions', []):
            uid = self._row_uid(forget)
            forget_iid = self._row_iid(forget)
            pred = pred_by_forget.get((uid, forget_iid))
            if not pred:
                continue

            scores_cf, ok, err = self.calibrator.recompute_scores(pred)
            if ok:
                cf_success += 1
            else:
                cf_failures += 1
                self.logs.setdefault('counterfactual_errors', []).append(err)

            before_scores = pred.get('scores', {})
            for candidate_iid in pred.get('candidate_items', []):
                candidate_iid = int(candidate_iid)
                if candidate_iid == int(forget_iid):
                    continue
                score_before = float(before_scores.get(str(candidate_iid), 0.0))
                score_counterfactual = float(scores_cf.get(str(candidate_iid), score_before))
                cf_residual = abs(score_before - score_counterfactual)
                collab = self.collaborative_proximity(uid, forget_iid, candidate_iid)
                boundary = self.calibrator.boundary_sensitivity({
                    **pred,
                    'target_iid': candidate_iid,
                    'target_rank': pred.get('ranks', {}).get(str(candidate_iid), 10**6),
                    'target_score': score_before,
                })
                raw_score = (
                    self.args.residual_alpha * collab +
                    self.args.residual_beta * boundary +
                    self.args.residual_gamma * cf_residual
                )
                records.append({
                    'uid': uid,
                    'forget_iid': int(forget_iid),
                    'candidate_iid': candidate_iid,
                    'collaborative_proximity': float(collab),
                    'boundary_sensitivity': float(boundary),
                    'counterfactual_residual': float(cf_residual),
                    'marginal_residual_score': float(raw_score),
                    'rank_before': int(pred.get('ranks', {}).get(str(candidate_iid), 10**6)),
                    'score_before': score_before,
                    'score_counterfactual': score_counterfactual,
                    'margin_to_topk_boundary': float(pred.get('margin_to_topk_boundary', 0.0)),
                })

        if records:
            vals = np.array([r['marginal_residual_score'] for r in records], dtype=np.float32)
            vmin, vmax = vals.min(), vals.max()
            for rec in records:
                rec['marginal_residual_score_raw'] = rec['marginal_residual_score']
                rec['marginal_residual_score'] = float(
                    0.0 if vmax <= vmin else
                    (rec['marginal_residual_score_raw'] - vmin) / (vmax - vmin)
                )

        records.sort(key=lambda r: r['marginal_residual_score'], reverse=True)
        self.logs['counterfactual_success'] = cf_success
        self.logs['counterfactual_failures'] = cf_failures
        self.logs['num_marginal_candidates'] = len(records)
        return {
            'records': records,
            'logs': self.logs,
        }


def _json_serialize(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
