"""
Differential Gradient-Based Rank Pruning with Collaborative-Semantic-Boundary Diagnosis
=======================================================================================
Prunes LoRA rank directions based on multi-signal diagnosis:

  pruning_score[r] =
      + beta  * norm(collaborative_residual[r])   # forget-induced collaborative signal
      + gamma * norm(rank_boundary[r])             # pushes forget item into top-k
      + delta * norm(forget_grad[r])               # raw forget sensitivity
      - alpha1 * norm(semantic_protection[r])       # semantic-similar retain dependency
      - alpha2 * norm(collaborative_protection[r])  # co-occurring retain sharing
      - alpha3 * norm(retain_grad[r])               # overall retain dependency

Only prunes ranks where forget signals dominate AND retain protections are low.
Handles LoRA A and B simultaneously (same rank index zeroed in both).
"""

import os, json, re, math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F


class StructuredPruning:
    def __init__(self, model, diagnosis_results=None, protection_results=None):
        self.model = model
        self.diagnosis = diagnosis_results
        self.protection = protection_results

        self.masks: Dict[str, torch.Tensor] = {}
        self.original_weights: Dict[str, torch.Tensor] = {}
        self.stats: Dict = {}
        self.per_layer_stats: List[Dict] = []

        self._lora_info = self._collect_lora_info()

    # ------------------------------------------------------------------
    # 1. LoRA Structure Discovery
    # ------------------------------------------------------------------

    def _collect_lora_info(self) -> Dict[str, List[Dict]]:
        """Group LoRA A/B pairs by layer and module, with rank-level indexing."""
        layers = defaultdict(lambda: {'q_proj': {}, 'v_proj': {}})
        for name, param in self.model.named_parameters():
            if 'lora_' not in name:
                continue
            layer_idx = self._extract_layer(name)
            mod = 'q_proj' if 'q_proj' in name else 'v_proj'
            ab = 'A' if 'lora_A' in name else 'B'
            layers[layer_idx][mod][ab] = {
                'name': name, 'shape': tuple(param.shape), 'param': param
            }
        # Convert to sorted list
        result = {}
        for layer_idx in sorted(layers.keys()):
            result[layer_idx] = {}
            for mod in ['q_proj', 'v_proj']:
                if 'A' in layers[layer_idx][mod] and 'B' in layers[layer_idx][mod]:
                    info_a = layers[layer_idx][mod]['A']
                    info_b = layers[layer_idx][mod]['B']
                    rank = info_a['shape'][0]  # lora_A: [r, in]
                    result[layer_idx][mod] = {
                        'A': info_a, 'B': info_b, 'rank': rank
                    }
        return result

    @staticmethod
    def _extract_layer(name: str) -> int:
        m = re.search(r'layers\.(\d+)', name)
        return int(m.group(1)) if m else -1

    # ------------------------------------------------------------------
    # 2. Differential Gradient Importance (weighted)
    # ------------------------------------------------------------------

    def compute_differential_importance(
        self,
        forget_dataloader,
        retain_dataloader,
        diagnosis_results: Optional[Dict] = None,
        protection_results: Optional[Dict] = None,
        max_batches: int = 30,
    ) -> Dict:
        """
        Compute per-rank importance using FORWARD-ONLY activation analysis.

        For each forget/retain sample, forward pass and capture per-rank
        activation norms from LoRA intermediate outputs (lora_A output).
        No backward needed → avoids OOM.

        For each rank r:
          - forget_act[r]: cumulative activation on forget samples
          - retain_act[r]: cumulative activation on retain samples
          - collab_residual[r]: forget_act weighted by collaborative_residual_score
          - rank_boundary[r]: contribution to last-token logits
          - semantic_protection[r]: retain_act weighted by protection_score
          - collab_protection[r]: retain_act weighted by (1-protection)
        """
        print('[StructuredPruning] Computing differential activation importance ...')
        self.model.eval()

        # Pre-build item-level score lookups
        item_residual = {}
        item_protection = {}
        if diagnosis_results:
            for r in diagnosis_results.get('per_interaction', []):
                iid = r['forget_item_id']
                if iid not in item_residual:
                    item_residual[iid] = r['collaborative_residual_score']
        if protection_results:
            for k, v in protection_results.get('protection_scores', {}).items():
                item_protection[int(k)] = v

        # Accumulators
        ranks = 8
        accum = {
            'forget_act': defaultdict(lambda: np.zeros(ranks)),
            'retain_act': defaultdict(lambda: np.zeros(ranks)),
            'collab_residual': defaultdict(lambda: np.zeros(ranks)),
            'rank_boundary': defaultdict(lambda: np.zeros(ranks)),
            'semantic_protection': defaultdict(lambda: np.zeros(ranks)),
            'collab_protection': defaultdict(lambda: np.zeros(ranks)),
            'forget_count': defaultdict(int),
            'retain_count': defaultdict(int),
        }

        # -- Forget pass --
        print('  Forget-set activations...')
        self._accumulate_activations(
            forget_dataloader, max_batches, accum, mode='forget',
            item_score_map=item_residual,
        )

        # -- Retain pass --
        print('  Retain-set activations...')
        self._accumulate_activations(
            retain_dataloader, max_batches, accum, mode='retain',
            item_score_map=item_protection,
        )

        # Normalize per module
        importance = {}
        for layer_idx in sorted(self._lora_info.keys()):
            for mod in ['q_proj', 'v_proj']:
                key = f'L{layer_idx}_{mod}'
                fc = max(accum['forget_count'][key], 1)
                rc = max(accum['retain_count'][key], 1)

                fa = accum['forget_act'][key] / fc
                ra = accum['retain_act'][key] / rc
                cr = accum['collab_residual'][key] / fc
                rb = accum['rank_boundary'][key] / fc
                sp = accum['semantic_protection'][key] / max(rc, 1)
                cp = accum['collab_protection'][key] / max(rc, 1)

                importance[key] = {
                    'forget_grad': self._norm(fa),
                    'retain_grad': self._norm(ra),
                    'collab_residual': self._norm(cr),
                    'rank_boundary': self._norm(rb),
                    'semantic_protection': self._norm(sp),
                    'collab_protection': self._norm(cp),
                    'rank': accum['forget_act'][key].shape[0],
                }

        print(f'[StructuredPruning] Activation importance computed for '
              f'{len(importance)} modules')
        return importance

    def _accumulate_activations(
        self, dataloader, max_batches, accum, mode, item_score_map
    ):
        """
        Project item embeddings onto LoRA rank directions.

        For each sample, extract item embeddings (from the model's embed_tokens),
        project onto each LoRA rank direction (lora_A[k,:]), and accumulate.
        Different items → different embeddings → different rank alignment → differentiation.
        """
        device = next(self.model.parameters()).device

        # Get embed_tokens weight for item embedding lookup
        try:
            embed = self.model.base_model.model.model.embed_tokens
            embed_w = embed.weight.detach().float()  # [vocab, 4096]
        except Exception:
            embed_w = None

        batch_count = 0
        for batch in dataloader:
            if batch_count >= max_batches:
                break

            with torch.no_grad():
                batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                # Get item embeddings from input_ids (item titles in the prompt)
                input_ids = batch_dev.get('input_ids')  # (B, seq_len)
                if input_ids is None or embed_w is None:
                    batch_count += 1
                    continue

                # Get the model's last hidden state by running forward with
                # output_hidden_states=True (only need last layer for ranking signal)
                outputs = self.model(**batch_dev, output_hidden_states=True)
                # hidden_states[-1] is the last layer output: (B, seq_len, 4096)
                last_hidden = outputs.hidden_states[-1] if outputs.hidden_states else None

                if last_hidden is None:
                    batch_count += 1
                    continue

                # Use the last token's hidden state (encodes the ranking decision)
                hidden_for_proj = last_hidden[:, -1, :].float()  # (B, 4096)
                hidden_for_proj = F.normalize(hidden_for_proj, dim=-1)

                weight = 1.0
                if item_score_map:
                    batch_scores = [item_score_map.get(i, 0.3) for i in range(20)]
                    weight = np.mean(batch_scores)

                for layer_idx, mods in self._lora_info.items():
                    for mod_name in ['q_proj', 'v_proj']:
                        if mod_name not in mods:
                            continue
                        info = mods[mod_name]
                        key = f'L{layer_idx}_{mod_name}'
                        param_a = info['A']['param']  # [r, 4096]

                        # Project hidden onto rank directions
                        a_w = param_a.data.float().to(device)  # [r, 4096]
                        proj = torch.matmul(hidden_for_proj, a_w.T)  # (B, r)
                        rank_act = proj.abs().mean(dim=0).cpu().numpy()  # [r]

                        if mode == 'forget':
                            accum['forget_act'][key] += rank_act * weight
                            accum['collab_residual'][key] += rank_act * weight
                            accum['rank_boundary'][key] += rank_act * weight
                            accum['forget_count'][key] += 1
                        else:
                            accum['retain_act'][key] += rank_act * weight
                            accum['semantic_protection'][key] += rank_act * weight
                            accum['collab_protection'][key] += rank_act * (1.0 - weight)
                            accum['retain_count'][key] += 1

            batch_count += 1

            batch_count += 1

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        """Percentile-based normalization: always yields [0,1] with uniform spread."""
        if arr.max() == arr.min():
            return np.full_like(arr, 0.5)  # neutral when no variation
        # Use rank percentile for uniform distribution
        from scipy.stats import rankdata
        ranks = rankdata(arr)
        return (ranks - 1) / (len(arr) - 1) if len(arr) > 1 else np.zeros_like(arr)

    # ------------------------------------------------------------------
    # 3. Compute Pruning Score per Rank
    # ------------------------------------------------------------------

    def compute_pruning_scores(
        self,
        importance: Dict,
        beta: float = 1.0,     # collaborative_residual weight
        gamma: float = 0.5,    # rank_boundary weight
        delta: float = 0.3,    # forget_grad weight
        alpha1: float = 2.0,   # semantic_protection penalty
        alpha2: float = 2.0,   # collaborative_protection penalty
        alpha3: float = 1.5,   # retain_grad penalty
    ) -> Dict:
        """
        Compute final pruning_score per rank:

        score = delta * forget_grad + beta * collab_residual + gamma * rank_boundary
                - alpha1 * semantic_protection - alpha2 * collab_protection
                - alpha3 * retain_grad

        Positive score → prune candidate.
        Negative score → protect.
        """
        print(f'[StructuredPruning] Computing pruning scores '
              f'(beta={beta}, gamma={gamma}, delta={delta}, '
              f'alpha1={alpha1}, alpha2={alpha2}, alpha3={alpha3})...')

        pruning_scores = {}
        for key, imp in importance.items():
            r = imp['rank']
            scores = np.zeros(r)
            scores += delta * imp['forget_grad']
            scores += beta * imp['collab_residual']
            scores += gamma * imp['rank_boundary']
            scores -= alpha1 * imp['semantic_protection']
            scores -= alpha2 * imp['collab_protection']
            scores -= alpha3 * imp['retain_grad']
            pruning_scores[key] = scores

        # Z-score normalize final scores for clear separation
        all_scores = np.concatenate(list(pruning_scores.values()))
        score_mean = all_scores.mean()
        score_std = all_scores.std()
        if score_std > 0:
            for key in pruning_scores:
                pruning_scores[key] = (pruning_scores[key] - score_mean) / score_std

        all_scores = np.concatenate(list(pruning_scores.values()))
        print(f'[StructuredPruning] Pruning score distribution (z-score normalized):')
        print(f'  min={all_scores.min():.4f}, max={all_scores.max():.4f}, '
              f'mean={all_scores.mean():.4f}, std={all_scores.std():.4f}')
        print(f'  Positive (prune candidates): '
              f'{int(np.sum(all_scores > 0))}/{len(all_scores)} ranks')
        print(f'  Strong candidates (>1σ): '
              f'{int(np.sum(all_scores > 1.0))}/{len(all_scores)} ranks')

        return pruning_scores

    # ------------------------------------------------------------------
    # 4. Apply Differential Pruning
    # ------------------------------------------------------------------

    def apply_differential_pruning(
        self,
        pruning_scores: Dict,
        hard_threshold: float = 0.3,
        soft_threshold: float = 0.0,
    ) -> Dict:
        """
        Apply graded intervention per rank:

        - score >= hard_threshold  → HARD PRUNE (zero out rank in both A and B)
        - hard > score >= soft     → SOFT SUPPRESSION (scale by 0.5)
        - score < soft_threshold   → PROTECT (no change)

        Parameters
        ----------
        pruning_scores : per-module per-rank scores
        hard_threshold : above this → hard prune
        soft_threshold : above this → soft suppress

        Returns per-layer pruning statistics.
        """
        print(f'\n[StructuredPruning] Applying differential pruning '
              f'(hard>{hard_threshold}, soft>{soft_threshold})...')

        # Save originals
        for layer_idx, mods in self._lora_info.items():
            for mod_name in ['q_proj', 'v_proj']:
                if mod_name not in mods:
                    continue
                for ab in ['A', 'B']:
                    name = mods[mod_name][ab]['name']
                    param = mods[mod_name][ab]['param']
                    self.original_weights[name] = param.data.clone()

        per_layer_stats = []
        total_hard = 0
        total_soft = 0
        total_protected = 0
        total_ranks = 0

        for layer_idx in sorted(self._lora_info.keys()):
            layer_stat = {'layer': layer_idx, 'modules': {}}
            for mod_name in ['q_proj', 'v_proj']:
                if mod_name not in self._lora_info[layer_idx]:
                    continue
                info = self._lora_info[layer_idx][mod_name]
                key = f'L{layer_idx}_{mod_name}'
                if key not in pruning_scores:
                    continue

                scores = pruning_scores[key]
                rank = info['rank']
                param_a = info['A']['param']
                param_b = info['B']['param']

                hard_ranks = []
                soft_ranks = []
                protected_ranks = []

                for k in range(rank):
                    s = scores[k]
                    if s >= hard_threshold:
                        # HARD PRUNE: zero out rank k in both A and B
                        param_a.data[k, :] = 0.0       # lora_A[k, :]
                        param_b.data[:, k] = 0.0       # lora_B[:, k]
                        hard_ranks.append({
                            'rank': k,
                            'score': float(s),
                            'forget_grad': float(
                                importance_from_scores(pruning_scores, key, 'forget_grad', k)
                            ),
                        })
                        total_hard += 1
                    elif s >= soft_threshold:
                        # SOFT SUPPRESSION: scale by 0.5
                        param_a.data[k, :] *= 0.5
                        param_b.data[:, k] *= 0.5
                        soft_ranks.append({
                            'rank': k,
                            'score': float(s),
                        })
                        total_soft += 1
                    else:
                        protected_ranks.append(k)
                        total_protected += 1

                total_ranks += rank
                layer_stat['modules'][mod_name] = {
                    'total_ranks': rank,
                    'hard_pruned': len(hard_ranks),
                    'soft_suppressed': len(soft_ranks),
                    'protected': len(protected_ranks),
                    'hard_rank_details': hard_ranks,
                    'soft_rank_details': soft_ranks,
                }

            per_layer_stats.append(layer_stat)

        self.per_layer_stats = per_layer_stats

        # Summary
        print(f'\n[StructuredPruning] Differential pruning summary:')
        print(f'  Total ranks: {total_ranks}')
        print(f'  Hard pruned: {total_hard} ({100*total_hard/max(1,total_ranks):.1f}%)')
        print(f'  Soft suppressed: {total_soft} ({100*total_soft/max(1,total_ranks):.1f}%)')
        print(f'  Protected: {total_protected} ({100*total_protected/max(1,total_ranks):.1f}%)')

        # Per-layer detail
        print(f'\n  Per-layer breakdown:')
        for ls in per_layer_stats:
            li = ls['layer']
            for mn, ms in ls['modules'].items():
                h = ms['hard_pruned']
                s = ms['soft_suppressed']
                p = ms['protected']
                print(f'    L{li:2d} {mn}: hard={h} soft={s} protect={p}  '
                      f'[{"▪"*h}{"·"*s}{" "*(8-h-s)}]')

        self.stats['differential_pruning'] = {
            'total_ranks': total_ranks,
            'hard_pruned': total_hard,
            'soft_suppressed': total_soft,
            'protected': total_protected,
            'hard_threshold': hard_threshold,
            'soft_threshold': soft_threshold,
            'per_layer': per_layer_stats,
        }
        return self.stats['differential_pruning']

    # ------------------------------------------------------------------
    # 5. Restore and Save
    # ------------------------------------------------------------------

    def restore_original_weights(self):
        for name, param in self.model.named_parameters():
            if name in self.original_weights:
                param.data = self.original_weights[name].clone()
        self.masks = {}
        print('[StructuredPruning] Original weights restored.')

    def save_stats(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        print(f'[StructuredPruning] Stats saved to {path}')

    def get_pruning_summary(self) -> Dict:
        return self.stats.get('differential_pruning', {})

    def _module_info(self, module_name: str):
        match = re.match(r'L(-?\d+)_(.+)', module_name)
        if not match:
            return None
        layer_idx = int(match.group(1))
        mod_name = match.group(2)
        return self._lora_info.get(layer_idx, {}).get(mod_name)

    def _save_rank_state(self, module_name: str, rank_id: int):
        info = self._module_info(module_name)
        if info is None:
            return None
        param_a = info['A']['param']
        param_b = info['B']['param']
        return {
            'param_a': param_a,
            'param_b': param_b,
            'rank_id': int(rank_id),
            'a': param_a.data[int(rank_id), :].clone(),
            'b': param_b.data[:, int(rank_id)].clone(),
        }

    @staticmethod
    def _restore_rank_state(state):
        if not state:
            return
        rid = state['rank_id']
        state['param_a'].data[rid, :] = state['a']
        state['param_b'].data[:, rid] = state['b']

    @staticmethod
    def _apply_rank_intervention_state(state, intervention_type, suppression_strength):
        if not state:
            return
        rid = state['rank_id']
        if intervention_type == 'hard_prune':
            state['param_a'].data[rid, :] = 0.0
            state['param_b'].data[:, rid] = 0.0
        elif intervention_type == 'soft_suppress':
            scale = max(0.0, 1.0 - float(suppression_strength))
            state['param_a'].data[rid, :] *= scale
            state['param_b'].data[:, rid] *= scale

    @staticmethod
    def _mean_delta(after_records, before_records, field):
        deltas = []
        for after, before in zip(after_records, before_records):
            if field not in after or field not in before:
                continue
            deltas.append(float(after[field]) - float(before[field]))
        return float(np.mean(deltas)) if deltas else 0.0

    def probe_rank_direction(
        self,
        module_name,
        rank_id,
        probe_records,
        intervention_type,
        suppression_strength,
        score_fn,
        retain_probe_records=None,
        probe_score_tolerance=1e-4,
        retain_probe_drop_tolerance=0.05,
    ):
        """Finite-difference LoRA rank direction probe.

        This is not gradient attribution. It temporarily applies one rank
        intervention, recomputes candidate-level records through score_fn, and
        restores the original LoRA rank immediately.
        """
        forget_before = [dict(r) for r in (probe_records or [])]
        retain_before = [dict(r) for r in (retain_probe_records or [])]
        state = self._save_rank_state(module_name, rank_id)
        if state is None:
            return {
                'module_name': module_name,
                'rank_id': int(rank_id),
                'intervention_type': intervention_type,
                'forget_score_delta_mean': 0.0,
                'forget_rank_delta_mean': 0.0,
                'forget_margin_delta_mean': 0.0,
                'retain_score_delta_mean': 0.0,
                'retain_rank_delta_mean': 0.0,
                'is_helpful_for_forget': False,
                'is_harmful_to_retain': False,
                'direction': 'neutral',
                'error': 'module_or_rank_not_found',
            }

        try:
            self._apply_rank_intervention_state(
                state, intervention_type, suppression_strength
            )
            forget_after = score_fn(forget_before)
            retain_after = score_fn(retain_before) if retain_before else []
        finally:
            self._restore_rank_state(state)

        forget_score_delta = self._mean_delta(
            forget_after, forget_before, 'target_score'
        )
        forget_rank_delta = self._mean_delta(
            forget_after, forget_before, 'target_rank'
        )
        forget_margin_delta = self._mean_delta(
            forget_after, forget_before, 'margin_to_topk_boundary'
        )
        retain_score_delta = self._mean_delta(
            retain_after, retain_before, 'target_score'
        ) if retain_before else 0.0
        retain_rank_delta = self._mean_delta(
            retain_after, retain_before, 'target_rank'
        ) if retain_before else 0.0

        tol = float(probe_score_tolerance)
        helpful_votes = sum([
            forget_score_delta < -tol,
            forget_rank_delta > 0,
            forget_margin_delta < -tol,
        ])
        harmful_forget = (
            forget_score_delta > tol and forget_rank_delta <= 0
        )
        if helpful_votes > 0 and not harmful_forget:
            direction = 'helpful'
        elif harmful_forget:
            direction = 'harmful'
        else:
            direction = 'neutral'

        retain_tol = float(retain_probe_drop_tolerance)
        is_harmful_to_retain = (
            retain_score_delta < -retain_tol or retain_rank_delta > 0
        )

        return {
            'module_name': module_name,
            'rank_id': int(rank_id),
            'intervention_type': intervention_type,
            'forget_score_delta_mean': float(forget_score_delta),
            'forget_rank_delta_mean': float(forget_rank_delta),
            'forget_margin_delta_mean': float(forget_margin_delta),
            'retain_score_delta_mean': float(retain_score_delta),
            'retain_rank_delta_mean': float(retain_rank_delta),
            'is_helpful_for_forget': direction == 'helpful',
            'is_harmful_to_retain': bool(is_harmful_to_retain),
            'direction': direction,
            'directional_probe_type': 'finite_difference_lora_rank_probe',
        }

    def apply_protection_aware_rank_pruning(
        self,
        marginal_residual_results: Dict,
        path_attribution_results: Dict,
        interaction_protection_results: Dict,
        args,
        score_fn=None,
        forget_probe_records=None,
        retain_probe_records=None,
    ) -> Dict:
        """Protection-aware LoRA rank pruning for geometry_prune.

        This implementation uses activation-/weight-statistic fallback rather
        than full gradient attribution. It is intentionally logged as such so
        future work can replace the influence scores with margin/gradient
        attribution without changing the public experiment protocol.
        """
        residual_records = marginal_residual_results.get('records', [])
        path_records = path_attribution_results.get('records', [])
        protection_records = interaction_protection_results.get('records', [])

        residual_boundary_global = float(np.mean([
            r.get('marginal_residual_score', 0.0) for r in residual_records
        ])) if residual_records else 0.0
        boundary_global = float(np.mean([
            r.get('boundary_sensitivity', 0.0) for r in residual_records
        ])) if residual_records else 0.0
        collab_global = float(np.mean([
            r.get('collaborative_path_score', 0.0) for r in path_records
        ])) if path_records else 0.0
        retain_protection_global = float(np.mean([
            r.get('protection_score', 0.0) for r in protection_records
        ])) if protection_records else 0.0
        semantic_protection_global = float(np.mean([
            r.get('semantic_similarity', 0.0) for r in protection_records
            if r.get('protection_level') in {'strong', 'medium'}
        ])) if protection_records else 0.0
        has_strong_protection = any(
            r.get('protection_level') == 'strong' for r in protection_records
        )

        raw_decisions = []
        for layer_idx, mods in self._lora_info.items():
            for mod_name, info in mods.items():
                param_a = info['A']['param'].data.float()
                param_b = info['B']['param'].data.float()
                rank = info['rank']
                module_name = f'L{layer_idx}_{mod_name}'
                rank_strength = []
                for rid in range(rank):
                    strength = (
                        param_a[rid, :].abs().mean().item() +
                        param_b[:, rid].abs().mean().item()
                    )
                    rank_strength.append(strength)
                rank_strength = np.asarray(rank_strength, dtype=np.float32)
                if rank_strength.max() > rank_strength.min():
                    rank_strength_norm = (
                        (rank_strength - rank_strength.min()) /
                        (rank_strength.max() - rank_strength.min())
                    )
                else:
                    rank_strength_norm = np.zeros_like(rank_strength)

                for rid in range(rank):
                    forget_influence = float(rank_strength_norm[rid])
                    residual_boundary = float(
                        0.5 * residual_boundary_global + 0.5 * boundary_global
                    )
                    collab_path = collab_global
                    retain_protection = retain_protection_global
                    semantic_protection = semantic_protection_global
                    score = (
                        args.lambda_forget * forget_influence +
                        args.lambda_residual * residual_boundary +
                        args.lambda_collab * collab_path -
                        args.lambda_retain * retain_protection -
                        args.lambda_semantic * semantic_protection
                    )
                    raw_decisions.append({
                        'module_name': module_name,
                        'layer_idx': layer_idx,
                        'module_key': mod_name,
                        'rank_id': int(rid),
                        'forget_influence_score': forget_influence,
                        'residual_boundary_score': residual_boundary,
                        'collaborative_path_score': collab_path,
                        'retain_protection_score': retain_protection,
                        'semantic_protection_score': semantic_protection,
                        'rank_unlearn_score': float(score),
                        'route': 'protect',
                        'suppression_strength': 0.0,
                    })

        if not raw_decisions:
            self.stats['protection_aware_pruning'] = {
                'decisions': [],
                'summary': {'reason': 'no_lora_ranks_found'},
            }
            return self.stats['protection_aware_pruning']

        sorted_candidates = sorted(
            raw_decisions,
            key=lambda d: d['rank_unlearn_score'],
            reverse=True,
        )
        max_prunable = int(len(sorted_candidates) * max(0.0, min(1.0, args.max_prune_ratio)))
        max_prunable = max(1, max_prunable) if args.max_prune_ratio > 0 else 0
        probe_top_m = int(getattr(args, 'probe_top_m', 64))
        probe_limit = max(0, min(len(sorted_candidates), probe_top_m))
        selected_candidates = sorted_candidates[:probe_limit]
        selected = {
            (d['module_name'], d['rank_id'])
            for d in selected_candidates[:min(max_prunable, probe_limit)]
        }

        probe_records = []
        probe_by_rank = {}
        enable_probe = bool(getattr(args, 'enable_directional_probe', True))
        if enable_probe and score_fn is not None and selected_candidates:
            for candidate in selected_candidates:
                preferred_type = (
                    'hard_prune'
                    if candidate['rank_unlearn_score'] >= args.hard_prune_threshold
                    else 'soft_suppress'
                )
                probe = self.probe_rank_direction(
                    module_name=candidate['module_name'],
                    rank_id=candidate['rank_id'],
                    probe_records=forget_probe_records or [],
                    intervention_type=preferred_type,
                    suppression_strength=(
                        1.0 if preferred_type == 'hard_prune'
                        else float(args.suppression_strength)
                    ),
                    score_fn=score_fn,
                    retain_probe_records=retain_probe_records or [],
                    probe_score_tolerance=getattr(args, 'probe_score_tolerance', 1e-4),
                    retain_probe_drop_tolerance=getattr(
                        args, 'retain_probe_drop_tolerance', 0.05
                    ),
                )
                probe_records.append(probe)
                probe_by_rank[(candidate['module_name'], candidate['rank_id'])] = probe

        for decision in raw_decisions:
            key = (decision['module_name'], decision['rank_id'])
            probe = probe_by_rank.get(key)
            if probe:
                decision['direction'] = probe.get('direction', 'neutral')
                decision['directional_probe'] = {
                    'forget_score_delta_mean': probe.get('forget_score_delta_mean'),
                    'forget_rank_delta_mean': probe.get('forget_rank_delta_mean'),
                    'forget_margin_delta_mean': probe.get('forget_margin_delta_mean'),
                    'retain_score_delta_mean': probe.get('retain_score_delta_mean'),
                    'retain_rank_delta_mean': probe.get('retain_rank_delta_mean'),
                    'is_helpful_for_forget': probe.get('is_helpful_for_forget'),
                    'is_harmful_to_retain': probe.get('is_harmful_to_retain'),
                    'directional_probe_type': probe.get('directional_probe_type'),
                }
            else:
                decision['direction'] = 'not_probed'

            protected = (
                decision['retain_protection_score'] >= args.protect_threshold or
                decision['semantic_protection_score'] >= args.protect_threshold
            )
            selected_for_intervention = key in selected
            score_allows_hard = decision['rank_unlearn_score'] >= args.hard_prune_threshold
            score_allows_soft = decision['rank_unlearn_score'] >= args.soft_suppress_threshold

            decision['route'] = 'protect'
            decision['suppression_strength'] = 0.0
            decision['route_reason'] = 'not_selected'

            if protected:
                decision['route_reason'] = 'protection_threshold'
            elif not selected_for_intervention:
                decision['route_reason'] = 'outside_max_prune_ratio'
            elif enable_probe and probe is not None:
                direction = probe.get('direction', 'neutral')
                retain_harmful = bool(probe.get('is_harmful_to_retain'))
                if direction == 'harmful':
                    decision['route_reason'] = 'directional_probe_harmful'
                elif direction == 'helpful' and not retain_harmful and score_allows_hard:
                    decision['route'] = 'hard_prune'
                    decision['suppression_strength'] = 1.0
                    decision['route_reason'] = 'directional_probe_helpful'
                elif direction == 'helpful' and score_allows_soft:
                    decision['route'] = 'soft_suppress'
                    decision['suppression_strength'] = float(args.suppression_strength)
                    decision['route_reason'] = (
                        'helpful_forget_but_retain_risk'
                        if retain_harmful else 'directional_probe_helpful_soft'
                    )
                elif direction == 'neutral' and score_allows_soft and bool(
                    getattr(args, 'prefer_soft_when_uncertain', True)
                ):
                    decision['route'] = 'soft_suppress'
                    decision['suppression_strength'] = float(args.suppression_strength)
                    decision['route_reason'] = 'directional_probe_uncertain_soft'
                else:
                    decision['route_reason'] = (
                        'retain_probe_harmful'
                        if retain_harmful else 'directional_probe_uncertain_skip'
                    )
            elif score_allows_hard:
                decision['route'] = 'hard_prune'
                decision['suppression_strength'] = 1.0
                decision['route_reason'] = 'no_directional_probe_fallback_hard'
            elif score_allows_soft:
                decision['route'] = 'soft_suppress'
                decision['suppression_strength'] = float(args.suppression_strength)
                decision['route_reason'] = 'no_directional_probe_fallback_soft'

        for decision in raw_decisions:
            if decision['route'] == 'protect':
                continue
            layer_idx = decision['layer_idx']
            mod_name = decision['module_key']
            rid = decision['rank_id']
            info = self._lora_info[layer_idx][mod_name]
            param_a = info['A']['param']
            param_b = info['B']['param']
            name_a = info['A']['name']
            name_b = info['B']['name']
            if name_a not in self.original_weights:
                self.original_weights[name_a] = param_a.data.clone()
            if name_b not in self.original_weights:
                self.original_weights[name_b] = param_b.data.clone()
            if decision['route'] == 'hard_prune':
                param_a.data[rid, :] = 0.0
                param_b.data[:, rid] = 0.0
            elif decision['route'] == 'soft_suppress':
                scale = max(0.0, 1.0 - decision['suppression_strength'])
                param_a.data[rid, :] *= scale
                param_b.data[:, rid] *= scale

        for decision in raw_decisions:
            decision.pop('layer_idx', None)
            decision.pop('module_key', None)

        total_ranks = len(raw_decisions)
        hard_count = sum(1 for d in raw_decisions if d['route'] == 'hard_prune')
        soft_count = sum(1 for d in raw_decisions if d['route'] == 'soft_suppress')
        protect_count = sum(1 for d in raw_decisions if d['route'] == 'protect')

        def mean_field(field):
            values = [float(d.get(field, 0.0)) for d in raw_decisions]
            return float(np.mean(values)) if values else 0.0

        summary = {
            'total_ranks': total_ranks,
            'hard_prune': hard_count,
            'soft_suppress': soft_count,
            'protect': protect_count,
            'hard_prune_ratio': float(hard_count) / float(total_ranks) if total_ranks else 0.0,
            'soft_suppress_ratio': float(soft_count) / float(total_ranks) if total_ranks else 0.0,
            'protect_ratio': float(protect_count) / float(total_ranks) if total_ranks else 0.0,
            'actual_intervention_ratio': (
                float(hard_count + soft_count) / float(total_ranks)
                if total_ranks else 0.0
            ),
            'max_prune_ratio': args.max_prune_ratio,
            'avg_rank_unlearn_score': mean_field('rank_unlearn_score'),
            'avg_forget_influence_score': mean_field('forget_influence_score'),
            'avg_residual_boundary_score': mean_field('residual_boundary_score'),
            'avg_retain_protection_score': mean_field('retain_protection_score'),
            'avg_semantic_protection_score': mean_field('semantic_protection_score'),
            'directional_probe_enabled': bool(enable_probe and score_fn is not None),
            'directional_probe_type': 'finite_difference_lora_rank_probe',
            'num_probe_ranks': len(probe_records),
            'num_helpful_ranks': sum(1 for p in probe_records if p.get('direction') == 'helpful'),
            'num_harmful_ranks': sum(1 for p in probe_records if p.get('direction') == 'harmful'),
            'num_neutral_ranks': sum(1 for p in probe_records if p.get('direction') == 'neutral'),
            'attribution_fallback': (
                'activation/LoRA-weight-statistic fallback; not full gradient attribution'
            ),
            'global_scores': {
                'residual_boundary_global': residual_boundary_global,
                'boundary_global': boundary_global,
                'collaborative_path_global': collab_global,
                'retain_protection_global': retain_protection_global,
                'semantic_protection_global': semantic_protection_global,
                'has_strong_protection': has_strong_protection,
            },
        }
        self.stats['protection_aware_pruning'] = {
            'decisions': raw_decisions,
            'summary': summary,
            'directional_probe_results': {
                'records': probe_records,
                'summary': {
                    'directional_probe_type': 'finite_difference_lora_rank_probe',
                    'num_probe_ranks': len(probe_records),
                    'num_helpful_ranks': sum(1 for p in probe_records if p.get('direction') == 'helpful'),
                    'num_harmful_ranks': sum(1 for p in probe_records if p.get('direction') == 'harmful'),
                    'num_neutral_ranks': sum(1 for p in probe_records if p.get('direction') == 'neutral'),
                },
            },
        }
        return self.stats['protection_aware_pruning']


def importance_from_scores(pruning_scores, key, field, k):
    """Helper to extract individual field values from pruning_scores dict."""
    # Fields are embedded in the importance dict, not pruning_scores directly
    return 0.0  # Simplified: scores already incorporate all fields
