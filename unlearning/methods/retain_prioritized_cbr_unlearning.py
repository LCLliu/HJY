import json
import math
import os
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .geometry_prune import (
    GeometryPruneMethod,
    _minmax,
    _row_iid,
    _row_position,
    _row_uid,
    remove_forget_item_from_history,
    save_json,
)


class RetainPrioritizedCBRUnlearningMethod(GeometryPruneMethod):
    """Retain-Prioritized Collaborative-Boundary Unlearning.

    This method intentionally does not call GeometryPruneMethod.run() or
    GeometryPruneMethod.build_residual_candidates(). The candidate flow keeps
    semantic/retain protection out of CBR construction, then applies it only
    during retain-aware region splitting.
    """

    method_name = "retain_prioritized_cbr"

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        method_split_data = self._method_split_data()
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})
        self.logs.update({
            "status": "running",
            "action": "retain_prioritized_collaborative_boundary_unlearning",
            "uses_fixed_split": True,
            "logit_correction_used": False,
            "candidate_level_exposure": True,
            "formal_run": bool(getattr(self.args, "formal_run", True)),
            "run_is_formal": bool(getattr(self.args, "run_is_formal", True)),
            "debug_split_sample_limit": getattr(self.args, "debug_split_sample_limit", None),
            "split_counts": self._split_counts(),
            "method_diagnostic_split_counts": self._split_counts_for(method_split_data),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get(
                "retain_train_excludes_forget_interactions"
            ),
            "fallbacks": [],
            "notes": [
                "S_CBR is built before retention protection; semantic protection never filters CBR.",
                "S_collab uses item-item NPMI and high-order BFS proximity in the user-item graph.",
                "LoRA interventions are rank-level and do not update full model parameters.",
            ],
        })
        save_json(os.path.join(self.output_dir, "run_config.json"), vars(self.args).copy())
        save_json(os.path.join(self.output_dir, "logit_correction.json"), {
            "enabled": False,
            "corrections": {},
            "note": "retain_prioritized_cbr disables logit correction; effects must come from LoRA rank interventions.",
        })

        self._init_similarity_fallbacks(method_split_data)
        self._enable_lora_gradients()
        self._build_interaction_graph_indices()
        self._load_retrieved_candidate_store()

        region_records, cbr_summary = self._build_all_regions(method_split_data)
        self._write_jsonl(os.path.join(self.output_dir, "cbr_candidates.jsonl"), region_records)
        save_json(os.path.join(self.output_dir, "cbr_summary.json"), cbr_summary)
        save_json(os.path.join(self.output_dir, "region_splits.json"), {
            "summary": cbr_summary.get("region_summary", {}),
            "records": region_records,
        })

        losses = self.build_region_aware_losses(region_records)
        save_json(os.path.join(self.output_dir, "region_loss_summary.json"), losses["summary"])

        rank_signals = self.compute_rank_level_signals(losses)
        save_json(os.path.join(self.output_dir, "rank_signals.json"), {
            "records": list(rank_signals.values()),
            "summary": self._rank_signal_summary(rank_signals),
        })

        routing = self.route_lora_ranks(rank_signals)
        save_json(os.path.join(self.output_dir, "rank_routing.json"), routing)

        before_records = [dict(r) for r in self.predictions_before.get("records", [])]
        if not before_records:
            before_records = self._score_records(self._all_protocol_records(method_split_data))
        feasible_state = self._save_trainable_state()
        rollback_log, pruning = self.rollback_or_accept_update(
            routing=routing,
            rank_signals=rank_signals,
            before_records=before_records,
            feasible_state=feasible_state,
        )

        after_records = self._score_records(self._all_protocol_records(method_split_data))
        method_metrics = self.evaluate_unlearning_metrics(
            None,
            None,
            method_split_data.get("forget_interactions", []),
            method_split_data.get("retain_interactions", []),
            self.args,
            residual_candidates=[r for r in region_records if r.get("in_S_CBR")],
            before_records=before_records,
            after_records=after_records,
        )

        save_json(os.path.join(self.output_dir, "pruning_decisions.json"), pruning)
        save_json(os.path.join(self.output_dir, "pruning_summary.json"), pruning.get("summary", {}))
        save_json(os.path.join(self.output_dir, "rollback_log.json"), rollback_log)
        save_json(os.path.join(self.output_dir, "unlearning_metrics.json"), method_metrics)
        self._save_cbr_rank_mask(pruning)

        self.logs.update({
            "status": "completed",
            "is_effective_unlearning_baseline": True,
            "cbr_summary": cbr_summary,
            "region_loss_summary": losses["summary"],
            "rank_signal_summary": self._rank_signal_summary(rank_signals),
            "rank_routing_summary": routing.get("summary", {}),
            "pruning_summary": pruning.get("summary", {}),
            "rollback": rollback_log,
            "metrics_file": os.path.join(self.output_dir, "unlearning_metrics.json"),
            "cbr_candidates_file": os.path.join(self.output_dir, "cbr_candidates.jsonl"),
            "rank_signals_file": os.path.join(self.output_dir, "rank_signals.json"),
        })
        self.save_logs()
        return self.logs

    # ------------------------------------------------------------------
    # Stage 1: collaborative boundary residual localization
    # ------------------------------------------------------------------

    def build_counterfactual_history(self, row: Dict) -> Tuple[List[int], List[int]]:
        """Step 1: build H_u and H_u^{-f} for one forget interaction."""
        uid = _row_uid(row)
        forget_iid = _row_iid(row)
        history = self._original_history_for_forget_row(row, uid, forget_iid)
        counterfactual = remove_forget_item_from_history(history, forget_iid)
        return history, counterfactual

    def compute_counterfactual_residual(
        self,
        uid: int,
        forget_iid: int,
        history: List[int],
        counterfactual_history: List[int],
        candidates: List[int],
    ) -> List[Dict]:
        """Step 2: compute Delta_i^f = s(i|H_u)-s(i|H_u^{-f})."""
        original = self._score_candidate_list(history, candidates, forget_iid, grad=False)
        counterfactual = self._score_candidate_list(
            counterfactual_history,
            candidates,
            forget_iid,
            grad=False,
        )
        records = []
        for iid in candidates:
            key = str(int(iid))
            score_original = float(original["scores"].get(key, 0.0))
            score_counterfactual = float(counterfactual["scores"].get(key, 0.0))
            records.append({
                "uid": int(uid),
                "user_id": int(uid),
                "forget_iid": int(forget_iid),
                "forget_item_id": int(forget_iid),
                "candidate_iid": int(iid),
                "candidate_item_id": int(iid),
                "score_original": score_original,
                "score_counterfactual": score_counterfactual,
                "residual_delta": float(score_original - score_counterfactual),
                "residual_score": float(score_original - score_counterfactual),
                "rank_original": int(original["ranks"].get(key, 10**9)),
                "rank_counterfactual": int(counterfactual["ranks"].get(key, 10**9)),
                "topk_boundary_score_original": float(original.get("topk_boundary_score", 0.0)),
                "margin_to_topk_boundary_original": float(
                    score_original - float(original.get("topk_boundary_score", 0.0))
                ),
            })
        return records

    def compute_null_control_baseline(
        self,
        uid: int,
        forget_iid: int,
        history: List[int],
        candidates: List[int],
        original_scores: Dict[str, float],
    ) -> Tuple[Dict[str, Dict], Dict]:
        """Step 3: build null-control residual distribution from retain removals."""
        sampled_items = self._sample_null_removal_items(uid, history, forget_iid, self.args)
        eps = float(getattr(self.args, "null_eps", 1e-8) or 1e-8)
        if not sampled_items:
            stats = {
                str(int(iid)): {"mean": 0.0, "std": eps, "num_null_removals": 0}
                for iid in candidates
            }
            return stats, {
                "skipped": True,
                "reason": "insufficient_retain_history_for_null_removals",
                "requested": int(getattr(self.args, "num_null_removals", 5) or 5),
                "actual": 0,
            }

        null_by_candidate = {str(int(iid)): [] for iid in candidates}
        for retain_iid in sampled_items:
            null_history = remove_forget_item_from_history(history, retain_iid)
            null_scores = self._score_candidate_list(
                null_history,
                candidates,
                forget_iid,
                grad=False,
            )
            for iid in candidates:
                key = str(int(iid))
                delta = float(original_scores.get(key, 0.0)) - float(
                    null_scores["scores"].get(key, 0.0)
                )
                null_by_candidate[key].append(delta)

        stats = {}
        for iid, values in null_by_candidate.items():
            arr = np.asarray(values, dtype=np.float64)
            stats[iid] = {
                "mean": float(np.mean(arr)) if arr.size else 0.0,
                "std": float(np.std(arr)) + eps if arr.size else eps,
                "num_null_removals": int(arr.size),
            }
        return stats, {
            "skipped": False,
            "reason": None,
            "requested": int(getattr(self.args, "num_null_removals", 5) or 5),
            "actual": len(sampled_items),
        }

    def build_residual_set(
        self,
        residual_records: List[Dict],
        null_stats: Dict[str, Dict],
    ) -> Tuple[set, Dict[int, Dict]]:
        """Step 4: obtain S_res using null-control z-score significance."""
        tau_z = float(getattr(self.args, "cbr_tau_z", getattr(self.args, "tau_residual_z", 1.0)))
        eps = float(getattr(self.args, "null_eps", 1e-8) or 1e-8)
        residual_set = set()
        per_item = {}
        for record in residual_records:
            iid = int(record["candidate_iid"])
            stat = null_stats.get(str(iid), {"mean": 0.0, "std": eps})
            z = (
                float(record.get("residual_delta", 0.0)) - float(stat.get("mean", 0.0))
            ) / max(float(stat.get("std", eps)), eps)
            passed = z > tau_z
            if passed:
                residual_set.add(iid)
            per_item[iid] = {
                "residual_z": float(z),
                "residual_null_mean": float(stat.get("mean", 0.0)),
                "residual_null_std": float(stat.get("std", eps)),
                "in_S_res": bool(passed),
            }
        return residual_set, per_item

    def build_topk_boundary_set(self, residual_records: List[Dict]) -> set:
        """Step 5: obtain S_bd using the Top-K boundary window."""
        top_k = getattr(self.args, "cbr_boundary_top_k", None)
        if top_k is None:
            top_k = getattr(self.args, "boundary_top_k", None)
        if top_k is None:
            top_k = getattr(self.args, "topk_boundary", 10)
        top_k = int(top_k)
        window = int(getattr(self.args, "cbr_boundary_window", getattr(self.args, "boundary_window", 10)))
        return {
            int(record["candidate_iid"])
            for record in residual_records
            if abs(int(record.get("rank_original", 10**9)) - top_k) <= window
        }

    def build_collaborative_neighborhood(
        self,
        forget_iid: int,
        candidates: List[int],
    ) -> Tuple[set, Dict[int, Dict]]:
        """Step 6: obtain S_collab from NPMI or high-order BFS proximity."""
        tau_c = float(getattr(self.args, "cbr_tau_c", 0.0))
        max_hops = int(getattr(self.args, "cbr_collab_max_hops", 4))
        collab_set = set()
        per_item = {}
        for iid in candidates:
            iid = int(iid)
            npmi = self._item_npmi(int(forget_iid), iid)
            distance = self._bipartite_shortest_path(int(forget_iid), iid, max_hops)
            pass_npmi = npmi > tau_c
            pass_path = distance is not None and distance <= max_hops
            if pass_npmi or pass_path:
                collab_set.add(iid)
            per_item[iid] = {
                "collab_npmi": float(npmi),
                "graph_distance": distance,
                "collab_pass_npmi": bool(pass_npmi),
                "collab_pass_path": bool(pass_path),
                "in_S_collab": bool(pass_npmi or pass_path),
            }
        return collab_set, per_item

    @staticmethod
    def build_cbr_set(S_res: set, S_bd: set, S_collab: set) -> set:
        """Step 7: construct S_CBR = S_res intersection S_bd intersection S_collab."""
        return set(S_res) & set(S_bd) & set(S_collab)

    # ------------------------------------------------------------------
    # Stage 2: retain-aware residual partition
    # ------------------------------------------------------------------

    def compute_retention_protection_score(
        self,
        uid: int,
        forget_iid: int,
        history: List[int],
        candidates: List[int],
    ) -> Tuple[set, Dict[int, Dict]]:
        """Step 8: compute P_i = alpha Sem + beta Collab + gamma Drop_R."""
        retain_history = self._retain_history_for_user(uid, exclude_item=forget_iid)
        sem_raw = {}
        collab_raw = {}
        for iid in candidates:
            iid = int(iid)
            sem_raw[iid] = self._semantic_retain_alignment(iid, retain_history)
            collab_raw[iid] = self._collab_retain_dependency(iid, retain_history)

        drop_raw = self._retain_history_drop(uid, forget_iid, retain_history, candidates)
        sem_norm = self._normalize_dict(sem_raw)
        collab_norm = self._normalize_dict(collab_raw)
        drop_norm = self._normalize_dict(drop_raw)

        alpha = float(getattr(self.args, "cbr_alpha_sem", 1.0))
        beta = float(getattr(self.args, "cbr_beta_collab", 1.0))
        gamma = float(getattr(self.args, "cbr_gamma_drop", 1.0))
        tau_p = float(getattr(self.args, "cbr_tau_p", getattr(self.args, "retain_protection_threshold", 0.3)))
        denom = max(alpha + beta + gamma, 1e-12)

        protect = set()
        per_item = {}
        for iid in candidates:
            iid = int(iid)
            score = (
                alpha * sem_norm.get(iid, 0.0) +
                beta * collab_norm.get(iid, 0.0) +
                gamma * drop_norm.get(iid, 0.0)
            ) / denom
            if score > tau_p:
                protect.add(iid)
            per_item[iid] = {
                "Sem_i_Ru": float(sem_raw.get(iid, 0.0)),
                "Collab_i_Ru": float(collab_raw.get(iid, 0.0)),
                "Drop_R_i": float(drop_raw.get(iid, 0.0)),
                "Sem_i_Ru_norm": float(sem_norm.get(iid, 0.0)),
                "Collab_i_Ru_norm": float(collab_norm.get(iid, 0.0)),
                "Drop_R_i_norm": float(drop_norm.get(iid, 0.0)),
                "retention_protection_score": float(score),
                "in_S_protect": bool(score > tau_p),
            }
        return protect, per_item

    @staticmethod
    def split_regions(S_CBR: set, S_protect: set) -> Dict[str, set]:
        """Step 8 continued: split S_FD, S_OA, and S_RP."""
        S_CBR = set(S_CBR)
        S_protect = set(S_protect)
        return {
            "S_FD": S_CBR - S_protect,
            "S_OA": S_CBR & S_protect,
            "S_RP": S_protect - S_CBR,
        }

    # ------------------------------------------------------------------
    # Stage 3: region-aware losses and rank-level interventions
    # ------------------------------------------------------------------

    def build_region_aware_losses(self, region_records: List[Dict]) -> Dict:
        """Step 9: construct L_FD, L_R, and overlap-aware forget/retain losses."""
        fd_loss, fd_summary = self._counterfactual_distill_loss(region_records, "S_FD")
        oa_forget_loss, oa_f_summary = self._counterfactual_distill_loss(region_records, "S_OA")
        rp_retain_loss, rp_summary = self._retention_anchor_loss(region_records, "S_RP")
        oa_retain_loss, oa_r_summary = self._retention_anchor_loss(region_records, "S_OA")
        retain_probe_loss, retain_probe_summary = self._retain_probe_anchor_loss()

        L_R = rp_retain_loss + retain_probe_loss
        forget_loss = fd_loss + oa_forget_loss
        retain_loss = L_R + oa_retain_loss
        summary = {
            "L_FD": float(fd_loss.detach().cpu()),
            "L_R": float(L_R.detach().cpu()),
            "L_OA_forget": float(oa_forget_loss.detach().cpu()),
            "L_OA_retain": float(oa_retain_loss.detach().cpu()),
            "forget_loss_total": float(forget_loss.detach().cpu()),
            "retain_loss_total": float(retain_loss.detach().cpu()),
            "fd": fd_summary,
            "oa_forget": oa_f_summary,
            "rp_retain": rp_summary,
            "oa_retain": oa_r_summary,
            "retain_probe": retain_probe_summary,
            "retain_loss_note": (
                "Retain gradients use a teacher-weighted score anchor so rank "
                "importance is non-zero before the candidate update; RetainDrop "
                "is still checked with ranking metrics."
            ),
        }
        return {
            "L_FD": fd_loss,
            "L_R": L_R,
            "L_OA_forget": oa_forget_loss,
            "L_OA_retain": oa_retain_loss,
            "forget_loss": forget_loss,
            "retain_loss": retain_loss,
            "summary": summary,
        }

    def compute_rank_level_signals(self, losses: Dict) -> Dict[str, Dict]:
        """Step 10: compute F_r, G_r, rho_r, and continuous O_r per LoRA rank."""
        forget_vecs = self._grad_vectors_for_loss(losses.get("forget_loss"))
        retain_vecs = self._grad_vectors_for_loss(losses.get("retain_loss"))
        pairs = self._collect_lora_pairs()
        keys = []
        for module_name, pair in pairs.items():
            for rid in range(pair["rank"]):
                keys.append(f"{module_name}:{rid}")

        eps = float(getattr(self.args, "rank_score_delta", 1e-8) or 1e-8)
        f_values = []
        g_values = []
        raw = {}
        for key in keys:
            f_vec = forget_vecs.get(key)
            r_vec = retain_vecs.get(key)
            if f_vec is None:
                f_vec = self._zero_rank_vector_for_key(key)
            if r_vec is None:
                r_vec = self._zero_rank_vector_for_key(key)
            F_r = float(torch.linalg.vector_norm(f_vec.float()).cpu())
            G_r = float(torch.linalg.vector_norm(r_vec.float()).cpu())
            dot = float(torch.dot(f_vec.float(), r_vec.float()).cpu())
            rho = dot / (F_r * G_r + eps)
            rho = float(np.clip(rho, -1.0, 1.0))
            module_name, rank_s = key.rsplit(":", 1)
            raw[key] = {
                "parameter_key": key,
                "module_name": module_name,
                "rank_id": int(rank_s),
                "F_r": F_r,
                "G_r": G_r,
                "rho_r": rho,
                "g_F_vector": f_vec.cpu(),
                "g_R_vector": r_vec.cpu(),
            }
            f_values.append(F_r)
            g_values.append(G_r)

        f_norm = _minmax(f_values)
        g_norm = _minmax(g_values)
        overlap_values = []
        for idx, key in enumerate(keys):
            O_r = float(f_norm[idx]) * float(g_norm[idx]) * (1.0 - raw[key]["rho_r"]) / 2.0
            raw[key]["F_r_norm"] = float(f_norm[idx])
            raw[key]["G_r_norm"] = float(g_norm[idx])
            raw[key]["O_r"] = float(O_r)
            overlap_values.append(O_r)
        o_norm = _minmax(overlap_values)
        for idx, key in enumerate(keys):
            raw[key]["O_r_norm"] = float(o_norm[idx])

        serializable = {}
        self._rank_gradient_vectors = {}
        for key, record in raw.items():
            self._rank_gradient_vectors[key] = {
                "g_F_vector": record.pop("g_F_vector"),
                "g_R_vector": record.pop("g_R_vector"),
            }
            serializable[key] = record
        return serializable

    def route_lora_ranks(self, rank_signals: Dict[str, Dict]) -> Dict:
        """Step 11: route LoRA ranks by quantile thresholds."""
        high_q = float(getattr(self.args, "cbr_rank_high_quantile", 0.75))
        low_q = float(getattr(self.args, "cbr_rank_low_quantile", 0.25))
        overlap_q = float(getattr(self.args, "cbr_overlap_quantile", 0.75))
        F = np.asarray([float(v.get("F_r_norm", 0.0)) for v in rank_signals.values()], dtype=np.float64)
        G = np.asarray([float(v.get("G_r_norm", 0.0)) for v in rank_signals.values()], dtype=np.float64)
        O = np.asarray([float(v.get("O_r_norm", 0.0)) for v in rank_signals.values()], dtype=np.float64)
        f_high = float(np.quantile(F, high_q)) if F.size else 1.0
        g_high = float(np.quantile(G, high_q)) if G.size else 1.0
        f_low = float(np.quantile(F, low_q)) if F.size else 0.0
        g_low = float(np.quantile(G, low_q)) if G.size else 0.0
        o_high = float(np.quantile(O, overlap_q)) if O.size else 1.0

        records = []
        counts = defaultdict(int)
        for key, signal in rank_signals.items():
            f = float(signal.get("F_r_norm", 0.0))
            g = float(signal.get("G_r_norm", 0.0))
            o = float(signal.get("O_r_norm", 0.0))
            raw_f = float(signal.get("F_r", 0.0))
            raw_g = float(signal.get("G_r", 0.0))
            has_f = raw_f > 0.0
            has_g = raw_g > 0.0
            if has_f and f >= f_high and g <= g_low:
                route = "forget_dominant"
                reason = "F_high_G_low"
            elif has_g and g >= g_high and f <= f_low:
                route = "retain_dominant"
                reason = "G_high_F_low"
            elif has_f and has_g and f >= f_high and g >= g_high:
                route = "overlap_aware"
                reason = "F_high_G_high"
            elif has_f and has_g and o >= o_high and f >= f_low and g >= g_low:
                route = "overlap_aware"
                reason = "continuous_overlap_risk"
            else:
                route = "neutral"
                reason = "below_route_thresholds"
            counts[route] += 1
            records.append({
                **signal,
                "route": route,
                "route_reason": reason,
            })

        summary = {
            "num_total_ranks": len(records),
            "forget_dominant": int(counts["forget_dominant"]),
            "retain_dominant": int(counts["retain_dominant"]),
            "overlap_aware": int(counts["overlap_aware"]),
            "neutral": int(counts["neutral"]),
            "thresholds": {
                "high_quantile": high_q,
                "low_quantile": low_q,
                "overlap_quantile": overlap_q,
                "F_high": f_high,
                "G_high": g_high,
                "F_low": f_low,
                "G_low": g_low,
                "O_high": o_high,
            },
        }
        return {"records": records, "summary": summary}

    def apply_forget_dominant_intervention(self, routing: Dict, config: Dict) -> Dict:
        """Step 12: prune or strongly suppress forget-dominant ranks."""
        pairs = self._collect_lora_pairs()
        total_ranks = sum(pair["rank"] for pair in pairs.values())
        budget_ratio = float(config.get("pruning_ratio", 0.0))
        max_ratio = float(getattr(self.args, "max_prune_ratio", 0.05) or 0.0)
        budget = min(int(total_ranks * max(0.0, budget_ratio)), int(total_ranks * max(0.0, max_ratio)))
        candidates = [
            dict(r) for r in routing.get("records", [])
            if r.get("route") == "forget_dominant"
        ]
        candidates.sort(key=lambda r: float(r.get("F_r_norm", 0.0)), reverse=True)
        selected = candidates[:budget] if budget > 0 else []
        intervention_type = getattr(self.args, "cbr_forget_intervention", "suppress")
        suppression = float(config.get("suppression_strength", 0.0))
        changed = []

        with torch.no_grad():
            for record in selected:
                module_name = record["module_name"]
                rid = int(record["rank_id"])
                pair = pairs.get(module_name)
                if pair is None:
                    continue
                if intervention_type == "prune":
                    pair["A"].data[rid, :] = 0.0
                    pair["B"].data[:, rid] = 0.0
                    route = "hard_prune"
                    strength = 1.0
                else:
                    scale = max(0.0, 1.0 - suppression)
                    pair["A"].data[rid, :] *= scale
                    pair["B"].data[:, rid] *= scale
                    route = "strong_suppress"
                    strength = suppression
                changed.append({
                    **record,
                    "applied_route": route,
                    "suppression_strength": float(strength),
                })
        return {
            "changed": changed,
            "budget": budget,
            "intervention_type": intervention_type,
            "pruning_ratio": budget_ratio,
            "suppression_strength": suppression,
            "hard_prune": sum(1 for r in changed if r["applied_route"] == "hard_prune"),
            "soft_suppress": sum(1 for r in changed if r["applied_route"] == "strong_suppress"),
        }

    def apply_retain_dominant_freezing(self, routing: Dict) -> Dict:
        """Step 12: freeze retain-dominant ranks by excluding them from updates."""
        frozen = [
            dict(r) for r in routing.get("records", [])
            if r.get("route") == "retain_dominant"
        ]
        return {
            "num_frozen": len(frozen),
            "records": frozen,
            "note": "Manual rank interventions skip retain-dominant ranks.",
        }

    def apply_overlap_aware_projection(self, routing: Dict, config: Dict) -> Dict:
        """Step 12: apply continuous gated projection to overlap-aware ranks."""
        pairs = self._collect_lora_pairs()
        vectors = getattr(self, "_rank_gradient_vectors", {})
        lr = float(getattr(self.args, "cbr_update_lr", getattr(self.args, "lora_suppression_rho", 0.05)))
        gate_scale = float(config.get("projection_gate", 1.0))
        eps = float(getattr(self.args, "rank_score_delta", 1e-8) or 1e-8)
        changed = []

        with torch.no_grad():
            for record in routing.get("records", []):
                if record.get("route") != "overlap_aware":
                    continue
                key = record["parameter_key"]
                module_name = record["module_name"]
                rid = int(record["rank_id"])
                pair = pairs.get(module_name)
                grads = vectors.get(key)
                if pair is None or grads is None:
                    continue
                g_f = grads["g_F_vector"].float()
                g_r = grads["g_R_vector"].float()
                denom = float(torch.dot(g_r, g_r).cpu()) + eps
                coeff = float(torch.dot(g_f, g_r).cpu()) / denom
                base_lambda = float(record.get("O_r_norm", 0.0))
                lambda_r = float(np.clip(base_lambda * gate_scale, 0.0, 1.0))
                g_tilde = g_f - lambda_r * coeff * g_r

                a_size = pair["A"].data[rid, :].numel()
                a_update = g_tilde[:a_size].view_as(pair["A"].data[rid, :])
                b_update = g_tilde[a_size:].view_as(pair["B"].data[:, rid])
                pair["A"].data[rid, :] -= lr * a_update.to(pair["A"].device, dtype=pair["A"].dtype)
                pair["B"].data[:, rid] -= lr * b_update.to(pair["B"].device, dtype=pair["B"].dtype)
                changed.append({
                    **record,
                    "applied_route": "overlap_gated_projection",
                    "base_lambda_r": base_lambda,
                    "effective_lambda_r": lambda_r,
                    "projection_coeff": coeff,
                    "update_lr": lr,
                    "g_tilde_norm": float(torch.linalg.vector_norm(g_tilde).cpu()),
                })
        return {
            "changed": changed,
            "projection_gate": gate_scale,
            "update_lr": lr,
            "lambda_stats": self._lambda_stats(changed),
        }

    def evaluate_retain_drop(self, before_records: List[Dict], after_records: List[Dict]) -> float:
        """Step 13: compute RetainDrop with the project's retain ranking metrics."""
        return float(self._max_retain_drop(before_records, after_records))

    def rollback_or_accept_update(
        self,
        routing: Dict,
        rank_signals: Dict[str, Dict],
        before_records: List[Dict],
        feasible_state: Dict[str, torch.Tensor],
    ) -> Tuple[Dict, Dict]:
        """Step 13: accept feasible updates or rollback and soften interventions."""
        tolerance = getattr(self.args, "epsilon_retain", None)
        if tolerance is None:
            tolerance = getattr(self.args, "retain_drop_tolerance", 0.05)
        tolerance = float(tolerance)
        max_attempts = max(1, int(getattr(self.args, "cbr_max_update_attempts", 3)))
        soften = float(getattr(self.args, "cbr_soften_factor", 0.5))
        retain_eval_records = [
            dict(r) for r in self._all_protocol_records(self._method_split_data())
            if r.get("split_tag") == "retain"
        ]
        before_retain_records = [dict(r) for r in before_records if r.get("split_tag") == "retain"]
        if not before_retain_records and retain_eval_records:
            before_retain_records = self._score_records(retain_eval_records)

        config = {
            "pruning_ratio": float(getattr(
                self.args,
                "cbr_pruning_budget_ratio",
                getattr(self.args, "prune_budget_ratio", None)
                if getattr(self.args, "prune_budget_ratio", None) is not None
                else getattr(self.args, "prune_ratio", 0.01),
            ) or 0.0),
            "suppression_strength": float(getattr(
                self.args,
                "cbr_suppression_strength",
                getattr(self.args, "suppression_strength", 0.7),
            )),
            "projection_gate": float(getattr(self.args, "cbr_projection_gate", 1.0)),
        }

        attempts = []
        accepted = False
        final_forget = {"changed": [], "hard_prune": 0, "soft_suppress": 0}
        final_freeze = {"num_frozen": 0, "records": []}
        final_overlap = {"changed": [], "lambda_stats": {}}
        retain_drop = 0.0

        for attempt_idx in range(max_attempts):
            self._restore_trainable_state(feasible_state)
            final_freeze = self.apply_retain_dominant_freezing(routing)
            final_forget = self.apply_forget_dominant_intervention(routing, config)
            final_overlap = self.apply_overlap_aware_projection(routing, config)
            after_retain_records = self._score_records(retain_eval_records) if retain_eval_records else []
            retain_drop = self.evaluate_retain_drop(before_retain_records, after_retain_records)
            accept = retain_drop <= tolerance
            attempts.append({
                "attempt": attempt_idx + 1,
                "accepted": bool(accept),
                "retain_drop": float(retain_drop),
                "retain_tolerance": tolerance,
                "pruning_ratio": float(config["pruning_ratio"]),
                "suppression_strength": float(config["suppression_strength"]),
                "projection_gate": float(config["projection_gate"]),
                "num_forget_intervened": len(final_forget.get("changed", [])),
                "num_overlap_projected": len(final_overlap.get("changed", [])),
                "rollback": not bool(accept),
            })
            if accept:
                accepted = True
                feasible_state = self._save_trainable_state()
                break
            self._restore_trainable_state(feasible_state)
            config["pruning_ratio"] *= soften
            config["suppression_strength"] *= soften
            config["projection_gate"] *= soften

        if not accepted:
            self._restore_trainable_state(feasible_state)
            final_forget = {"changed": [], "hard_prune": 0, "soft_suppress": 0}
            final_overlap = {"changed": [], "lambda_stats": {}}

        total_ranks = int(routing.get("summary", {}).get("num_total_ranks", len(rank_signals)))
        hard = int(final_forget.get("hard_prune", 0))
        soft = int(final_forget.get("soft_suppress", 0))
        overlap = len(final_overlap.get("changed", []))
        frozen = int(final_freeze.get("num_frozen", 0))
        decisions = []
        applied_by_key = {
            r["parameter_key"]: r for r in final_forget.get("changed", []) + final_overlap.get("changed", [])
        }
        for record in routing.get("records", []):
            key = record["parameter_key"]
            applied = applied_by_key.get(key, {})
            decisions.append({
                **record,
                "applied_route": applied.get(
                    "applied_route",
                    "frozen" if record.get("route") == "retain_dominant" else "no_update",
                ),
                "suppression_strength": applied.get("suppression_strength", 0.0),
                "effective_lambda_r": applied.get("effective_lambda_r", 0.0),
            })

        summary = {
            "enabled": True,
            "total_ranks": total_ranks,
            "hard_prune": hard,
            "soft_suppress": soft,
            "overlap_project": overlap,
            "protect": frozen,
            "neutral": int(routing.get("summary", {}).get("neutral", 0)),
            "hard_prune_ratio": float(hard) / float(total_ranks) if total_ranks else 0.0,
            "soft_suppress_ratio": float(soft) / float(total_ranks) if total_ranks else 0.0,
            "actual_intervention_ratio": float(hard + soft + overlap) / float(total_ranks) if total_ranks else 0.0,
            "retain_drop": float(retain_drop),
            "retain_drop_tolerance": tolerance,
            "rollback": not accepted,
            "accepted": accepted,
            "final_pruning_ratio": float(config["pruning_ratio"]),
            "final_suppression_strength": float(config["suppression_strength"]),
            "final_projection_gate": float(config["projection_gate"]),
            "lambda_stats": final_overlap.get("lambda_stats", {}),
            "routing_counts": routing.get("summary", {}),
        }
        rollback_log = {
            "enabled": True,
            "accepted": accepted,
            "rollback_applied": not accepted,
            "retain_drop": float(retain_drop),
            "retain_drop_tolerance": tolerance,
            "attempts": attempts,
            "final_intervention_strengths": {
                "pruning_ratio": float(config["pruning_ratio"]),
                "suppression_strength": float(config["suppression_strength"]),
                "projection_gate": float(config["projection_gate"]),
            },
        }
        return rollback_log, {"decisions": decisions, "summary": summary}

    # ------------------------------------------------------------------
    # Region construction helpers
    # ------------------------------------------------------------------

    def _build_all_regions(self, split_data: Dict) -> Tuple[List[Dict], Dict]:
        records = []
        summary = {
            "num_forget_interactions": len(split_data.get("forget_interactions", [])),
            "candidate_sources": defaultdict(int),
            "fallback_used": False,
            "num_null_control_skipped": 0,
            "set_counts": defaultdict(int),
            "region_summary": defaultdict(int),
        }
        for row in split_data.get("forget_interactions", []):
            interaction_records, interaction_summary = self._build_regions_for_forget(row)
            records.extend(interaction_records)
            summary["candidate_sources"][interaction_summary["candidate_source"]] += 1
            summary["fallback_used"] = bool(summary["fallback_used"] or interaction_summary["fallback_used"])
            summary["num_null_control_skipped"] += int(interaction_summary["null_skipped"])
            for key, value in interaction_summary["set_counts"].items():
                summary["set_counts"][key] += int(value)
            for key, value in interaction_summary["region_counts"].items():
                summary["region_summary"][key] += int(value)

        summary["candidate_sources"] = dict(summary["candidate_sources"])
        summary["set_counts"] = dict(summary["set_counts"])
        summary["region_summary"] = dict(summary["region_summary"])
        if summary["fallback_used"] and bool(getattr(self.args, "formal_run", True)):
            self.logs.setdefault("warnings", []).append(
                "deterministic_random_candidate_fallback_used_in_formal_run"
            )
        return records, summary

    def _build_regions_for_forget(self, row: Dict) -> Tuple[List[Dict], Dict]:
        uid = _row_uid(row)
        forget_iid = _row_iid(row)
        position = _row_position(row)
        history, counterfactual_history = self.build_counterfactual_history(row)
        candidates, source, fallback_used = self._get_cbr_candidate_set(row, uid, forget_iid, history)

        residual_records = self.compute_counterfactual_residual(
            uid,
            forget_iid,
            history,
            counterfactual_history,
            candidates,
        )
        original_scores = {
            str(int(r["candidate_iid"])): float(r.get("score_original", 0.0))
            for r in residual_records
        }
        null_stats, null_log = self.compute_null_control_baseline(
            uid,
            forget_iid,
            history,
            candidates,
            original_scores,
        )
        S_res, residual_meta = self.build_residual_set(residual_records, null_stats)
        S_bd = self.build_topk_boundary_set(residual_records)
        S_collab, collab_meta = self.build_collaborative_neighborhood(forget_iid, candidates)
        S_CBR = self.build_cbr_set(S_res, S_bd, S_collab)
        S_protect, protect_meta = self.compute_retention_protection_score(
            uid,
            forget_iid,
            history,
            candidates,
        )
        regions = self.split_regions(S_CBR, S_protect)

        by_iid = {int(r["candidate_iid"]): dict(r) for r in residual_records}
        interaction_id = f"{uid}:{forget_iid}:{position}"
        output = []
        for iid in candidates:
            iid = int(iid)
            region = "other"
            for name, values in regions.items():
                if iid in values:
                    region = name
                    break
            record = {
                **by_iid[iid],
                **residual_meta.get(iid, {}),
                **collab_meta.get(iid, {}),
                **protect_meta.get(iid, {}),
                "interaction_key": interaction_id,
                "position": position,
                "history": [int(i) for i in history],
                "counterfactual_history": [int(i) for i in counterfactual_history],
                "candidate_items": [int(i) for i in candidates],
                "candidate_source": source,
                "candidate_fallback_used": bool(fallback_used),
                "in_S_bd": bool(iid in S_bd),
                "in_S_CBR": bool(iid in S_CBR),
                "region": region,
                "target_iid": iid,
            }
            output.append(record)

        region_counts = {name: len(values) for name, values in regions.items()}
        region_counts["other"] = sum(1 for r in output if r["region"] == "other")
        interaction_summary = {
            "candidate_source": source,
            "fallback_used": bool(fallback_used),
            "null_skipped": bool(null_log.get("skipped")),
            "set_counts": {
                "|S_res|": len(S_res),
                "|S_bd|": len(S_bd),
                "|S_collab|": len(S_collab),
                "|S_CBR|": len(S_CBR),
                "|S_protect|": len(S_protect),
            },
            "region_counts": {
                "|S_FD|": region_counts.get("S_FD", 0),
                "|S_OA|": region_counts.get("S_OA", 0),
                "|S_RP|": region_counts.get("S_RP", 0),
                "|other|": region_counts.get("other", 0),
            },
        }
        return output, interaction_summary

    # ------------------------------------------------------------------
    # Candidate, graph, and score helpers
    # ------------------------------------------------------------------

    def _get_cbr_candidate_set(
        self,
        row: Dict,
        uid: int,
        forget_iid: int,
        history: List[int],
    ) -> Tuple[List[int], str, bool]:
        size = int(getattr(self.args, "llm_negative_sample_size", 19)) + 1
        record = self._prediction_record_for("forget", uid, forget_iid, row)
        if record and record.get("candidate_items"):
            return self._dedupe_candidates(record["candidate_items"], forget_iid, size), "predictions_before", False

        explicit = row.get("candidate_items", row.get("candidates"))
        parsed = self._parse_candidate_items(explicit)
        if parsed:
            return self._dedupe_candidates(parsed, forget_iid, size), "row_candidate_items", False

        retrieved = self._retrieved_user_candidates(uid, forget_iid, size)
        if retrieved:
            return self._dedupe_candidates(retrieved, forget_iid, size), "retrieved_pkl_user_top_items", False

        return self._fallback_candidate_set(uid, forget_iid, history, "forget"), "deterministic_random_fallback", True

    @staticmethod
    def _dedupe_candidates(candidates: List[int], required_iid: int, size: int) -> List[int]:
        candidates = [int(iid) for iid in candidates if int(iid) != 0]
        if int(required_iid) in candidates:
            seed = candidates
        else:
            seed = [int(required_iid)] + candidates
        out = []
        for iid in seed:
            iid = int(iid)
            if iid == 0 or iid in out:
                continue
            out.append(iid)
            if len(out) >= size:
                break
        return out

    def _load_retrieved_candidate_store(self):
        self._retrieved_store = None
        path = getattr(self.args, "llm_retrieved_path", None)
        if not path:
            self.logs["retrieved_candidate_store"] = {"available": False, "reason": "path_missing"}
            return
        file_path = os.path.join(path, "retrieved.pkl")
        try:
            with open(file_path, "rb") as f:
                self._retrieved_store = pickle.load(f)
            self.logs["retrieved_candidate_store"] = {"available": True, "path": file_path}
        except Exception as exc:
            self.logs["retrieved_candidate_store"] = {
                "available": False,
                "path": file_path,
                "error": str(exc),
            }

    def _retrieved_user_candidates(self, uid: int, forget_iid: int, size: int) -> List[int]:
        store = getattr(self, "_retrieved_store", None)
        if not isinstance(store, dict):
            return []
        probs = store.get("val_probs")
        if probs is None:
            probs = store.get("test_probs")
        if probs is None:
            return []
        try:
            user_probs = torch.tensor(probs[int(uid) - 1])
            topk = torch.topk(user_probs, min(size, int(user_probs.numel()))).indices.tolist()
            return [int(forget_iid)] + [int(i) for i in topk if int(i) != 0]
        except Exception:
            return []

    def _build_interaction_graph_indices(self):
        train = self.dataset_data.get("train", {})
        self._graph_user_items = {
            int(uid): set(int(i) for i in seq if int(i) != 0)
            for uid, seq in train.items()
        }
        item_users = defaultdict(set)
        for uid, items in self._graph_user_items.items():
            for iid in items:
                item_users[int(iid)].add(int(uid))
        self._graph_item_users = dict(item_users)
        self._graph_num_users = max(1, len(self._graph_user_items))

    def _item_npmi(self, item_a: int, item_b: int) -> float:
        if item_a == item_b:
            return 1.0 if item_a in self._graph_item_users else 0.0
        users_a = self._graph_item_users.get(int(item_a), set())
        users_b = self._graph_item_users.get(int(item_b), set())
        if not users_a or not users_b:
            return 0.0
        co_users = users_a & users_b
        if not co_users:
            return 0.0
        n = float(self._graph_num_users)
        p_a = len(users_a) / n
        p_b = len(users_b) / n
        p_ab = len(co_users) / n
        if p_a <= 0 or p_b <= 0 or p_ab <= 0:
            return 0.0
        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)
        if denom <= 0:
            return 0.0
        return float(pmi / denom)

    def _bipartite_shortest_path(self, source_item: int, target_item: int, max_hops: int) -> Optional[int]:
        source_item = int(source_item)
        target_item = int(target_item)
        if source_item == target_item:
            return 0
        if max_hops <= 0:
            return None
        q = deque([("i", source_item, 0)])
        seen_items = {source_item}
        seen_users = set()
        while q:
            node_type, node_id, dist = q.popleft()
            if dist >= max_hops:
                continue
            if node_type == "i":
                for uid in self._graph_item_users.get(node_id, set()):
                    if uid in seen_users:
                        continue
                    seen_users.add(uid)
                    q.append(("u", uid, dist + 1))
            else:
                for iid in self._graph_user_items.get(node_id, set()):
                    if iid == target_item:
                        return dist + 1
                    if iid in seen_items:
                        continue
                    seen_items.add(iid)
                    q.append(("i", iid, dist + 1))
        return None

    def _semantic_retain_alignment(self, candidate_iid: int, retain_history: List[int]) -> float:
        return float(self.compute_semantic_protection(candidate_iid, retain_history))

    def _collab_retain_dependency(self, candidate_iid: int, retain_history: List[int]) -> float:
        best = 0.0
        max_hops = int(getattr(self.args, "cbr_collab_max_hops", 4))
        for retain_iid in retain_history:
            npmi = max(0.0, self._item_npmi(candidate_iid, int(retain_iid)))
            dist = self._bipartite_shortest_path(candidate_iid, int(retain_iid), max_hops)
            proximity = 0.0 if dist is None else 1.0 / (1.0 + float(dist))
            best = max(best, npmi, proximity)
        return float(best)

    def _retain_history_drop(
        self,
        uid: int,
        forget_iid: int,
        retain_history: List[int],
        candidates: List[int],
    ) -> Dict[int, float]:
        if not candidates:
            return {}
        if not retain_history:
            return {int(iid): 0.0 for iid in candidates}
        full_scores = self._score_candidate_list(retain_history, candidates, forget_iid, grad=False)
        masked_scores = self._score_candidate_list([], candidates, forget_iid, grad=False)
        drops = {}
        for iid in candidates:
            key = str(int(iid))
            drops[int(iid)] = max(
                0.0,
                float(full_scores["scores"].get(key, 0.0)) -
                float(masked_scores["scores"].get(key, 0.0)),
            )
        return drops

    @staticmethod
    def _normalize_dict(values: Dict[int, float]) -> Dict[int, float]:
        if not values:
            return {}
        keys = list(values.keys())
        norm = _minmax([values[k] for k in keys])
        return {int(k): float(norm[idx]) for idx, k in enumerate(keys)}

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    def _zero_loss(self):
        for _, param in self.model.named_parameters():
            if "lora_" in _ and param.requires_grad:
                return param.sum() * 0.0
        device = next(self.model.parameters()).device
        return torch.tensor(0.0, device=device)

    def _records_by_interaction(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        groups = defaultdict(list)
        for record in records:
            groups[str(record.get("interaction_key"))].append(record)
        return dict(groups)

    def _counterfactual_distill_loss(self, records: List[Dict], region: str) -> Tuple[torch.Tensor, Dict]:
        losses = []
        max_groups = int(getattr(self.args, "cbr_loss_max_groups", getattr(self.args, "residual_top_m", 64)))
        temperature = float(getattr(self.args, "cbr_distill_temperature", 1.0))
        for _, group in list(self._records_by_interaction(records).items())[:max_groups]:
            selected = [r for r in group if r.get("region") == region]
            if not selected:
                continue
            base = group[0]
            candidates = [int(i) for i in base["candidate_items"]]
            scores = self._score_candidate_list(
                base["history"],
                candidates,
                int(base["forget_iid"]),
                grad=True,
            )
            score_tensor = scores["score_tensor"]
            cf_map = {int(r["candidate_iid"]): float(r["score_counterfactual"]) for r in group}
            teacher_scores = torch.tensor(
                [cf_map.get(int(iid), 0.0) for iid in candidates],
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            mask_ids = {int(r["candidate_iid"]) for r in selected}
            mask = torch.tensor(
                [1.0 if int(iid) in mask_ids else 0.0 for iid in candidates],
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            teacher_probs = F.softmax(teacher_scores / temperature, dim=0)
            log_probs = F.log_softmax(score_tensor / temperature, dim=0)
            weight = float(np.mean([
                max(0.0, float(r.get("residual_z", 0.0)))
                for r in selected
            ])) or 1.0
            losses.append(weight * (-(teacher_probs * log_probs * mask).sum() / mask.sum().clamp_min(1.0)))
        if not losses:
            return self._zero_loss(), {"region": region, "num_groups": 0, "num_terms": 0}
        loss = torch.stack(losses).mean()
        return loss, {"region": region, "num_groups": len(losses), "num_terms": len(losses)}

    def _retention_anchor_loss(self, records: List[Dict], region: str) -> Tuple[torch.Tensor, Dict]:
        losses = []
        max_groups = int(getattr(self.args, "cbr_loss_max_groups", getattr(self.args, "residual_top_m", 64)))
        for _, group in list(self._records_by_interaction(records).items())[:max_groups]:
            selected = [r for r in group if r.get("region") == region]
            if not selected:
                continue
            base = group[0]
            candidates = [int(i) for i in base["candidate_items"]]
            scores = self._score_candidate_list(
                base["counterfactual_history"],
                candidates,
                int(base["forget_iid"]),
                grad=True,
            )
            score_tensor = scores["score_tensor"]
            teacher_map = {int(r["candidate_iid"]): float(r["score_counterfactual"]) for r in group}
            teacher_scores = torch.tensor(
                [teacher_map.get(int(iid), 0.0) for iid in candidates],
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            mask_ids = {int(r["candidate_iid"]) for r in selected}
            mask = torch.tensor(
                [1.0 if int(iid) in mask_ids else 0.0 for iid in candidates],
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            teacher_probs = F.softmax(teacher_scores, dim=0)
            losses.append(-((teacher_probs * score_tensor * mask).sum() / mask.sum().clamp_min(1.0)))
        if not losses:
            return self._zero_loss(), {"region": region, "num_groups": 0, "num_terms": 0}
        loss = torch.stack(losses).mean()
        return loss, {"region": region, "num_groups": len(losses), "num_terms": len(losses)}

    def _retain_probe_anchor_loss(self) -> Tuple[torch.Tensor, Dict]:
        retain_records = self._retain_probe_records()
        max_samples = int(getattr(self.args, "cbr_retain_loss_samples", getattr(self.args, "probe_retain_samples", 8)))
        retain_records = retain_records[:max_samples]
        losses = []
        for record in retain_records:
            candidates = [int(i) for i in record.get("candidate_items", [])]
            if not candidates:
                continue
            target_iid = int(record["target_iid"])
            scores = self._score_candidate_list(
                record.get("context_items", []),
                candidates,
                target_iid,
                grad=True,
            )
            score_tensor = scores["score_tensor"]
            teacher_map = record.get("scores", {})
            if not teacher_map:
                with torch.no_grad():
                    teacher_rank = self._score_candidate_list(
                        record.get("context_items", []),
                        candidates,
                        target_iid,
                        grad=False,
                    )
                teacher_map = teacher_rank.get("scores", {})
            teacher_scores = torch.tensor(
                [float(teacher_map.get(str(int(iid)), 0.0)) for iid in candidates],
                dtype=score_tensor.dtype,
                device=score_tensor.device,
            )
            teacher_probs = F.softmax(teacher_scores, dim=0)
            losses.append(-(teacher_probs * score_tensor).sum())
        if not losses:
            return self._zero_loss(), {"num_samples": 0}
        return torch.stack(losses).mean(), {"num_samples": len(losses)}

    # ------------------------------------------------------------------
    # LoRA rank gradient and update helpers
    # ------------------------------------------------------------------

    def _grad_vectors_for_loss(self, loss: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        self._zero_trainable_grads()
        if loss is None or not getattr(loss, "requires_grad", False):
            return {}
        loss.backward()
        out = self._collect_lora_rank_grad_vectors()
        self._zero_trainable_grads()
        return out

    def _collect_lora_rank_grad_vectors(self) -> Dict[str, torch.Tensor]:
        pairs = self._collect_lora_pairs()
        vectors = {}
        for module_name, pair in pairs.items():
            grad_a = pair["A"].grad
            grad_b = pair["B"].grad
            for rid in range(pair["rank"]):
                if grad_a is None:
                    a = torch.zeros_like(pair["A"].data[rid, :], dtype=torch.float32, device="cpu")
                else:
                    a = grad_a[rid, :].detach().float().cpu().flatten()
                if grad_b is None:
                    b = torch.zeros_like(pair["B"].data[:, rid], dtype=torch.float32, device="cpu")
                else:
                    b = grad_b[:, rid].detach().float().cpu().flatten()
                vectors[f"{module_name}:{rid}"] = torch.cat([a, b], dim=0)
        return vectors

    def _zero_rank_vector_for_key(self, key: str) -> torch.Tensor:
        module_name, rank_s = key.rsplit(":", 1)
        pair = self._collect_lora_pairs().get(module_name)
        if pair is None:
            return torch.zeros(1, dtype=torch.float32)
        rid = int(rank_s)
        a_size = pair["A"].data[rid, :].numel()
        b_size = pair["B"].data[:, rid].numel()
        return torch.zeros(a_size + b_size, dtype=torch.float32)

    @staticmethod
    def _lambda_stats(records: List[Dict]) -> Dict:
        values = [float(r.get("effective_lambda_r", 0.0)) for r in records]
        if not values:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
        }

    @staticmethod
    def _rank_signal_summary(rank_signals: Dict[str, Dict]) -> Dict:
        if not rank_signals:
            return {"num_ranks": 0}
        return {
            "num_ranks": len(rank_signals),
            "mean_F_r": float(np.mean([r.get("F_r", 0.0) for r in rank_signals.values()])),
            "mean_G_r": float(np.mean([r.get("G_r", 0.0) for r in rank_signals.values()])),
            "mean_rho_r": float(np.mean([r.get("rho_r", 0.0) for r in rank_signals.values()])),
            "mean_O_r": float(np.mean([r.get("O_r", 0.0) for r in rank_signals.values()])),
            "max_O_r": float(np.max([r.get("O_r", 0.0) for r in rank_signals.values()])),
        }

    def _save_cbr_rank_mask(self, pruning: Dict):
        mask = {}
        for decision in pruning.get("decisions", []):
            key = f"{decision['module_name']}:{decision['rank_id']}"
            mask[key] = 0 if decision.get("applied_route") == "hard_prune" else 1
        torch.save(mask, os.path.join(self.output_dir, "prune_mask.pt"))
