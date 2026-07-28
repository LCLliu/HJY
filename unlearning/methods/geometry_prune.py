import json
import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataloader.llm import seq_to_token_ids
from dataloader.utils import Prompter
from trainer.verb import ManualVerbalizer

from ..evaluation import (
    _candidate_items as evaluation_candidate_items,
    _context_items_for_record as evaluation_context_items_for_record,
)
from .base import BaseUnlearningMethod


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_forget_retain_split(split_metadata_path):
    """Load common inline split metadata schemas.

    run_unlearning.py already resolves split files. This helper is kept here
    for the method-level API and direct unit/smoke use.
    """
    if not split_metadata_path:
        return {"forget_interactions": [], "retain_interactions": []}
    with open(split_metadata_path, "r") as f:
        metadata = json.load(f)

    def rows(*keys):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for inner in ("interactions", key):
                    if isinstance(value.get(inner), list):
                        return value[inner]
        return []

    return {
        "forget_interactions": rows("forget_interactions", "forget", "forget_set"),
        "retain_interactions": rows("retain_interactions", "retain", "retain_set"),
        "metadata": metadata,
    }


def get_user_history(dataset, user_id):
    train = dataset.get("train", {})
    retain = dataset.get("retain_train", {})
    return [int(i) for i in train.get(int(user_id), retain.get(int(user_id), []))]


def remove_forget_item_from_history(history, forget_item_id):
    forget_item_id = int(forget_item_id)
    out = [int(iid) for iid in history]
    for idx in range(len(out) - 1, -1, -1):
        if int(out[idx]) == forget_item_id:
            return out[:idx] + out[idx + 1:]
    return out


def compute_boundary_sensitivity(rank, top_k, boundary_tau):
    tau = max(float(boundary_tau), 1e-6)
    return float(math.exp(-abs(float(rank) - float(top_k)) / tau))


def _minmax(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _as_optional_float(value):
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tokenize_title(title):
    import re

    return set(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _row_uid(row):
    return int(row.get("uid", row.get("user_id")))


def _row_iid(row):
    return int(row.get("iid", row.get("item_id")))


def _row_position(row):
    value = row.get("position", row.get("sequence_index"))
    if value is None or value == "" or value == "null":
        return None
    return int(value)


def _ranking_metrics(records, ks):
    metrics = {}
    ranks = [int(r.get("target_rank", 10**9)) for r in records]
    for k in ks:
        if not ranks:
            metrics[f"Recall@{k}"] = None
            metrics[f"NDCG@{k}"] = None
            continue
        metrics[f"Recall@{k}"] = float(np.mean([1.0 if rank <= k else 0.0 for rank in ranks]))
        metrics[f"NDCG@{k}"] = float(np.mean([
            (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
            for rank in ranks
        ]))
    return metrics


class GeometryPruneMethod(BaseUnlearningMethod):
    """Semantic-protected geometry-aware interaction unlearning.

    The public method name stays ``geometry_prune`` for script compatibility.
    Internally this is the semantic_geometry_prune implementation requested by
    the current protocol.
    """

    method_name = "geometry_prune"

    def run(self):
        method_split_data = self._method_split_data()
        full_split_counts = self._split_counts()
        method_split_counts = self._split_counts_for(method_split_data)
        debug_limit = getattr(self.args, "debug_split_sample_limit", None)
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})
        self.logs.update({
            "status": "running",
            "action": "semantic_geometry_prune",
            "semantic_geometry_prune": True,
            "uses_fixed_split": True,
            "formal_run": bool(getattr(self.args, "formal_run", True)),
            "run_is_formal": bool(getattr(self.args, "run_is_formal", True)),
            "non_formal_reason": getattr(self.args, "non_formal_reason", None),
            "debug_split_sample_limit": debug_limit,
            "debug_split_sample_limit_applied": debug_limit is not None,
            "candidate_level_exposure": True,
            "not_full_item_exposure": True,
            "num_forget_interactions": len(method_split_data.get("forget_interactions", [])),
            "num_retain_interactions": len(method_split_data.get("retain_interactions", [])),
            "num_overlap_retain_interactions": len(method_split_data.get("overlap_retain_interactions", [])),
            "num_semantic_neighbor_retain": len(method_split_data.get("semantic_neighbor_retain", [])),
            "num_collaborative_neighbor_retain": len(method_split_data.get("collaborative_neighbor_retain", [])),
            "loaded_split_fingerprint": (method_split_data.get("metadata") or {}).get("split_fingerprint"),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get("retain_train_excludes_forget_interactions"),
            "forgotten_interactions_in_retain_train": split_diag.get("forgotten_interactions_in_retain_train"),
            "max_eval_samples_applied_to_split": False,
            "split_counts": full_split_counts,
            "full_split_counts": full_split_counts,
            "method_diagnostic_split_counts": method_split_counts,
            "fallbacks": [],
            "notes": [
                "Counterfactual residuals use the LLM ranker on paired original/minus-forget histories.",
                "Parameter intervention is restricted to LoRA rank directions when available.",
            ],
        })
        run_config_path = os.path.join(self.output_dir, "run_config.json")
        if not os.path.exists(run_config_path):
            save_json(run_config_path, vars(self.args).copy())

        self._init_similarity_fallbacks(method_split_data)
        self._enable_lora_gradients()
        residual_candidates = self.build_residual_candidates(method_split_data, self.args)
        self._write_jsonl(
            os.path.join(self.output_dir, "residual_diagnostics.jsonl"),
            residual_candidates,
        )
        save_json(
            os.path.join(self.output_dir, "residual_candidates.json"),
            {"records": residual_candidates, "summary": self._residual_summary(residual_candidates)},
        )

        correction = self.apply_logit_correction(None, residual_candidates, self.args)
        save_json(os.path.join(self.output_dir, "logit_correction.json"), correction)

        before_records = self._score_records(self._all_protocol_records(method_split_data))
        intervention_state = self._save_trainable_state()
        path_grads, path_summary = self.compute_path_loss(
            self.model, residual_candidates, self.args, backward=True
        )
        lora_result = {
            "enabled": False,
            "reason": "direction_suppression_disabled_for_constrained_rank_pruning",
            "num_parameters": 0,
        }
        retain_grads, retain_summary = self.compute_retain_protection(
            self.model,
            self._retain_probe_records(),
            self.args,
        )
        importance = self.compute_parameter_importance(path_grads, retain_grads, self.args)
        pruning = self.apply_semantic_protected_pruning(
            self.model,
            importance,
            {k: v.get("retain_protection", 0.0) for k, v in importance.items()},
            self.args,
            before_records=before_records,
            original_state=intervention_state,
        )
        after_records = self._score_records(
            self._all_protocol_records(method_split_data),
            apply_logit_correction=True,
        )
        rollback_log = self._maybe_rollback(
            before_records=before_records,
            after_records=after_records,
            original_state=intervention_state,
            pruning=pruning,
            lora_result=lora_result,
        )
        if rollback_log["rollback_applied"]:
            after_records = self._score_records(
                self._all_protocol_records(method_split_data),
                apply_logit_correction=True,
            )

        metrics = self.evaluate_unlearning_metrics(
            None, None,
            method_split_data.get("forget_interactions", []),
            method_split_data.get("retain_interactions", []),
            self.args,
            residual_candidates=residual_candidates,
            before_records=before_records,
            after_records=after_records,
        )

        save_json(os.path.join(self.output_dir, "unlearning_metrics.json"), metrics)
        save_json(os.path.join(self.output_dir, "pruning_decisions.json"), pruning)
        save_json(os.path.join(self.output_dir, "pruning_summary.json"), pruning.get("summary", {}))
        save_json(os.path.join(self.output_dir, "rollback_log.json"), rollback_log)
        save_json(os.path.join(self.output_dir, "path_loss_summary.json"), path_summary)
        save_json(os.path.join(self.output_dir, "retain_protection_summary.json"), retain_summary)
        save_json(os.path.join(self.output_dir, "parameter_importance.json"), {
            "records": list(importance.values()),
            "summary": self._importance_summary(importance),
        })
        self._save_prune_mask(pruning)

        self.logs.update({
            "status": "completed",
            "num_residual_candidates": len(residual_candidates),
            "path_loss_summary": path_summary,
            "retain_protection_summary": retain_summary,
            "pruning_summary": pruning.get("summary", {}),
            "lora_suppression_summary": lora_result,
            "rollback": rollback_log,
            "metrics_file": os.path.join(self.output_dir, "unlearning_metrics.json"),
            "residual_diagnostics_file": os.path.join(self.output_dir, "residual_diagnostics.jsonl"),
        })
        self.save_logs()
        self._print_required_metrics(metrics, pruning, rollback_log)
        return self.logs

    def get_candidate_set(self, user_id, forget_item_id, dataset, args, row=None):
        record = self._prediction_record_for("forget", int(user_id), int(forget_item_id), row)
        if record:
            return [int(i) for i in record.get("candidate_items", [])]
        explicit = None
        if row:
            explicit = row.get("candidate_items", row.get("candidates"))
        parsed = self._parse_candidate_items(explicit)
        if parsed:
            if int(forget_item_id) not in parsed:
                parsed = [int(forget_item_id)] + parsed
            return parsed[: int(getattr(args, "llm_negative_sample_size", 19)) + 1]
        return self._fallback_candidate_set(
            int(user_id), int(forget_item_id), self._history_for_row(row, int(user_id)), "forget"
        )

    def ranker_score_candidates(self, model, user_history, candidates, args, target_iid=None, grad=False):
        target_iid = int(target_iid if target_iid is not None else candidates[0])
        return self._score_candidate_list(
            user_history=user_history,
            candidates=candidates,
            target_iid=target_iid,
            grad=grad,
        )

    def compute_counterfactual_residuals(
        self, model, user_id, forget_item_id, history, candidates, args
    ):
        original = self.ranker_score_candidates(
            model, history, candidates, args, target_iid=forget_item_id, grad=False
        )
        cf_history = remove_forget_item_from_history(history, forget_item_id)
        counterfactual = self.ranker_score_candidates(
            model, cf_history, candidates, args, target_iid=forget_item_id, grad=False
        )
        records = []
        target_key = str(int(forget_item_id))
        target_original = float(original["scores"].get(target_key, 0.0))
        target_counterfactual = float(counterfactual["scores"].get(target_key, 0.0))
        target_delta = float(target_original - target_counterfactual)
        for iid in candidates:
            key = str(int(iid))
            score_original = float(original["scores"].get(key, 0.0))
            score_counterfactual = float(counterfactual["scores"].get(key, 0.0))
            residual_delta = float(score_original - score_counterfactual)
            records.append({
                "user_id": int(user_id),
                "uid": int(user_id),
                "forget_item_id": int(forget_item_id),
                "forget_iid": int(forget_item_id),
                "candidate_item_id": int(iid),
                "candidate_iid": int(iid),
                "score_original": score_original,
                "score_counterfactual": score_counterfactual,
                "residual_delta": residual_delta,
                "residual_score": residual_delta,
                "rank_original": int(original["ranks"].get(key, 10**9)),
                "rank_counterfactual": int(counterfactual["ranks"].get(key, 10**9)),
                "topk_boundary_score_original": float(original.get("topk_boundary_score", 0.0)),
                "margin_to_topk_boundary_original": float(
                    score_original - float(original.get("topk_boundary_score", 0.0))
                ),
                "original_history_len": len(history),
                "counterfactual_history_len": len(cf_history),
                "target_score_original": target_original,
                "target_score_counterfactual": target_counterfactual,
                "target_residual_delta": target_delta,
            })
        return records

    def compute_collaborative_similarity(
        self, forget_item_id, candidate_item_id, collab_emb=None, interaction_graph=None
    ):
        if collab_emb is not None:
            try:
                a = collab_emb[int(forget_item_id)]
                b = collab_emb[int(candidate_item_id)]
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                return float(np.dot(a, b) / denom) if denom > 0 else 0.0
            except Exception:
                pass
        users_f = self._item_users.get(int(forget_item_id), set())
        users_c = self._item_users.get(int(candidate_item_id), set())
        if not users_f or not users_c:
            return 0.0
        inter = len(users_f & users_c)
        union = len(users_f | users_c)
        return float(inter / union) if union else 0.0

    def compute_semantic_protection(
        self, candidate_item_id, retain_history_items, sem_emb=None, item_texts=None
    ):
        if sem_emb is not None:
            best = 0.0
            try:
                a = sem_emb[int(candidate_item_id)]
                for iid in retain_history_items:
                    b = sem_emb[int(iid)]
                    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                    if denom > 0:
                        best = max(best, float(np.dot(a, b) / denom))
                return best
            except Exception:
                pass
        cand_tokens = self._title_tokens.get(int(candidate_item_id), set())
        if not cand_tokens:
            return 0.0
        best = 0.0
        for iid in retain_history_items:
            other = self._title_tokens.get(int(iid), set())
            if not other:
                continue
            union = len(cand_tokens | other)
            best = max(best, float(len(cand_tokens & other) / union) if union else 0.0)
        return best

    def _boundary_top_k(self, args):
        value = getattr(args, "boundary_top_k", None)
        if value is None:
            value = getattr(args, "topk_boundary", None)
        if value is None:
            value = max(getattr(args, "rerank_metric_ks", [10]))
        return max(1, int(value))

    def _candidate_boundary_pass(self, info, args):
        margin = _as_optional_float(getattr(args, "boundary_score_margin", None))
        if margin is not None:
            return abs(float(info.get("margin_to_topk_boundary_original", 0.0))) <= margin
        top_k = self._boundary_top_k(args)
        window = int(getattr(args, "boundary_window", 10) or 10)
        return abs(int(info.get("rank_original", 10**9)) - top_k) <= max(0, window)

    def _sample_null_removal_items(self, uid, history, forget_iid, args):
        requested = max(0, int(getattr(args, "num_null_removals", 5) or 5))
        if requested <= 0:
            return []
        retain_seq = self.dataset_data.get("retain_train", self.dataset_data.get("train", {})).get(int(uid), [])
        retain_items = [int(i) for i in retain_seq if int(i) != int(forget_iid)]
        history_items = [int(i) for i in history if int(i) != int(forget_iid)]
        pool = sorted(set(retain_items) & set(history_items))
        if not pool:
            return []
        seed_payload = f"{getattr(args, 'seed', 42)}|null|{int(uid)}|{int(forget_iid)}"
        import hashlib

        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        replace = len(pool) < requested
        sampled = rng.choice(pool, size=requested if replace else min(requested, len(pool)), replace=replace)
        return [int(i) for i in sampled.tolist()]

    def _null_residual_stats(self, uid, forget_iid, history, candidates, original_scores, args):
        sampled_items = self._sample_null_removal_items(uid, history, forget_iid, args)
        if not sampled_items:
            return None, {
                "skipped": True,
                "reason": "insufficient_retain_history_for_null_removals",
                "requested": int(getattr(args, "num_null_removals", 5) or 5),
                "actual": 0,
            }

        null_by_candidate = {str(int(iid)): [] for iid in candidates}
        for retain_iid in sampled_items:
            null_history = remove_forget_item_from_history(history, retain_iid)
            scores = self._score_candidate_list(
                null_history,
                candidates,
                int(forget_iid),
                grad=False,
            )
            for iid in candidates:
                key = str(int(iid))
                null_delta = float(original_scores.get(key, 0.0)) - float(scores["scores"].get(key, 0.0))
                null_by_candidate[key].append(null_delta)

        eps = float(getattr(args, "null_eps", 1e-8) or 1e-8)
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
            "requested": int(getattr(args, "num_null_removals", 5) or 5),
            "actual": len(sampled_items),
        }

    def _collab_gate_items(self, residual_info, collab_scores, args):
        positive = [
            (int(info["candidate_item_id"]), float(collab_scores.get(int(info["candidate_item_id"]), 0.0)))
            for info in residual_info
            if float(collab_scores.get(int(info["candidate_item_id"]), 0.0)) > 0.0
        ]
        if not positive:
            return set()
        positive.sort(key=lambda x: x[1], reverse=True)
        top_n = getattr(args, "collab_top_n", None)
        if top_n is not None:
            try:
                keep = max(0, min(int(top_n), len(positive)))
            except (TypeError, ValueError):
                keep = 0
        else:
            top_q = float(getattr(args, "collab_top_q", 0.2) or 0.2)
            top_q = max(0.0, min(1.0, top_q))
            keep = int(math.ceil(len(positive) * top_q))
        keep = max(1, keep) if positive and keep > 0 else keep
        return {iid for iid, _ in positive[:keep]}

    def build_residual_candidates(self, split_data, args):
        final_records = []
        forget_rows = split_data.get("forget_interactions", [])
        tau_z = float(getattr(args, "tau_residual_z", 1.0) or 1.0)
        tau_sem = float(getattr(args, "tau_semantic_protect", 0.5) or 0.5)
        stats = {
            "num_candidates_total": 0,
            "num_residual_z_pass": 0,
            "num_boundary_pass": 0,
            "num_collab_pass": 0,
            "num_semantic_protected": 0,
            "num_final_S_res": 0,
            "num_forget_samples": len(forget_rows),
            "num_forget_samples_skipped_null": 0,
            "skip_reasons": {},
        }
        null_summary = {
            "num_null_removals": int(getattr(args, "num_null_removals", 5) or 5),
            "tau_residual_z": tau_z,
            "mean_z": 0.0,
            "max_z": 0.0,
            "num_z_values": 0,
            "num_null_samples_used": 0,
            "num_null_samples_skipped": 0,
        }
        z_values = []
        for row in forget_rows:
            uid = _row_uid(row)
            forget_iid = _row_iid(row)
            history = self._original_history_for_forget_row(row, uid, forget_iid)
            counterfactual_history = remove_forget_item_from_history(history, forget_iid)
            candidates = self.get_candidate_set(uid, forget_iid, self.dataset_data, args, row=row)
            residual_info = self.compute_counterfactual_residuals(
                self.model, uid, forget_iid, history, candidates, args
            )
            self._record_residual_diagnostic(uid, forget_iid, residual_info, history, counterfactual_history)
            retain_history = self._retain_history_for_user(uid, exclude_item=forget_iid)
            stats["num_candidates_total"] += len(residual_info)
            original_scores = {
                str(int(info["candidate_item_id"])): float(info.get("score_original", 0.0))
                for info in residual_info
            }
            null_stats, null_log = self._null_residual_stats(
                uid, forget_iid, history, candidates, original_scores, args
            )
            if null_log["skipped"]:
                stats["num_forget_samples_skipped_null"] += 1
                reason = null_log["reason"]
                stats["skip_reasons"][reason] = int(stats["skip_reasons"].get(reason, 0)) + 1
                null_summary["num_null_samples_skipped"] += 1
                continue
            null_summary["num_null_samples_used"] += int(null_log["actual"])

            collab_scores = {
                int(info["candidate_item_id"]): self.compute_collaborative_similarity(
                    forget_iid, info["candidate_item_id"]
                )
                for info in residual_info
            }
            collab_gate = self._collab_gate_items(residual_info, collab_scores, args)

            for info in residual_info:
                candidate_iid = int(info["candidate_item_id"])
                key = str(candidate_iid)
                nstat = null_stats.get(key, {"mean": 0.0, "std": float(getattr(args, "null_eps", 1e-8))})
                z_score = (
                    (float(info.get("residual_score", 0.0)) - float(nstat["mean"])) /
                    max(float(nstat["std"]), float(getattr(args, "null_eps", 1e-8) or 1e-8))
                )
                z_values.append(float(z_score))
                boundary_pass = self._candidate_boundary_pass(info, args)
                collab_pass = candidate_iid in collab_gate
                semantic_protection = self.compute_semantic_protection(candidate_iid, retain_history)
                semantic_protected = semantic_protection > tau_sem
                residual_pass = z_score > tau_z

                stats["num_residual_z_pass"] += int(residual_pass)
                stats["num_boundary_pass"] += int(boundary_pass)
                stats["num_collab_pass"] += int(collab_pass)
                stats["num_semantic_protected"] += int(semantic_protected)
                if not (residual_pass and boundary_pass and collab_pass and not semantic_protected):
                    continue

                enriched = {
                    **info,
                    "history": [int(i) for i in history[-int(getattr(args, "llm_max_history", 20)):]],
                    "counterfactual_history": [
                        int(i) for i in counterfactual_history[-int(getattr(args, "llm_max_history", 20)):]
                    ],
                    "candidate_items": [int(i) for i in candidates],
                    "residual_score": float(info.get("residual_score", 0.0)),
                    "residual_null_mean": float(nstat["mean"]),
                    "residual_null_std": float(nstat["std"]),
                    "residual_z": float(z_score),
                    "boundary_pass": bool(boundary_pass),
                    "collab_pass": bool(collab_pass),
                    "collaborative_similarity": float(collab_scores.get(candidate_iid, 0.0)),
                    "semantic_protection": float(semantic_protection),
                    "semantic_protected": bool(semantic_protected),
                    "is_final_residual_candidate": True,
                    "target_iid": candidate_iid,
                }
                final_records.append(enriched)
        final_records.sort(key=lambda r: float(r.get("residual_z", 0.0)), reverse=True)
        top_m = int(getattr(args, "residual_top_m", 64) or 64)
        if top_m > 0:
            final_records = final_records[:top_m]
        stats["num_final_S_res"] = len(final_records)
        if z_values:
            null_summary["mean_z"] = float(np.mean(z_values))
            null_summary["max_z"] = float(np.max(z_values))
            null_summary["num_z_values"] = len(z_values)
        if not final_records:
            self.logs.setdefault("warnings", []).append(
                "no residual candidates after constrained filtering"
            )
        self._candidate_selection_summary = stats
        self._null_residual_summary = null_summary
        return final_records

    def compute_path_loss(self, model, residual_candidates, args, backward=False):
        if backward:
            self._zero_trainable_grads()
        if not residual_candidates:
            return {}, {"num_candidates": 0, "loss": 0.0, "path_weight": 0.0}
        max_items = int(getattr(args, "residual_top_m", 64) or 64)
        selected_candidates = residual_candidates[:max_items]
        loss_value = 0.0
        num_terms = 0
        weights = []
        for cand in selected_candidates:
            target_iid = int(cand["candidate_item_id"])
            scores = self._score_candidate_list(
                cand["history"],
                [target_iid],
                target_iid,
                grad=backward,
                candidate_chunk_size=1,
            )
            key = str(target_iid)
            score = scores["score_tensor"][scores["candidate_index"][key]]
            z_clip = float(getattr(args, "z_clip", 5.0) or 5.0)
            weight = float(np.clip(float(cand.get("residual_z", 0.0)), 0.0, z_clip))
            margin = float(getattr(args, "path_margin", 0.0))
            if margin > 0:
                cf_score = float(cand.get("score_counterfactual", 0.0))
                term = weight * F.relu(score - score.new_tensor(cf_score) + margin)
            else:
                term = weight * score
            loss_value += float(term.detach().cpu())
            num_terms += 1
            weights.append(weight)
            if backward and term.requires_grad:
                term.backward()
            del scores, score, term
            torch.cuda.empty_cache()
        grads = {}
        if backward and num_terms > 0:
            grads = self._collect_lora_rank_grad_norms()
        summary = {
            "num_candidates": num_terms,
            "loss": loss_value,
            "path_weight": float(np.mean(weights)) if weights else 0.0,
        }
        return grads, summary

    def compute_retain_protection(self, model, retain_samples, args):
        self._zero_trainable_grads()
        if not retain_samples:
            return {}, {"num_samples": 0, "loss": 0.0}
        losses = []
        for record in retain_samples:
            scores = self._score_candidate_list(
                record.get("context_items", []),
                record.get("candidate_items", []),
                int(record["target_iid"]),
                grad=True,
            )
            key = str(int(record["target_iid"]))
            idx = scores["candidate_index"][key]
            label = torch.tensor([idx], dtype=torch.long, device=scores["score_tensor"].device)
            losses.append(F.cross_entropy(scores["score_tensor"].unsqueeze(0), label))
        loss = torch.stack(losses).mean()
        loss.backward()
        grads = self._collect_lora_rank_grad_norms()
        self._zero_trainable_grads()
        return grads, {"num_samples": len(losses), "loss": float(loss.detach().cpu())}

    def compute_parameter_importance(self, path_grads, retain_grads_or_activations, args):
        keys = sorted(set(path_grads.keys()) | set(retain_grads_or_activations.keys()))
        forget_values = np.asarray([float(path_grads.get(k, 0.0)) for k in keys], dtype=np.float64)
        retain_values = np.asarray([float(retain_grads_or_activations.get(k, 0.0)) for k in keys], dtype=np.float64)
        forget_norm = _minmax(forget_values)
        retain_norm = _minmax(retain_values)
        delta = float(getattr(args, "rank_score_delta", 1e-8) or 1e-8)
        out = {}
        for idx, key in enumerate(keys):
            module_name, rank_s = key.rsplit(":", 1)
            forget_grad = float(forget_values[idx])
            retain_grad = float(retain_values[idx])
            # F_r is the forget residual-path rank gradient norm. G_r is the
            # retain-loss rank gradient norm. P_r conservatively reuses
            # normalized G_r as rank-level protection when no separate
            # semantic rank score is available.
            rank_protection = float(retain_norm[idx])
            out[key] = {
                "parameter_key": key,
                "module_name": module_name,
                "rank_id": int(rank_s),
                "F_r": forget_grad,
                "G_r": retain_grad,
                "P_r": rank_protection,
                "forget_grad_norm": forget_grad,
                "retain_grad_norm": retain_grad,
                "forget_grad_norm_normalized": float(forget_norm[idx]),
                "retain_grad_norm_normalized": float(retain_norm[idx]),
                "rank_score": float(forget_grad / (retain_grad + delta)),
                "retain_protection": rank_protection,
            }
        return out

    def apply_logit_correction(self, scores, residual_candidates, args):
        enabled = bool(getattr(args, "enable_logit_correction", True))
        eta = float(getattr(args, "eta_logit", 0.1))
        corrections = {}
        if enabled:
            for cand in residual_candidates:
                uid = str(int(cand["user_id"]))
                iid = str(int(cand["candidate_item_id"]))
                penalty = eta * max(0.0, float(cand.get("residual_z", 0.0)))
                corrections.setdefault(uid, {})
                corrections[uid][iid] = max(float(corrections[uid].get(iid, 0.0)), float(penalty))
        return {
            "enabled": enabled,
            "eta_logit": eta,
            "corrections": corrections,
            "num_users": len(corrections),
            "num_items": sum(len(v) for v in corrections.values()),
            "note": "Applied by unlearning.evaluation.collect_predictions during after-stage scoring.",
        }

    def apply_lora_direction_suppression(self, model, path_loss, retain_loss, args):
        enabled = bool(getattr(args, "enable_lora_suppression", True))
        rho = float(getattr(args, "lora_suppression_rho", 0.05))
        if not enabled:
            return {"enabled": False, "num_parameters": 0}
        changed = 0
        total_update_norm = 0.0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_" not in name or param.grad is None:
                    continue
                if name not in self._original_trainable:
                    self._original_trainable[name] = param.data.detach().clone()
                grad = param.grad.detach()
                denom = torch.sum(grad.float() * grad.float())
                if denom <= 0:
                    continue
                coeff = torch.sum(param.data.float() * grad.float()) / (denom + 1e-12)
                update = rho * coeff.to(param.data.dtype) * grad
                param.data -= update
                total_update_norm += float(update.float().norm().detach().cpu())
                changed += 1
        self._zero_trainable_grads()
        return {
            "enabled": True,
            "rho": rho,
            "num_parameters": changed,
            "total_update_norm": total_update_norm,
            "fallback": changed == 0,
        }

    def apply_semantic_protected_pruning(
        self,
        model,
        importance_scores,
        retain_protection,
        args,
        before_records=None,
        original_state=None,
    ):
        enabled = bool(getattr(args, "enable_semantic_protected_prune", True))
        pairs = self._collect_lora_pairs()
        decisions = []
        total_ranks = sum(pair["rank"] for pair in pairs.values())
        candidate_summary = getattr(self, "_candidate_selection_summary", {})
        null_summary = getattr(self, "_null_residual_summary", {})
        method_flags = {
            "use_weighted_candidate_score": False,
            "use_constrained_candidate_filtering": True,
            "use_constrained_rank_pruning": True,
        }
        if not enabled or not pairs or not importance_scores:
            reason = "disabled_or_no_lora_or_no_importance"
            if candidate_summary.get("num_final_S_res") == 0:
                reason = "no_residual_candidates_after_constrained_filtering"
            if not enabled:
                reason = "hard_prune_disabled"
            epsilon_value = getattr(args, "epsilon_retain", None)
            if epsilon_value is None:
                epsilon_value = getattr(args, "retain_drop_tolerance", 0.05)
            summary = {
                "enabled": enabled,
                "total_ranks": total_ranks,
                "hard_prune": 0,
                "soft_suppress": 0,
                "protect": total_ranks,
                "actual_intervention_ratio": 0.0,
                "reason": reason,
                "candidate_selection": candidate_summary,
                "null_residual": null_summary,
                "rank_pruning": {
                    "total_ranks": total_ranks,
                    "protected_ranks": total_ranks,
                    "candidate_ranks": 0,
                    "hard_pruned_ranks": 0,
                    "prune_budget": 0,
                    "epsilon_retain": float(epsilon_value),
                    "retain_drop": 0.0,
                    "rollback": False,
                    "top_pruned_rank_scores": [],
                    "reason": reason,
                },
                "method_flags": method_flags,
            }
            return {"decisions": decisions, "summary": summary}

        tau_rank_protect = float(getattr(args, "tau_rank_protect", 0.7) or 0.7)
        delta = float(getattr(args, "rank_score_delta", 1e-8) or 1e-8)
        epsilon_value = getattr(args, "epsilon_retain", None)
        if epsilon_value is None:
            epsilon_value = getattr(args, "retain_drop_tolerance", 0.05)
        epsilon = float(epsilon_value)
        budget_ratio_value = getattr(args, "prune_budget_ratio", None)
        if budget_ratio_value is None:
            budget_ratio_value = getattr(args, "prune_ratio", 0.01)
        budget_ratio = float(budget_ratio_value or 0.0)
        max_ratio = float(getattr(args, "max_prune_ratio", 0.05) or 0.0)
        prune_budget = min(
            int(total_ranks * max(0.0, budget_ratio)),
            int(total_ranks * max(0.0, max_ratio)),
        )
        batch_size = max(1, int(getattr(args, "prune_batch_size", 1) or 1))

        rank_records = []
        for module_name, pair in pairs.items():
            for rid in range(pair["rank"]):
                key = f"{module_name}:{rid}"
                imp = importance_scores.get(key, {})
                f_r = float(imp.get("F_r", 0.0))
                g_r = float(imp.get("G_r", 0.0))
                p_r = float(imp.get("P_r", imp.get("retain_protection", 1.0)))
                protected = p_r > tau_rank_protect
                score = f_r / (g_r + delta)
                record = {
                    "module_name": module_name,
                    "rank_id": int(rid),
                    "parameter_key": key,
                    "F_r": f_r,
                    "G_r": g_r,
                    "P_r": p_r,
                    "rank_score": float(score),
                    "route": "protect",
                    "suppression_strength": 0.0,
                    "route_reason": "rank_level_protected" if protected else "not_selected",
                    "protected": bool(protected),
                }
                rank_records.append(record)

        candidates = [
            r for r in rank_records
            if not r["protected"] and float(r.get("F_r", 0.0)) > 0.0
        ]
        candidates.sort(key=lambda r: float(r.get("rank_score", 0.0)), reverse=True)
        selected_pool = candidates[:prune_budget] if prune_budget > 0 else []
        selected_keys = {(r["module_name"], int(r["rank_id"])) for r in selected_pool}

        for record in rank_records:
            key = (record["module_name"], int(record["rank_id"]))
            if key in selected_keys:
                record["route_reason"] = "selected_by_constrained_greedy_score"
            decisions.append(record)

        hard_keys = set()
        attempted = []
        rollback = False
        retain_drop = 0.0
        rollback_reason = None
        state_for_rollback = original_state or self._save_trainable_state()
        retain_eval_records = [
            dict(r) for r in self._all_protocol_records(self._method_split_data())
            if r.get("split_tag") == "retain"
        ]
        before_retain_records = [
            dict(r) for r in (before_records or [])
            if r.get("split_tag") == "retain"
        ]
        if not before_retain_records and retain_eval_records:
            before_retain_records = self._score_records(retain_eval_records)

        for start in range(0, len(selected_pool), batch_size):
            batch = selected_pool[start:start + batch_size]
            if not batch:
                continue
            with torch.no_grad():
                for cand in batch:
                    module_name = cand["module_name"]
                    rid = int(cand["rank_id"])
                    pair = pairs[module_name]
                    if pair["A_name"] not in self._original_trainable:
                        self._original_trainable[pair["A_name"]] = pair["A"].data.detach().clone()
                    if pair["B_name"] not in self._original_trainable:
                        self._original_trainable[pair["B_name"]] = pair["B"].data.detach().clone()
                    # PEFT LoRA uses A:[rank, in_features], B:[out_features, rank],
                    # so the same rank direction is A[rid, :] and B[:, rid].
                    pair["A"].data[rid, :] = 0.0
                    pair["B"].data[:, rid] = 0.0
                    hard_keys.add((module_name, rid))
                    attempted.append(dict(cand))

            after_retain_records = (
                self._score_records(retain_eval_records)
                if retain_eval_records else []
            )
            retain_drop = self._max_retain_drop(before_retain_records, after_retain_records)
            if retain_drop > epsilon:
                rollback = True
                rollback_reason = "retain_drop_exceeded"
                self._restore_trainable_state(state_for_rollback)
                hard_keys.clear()
                break

        hard = len(hard_keys)
        soft = 0
        protect = total_ranks - hard
        for decision in decisions:
            key = (decision["module_name"], int(decision["rank_id"]))
            if key in hard_keys:
                decision["route"] = "hard_prune"
                decision["suppression_strength"] = 1.0
                decision["route_reason"] = "constrained_greedy_within_retain_budget"
            elif rollback and key in selected_keys:
                decision["route"] = "protect"
                decision["suppression_strength"] = 0.0
                decision["attempted_route"] = "hard_prune"
                decision["route_reason"] = "rollback_retain_drop_exceeded"

        if candidate_summary.get("num_final_S_res") == 0:
            reason = "S_res_empty"
        elif prune_budget <= 0:
            reason = "prune_budget_zero"
        elif not candidates:
            reason = "all_ranks_protected_or_zero_forget_gradient"
        elif rollback:
            reason = "retain_drop_exceeded_after_rollback"
        elif hard == 0:
            reason = "no_rank_selected"
        else:
            reason = "completed"

        total = len(decisions)
        top_pruned = [
            {
                "module_name": d["module_name"],
                "rank_id": int(d["rank_id"]),
                "rank_score": float(d.get("rank_score", 0.0)),
                "F_r": float(d.get("F_r", 0.0)),
                "G_r": float(d.get("G_r", 0.0)),
                "P_r": float(d.get("P_r", 0.0)),
            }
            for d in decisions
            if d.get("route") == "hard_prune"
        ][:20]
        top_attempted = [
            {
                "module_name": d["module_name"],
                "rank_id": int(d["rank_id"]),
                "rank_score": float(d.get("rank_score", 0.0)),
                "F_r": float(d.get("F_r", 0.0)),
                "G_r": float(d.get("G_r", 0.0)),
                "P_r": float(d.get("P_r", 0.0)),
            }
            for d in attempted
        ][:20]
        summary = {
            "enabled": True,
            "total_ranks": total,
            "hard_prune": hard,
            "soft_suppress": soft,
            "protect": protect,
            "hard_prune_ratio": float(hard) / float(total) if total else 0.0,
            "soft_suppress_ratio": 0.0,
            "protect_ratio": float(protect) / float(total) if total else 0.0,
            "actual_intervention_ratio": float(hard + soft) / float(total) if total else 0.0,
            "prune_ratio": budget_ratio,
            "prune_budget_ratio": budget_ratio,
            "max_prune_ratio": max_ratio,
            "tau_rank_protect": tau_rank_protect,
            "rank_score_delta": delta,
            "reason": reason,
            "candidate_selection": candidate_summary,
            "null_residual": null_summary,
            "rank_pruning": {
                "total_ranks": total,
                "protected_ranks": sum(1 for d in decisions if d.get("protected")),
                "candidate_ranks": len(candidates),
                "hard_pruned_ranks": hard,
                "attempted_hard_pruned_ranks": len(attempted),
                "prune_budget": prune_budget,
                "epsilon_retain": epsilon,
                "retain_drop": retain_drop,
                "rollback": rollback,
                "rollback_reason": rollback_reason,
                "top_pruned_rank_scores": top_pruned,
                "top_attempted_rank_scores": top_attempted,
            },
            "method_flags": method_flags,
        }
        if rollback:
            summary["rolled_back"] = True
            summary["rollback_reason"] = rollback_reason
        return {"decisions": decisions, "summary": summary}

    def evaluate_unlearning_metrics(
        self, model_before, model_after, forget_set, retain_set, args,
        residual_candidates=None, before_records=None, after_records=None,
    ):
        ks = list(getattr(args, "rerank_metric_ks", [1, 5, 10]))
        before_records = before_records or self._score_records(self._all_protocol_records())
        after_records = after_records or self._score_records(self._all_protocol_records())
        before_by_id = {r["prediction_id"]: r for r in before_records}
        paired_after = [r for r in after_records if r.get("prediction_id") in before_by_id]

        def by_tag(records, tag):
            return [r for r in records if r.get("split_tag") == tag]

        before_forget = by_tag(before_records, "forget")
        after_forget = by_tag(after_records, "forget")
        before_retain = by_tag(before_records, "retain")
        after_retain = by_tag(after_records, "retain")

        crg_before = self._counterfactual_residual_gap(residual_candidates or [], use_after=False)
        crg_after = self._counterfactual_residual_gap(residual_candidates or [], use_after=True)
        rphr_before = self._residual_path_hit_rate(residual_candidates or [], use_after=False)
        rphr_after = self._residual_path_hit_rate(residual_candidates or [], use_after=True)
        spd = self._semantic_protected_drop(before_records, after_records)
        nrd = self._neighbor_retain_drop(before_records, after_records)
        bfr = self._boundary_flip_rate(before_by_id, paired_after, ks)

        metrics = {
            "Forget Recall@K before": {k: _ranking_metrics(before_forget, [k])[f"Recall@{k}"] for k in ks},
            "Forget Recall@K after": {k: _ranking_metrics(after_forget, [k])[f"Recall@{k}"] for k in ks},
            "Forget NDCG@K before": {k: _ranking_metrics(before_forget, [k])[f"NDCG@{k}"] for k in ks},
            "Forget NDCG@K after": {k: _ranking_metrics(after_forget, [k])[f"NDCG@{k}"] for k in ks},
            "Retain Recall@K before": {k: _ranking_metrics(before_retain, [k])[f"Recall@{k}"] for k in ks},
            "Retain Recall@K after": {k: _ranking_metrics(after_retain, [k])[f"Recall@{k}"] for k in ks},
            "Retain NDCG@K before": {k: _ranking_metrics(before_retain, [k])[f"NDCG@{k}"] for k in ks},
            "Retain NDCG@K after": {k: _ranking_metrics(after_retain, [k])[f"NDCG@{k}"] for k in ks},
            "CRG before": crg_before,
            "CRG after": crg_after,
            "RPHR before": rphr_before,
            "RPHR after": rphr_after,
            "SPD": spd,
            "NRD": nrd,
            "BFR": bfr,
            "metric_ks": ks,
            "num_residual_candidates": len(residual_candidates or []),
        }
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_similarity_fallbacks(self, split_data):
        self._item_users = {}
        retain_train = self.dataset_data.get("retain_train", self.dataset_data.get("train", {}))
        for uid, seq in retain_train.items():
            for iid in set(seq):
                self._item_users.setdefault(int(iid), set()).add(int(uid))
        self._title_tokens = {
            int(iid): _tokenize_title(title)
            for iid, title in self.dataset_data.get("meta", {}).items()
        }
        if not self._title_tokens:
            self.logs["fallbacks"].append("semantic_protection_no_item_text_uses_zero")
        self._prediction_lookup = {}
        for record in self.predictions_before.get("records", []):
            key = (
                record.get("split_tag"),
                int(record.get("uid")),
                int(record.get("target_iid")),
                record.get("position"),
            )
            self._prediction_lookup[key] = record
        self._original_trainable = {}

    def _enable_lora_gradients(self):
        enabled = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                enabled += 1
        if enabled == 0:
            self.logs["fallbacks"].append("no_trainable_lora_parameters_for_path_gradients")
        self.logs["num_lora_gradient_parameters"] = enabled

    def _method_split_data(self):
        limit = getattr(self.args, "debug_split_sample_limit", None)
        if limit is None:
            return self.split_data
        limit = int(limit)
        out = dict(self.split_data)
        for key in (
            "forget_interactions",
            "retain_interactions",
            "overlap_retain_interactions",
            "semantic_neighbor_retain",
            "collaborative_neighbor_retain",
        ):
            out[key] = list(self.split_data.get(key, []))[:limit]
        return out

    @staticmethod
    def _split_counts_for(split_data):
        return {
            "forget": len(split_data.get("forget_interactions", [])),
            "retain": len(split_data.get("retain_interactions", [])),
            "overlap_retain": len(split_data.get("overlap_retain_interactions", [])),
            "semantic_neighbor_retain": len(split_data.get("semantic_neighbor_retain", [])),
            "collaborative_neighbor_retain": len(split_data.get("collaborative_neighbor_retain", [])),
        }

    def _history_for_row(self, row, uid):
        if row:
            parsed = self._parse_candidate_items(row.get("history", row.get("context_items")))
            if parsed:
                return parsed[-int(getattr(self.args, "llm_max_history", 20)):]
        history = get_user_history(self.dataset_data, uid)
        pos = _row_position(row or {})
        if pos is not None and history:
            return [int(i) for i in history[:pos]][-int(getattr(self.args, "llm_max_history", 20)):]
        return [int(i) for i in history][-int(getattr(self.args, "llm_max_history", 20)):]

    def _original_history_for_forget_row(self, row, uid, forget_iid):
        max_history = int(getattr(self.args, "llm_max_history", 20))
        if row:
            parsed = self._parse_candidate_items(row.get("history", row.get("context_items")))
            if parsed:
                if int(forget_iid) not in [int(i) for i in parsed]:
                    parsed = [*parsed, int(forget_iid)]
                return [int(i) for i in parsed][-max_history:]
        history = get_user_history(self.dataset_data, uid)
        pos = _row_position(row or {})
        if pos is not None and history:
            end = min(len(history), max(0, pos) + 1)
            original = [int(i) for i in history[:end]]
            if pos < len(history) and int(history[pos]) != int(forget_iid):
                original.append(int(forget_iid))
            return original[-max_history:]
        original = [int(i) for i in history]
        if int(forget_iid) not in original:
            original.append(int(forget_iid))
        return original[-max_history:]

    def _record_residual_diagnostic(self, uid, forget_iid, residual_info, original_history, counterfactual_history):
        match = next(
            (r for r in residual_info if int(r.get("candidate_item_id")) == int(forget_iid)),
            residual_info[0] if residual_info else {},
        )
        diagnostic = {
            "user_id": int(uid),
            "forget_item_id": int(forget_iid),
            "original_history_len": len(original_history),
            "counterfactual_history_len": len(counterfactual_history),
            "score_original": float(match.get("score_original", 0.0)),
            "score_counterfactual": float(match.get("score_counterfactual", 0.0)),
            "residual_delta": float(match.get("residual_delta", 0.0)),
        }
        self.logs["last_counterfactual_residual"] = diagnostic
        summary = self.logs.setdefault("counterfactual_residual_summary", {
            "num_records": 0,
            "nonzero_residual_delta": 0,
            "residual_delta_abs_sum": 0.0,
        })
        summary["num_records"] += 1
        if abs(diagnostic["residual_delta"]) > 1e-12:
            summary["nonzero_residual_delta"] += 1
        summary["residual_delta_abs_sum"] += abs(diagnostic["residual_delta"])
        summary["mean_abs_residual_delta"] = (
            summary["residual_delta_abs_sum"] / max(1, summary["num_records"])
        )

    def _retain_history_for_user(self, uid, exclude_item=None):
        seq = self.dataset_data.get("retain_train", self.dataset_data.get("train", {})).get(int(uid), [])
        if exclude_item is not None:
            seq = [int(i) for i in seq if int(i) != int(exclude_item)]
        return [int(i) for i in seq][-int(getattr(self.args, "llm_max_history", 20)):]

    def _prediction_record_for(self, split_tag, uid, iid, row=None):
        pos = _row_position(row or {})
        key = (split_tag, int(uid), int(iid), pos)
        prediction_lookup = getattr(self, "_prediction_lookup", {})
        if key in prediction_lookup:
            return prediction_lookup[key]
        for record in self.predictions_before.get("records", []):
            if (
                record.get("split_tag") == split_tag and
                int(record.get("uid")) == int(uid) and
                int(record.get("target_iid")) == int(iid)
            ):
                return record
        return None

    @staticmethod
    def _parse_candidate_items(value):
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, tuple):
            return [int(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [int(v) for v in parsed]
            except Exception:
                pass
            import re

            return [int(v) for v in re.findall(r"\d+", text)]
        return []

    def _fallback_candidate_set(self, uid, target_iid, context_items, split_tag):
        import hashlib

        num_items = len(self.dataset_data.get("meta", {}))
        size = int(getattr(self.args, "llm_negative_sample_size", 19)) + 1
        seed_payload = f"{getattr(self.args, 'seed', 42)}|{uid}|{target_iid}|{split_tag}"
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        candidates = [int(target_iid)]
        blocked = set(int(i) for i in context_items) | {int(target_iid), 0}
        while len(candidates) < size and num_items > 0:
            iid = int(rng.randint(1, num_items + 1))
            if iid not in blocked and iid not in candidates:
                candidates.append(iid)
        rng.shuffle(candidates)
        return candidates

    def _verbalizer(self):
        return ManualVerbalizer(
            tokenizer=self.tokenizer,
            prefix="",
            post_log_softmax=False,
            classes=list(range(self.args.llm_negative_sample_size + 1)),
            label_words={
                i: chr(ord("A") + i)
                for i in range(self.args.llm_negative_sample_size + 1)
            },
        )

    def _score_candidate_list(
        self,
        user_history,
        candidates,
        target_iid,
        grad=False,
        score_batch_size=None,
        candidate_chunk_size=None,
    ):
        if not candidates:
            raise ValueError("candidate list must be non-empty")
        meta = self.dataset_data.setdefault("meta", {})
        candidates = [int(i) for i in candidates]
        target_iid = int(target_iid)
        if target_iid not in candidates:
            candidates = [target_iid] + candidates
        for iid in list(candidates) + [int(i) for i in user_history]:
            meta.setdefault(int(iid), f"Item {int(iid)}")
        device = next(self.model.parameters()).device
        prompter = Prompter()
        verbalizer = self._verbalizer()
        if grad:
            verbalizer = verbalizer.to(device)

        candidate_count = len(candidates)
        score_batch_size = candidate_count
        self.model.eval()
        tokenized = seq_to_token_ids(
            self.args,
            [int(i) for i in user_history],
            candidates,
            target_iid,
            meta,
            self.tokenizer,
            prompter,
            eval=True,
        )
        batch = {
            "input_ids": torch.tensor([tokenized["input_ids"]], dtype=torch.long).to(device),
            "attention_mask": torch.tensor([tokenized["attention_mask"]], dtype=torch.long).to(device),
        }
        if grad:
            outputs = self.model(**batch)
            class_scores = verbalizer.process_logits(outputs.logits.float())[0]
            score_tensor = class_scores[:candidate_count]
            detached = score_tensor.detach().float().cpu().tolist()
        else:
            with torch.no_grad():
                outputs = self.model(**batch)
                class_scores = verbalizer.process_logits(outputs.logits.float().cpu())[0]
                score_tensor = class_scores[:candidate_count].detach().float().cpu()
                detached = score_tensor.tolist()
        self._log_score_chunk(candidate_count, score_batch_size, 0)
        del batch, outputs
        torch.cuda.empty_cache()
        rank_info = self._rank_scores(detached, candidates, target_iid, getattr(self.args, "rerank_metric_ks", [1, 5, 10]))
        rank_info["score_tensor"] = score_tensor
        rank_info["candidate_index"] = {str(int(iid)): idx for idx, iid in enumerate(candidates)}
        return rank_info

    def _log_score_chunk(self, candidate_count, score_batch_size, chunk_index):
        max_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        self.logs["last_score_chunk"] = {
            "candidate_count": int(candidate_count),
            "score_batch_size": int(score_batch_size),
            "current_chunk_index": int(chunk_index),
            "max_memory_allocated": max_memory,
        }
        self.logs["score_chunk_log_count"] = int(self.logs.get("score_chunk_log_count", 0)) + 1
        print(
            "[GeometryPrune] _score_candidate_list "
            f"candidate_count={candidate_count} "
            f"score_batch_size={score_batch_size} "
            f"current_chunk_index={chunk_index} "
            f"torch.cuda.max_memory_allocated()={max_memory}"
        )

    @staticmethod
    def _rank_scores(scores, candidate_items, target_iid, metric_ks):
        order = np.argsort(-np.asarray(scores)).tolist()
        ranked_items = [int(candidate_items[idx]) for idx in order]
        ranks = {str(int(candidate_items[idx])): rank + 1 for rank, idx in enumerate(order)}
        score_map = {str(int(iid)): float(score) for iid, score in zip(candidate_items, scores)}
        target_rank = int(ranks[str(int(target_iid))])
        target_score = float(score_map[str(int(target_iid))])
        max_k = min(max(metric_ks), len(ranked_items))
        boundary_item = ranked_items[max_k - 1]
        boundary_score = float(score_map[str(boundary_item)])
        return {
            "scores": score_map,
            "ranks": ranks,
            "topk_items": ranked_items[:max_k],
            "target_rank": target_rank,
            "target_score": target_score,
            "topk_boundary_score": boundary_score,
            "margin_to_topk_boundary": float(target_score - boundary_score),
        }

    def _score_records(self, records, apply_logit_correction=False):
        scored = []
        for record in records:
            result = self._score_candidate_list(
                record.get("context_items", []),
                record.get("candidate_items", []),
                int(record["target_iid"]),
                grad=False,
            )
            if apply_logit_correction:
                result = self._apply_correction_to_rank_info(result, record)
            result.pop("score_tensor", None)
            result.pop("candidate_index", None)
            updated = dict(record)
            updated.update(result)
            scored.append(updated)
        return scored

    def _apply_correction_to_rank_info(self, rank_info, record):
        path = os.path.join(self.output_dir, "logit_correction.json")
        if not os.path.exists(path):
            return rank_info
        with open(path, "r") as f:
            payload = json.load(f)
        corrections = payload.get("corrections", {}) if isinstance(payload, dict) else {}
        user_map = corrections.get(str(int(record.get("uid"))), {})
        if not user_map:
            return rank_info
        candidate_items = [int(i) for i in record.get("candidate_items", [])]
        scores = [float(rank_info["scores"].get(str(i), 0.0)) for i in candidate_items]
        for idx, iid in enumerate(candidate_items):
            scores[idx] -= float(user_map.get(str(int(iid)), 0.0))
        updated = dict(rank_info)
        updated.update(self._rank_scores(
            scores,
            candidate_items,
            int(record["target_iid"]),
            getattr(self.args, "rerank_metric_ks", [1, 5, 10]),
        ))
        return updated

    def _all_protocol_records(self, split_data=None):
        split_data = split_data or self._method_split_data()
        records = []
        split_groups = [
            ("forget", "forget_interactions"),
            ("retain", "retain_interactions"),
            ("overlap", "overlap_retain_interactions"),
            ("semantic_neighbor", "semantic_neighbor_retain"),
            ("collaborative_neighbor", "collaborative_neighbor_retain"),
        ]
        for tag, key in split_groups:
            for local_idx, row in enumerate(split_data.get(key, [])):
                records.append(self._protocol_record_from_row(tag, row, local_idx, split_data))
        return records

    def _protocol_record_from_row(self, split_tag, row, local_idx, split_data):
        uid = _row_uid(row)
        target_iid = _row_iid(row)
        pos = _row_position(row)
        existing = getattr(self, "_prediction_lookup", {}).get(
            (split_tag, int(uid), int(target_iid), pos)
        )
        if existing:
            return dict(existing)

        context_items = evaluation_context_items_for_record(
            row,
            split_tag,
            self.dataset_data,
            self.split_data,
            int(getattr(self.args, "llm_max_history", 20)),
        )
        num_items = len(self.dataset_data.get("meta", {}))
        if num_items > 0:
            candidate_items = evaluation_candidate_items(
                self.args,
                uid,
                target_iid,
                context_items,
                split_tag,
                num_items,
            )
        else:
            candidate_items = [int(target_iid)]
        return {
            "prediction_id": f"{split_tag}:{uid}:{target_iid}:{pos}:{local_idx}",
            "source_row": row,
            "uid": uid,
            "target_iid": target_iid,
            "position": pos,
            "split_tag": split_tag,
            "context_items": [int(i) for i in context_items],
            "candidate_items": [int(i) for i in candidate_items],
        }

    def _retain_probe_records(self):
        limit = int(getattr(self.args, "probe_retain_samples", 8) or 8)
        records = [
            dict(r) for r in self._all_protocol_records(self._method_split_data())
            if r.get("split_tag") == "retain"
        ]
        return records[:limit]

    def _zero_trainable_grads(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = None

    def _collect_lora_rank_grad_norms(self):
        pairs = self._collect_lora_pairs()
        grads = {}
        for module_name, pair in pairs.items():
            rank = pair["rank"]
            grad_a = pair["A"].grad
            grad_b = pair["B"].grad
            for rid in range(rank):
                val = 0.0
                if grad_a is not None:
                    val += float(grad_a[rid, :].detach().float().abs().mean().cpu())
                if grad_b is not None:
                    val += float(grad_b[:, rid].detach().float().abs().mean().cpu())
                grads[f"{module_name}:{rid}"] = val
        return grads

    def _collect_lora_pairs(self):
        raw = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                key = self._module_key_from_lora_name(name)
                raw.setdefault(key, {})["A"] = param
                raw[key]["A_name"] = name
            elif "lora_B" in name:
                key = self._module_key_from_lora_name(name)
                raw.setdefault(key, {})["B"] = param
                raw[key]["B_name"] = name
        pairs = {}
        for key, value in raw.items():
            if "A" in value and "B" in value:
                value["rank"] = int(value["A"].shape[0])
                pairs[key] = value
        if not pairs and "no_lora_adapters_found_safe_fallback" not in self.logs.get("fallbacks", []):
            self.logs["fallbacks"].append("no_lora_adapters_found_safe_fallback")
        return pairs

    @staticmethod
    def _module_key_from_lora_name(name):
        import re

        layer = -1
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer = int(match.group(1))
        mod = "adapter"
        for candidate in ("q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "down_proj", "gate_proj"):
            if candidate in name:
                mod = candidate
                break
        return f"L{layer}_{mod}"

    def _save_trainable_state(self):
        state = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                state[name] = param.data.detach().clone()
        self._original_trainable.update({k: v.clone() for k, v in state.items() if k not in self._original_trainable})
        return state

    def _restore_trainable_state(self, state):
        if not state:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.data.copy_(state[name].to(param.device, dtype=param.dtype))

    def _maybe_rollback(self, before_records, after_records, original_state, pruning, lora_result):
        enabled = bool(getattr(self.args, "enable_prune_rollback", True))
        retain_drop = self._max_retain_drop(before_records, after_records)
        tolerance = float(getattr(self.args, "retain_drop_tolerance", 0.05))
        rollback = bool(enabled and retain_drop > tolerance)
        if rollback:
            self._restore_trainable_state(original_state)
            pruning.setdefault("summary", {})["rolled_back"] = True
            pruning["summary"]["rollback_reason"] = "retain_drop_exceeded"
        return {
            "enabled": enabled,
            "rollback_applied": rollback,
            "retain_drop": retain_drop,
            "retain_drop_tolerance": tolerance,
            "lora_suppression": lora_result,
            "num_pruned_before_rollback": int(pruning.get("summary", {}).get("hard_prune", 0)),
        }

    @staticmethod
    def _max_retain_drop(before_records, after_records):
        before = [r for r in before_records if r.get("split_tag") == "retain"]
        after = [r for r in after_records if r.get("split_tag") == "retain"]
        ks = [1, 5, 10]
        b = _ranking_metrics(before, ks)
        a = _ranking_metrics(after, ks)
        drops = []
        for key, value in b.items():
            if value is not None and a.get(key) is not None:
                drops.append(float(value) - float(a[key]))
        return max(drops) if drops else 0.0

    def _counterfactual_residual_gap(self, residual_candidates, use_after):
        if not residual_candidates:
            return 0.0
        gaps = []
        for cand in residual_candidates:
            if use_after:
                info = self.compute_counterfactual_residuals(
                    self.model,
                    cand["user_id"],
                    cand["forget_item_id"],
                    cand["history"],
                    cand["candidate_items"],
                    self.args,
                )
                match = next((r for r in info if int(r["candidate_item_id"]) == int(cand["candidate_item_id"])), None)
                if match:
                    gaps.append(abs(float(match["residual_score"])))
            else:
                gaps.append(abs(float(cand.get("residual_score", 0.0))))
        return float(np.mean(gaps)) if gaps else 0.0

    def _residual_path_hit_rate(self, residual_candidates, use_after):
        if not residual_candidates:
            return 0.0
        k = int(getattr(self.args, "topk_boundary", 10))
        hits = []
        for cand in residual_candidates:
            if use_after:
                info = self._score_candidate_list(
                    cand["history"],
                    cand["candidate_items"],
                    int(cand["candidate_item_id"]),
                    grad=False,
                )
                rank = int(info["target_rank"])
            else:
                rank = int(cand.get("rank_original", 10**9))
            hits.append(1.0 if rank <= k else 0.0)
        return float(np.mean(hits)) if hits else 0.0

    @staticmethod
    def _semantic_protected_drop(before_records, after_records):
        before_by_id = {r["prediction_id"]: r for r in before_records}
        drops = []
        for after in after_records:
            if after.get("split_tag") != "semantic_neighbor":
                continue
            before = before_by_id.get(after.get("prediction_id"))
            if before:
                drops.append(float(before.get("target_score", 0.0)) - float(after.get("target_score", 0.0)))
        return float(np.mean(drops)) if drops else 0.0

    @staticmethod
    def _neighbor_retain_drop(before_records, after_records):
        before_by_id = {r["prediction_id"]: r for r in before_records}
        drops = []
        for after in after_records:
            if after.get("split_tag") not in {"collaborative_neighbor", "overlap"}:
                continue
            before = before_by_id.get(after.get("prediction_id"))
            if before:
                drops.append(float(before.get("target_score", 0.0)) - float(after.get("target_score", 0.0)))
        return float(np.mean(drops)) if drops else 0.0

    @staticmethod
    def _boundary_flip_rate(before_by_id, after_records, ks):
        k = max(ks) if ks else 10
        flips = []
        for after in after_records:
            before = before_by_id.get(after.get("prediction_id"))
            if not before:
                continue
            before_rank = int(before.get("target_rank", 10**9))
            after_rank = int(after.get("target_rank", 10**9))
            if abs(before_rank - k) <= 2:
                flips.append(1.0 if (before_rank <= k) != (after_rank <= k) else 0.0)
        return float(np.mean(flips)) if flips else 0.0

    @staticmethod
    def _write_jsonl(path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    def _residual_summary(records):
        return {
            "num_records": len(records),
            "avg_residual_z": float(np.mean([r.get("residual_z", 0.0) for r in records])) if records else 0.0,
            "max_residual_z": float(np.max([r.get("residual_z", 0.0) for r in records])) if records else 0.0,
            "avg_collaborative_similarity": (
                float(np.mean([r.get("collaborative_similarity", 0.0) for r in records]))
                if records else 0.0
            ),
            "avg_semantic_protection": (
                float(np.mean([r.get("semantic_protection", 0.0) for r in records]))
                if records else 0.0
            ),
        }

    @staticmethod
    def _importance_summary(importance):
        scores = [float(v.get("rank_score", 0.0)) for v in importance.values()]
        forget = [float(v.get("F_r", 0.0)) for v in importance.values()]
        retain = [float(v.get("G_r", 0.0)) for v in importance.values()]
        return {
            "num_ranks": len(scores),
            "max_rank_score": max(scores) if scores else 0.0,
            "mean_rank_score": float(np.mean(scores)) if scores else 0.0,
            "mean_F_r": float(np.mean(forget)) if forget else 0.0,
            "mean_G_r": float(np.mean(retain)) if retain else 0.0,
        }

    def _save_prune_mask(self, pruning):
        mask = {}
        for decision in pruning.get("decisions", []):
            key = f"{decision['module_name']}:{decision['rank_id']}"
            mask[key] = 0 if decision.get("route") == "hard_prune" else 1
        torch.save(mask, os.path.join(self.output_dir, "prune_mask.pt"))

    @staticmethod
    def _print_required_metrics(metrics, pruning, rollback_log):
        print("Forget Recall@K before/after")
        print(json.dumps({
            "before": metrics.get("Forget Recall@K before"),
            "after": metrics.get("Forget Recall@K after"),
        }, indent=2))
        print("Forget NDCG@K before/after")
        print(json.dumps({
            "before": metrics.get("Forget NDCG@K before"),
            "after": metrics.get("Forget NDCG@K after"),
        }, indent=2))
        print("Retain Recall@K before/after")
        print(json.dumps({
            "before": metrics.get("Retain Recall@K before"),
            "after": metrics.get("Retain Recall@K after"),
        }, indent=2))
        print("Retain NDCG@K before/after")
        print(json.dumps({
            "before": metrics.get("Retain NDCG@K before"),
            "after": metrics.get("Retain NDCG@K after"),
        }, indent=2))
        print(f"CRG before/after: {metrics.get('CRG before')} -> {metrics.get('CRG after')}")
        print(f"RPHR before/after: {metrics.get('RPHR before')} -> {metrics.get('RPHR after')}")
        print(f"SPD: {metrics.get('SPD')}")
        print(f"NRD: {metrics.get('NRD')}")
        print(f"BFR: {metrics.get('BFR')}")
        print(f"Pruned parameter ratio: {pruning.get('summary', {}).get('hard_prune_ratio', 0.0)}")
        print(f"Rollback status: {rollback_log.get('rollback_applied')}")


SemanticGeometryPruneMethod = GeometryPruneMethod
