import hashlib
import json
import math
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .geometry_prune import (
    GeometryPruneMethod,
    _ranking_metrics,
    _row_iid,
    _row_position,
    _row_uid,
    remove_forget_item_from_history,
    save_json,
)


MASK_ATTR = "_retain_prior_cf_rank_mask"
HOOK_ATTR = "_retain_prior_cf_rank_mask_hook"


@dataclass
class LoRALayerHandle:
    module_name: str
    adapter_name: str
    module: torch.nn.Module
    lora_A: torch.nn.Module
    lora_B: torch.nn.Module
    A: torch.nn.Parameter
    B: torch.nn.Parameter
    rank: int
    rank_mask: torch.Tensor
    temp_values: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        if self.adapter_name and self.adapter_name != "default":
            return f"{self.module_name}.{self.adapter_name}"
        return self.module_name


def _is_adapter_mapping(value) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__") and not hasattr(value, "weight")


def _weight_parameter(module) -> Optional[torch.nn.Parameter]:
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.nn.Parameter):
        return weight
    return None


def _rank_mask_pre_hook(module, inputs):
    if not inputs:
        return inputs
    hidden = inputs[0]
    if not torch.is_tensor(hidden):
        return inputs
    rank_mask = getattr(module, MASK_ATTR, None)
    if rank_mask is None or int(hidden.shape[-1]) != int(rank_mask.numel()):
        return inputs
    view_shape = [1] * (hidden.dim() - 1) + [int(rank_mask.numel())]
    masked_hidden = hidden * rank_mask.to(device=hidden.device, dtype=hidden.dtype).view(*view_shape)
    return (masked_hidden, *inputs[1:])


def _ensure_rank_mask(module: torch.nn.Module, rank: int, device) -> torch.Tensor:
    existing = getattr(module, MASK_ATTR, None)
    if torch.is_tensor(existing) and int(existing.numel()) == int(rank):
        return existing
    if hasattr(module, MASK_ATTR):
        delattr(module, MASK_ATTR)
    module.register_buffer(
        MASK_ATTR,
        torch.ones(int(rank), dtype=torch.float32, device=device),
        persistent=True,
    )
    return getattr(module, MASK_ATTR)


def collect_lora_layers(model) -> List[LoRALayerHandle]:
    """Collect PEFT/LoRA modules with paired A/B projections.

    The hook on lora_B multiplies its input by rank_mask, implementing
    B @ diag(rank_mask) @ A without changing base model weights.
    """
    layers: List[LoRALayerHandle] = []
    for module_name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue
        lora_A = getattr(module, "lora_A")
        lora_B = getattr(module, "lora_B")
        if _is_adapter_mapping(lora_A) and _is_adapter_mapping(lora_B):
            adapter_names = sorted(set(lora_A.keys()) & set(lora_B.keys()))
        else:
            adapter_names = ["default"]

        for adapter_name in adapter_names:
            A_module = lora_A[adapter_name] if _is_adapter_mapping(lora_A) else lora_A
            B_module = lora_B[adapter_name] if _is_adapter_mapping(lora_B) else lora_B
            A = _weight_parameter(A_module)
            B = _weight_parameter(B_module)
            if A is None or B is None or A.ndim != 2 or B.ndim != 2:
                continue
            rank = int(A.shape[0])
            if rank <= 0 or int(B.shape[1]) != rank:
                continue
            rank_mask = _ensure_rank_mask(B_module, rank, A.device)
            if not hasattr(B_module, HOOK_ATTR):
                setattr(B_module, HOOK_ATTR, B_module.register_forward_pre_hook(_rank_mask_pre_hook))
            layers.append(LoRALayerHandle(
                module_name=module_name,
                adapter_name=str(adapter_name),
                module=module,
                lora_A=A_module,
                lora_B=B_module,
                A=A,
                B=B,
                rank=rank,
                rank_mask=rank_mask,
            ))
    return layers


def temporary_mask_rank(layer: LoRALayerHandle, rank_id: int):
    rank_id = int(rank_id)
    if rank_id < 0 or rank_id >= int(layer.rank):
        raise IndexError(f"rank_id {rank_id} out of range for {layer.key}")
    previous = layer.rank_mask[rank_id].detach().clone()
    layer.temp_values.setdefault(rank_id, []).append(previous)
    with torch.no_grad():
        layer.rank_mask[rank_id].fill_(0.0)


def restore_rank(layer: LoRALayerHandle, rank_id: int):
    rank_id = int(rank_id)
    previous_stack = layer.temp_values.get(rank_id, [])
    previous = previous_stack.pop() if previous_stack else torch.ones_like(layer.rank_mask[rank_id])
    with torch.no_grad():
        layer.rank_mask[rank_id].copy_(previous.to(layer.rank_mask.device, dtype=layer.rank_mask.dtype))


def lora_rank_key(layer: LoRALayerHandle, rank_id: int) -> str:
    return f"{layer.key}:{int(rank_id)}"


def _lora_layer_index(layers: List[LoRALayerHandle]) -> Dict[str, Tuple[LoRALayerHandle, int]]:
    return {
        lora_rank_key(layer, rank_id): (layer, int(rank_id))
        for layer in layers
        for rank_id in range(int(layer.rank))
    }


def get_current_rank_mask_state(layers: List[LoRALayerHandle]) -> Dict[str, int]:
    state = {}
    for layer in layers:
        mask = layer.rank_mask.detach().float().cpu()
        for rank_id in range(int(layer.rank)):
            state[lora_rank_key(layer, rank_id)] = 0 if float(mask[int(rank_id)]) <= 0.0 else 1
    return state


def restore_rank_masks(layers: List[LoRALayerHandle], mask_state: Dict[str, int]):
    index = _lora_layer_index(layers)
    with torch.no_grad():
        for key, value in mask_state.items():
            if key not in index:
                continue
            layer, rank_id = index[key]
            layer.rank_mask[rank_id].fill_(float(int(value)))


def apply_permanent_rank_masks(layers: List[LoRALayerHandle], rank_ids: List[str]):
    index = _lora_layer_index(layers)
    with torch.no_grad():
        for rank_id in rank_ids:
            if rank_id not in index:
                continue
            layer, local_rank = index[rank_id]
            layer.rank_mask[local_rank].fill_(0.0)


@contextmanager
def temporary_rank_mask(layers: List[LoRALayerHandle], rank_id: str):
    state = get_current_rank_mask_state(layers)
    index = _lora_layer_index(layers)
    if rank_id not in index:
        raise KeyError(f"Unknown LoRA rank id: {rank_id}")
    layer, local_rank = index[rank_id]
    with torch.no_grad():
        layer.rank_mask[local_rank].fill_(0.0)
    try:
        yield
    finally:
        restore_rank_masks(layers, state)


def atomic_save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp_path, path)


def lora_rank_mask_checksums(layers: List[LoRALayerHandle]) -> Dict:
    raw_abs_sum = 0.0
    active_rank_abs_sum = 0.0
    mask_sum = 0.0
    zero_count = 0
    total_ranks = 0
    for layer in layers:
        mask = layer.rank_mask.detach().to(device=layer.A.device, dtype=layer.A.dtype)
        raw_abs_sum += float(layer.A.detach().abs().float().sum().cpu())
        raw_abs_sum += float(layer.B.detach().abs().float().sum().cpu())
        active_rank_abs_sum += float(
            (layer.A.detach().abs() * mask.view(-1, 1)).float().sum().cpu()
        )
        active_rank_abs_sum += float(
            (layer.B.detach().abs() * mask.view(1, -1)).float().sum().cpu()
        )
        mask_cpu = layer.rank_mask.detach().float().cpu()
        mask_sum += float(mask_cpu.sum())
        zero_count += int((mask_cpu == 0).sum())
        total_ranks += int(layer.rank)
    return {
        "raw_lora_parameter_abs_sum": raw_abs_sum,
        "active_rank_lora_parameter_abs_sum": active_rank_abs_sum,
        "rank_mask_sum": mask_sum,
        "rank_mask_zero_count": zero_count,
        "total_ranks": total_ranks,
    }


def _normalize_rank_mask(rank_mask: Dict) -> Dict[str, int]:
    if not isinstance(rank_mask, dict):
        raise TypeError(f"LoRA rank mask must be a dict, got {type(rank_mask).__name__}")
    normalized = {}
    for key, value in rank_mask.items():
        if not isinstance(key, str) or ":" not in key:
            raise ValueError(f"Invalid rank mask key: {key!r}")
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rank mask value for {key}: {value!r}") from exc
        if int_value not in (0, 1):
            raise ValueError(f"Rank mask value for {key} must be 0 or 1, got {int_value}")
        normalized[str(key)] = int_value
    return normalized


def expected_lora_rank_mask_keys(layers: List[LoRALayerHandle]) -> List[str]:
    return [
        f"{layer.key}:{int(rank_id)}"
        for layer in layers
        for rank_id in range(int(layer.rank))
    ]


def apply_lora_rank_mask(
    layers: List[LoRALayerHandle],
    rank_mask: Dict,
    strict: bool = True,
) -> Dict:
    normalized = _normalize_rank_mask(rank_mask)
    expected_keys = set(expected_lora_rank_mask_keys(layers))
    mask_keys = set(normalized.keys())
    missing = sorted(expected_keys - mask_keys)
    extra = sorted(mask_keys - expected_keys)
    if strict and (missing or extra):
        raise RuntimeError(
            "LoRA rank mask keys do not match current model LoRA ranks. "
            f"missing={len(missing)} extra={len(extra)} "
            f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
        )

    before = lora_rank_mask_checksums(layers)
    applied = 0
    applied_zero = 0
    per_module_applied = defaultdict(int)
    per_module_zero = defaultdict(int)
    with torch.no_grad():
        for layer in layers:
            for rank_id in range(int(layer.rank)):
                key = f"{layer.key}:{int(rank_id)}"
                if key not in normalized:
                    continue
                value = int(normalized[key])
                layer.rank_mask[int(rank_id)].fill_(float(value))
                applied += 1
                applied_zero += int(value == 0)
                per_module_applied[layer.key] += 1
                per_module_zero[layer.key] += int(value == 0)
    after = lora_rank_mask_checksums(layers)
    return {
        "matched_lora_module_count": len(layers),
        "total_lora_rank_count": int(sum(layer.rank for layer in layers)),
        "applied_rank_mask_count": int(applied),
        "applied_pruned_rank_count": int(applied_zero),
        "applied_protected_rank_count": int(applied - applied_zero),
        "missing_rank_mask_count": len(missing),
        "extra_rank_mask_count": len(extra),
        "per_module_applied_rank_count": dict(per_module_applied),
        "per_module_pruned_rank_count": dict(per_module_zero),
        "checksum_before": before,
        "checksum_after": after,
    }


def load_and_apply_lora_rank_mask(model, mask_path: str, strict: bool = True) -> Tuple[List[LoRALayerHandle], Dict]:
    layers = collect_lora_layers(model)
    rank_mask = torch.load(mask_path, map_location="cpu")
    summary = apply_lora_rank_mask(layers, rank_mask, strict=strict)
    summary["rank_mask_path"] = mask_path
    return layers, summary


def materialize_lora_rank_mask(layers: List[LoRALayerHandle]) -> Dict:
    before = lora_rank_mask_checksums(layers)
    with torch.no_grad():
        for layer in layers:
            mask = layer.rank_mask.to(device=layer.B.device, dtype=layer.B.dtype)
            layer.B.mul_(mask.view(1, -1))
    after = lora_rank_mask_checksums(layers)
    return {
        "materialized_lora_B_columns": True,
        "checksum_before_materialize": before,
        "checksum_after_materialize": after,
    }


def robust_normalize_per_layer(values, eps: float = 1e-8) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    if mad <= eps:
        q75, q25 = np.percentile(arr, [75, 25])
        scale = float(q75 - q25)
    else:
        scale = 1.4826 * mad
    if scale <= eps:
        return [0.0 for _ in arr.tolist()]
    return [float(v) for v in ((arr - median) / (scale + eps)).tolist()]


def _safe_quantile(values, q: float, default: float = 0.0) -> float:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, min(1.0, max(0.0, float(q)))))


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -50.0, 50.0))
    return float(1.0 / (1.0 + math.exp(-x)))


class RetainPriorCFLoraPruneMethod(GeometryPruneMethod):
    """Retain-prior counterfactual LoRA rank pruning.

    This is an interaction-level pruning method. It localizes abnormal
    counterfactual ranking residuals, calibrates them by retain support, then
    masks LoRA rank directions only when forget gain dominates retain and
    semantic risks.
    """

    method_name = "retain_prior_cf_lora_prune"

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        method_split_data = self._method_split_data()
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})
        self.logs.update({
            "status": "running",
            "action": "retain_prior_cf_lora_rank_prune",
            "uses_fixed_split": True,
            "candidate_level_exposure": True,
            "logit_correction_used": False,
            "full_model_parameter_updates": False,
            "gradient_updates": False,
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
                "Stage 1 localizes interaction-induced residuals with a null retain-removal control.",
                "Stage 2 computes W_unl and W_prot as soft retain-aware weights, not hard candidate splitting.",
                "Stage 3 applies no-gradient LoRA rank masks B @ diag(mask) @ A; coarse_to_fine_pruning uses one-shot joint-score Top-K selection.",
            ],
            "residual_selection_mode": getattr(self.args, "rp_cf_residual_selection_mode", "topk"),
            "residual_threshold": (
                getattr(self.args, "rp_cf_residual_threshold", None)
                if getattr(self.args, "rp_cf_residual_threshold", None) is not None
                else getattr(self.args, "tau_residual_z", 1.0)
            ),
            "residual_top_k": self._residual_top_m(),
            "retain_support_gamma": float(getattr(self.args, "rp_cf_gamma", 1.0) or 1.0),
            "protection_quantile": float(getattr(self.args, "rp_cf_protection_quantile", 0.95) or 0.95),
            "robust_distance": self._robust_distance_name(),
            "boundary_gap_definition": self._boundary_gap_definition(),
        })
        save_json(os.path.join(self.output_dir, "run_config.json"), vars(self.args).copy())
        save_json(os.path.join(self.output_dir, "logit_correction.json"), {
            "enabled": False,
            "corrections": {},
            "note": "retain_prior_cf_lora_prune changes LoRA rank masks only.",
        })

        self._init_similarity_fallbacks(method_split_data)
        self._freeze_all_parameters()

        before_records = [dict(r) for r in self.predictions_before.get("records", [])]
        if not before_records:
            before_records = self._score_records(self._all_protocol_records(method_split_data))

        localization_path = os.path.join(self.output_dir, "residual_localization.json")
        calibration_path = os.path.join(self.output_dir, "retain_calibration.json")

        localization = self._load_resume_json(localization_path, "Stage1")
        if localization is None:
            print("[retain_prior_cf_lora_prune] Stage 1: residual localization")
            localization = self.localize_interaction_residuals(method_split_data)
            atomic_save_json(localization_path, localization)
            self._write_jsonl(
                os.path.join(self.output_dir, "residual_targets.jsonl"),
                localization.get("residual_targets", []),
            )
        else:
            print("[retain_prior_cf_lora_prune][Stage1][Cache] residual_localization.json")

        calibration = self._load_resume_json(calibration_path, "Stage2")
        if calibration is None:
            print("[retain_prior_cf_lora_prune] Stage 2: retain-aware calibration")
            calibration = self.calibrate_retain_support(localization.get("records", []))
            atomic_save_json(calibration_path, calibration)
            self._write_jsonl(
                os.path.join(self.output_dir, "calibrated_residual_targets.jsonl"),
                calibration.get("residual_targets", []),
            )
            self._write_jsonl(
                os.path.join(self.output_dir, "calibrated_protection_targets.jsonl"),
                calibration.get("protection_targets", []),
            )
        else:
            print("[retain_prior_cf_lora_prune][Stage2][Cache] retain_calibration.json")
        calibrated_targets = calibration.get("residual_targets", [])
        protection_targets = calibration.get("protection_targets", [])

        before_residual_summary = self._forget_residual_summary(calibrated_targets)

        print("[retain_prior_cf_lora_prune] Stage 3: retention-first LoRA rank pruning")
        pruning = self.retention_first_domination_prune(
            calibrated_targets,
            protection_targets,
            method_split_data,
        )
        pruning_to_save = {k: v for k, v in pruning.items() if not str(k).startswith("_")}
        atomic_save_json(os.path.join(self.output_dir, "pruning_decisions.json"), pruning_to_save)
        atomic_save_json(os.path.join(self.output_dir, "pruning_summary.json"), pruning.get("summary", {}))
        self._save_rank_mask(pruning)

        after_target_scores = pruning.get("_after_residual_score_overrides")
        if after_target_scores is None:
            after_target_scores = self._score_residual_targets(calibrated_targets) if calibrated_targets else {}
        after_residual_summary = self._forget_residual_summary(
            calibrated_targets,
            score_overrides=after_target_scores,
        )
        atomic_save_json(os.path.join(self.output_dir, "forget_residual_before_after.json"), {
            "before": before_residual_summary,
            "after": after_residual_summary,
        })

        after_records = self._score_records(self._all_protocol_records(method_split_data))
        method_metrics = self.evaluate_unlearning_metrics(
            None,
            None,
            method_split_data.get("forget_interactions", []),
            method_split_data.get("retain_interactions", []),
            self.args,
            residual_candidates=calibrated_targets,
            before_records=before_records,
            after_records=after_records,
        )
        atomic_save_json(os.path.join(self.output_dir, "unlearning_metrics.json"), method_metrics)
        atomic_save_json(os.path.join(self.output_dir, "metrics_unlearning.json"), method_metrics)

        retain_metrics = {
            "before": _ranking_metrics(
                [r for r in before_records if r.get("split_tag") == "retain"],
                getattr(self.args, "rerank_metric_ks", [1, 5, 10]),
            ),
            "after": _ranking_metrics(
                [r for r in after_records if r.get("split_tag") == "retain"],
                getattr(self.args, "rerank_metric_ks", [1, 5, 10]),
            ),
        }

        self.logs.update({
            "status": "completed",
            "is_effective_unlearning_baseline": True,
            "processed_users": localization.get("summary", {}).get("processed_users", 0),
            "residual_target_count": localization.get("summary", {}).get("residual_target_count", 0),
            "avg_residual_z": localization.get("summary", {}).get("avg_residual_z", 0.0),
            "avg_retain_support": calibration.get("summary", {}).get("avg_retain_support", 0.0),
            "final_prune_ratio": pruning.get("summary", {}).get("final_prune_ratio", 0.0),
            "final_stop_reason": pruning.get("summary", {}).get("reason"),
            "stage1_residual_localization": localization.get("summary", {}),
            "stage2_retain_calibration": calibration.get("summary", {}),
            "stage3_rank_pruning": pruning.get("summary", {}),
            "before_after_forget_residual_score": {
                "before": before_residual_summary,
                "after": after_residual_summary,
            },
            "before_after_retain_ranking_metric": retain_metrics,
            "metrics_file": os.path.join(self.output_dir, "unlearning_metrics.json"),
            "residual_targets_file": os.path.join(self.output_dir, "residual_targets.jsonl"),
            "calibrated_residual_targets_file": os.path.join(
                self.output_dir,
                "calibrated_residual_targets.jsonl",
            ),
            "calibrated_protection_targets_file": os.path.join(
                self.output_dir,
                "calibrated_protection_targets.jsonl",
            ),
        })
        self.save_logs()
        return self.logs

    # ------------------------------------------------------------------
    # Stage 1: interaction-induced residual localization
    # ------------------------------------------------------------------

    def localize_interaction_residuals(self, split_data: Dict) -> Dict:
        selection_mode = str(getattr(self.args, "rp_cf_residual_selection_mode", "topk") or "topk")
        if selection_mode not in {"topk", "threshold"}:
            raise ValueError(f"Unsupported rp_cf_residual_selection_mode={selection_mode!r}")
        top_m = self._residual_top_m()
        threshold = getattr(self.args, "rp_cf_residual_threshold", None)
        if threshold is None:
            threshold = getattr(self.args, "tau_residual_z", 1.0)
        threshold = float(threshold)
        eps = float(getattr(self.args, "null_eps", 1e-8) or 1e-8)
        all_records: List[Dict] = []
        residual_targets: List[Dict] = []
        interactions: List[Dict] = []
        skip_reasons = defaultdict(int)
        processed_users = set()
        z_values = []
        null_control_values = []

        for local_idx, row in enumerate(split_data.get("forget_interactions", [])):
            uid = _row_uid(row)
            forget_iid = _row_iid(row)
            pos = _row_position(row)
            processed_users.add(int(uid))
            history = self._original_history_for_forget_row(row, uid, forget_iid)
            cf_history = remove_forget_item_from_history(history, forget_iid)
            candidates = self.get_candidate_set(uid, forget_iid, self.dataset_data, self.args, row=row)
            if not candidates:
                skip_reasons["empty_candidate_set"] += 1
                continue

            original = self._score_candidate_list(history, candidates, forget_iid, grad=False)
            counterfactual = self._score_candidate_list(cf_history, candidates, forget_iid, grad=False)
            score_original = {str(int(i)): float(original["scores"].get(str(int(i)), 0.0)) for i in candidates}
            score_cf = {str(int(i)): float(counterfactual["scores"].get(str(int(i)), 0.0)) for i in candidates}

            sampled_retain = self._sample_retain_interactions(
                uid=uid,
                history=history,
                forget_iid=forget_iid,
                requested=int(getattr(self.args, "num_null_removals", 5) or 5),
                salt="stage1-null",
            )
            null_deltas = {str(int(iid)): [] for iid in candidates}
            for retain_iid in sampled_retain:
                minus_history = remove_forget_item_from_history(history, retain_iid)
                minus_scores = self._score_candidate_list(minus_history, candidates, forget_iid, grad=False)
                for iid in candidates:
                    key = str(int(iid))
                    delta = score_original[key] - float(minus_scores["scores"].get(key, 0.0))
                    null_deltas[key].append(float(delta))
                    null_control_values.append(self._robust_distance(float(delta), 0.0))

            null_mean = {}
            null_std = {}
            group_records = []
            group_id = f"{int(uid)}:{int(forget_iid)}:{pos}:{int(local_idx)}"
            for iid in candidates:
                key = str(int(iid))
                arr = np.asarray(null_deltas.get(key, []), dtype=np.float64)
                mean = float(np.mean(arr)) if arr.size else 0.0
                std = float(np.std(arr)) if arr.size else 0.0
                raw = float(score_original[key] - score_cf[key])
                z_score = float((raw - mean) / (std + eps))
                null_mean[key] = mean
                null_std[key] = std
                z_values.append(z_score)
                original_gap = self._boundary_gap_from_rank_info(original, candidates, iid)
                cf_gap = self._boundary_gap_from_rank_info(counterfactual, candidates, iid)
                record = {
                    "record_id": f"{group_id}:{int(iid)}",
                    "group_id": group_id,
                    "uid": int(uid),
                    "user_id": int(uid),
                    "forget_iid": int(forget_iid),
                    "forget_item_id": int(forget_iid),
                    "candidate_iid": int(iid),
                    "candidate_item_id": int(iid),
                    "target_iid": int(iid),
                    "scoring_label_iid": int(forget_iid),
                    "position": pos,
                    "history": [int(x) for x in history],
                    "counterfactual_history": [int(x) for x in cf_history],
                    "retain_history": [int(x) for x in cf_history],
                    "candidate_items": [int(x) for x in candidates],
                    "score_original": score_original[key],
                    "score_cf_forget": score_cf[key],
                    "score_counterfactual": score_cf[key],
                    "residual_raw": self.compute_raw_counterfactual_residual(
                        score_original[key],
                        score_cf[key],
                    ),
                    "residual_delta": raw,
                    "residual_score": raw,
                    "residual_z": z_score,
                    "null_mean": mean,
                    "null_std": std,
                    "num_null_controls": int(arr.size),
                    "null_control_deltas": [float(x) for x in null_deltas.get(key, [])],
                    "sampled_null_retain_items": [int(x) for x in sampled_retain],
                    "rank_original": int(original["ranks"].get(key, 10**9)),
                    "rank_counterfactual": int(counterfactual["ranks"].get(key, 10**9)),
                    "boundary_gap_original": float(original_gap),
                    "boundary_gap_cf": float(cf_gap),
                    "topk_boundary_score_original": float(original.get("topk_boundary_score", 0.0)),
                    "topk_boundary_score_cf": float(counterfactual.get("topk_boundary_score", 0.0)),
                    "is_residual_target": False,
                }
                group_records.append(record)
                all_records.append(record)

            group_records.sort(key=lambda r: float(r.get("residual_z", 0.0)), reverse=True)
            if selection_mode == "threshold":
                selected = [
                    r for r in group_records
                    if float(r.get("residual_z", 0.0)) > threshold
                ]
            else:
                selected = group_records[:top_m] if top_m > 0 else []
            for record in selected:
                record["is_residual_target"] = True
            residual_targets.extend(selected)
            interactions.append({
                "group_id": group_id,
                "uid": int(uid),
                "forget_iid": int(forget_iid),
                "position": pos,
                "candidate_items": [int(x) for x in candidates],
                "history": [int(x) for x in history],
                "counterfactual_history": [int(x) for x in cf_history],
                "score_original": score_original,
                "score_cf_forget": score_cf,
                "null_mean": null_mean,
                "null_std": null_std,
                "sampled_null_retain_items": [int(x) for x in sampled_retain],
                "residual_target_items": [int(r["candidate_item_id"]) for r in selected],
                "residual_selection_mode": selection_mode,
                "residual_threshold": threshold,
                "residual_top_k": int(top_m),
            })
            if not sampled_retain:
                skip_reasons["no_retain_history_for_null_control"] += 1

        residual_targets.sort(key=lambda r: float(r.get("residual_z", 0.0)), reverse=True)
        summary = {
            "stage": "Stage 1: interaction-induced residual localization",
            "processed_users": int(len(processed_users)),
            "processed_forget_interactions": int(len(interactions)),
            "residual_selection_mode": selection_mode,
            "residual_threshold": float(threshold),
            "top_m": int(top_m),
            "num_all_candidate_records": int(len(all_records)),
            "residual_target_count": int(len(residual_targets)),
            "avg_residual_z": float(np.mean([r["residual_z"] for r in residual_targets])) if residual_targets else 0.0,
            "avg_residual_z_all_candidates": float(np.mean(z_values)) if z_values else 0.0,
            "null_control_robust_distance": self._value_stats(null_control_values),
            "skip_reasons": dict(skip_reasons),
            "top_residual_target_examples": self._top_examples(residual_targets),
        }
        self.logs["stage1_residual_localization"] = summary
        return {
            "summary": summary,
            "residual_targets": residual_targets,
            "records": all_records,
            "interactions": interactions,
        }

    # ------------------------------------------------------------------
    # Stage 2: retain-aware residual strength calibration
    # ------------------------------------------------------------------

    def calibrate_retain_support(self, candidate_records: List[Dict]) -> Dict:
        eps = float(getattr(self.args, "null_eps", 1e-8) or 1e-8)
        requested = getattr(self.args, "retain_support_samples", None)
        if requested is None:
            requested = getattr(self.args, "num_null_removals", 5)
        requested = int(requested or 0)
        by_group = defaultdict(list)
        for record in candidate_records:
            by_group[record["group_id"]].append(dict(record))

        calibrated: List[Dict] = []
        support_raw_values = []
        residual_raw_values = []
        group_logs = []
        aggregation = str(getattr(self.args, "rp_cf_retain_aggregation", "mean_positive_delta") or "mean_positive_delta")

        for group_id, records in by_group.items():
            first = records[0]
            sampled_retain = self._sample_retain_interactions(
                uid=int(first["uid"]),
                history=first["history"],
                forget_iid=int(first["forget_item_id"]),
                requested=requested,
                salt="stage2-retain-support",
            )
            support_by_id = {r["record_id"]: [] for r in records}
            for retain_iid in sampled_retain:
                minus_history = remove_forget_item_from_history(first["history"], retain_iid)
                scores = self._score_candidate_list(
                    minus_history,
                    first["candidate_items"],
                    int(first["scoring_label_iid"]),
                    grad=False,
                )
                for record in records:
                    key = str(int(record["candidate_item_id"]))
                    delta = float(record["score_original"]) - float(scores["scores"].get(key, 0.0))
                    support_by_id[record["record_id"]].append(max(delta, 0.0))

            retain_support = []
            for record in records:
                contributions = support_by_id[record["record_id"]]
                support = self.aggregate_retain_support(contributions, aggregation)
                retain_support.append(float(support))

            for idx, record in enumerate(records):
                support = float(retain_support[idx])
                raw_residual = (
                    max(0.0, float(record.get("residual_z", 0.0)))
                    if bool(record.get("is_residual_target", False))
                    else 0.0
                )
                record.update({
                    "retain_support_raw": support,
                    "I_unl_raw": raw_residual,
                    "sampled_retain_support_items": [int(x) for x in sampled_retain],
                    "num_retain_support_samples": int(len(sampled_retain)),
                    "retain_support_contributions": [
                        float(x) for x in support_by_id[record["record_id"]]
                    ],
                })
                calibrated.append(record)
                support_raw_values.append(support)
                residual_raw_values.append(raw_residual)

            group_logs.append({
                "group_id": group_id,
                "uid": int(first["uid"]),
                "forget_iid": int(first["forget_item_id"]),
                "num_targets": int(len(records)),
                "num_retain_support_samples": int(len(sampled_retain)),
                "avg_retain_support": float(np.mean(retain_support)) if retain_support else 0.0,
            })

        s_ret_norm = self._normalize_global_nonnegative(support_raw_values)
        i_unl_norm = self._normalize_global_nonnegative(residual_raw_values)
        gamma = float(getattr(self.args, "rp_cf_gamma", 1.0) or 1.0)
        w_unl_values = []
        w_prot_values = []
        s_ret_values = []
        i_values = []
        for idx, record in enumerate(calibrated):
            s_ret = float(s_ret_norm[idx]) if idx < len(s_ret_norm) else 0.0
            i_unl = float(i_unl_norm[idx]) if idx < len(i_unl_norm) else 0.0
            w_unl = float(i_unl * math.exp(-gamma * s_ret))
            w_unl = float(np.clip(w_unl, 0.0, 1.0))
            w_prot = float((1.0 - w_unl) * _sigmoid(s_ret))
            w_prot = float(np.clip(w_prot, 0.0, 1.0))
            record.update({
                "S_ret": s_ret,
                "I_unl": i_unl,
                "W_unl": w_unl,
                "W_prot": w_prot,
                # Backward-compatible names used by older summaries.
                "retain_support": s_ret,
                "normalized_residual": i_unl,
                "normalized_retain_support": s_ret,
                "forget_weight": w_unl,
                "retain_weight": w_prot,
                "weight_formula": "W_unl=I_unl*exp(-gamma*S_ret); W_prot=(1-W_unl)*sigmoid(S_ret)",
            })
            s_ret_values.append(s_ret)
            i_values.append(i_unl)
            w_unl_values.append(w_unl)
            w_prot_values.append(w_prot)

        calibrated.sort(key=lambda r: float(r.get("residual_z", 0.0)), reverse=True)
        residual_targets = [r for r in calibrated if bool(r.get("is_residual_target", False))]
        protection_targets = [r for r in calibrated if float(r.get("W_prot", 0.0)) > 0.0]
        summary = {
            "stage": "Stage 2: retain-aware residual strength calibration",
            "candidate_record_count": int(len(calibrated)),
            "residual_target_count": int(len(residual_targets)),
            "protection_target_count": int(len(protection_targets)),
            "avg_retain_support": float(np.mean(s_ret_values)) if s_ret_values else 0.0,
            "avg_forget_weight": float(np.mean(w_unl_values)) if w_unl_values else 0.0,
            "avg_retain_weight": float(np.mean(w_prot_values)) if w_prot_values else 0.0,
            "retain_support_samples": int(requested),
            "gamma": gamma,
            "S_ret_aggregation": aggregation,
            "S_ret_formula_status": (
                "original_non_linear_formula_not_found_in_repository; "
                "using explicit configured fallback aggregation"
            ),
            "I_unl_mapping": "global min-max normalization of positive selected residual_z; non-selected candidates use 0",
            "W_unl_formula": "I_unl * exp(-gamma * S_ret)",
            "W_prot_formula": "(1 - W_unl) * sigmoid(S_ret)",
            "S_ret_stats": self._value_stats(s_ret_values),
            "I_unl_stats": self._value_stats(i_values),
            "W_unl_stats": self._value_stats(w_unl_values),
            "W_prot_stats": self._value_stats(w_prot_values),
            "top_calibrated_examples": self._top_examples(residual_targets),
        }
        self.logs["stage2_retain_calibration"] = summary
        return {
            "summary": summary,
            "residual_targets": residual_targets,
            "protection_targets": protection_targets,
            "all_calibrated_records": calibrated,
            "groups": group_logs,
        }

    # ------------------------------------------------------------------
    # Stage 3: retention-first counterfactual LoRA rank pruning
    # ------------------------------------------------------------------

    def retention_first_domination_prune(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
        split_data: Dict,
    ) -> Dict:
        mode = str(getattr(self.args, "stage3_search_mode", "coarse_to_fine_pruning") or "coarse_to_fine_pruning")
        if mode == "full_greedy":
            print("[Stage3][Legacy] stage3_search_mode=full_greedy")
            return self._retention_first_domination_prune_full_greedy(
                residual_targets,
                protection_targets,
                split_data,
            )
        if mode != "coarse_to_fine_pruning":
            raise ValueError(f"Unsupported stage3_search_mode={mode!r}")
        return self._retention_first_coarse_to_fine_prune(
            residual_targets,
            protection_targets,
            split_data,
        )

    def _retention_first_domination_prune_full_greedy(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
        split_data: Dict,
    ) -> Dict:
        layers = collect_lora_layers(self.model)
        total_ranks = int(sum(layer.rank for layer in layers))
        max_prune_ratio = max(0.0, float(getattr(self.args, "max_prune_ratio", 0.05) or 0.0))
        min_forget_gain = float(getattr(self.args, "min_forget_gain", 0.0) or 0.0)
        global_budget = int(math.floor(total_ranks * max_prune_ratio))
        tau_info = self._derive_null_control_thresholds(residual_targets, protection_targets)
        tau_prot = float(tau_info["tau_prot"])
        residual_stop_threshold = float(tau_info["residual_stop_threshold"])

        if not layers or not residual_targets or global_budget <= 0:
            reason = "no_lora_layers" if not layers else "no_residual_targets" if not residual_targets else "prune_budget_zero"
            decisions = []
            for layer in layers:
                for rid in range(layer.rank):
                    decisions.append({
                        "module_name": layer.key,
                        "rank_id": int(rid),
                        "route": "protect",
                        "route_reason": reason,
                    })
            summary = self._greedy_pruning_summary(
                decisions,
                layers,
                iterations=[],
                reason=reason,
                global_budget=global_budget,
                max_prune_ratio=max_prune_ratio,
                min_forget_gain=min_forget_gain,
                tau_info=tau_info,
                residual_before=0.0,
                residual_after=0.0,
                protection_after=0.0,
            )
            return {"decisions": decisions, "summary": summary}

        layer_by_key = {layer.key: layer for layer in layers}
        accepted: List[Dict] = []
        rejected = set()
        latest_eval_by_key: Dict[str, Dict] = {}
        iterations: List[Dict] = []
        accept_tol = float(getattr(self.args, "rp_cf_accept_tol", 1e-12) or 1e-12)

        current_residual_scores = self._score_calibrated_records(residual_targets)
        current_protection_scores = self._score_calibrated_records(protection_targets)
        initial_protection_scores = dict(current_protection_scores)
        baseline_residual_energy = self._weighted_residual_energy(
            residual_targets,
            current_residual_scores,
        )
        current_residual_energy = baseline_residual_energy
        current_global_protection = 0.0
        stop_reason = None

        while len(accepted) < global_budget:
            iteration = len(iterations) + 1
            if current_residual_energy <= residual_stop_threshold:
                stop_reason = "residual_within_null_control_natural_range"
                break

            rank_records: List[Dict] = []
            for layer in layers:
                for rid in range(int(layer.rank)):
                    rank_key = f"{layer.key}:{int(rid)}"
                    if float(layer.rank_mask[int(rid)].detach().float().cpu()) <= 0.0:
                        continue
                    if rank_key in rejected:
                        continue

                    temporary_mask_rank(layer, rid)
                    try:
                        masked_residual_scores = self._score_calibrated_records(residual_targets)
                        masked_protection_scores = self._score_calibrated_records(protection_targets)
                    finally:
                        restore_rank(layer, rid)

                    masked_residual_energy = self._weighted_residual_energy(
                        residual_targets,
                        masked_residual_scores,
                    )
                    gain = float(current_residual_energy - masked_residual_energy)
                    incremental_protection = self._weighted_protection_perturbation(
                        protection_targets,
                        current_protection_scores,
                        masked_protection_scores,
                    )
                    feasible = (
                        gain > max(0.0, min_forget_gain) and
                        incremental_protection <= tau_prot
                    )
                    importance = gain if feasible else float("-inf")
                    if gain <= 0.0:
                        route_reason = "non_positive_forget_gain"
                    elif gain <= min_forget_gain:
                        route_reason = "below_min_forget_gain"
                    elif incremental_protection > tau_prot:
                        route_reason = "protection_constraint_failed"
                    else:
                        route_reason = "eligible_for_greedy_prune"
                    record = {
                        "iteration": int(iteration),
                        "module_name": layer.key,
                        "rank_id": int(rid),
                        "parameter_key": rank_key,
                        "G_lk": float(gain),
                        "P_lk": float(incremental_protection),
                        "I_lk": float(importance) if np.isfinite(importance) else "-inf",
                        "residual_energy_before": float(current_residual_energy),
                        "residual_energy_after_if_masked": float(masked_residual_energy),
                        "tau_prot": float(tau_prot),
                        "passes_forget_gain": bool(gain > max(0.0, min_forget_gain)),
                        "passes_protection": bool(incremental_protection <= tau_prot),
                        "candidate_prune": bool(feasible),
                        "route_reason": route_reason,
                    }
                    rank_records.append(record)
                    latest_eval_by_key[rank_key] = record

            feasible_records = [
                r for r in rank_records
                if r.get("candidate_prune") and float(r.get("G_lk", 0.0)) > 0.0
            ]
            if not feasible_records:
                stop_reason = "completed_no_valid_rank" if not accepted else "no_valid_rank_remaining"
                iterations.append({
                    "iteration": int(iteration),
                    "accepted": False,
                    "rollback": False,
                    "stop_reason": stop_reason,
                    "current_pruned_rank_count": int(len(accepted)),
                    "current_prune_ratio": float(len(accepted)) / float(total_ranks) if total_ranks else 0.0,
                    "rank_evaluations": rank_records,
                })
                break

            selected = max(
                feasible_records,
                key=lambda r: (float(r.get("G_lk", 0.0)), -float(r.get("P_lk", 0.0))),
            )
            selected_layer = layer_by_key[selected["module_name"]]
            selected_rank = int(selected["rank_id"])
            rank_key = selected["parameter_key"]
            with torch.no_grad():
                selected_layer.rank_mask[selected_rank].fill_(0.0)

            verify_residual_scores = self._score_calibrated_records(residual_targets)
            verify_protection_scores = self._score_calibrated_records(protection_targets)
            verify_residual_energy = self._weighted_residual_energy(
                residual_targets,
                verify_residual_scores,
            )
            incremental_protection = self._weighted_protection_perturbation(
                protection_targets,
                current_protection_scores,
                verify_protection_scores,
            )
            global_protection = self._weighted_protection_perturbation(
                protection_targets,
                initial_protection_scores,
                verify_protection_scores,
            )
            next_ratio = float(len(accepted) + 1) / float(total_ranks) if total_ranks else 0.0
            accept = (
                verify_residual_energy < current_residual_energy - accept_tol and
                incremental_protection <= tau_prot and
                global_protection <= tau_prot and
                next_ratio <= max_prune_ratio + 1e-12
            )

            iteration_log = {
                "iteration": int(iteration),
                "current_pruned_rank_count": int(len(accepted)),
                "current_prune_ratio": float(len(accepted)) / float(total_ranks) if total_ranks else 0.0,
                "selected_rank": {
                    "module_name": selected["module_name"],
                    "rank_id": int(selected_rank),
                    "parameter_key": rank_key,
                },
                "selected_G_lk": float(selected.get("G_lk", 0.0)),
                "selected_P_lk": float(selected.get("P_lk", 0.0)),
                "selected_I_lk": float(selected.get("G_lk", 0.0)),
                "residual_energy_before": float(current_residual_energy),
                "residual_energy_after": float(verify_residual_energy),
                "incremental_protection_perturbation": float(incremental_protection),
                "global_protection_perturbation": float(global_protection),
                "tau_prot": float(tau_prot),
                "accepted": bool(accept),
                "rollback": not bool(accept),
                "rank_evaluations": rank_records,
            }
            if accept:
                selected.update({
                    "accepted_iteration": int(iteration),
                    "route": "hard_prune",
                    "route_reason": "greedy_gain_positive_and_global_validation_passed",
                    "rank_mask_value": 0.0,
                    "pruned_by_rank_mask": True,
                    "global_protection_perturbation": float(global_protection),
                    "verified_residual_energy_after": float(verify_residual_energy),
                })
                accepted.append(selected)
                current_residual_scores = verify_residual_scores
                current_protection_scores = verify_protection_scores
                current_residual_energy = verify_residual_energy
                current_global_protection = global_protection
            else:
                with torch.no_grad():
                    selected_layer.rank_mask[selected_rank].fill_(1.0)
                rejected.add(rank_key)
                rollback_reason = "global_validation_failed"
                if verify_residual_energy >= current_residual_energy - accept_tol:
                    rollback_reason = "residual_energy_not_decreased"
                elif incremental_protection > tau_prot:
                    rollback_reason = "incremental_protection_exceeded"
                elif global_protection > tau_prot:
                    rollback_reason = "global_protection_exceeded"
                elif next_ratio > max_prune_ratio + 1e-12:
                    rollback_reason = "max_prune_ratio_exceeded"
                iteration_log["rollback_reason"] = rollback_reason
                selected.update({
                    "route": "protect",
                    "route_reason": rollback_reason,
                    "rank_mask_value": 1.0,
                    "attempted_route": "hard_prune",
                })
            iterations.append(iteration_log)

        if stop_reason is None:
            if len(accepted) >= global_budget:
                stop_reason = "max_prune_ratio_reached"
            else:
                stop_reason = "completed"

        decisions = []
        accepted_by_key = {r["parameter_key"]: r for r in accepted}
        for layer in layers:
            for rid in range(int(layer.rank)):
                rank_key = f"{layer.key}:{int(rid)}"
                latest = dict(latest_eval_by_key.get(rank_key, {}))
                if rank_key in accepted_by_key:
                    accepted_record = accepted_by_key[rank_key]
                    decision = {
                        **latest,
                        **accepted_record,
                        "module_name": layer.key,
                        "rank_id": int(rid),
                        "parameter_key": rank_key,
                        "route": "hard_prune",
                        "rank_mask_value": 0.0,
                        "pruned_by_rank_mask": True,
                    }
                else:
                    decision = {
                        **latest,
                        "module_name": layer.key,
                        "rank_id": int(rid),
                        "parameter_key": rank_key,
                        "route": "protect",
                        "rank_mask_value": 1.0,
                        "pruned_by_rank_mask": False,
                    }
                    if rank_key in rejected:
                        decision["route_reason"] = latest.get("route_reason", "rejected_after_rollback")
                    else:
                        decision["route_reason"] = latest.get("route_reason", "not_selected_before_stop")
                decisions.append(decision)

        final_residual_scores = self._score_calibrated_records(residual_targets)
        final_protection_scores = self._score_calibrated_records(protection_targets)
        final_residual_energy = self._weighted_residual_energy(
            residual_targets,
            final_residual_scores,
        )
        final_protection = self._weighted_protection_perturbation(
            protection_targets,
            initial_protection_scores,
            final_protection_scores,
        )
        summary = self._greedy_pruning_summary(
            decisions,
            layers,
            iterations=iterations,
            reason=stop_reason,
            global_budget=global_budget,
            max_prune_ratio=max_prune_ratio,
            min_forget_gain=min_forget_gain,
            tau_info=tau_info,
            residual_before=baseline_residual_energy,
            residual_after=final_residual_energy,
            protection_after=final_protection,
        )
        self.logs["stage3_rank_pruning"] = summary
        return {"decisions": decisions, "summary": summary}

    def _retention_first_coarse_to_fine_prune(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
        split_data: Dict,
    ) -> Dict:
        del split_data
        stage_start = time.time()
        layers = collect_lora_layers(self.model)
        total_ranks = int(sum(layer.rank for layer in layers))
        max_prune_ratio = max(0.0, float(getattr(self.args, "max_prune_ratio", 0.05) or 0.0))
        min_forget_gain = float(getattr(self.args, "min_forget_gain", 0.0) or 0.0)
        global_budget = int(math.floor(total_ranks * max_prune_ratio))
        lambda_retain = max(0.0, float(getattr(self.args, "stage3_lambda_retain", 1.0) or 0.0))
        tau_info = self._derive_null_control_thresholds(residual_targets, protection_targets)
        block_schedule = self._stage3_block_schedule(global_budget)
        config_payload = self._stage3_config_payload(
            residual_targets=residual_targets,
            protection_targets=protection_targets,
            layers=layers,
            block_schedule=block_schedule,
            global_budget=global_budget,
        )
        config_hash = self._stage3_config_hash(config_payload)

        print(f"[Stage3][Backup] backup_path={self._stage3_backup_path()}")
        print("[Stage3][Probe] mode=no_gradient_mask_ablation")
        print(f"[Stage3][Probe] total_ranks={total_ranks}")
        print("[Stage3][OneShot] mode=joint_score_topk")

        if not layers or not residual_targets or global_budget <= 0:
            reason = "no_lora_layers" if not layers else "no_residual_targets" if not residual_targets else "prune_budget_zero"
            decisions = self._stage3_decisions_from_selected(layers, [], {}, reason=reason)
            summary = self._coarse_pruning_summary(
                layers=layers,
                decisions=decisions,
                rounds=[],
                reason=reason,
                global_budget=global_budget,
                max_prune_ratio=max_prune_ratio,
                min_forget_gain=min_forget_gain,
                tau_info=tau_info,
                block_schedule=block_schedule,
                full_before=None,
                full_after=None,
                config_hash=config_hash,
                elapsed_seconds=time.time() - stage_start,
            )
            summary.update({
                "stage": "Stage 3: one-shot joint-score LoRA rank pruning",
                "stage3_mode": "one_shot_joint_score_pruning",
                "block_pruning": False,
                "selection_rule": (
                    "static joint_score=normalize(forget_gain)-lambda_retain*normalize(retain_risk), "
                    "then one-shot Top-K rank masking"
                ),
                "stage3_lambda_retain": float(lambda_retain),
                "one_shot_top_k": int(global_budget),
            })
            return {"decisions": decisions, "summary": summary}

        completed = self._load_completed_stage3(layers, config_hash)
        if completed is not None:
            return completed

        all_one_state = {key: 1 for key in expected_lora_rank_mask_keys(layers)}
        restore_rank_masks(layers, all_one_state)

        print(
            f"[Stage3][FinalValidation] baseline_forget_targets={len(residual_targets)} "
            f"baseline_protection_targets={len(protection_targets)}"
        )
        full_before = self._stage3_measure_metrics(
            residual_targets,
            protection_targets,
            protection_reference_scores=None,
        )

        calibration = self._get_or_build_stage3_calibration(
            residual_targets,
            protection_targets,
            config_hash,
        )
        forget_calibration = self._stage3_records_by_ids(
            residual_targets,
            calibration.get("forget_record_ids", []),
        )
        protection_calibration = self._stage3_records_by_ids(
            protection_targets,
            calibration.get("protection_record_ids", []),
        )
        print(
            f"[Stage3][Calibration] forget selected={len(forget_calibration)}/{len(residual_targets)}"
        )
        print(
            f"[Stage3][Calibration] protection selected={len(protection_calibration)}/{len(protection_targets)}"
        )
        print(
            f"[Stage3][Probe] calibration_targets={len(forget_calibration) + len(protection_calibration)}"
        )

        probe_payload = self._get_or_build_stage3_probe_scores(
            layers=layers,
            forget_calibration=forget_calibration,
            protection_calibration=protection_calibration,
            config_hash=config_hash,
        )
        probe_records = probe_payload.get("records", [])

        shortlist_payload, reserve_payload = self._get_or_build_stage3_shortlist(
            probe_records,
            config_hash,
        )
        active_shortlist = [str(x) for x in shortlist_payload.get("rank_ids", [])]
        reserve_pool = [str(x) for x in reserve_payload.get("rank_ids", [])]
        print(
            f"[Stage3][Shortlist] shortlist={len(active_shortlist)} "
            f"reserve={len(reserve_pool)} "
            f"protection_quantile={float(getattr(self.args, 'protection_risk_quantile', 0.70) or 0.70):.2f}"
        )

        restore_rank_masks(layers, all_one_state)
        probe_by_id = {
            str(record.get("rank_id")): dict(record)
            for record in probe_records
            if record.get("rank_id") is not None
        }
        ordered_candidate_ids = []
        seen_candidate_ids = set()
        for source in (
            active_shortlist,
            reserve_pool,
            [str(record.get("rank_id")) for record in probe_records if record.get("rank_id") is not None],
        ):
            for rank_id in source:
                rank_id = str(rank_id)
                if rank_id in seen_candidate_ids:
                    continue
                if rank_id not in probe_by_id:
                    continue
                if not self._stage3_rank_is_active(layers, rank_id):
                    continue
                ordered_candidate_ids.append(rank_id)
                seen_candidate_ids.add(rank_id)

        candidate_records = [dict(probe_by_id[rank_id]) for rank_id in ordered_candidate_ids]
        if candidate_records:
            forget_norm = self._stage3_robust_unit_values([
                float(record.get("forget_gain", 0.0)) for record in candidate_records
            ])
            risk_values = [
                max(0.0, float(record.get("protection_damage", 0.0)))
                for record in candidate_records
            ]
            risk_norm = self._stage3_robust_unit_values(risk_values)
            min_gain_threshold = max(0.0, float(min_forget_gain))
            for index, record in enumerate(candidate_records):
                retain_risk = float(risk_values[index])
                record["retain_risk"] = retain_risk
                record["retain_risk_normalized"] = float(risk_norm[index])
                record["forget_gain_normalized"] = float(forget_norm[index])
                record["joint_score"] = float(forget_norm[index] - lambda_retain * risk_norm[index])
                record["joint_score_formula"] = (
                    "normalize(forget_gain)-lambda_retain*normalize(retain_risk)"
                )
                record["stage3_lambda_retain"] = float(lambda_retain)
                record["passes_min_forget_gain"] = bool(
                    float(record.get("forget_gain", 0.0)) > min_gain_threshold
                )
        sorted_candidates = sorted(
            candidate_records,
            key=lambda record: (
                -float(record.get("joint_score", 0.0)),
                -float(record.get("forget_gain", 0.0)),
                float(record.get("retain_risk", 0.0)),
                str(record.get("rank_id")),
            ),
        )
        top_k = min(int(global_budget), int(len(sorted_candidates)))
        selected_records = [dict(record) for record in sorted_candidates[:top_k]]
        selected_rank_ids = [str(record.get("rank_id")) for record in selected_records]
        selected_rank_id_set = set(selected_rank_ids)
        for selection_index, record in enumerate(selected_records, start=1):
            record["route"] = "hard_prune"
            record["route_reason"] = "one_shot_joint_score_topk"
            record["rank_mask_value"] = 0.0
            record["pruned_by_rank_mask"] = True
            record["one_shot_selection_rank"] = int(selection_index)
        for record in candidate_records:
            record["selected_one_shot"] = bool(str(record.get("rank_id")) in selected_rank_id_set)
            if not record["selected_one_shot"]:
                record["route"] = "protect"
                record["route_reason"] = "one_shot_joint_score_not_topk"
                record["rank_mask_value"] = 1.0
                record["pruned_by_rank_mask"] = False

        print(
            f"[Stage3][OneShot] candidate_count={len(candidate_records)} "
            f"top_k={top_k} lambda_retain={lambda_retain:.4f}"
        )
        restore_rank_masks(layers, all_one_state)
        apply_permanent_rank_masks(layers, selected_rank_ids)
        selected_score_by_id = {str(record.get("rank_id")): dict(record) for record in selected_records}
        stop_reason = "max_prune_ratio_reached" if len(selected_rank_ids) >= global_budget else "completed"
        print(
            f"[Stage3][FinalValidation] forget_targets={len(residual_targets)} "
            f"protection_targets={len(protection_targets)}"
        )
        full_after = self._stage3_measure_metrics(
            residual_targets,
            protection_targets,
            protection_reference_scores=full_before["protection_scores"],
        )
        one_shot_payload = {
            "round": 1,
            "config_hash": config_hash,
            "mode": "one_shot_joint_score_topk",
            "requested_block_size": int(global_budget),
            "actual_selected_count": int(len(selected_rank_ids)),
            "baseline_forget_metric": float(full_before["forget_metric"]),
            "baseline_protection_metric": float(full_before["protection_metric"]),
            "candidate_count": int(len(candidate_records)),
            "lambda_retain": float(lambda_retain),
            "joint_score_formula": (
                "normalize(forget_gain)-lambda_retain*normalize(retain_risk)"
            ),
            "selected": selected_records,
            "block_joint_validation": {
                "forget_metric": float(full_after["forget_metric"]),
                "protection_metric": float(full_after["protection_metric"]),
                "forget_gain": float(full_before["forget_metric"] - full_after["forget_metric"]),
            },
            "fallback_used": False,
            "elapsed_seconds": float(time.time() - stage_start),
            "rank_evaluations": candidate_records,
        }
        rounds = [one_shot_payload]
        self._save_stage3_cache("rank_exact_scores_round1.json", one_shot_payload)
        self._save_stage3_selected_masks(layers, selected_rank_ids, selected_records, config_hash)
        torch.save(
            get_current_rank_mask_state(layers),
            os.path.join(self.output_dir, "prune_mask_round1.pt"),
        )

        decisions = self._stage3_decisions_from_selected(
            layers,
            selected_rank_ids,
            selected_score_by_id,
            reason="one_shot_joint_score_not_topk",
        )
        summary = self._coarse_pruning_summary(
            layers=layers,
            decisions=decisions,
            rounds=rounds,
            reason=stop_reason,
            global_budget=global_budget,
            max_prune_ratio=max_prune_ratio,
            min_forget_gain=min_forget_gain,
            tau_info=tau_info,
            block_schedule=block_schedule,
            full_before=full_before,
            full_after=full_after,
            config_hash=config_hash,
            elapsed_seconds=time.time() - stage_start,
        )
        summary.update({
            "stage": "Stage 3: one-shot joint-score LoRA rank pruning",
            "stage3_mode": "one_shot_joint_score_pruning",
            "block_pruning": False,
            "selection_rule": (
                "static joint_score=normalize(forget_gain)-lambda_retain*normalize(retain_risk), "
                "then one-shot Top-K rank masking"
            ),
            "stage3_lambda_retain": float(lambda_retain),
            "one_shot_top_k": int(global_budget),
            "one_shot_candidate_count": int(len(candidate_records)),
            "one_shot_rank_exact_scores_file": "rank_exact_scores_round1.json",
        })
        pruning = {
            "config_hash": config_hash,
            "decisions": decisions,
            "summary": summary,
            "rounds": rounds,
            "config": config_payload,
            "_after_residual_score_overrides": {
                record_id: float(value.get("score", 0.0))
                for record_id, value in full_after.get("forget_scores", {}).items()
            },
        }
        atomic_save_json(
            os.path.join(self.output_dir, "pruning_summary.json"),
            summary,
        )
        print(
            f"[Stage3][Completed] selected={len(selected_rank_ids)}/{total_ranks} "
            f"summary_path={os.path.join(self.output_dir, 'pruning_summary.json')}"
        )
        self.logs["stage3_rank_pruning"] = summary
        return pruning

    def _stage3_config_payload(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
        layers: List[LoRALayerHandle],
        block_schedule: List[int],
        global_budget: int,
    ) -> Dict:
        return {
            "stage3_search_mode": "coarse_to_fine_pruning",
            "stage3_selection_mode": "one_shot_joint_score_pruning",
            "stage3_lambda_retain": float(getattr(self.args, "stage3_lambda_retain", 1.0) or 0.0),
            "forget_calibration_size": int(getattr(self.args, "forget_calibration_size", 128) or 128),
            "protection_calibration_size": int(getattr(self.args, "protection_calibration_size", 512) or 512),
            "rank_shortlist_size": int(getattr(self.args, "rank_shortlist_size", 64) or 64),
            "rank_reserve_size": int(getattr(self.args, "rank_reserve_size", 128) or 128),
            "protection_risk_quantile": float(getattr(self.args, "protection_risk_quantile", 0.70) or 0.70),
            "prune_block_schedule": [int(x) for x in block_schedule],
            "stage3_full_eval_frequency": str(getattr(self.args, "stage3_full_eval_frequency", "final_only") or "final_only"),
            "max_prune_ratio": float(getattr(self.args, "max_prune_ratio", 0.05) or 0.0),
            "min_forget_gain": float(getattr(self.args, "min_forget_gain", 0.0) or 0.0),
            "target_prune_count": int(global_budget),
            "seed": int(getattr(self.args, "seed", 42) or 42),
            "full_forget_target_count": int(len(residual_targets)),
            "full_protection_target_count": int(len(protection_targets)),
            "lora_rank_fingerprint": self._stage3_fingerprint(expected_lora_rank_mask_keys(layers)),
            "forget_target_fingerprint": self._stage3_fingerprint(
                [str(r.get("record_id", idx)) for idx, r in enumerate(residual_targets)]
            ),
            "protection_target_fingerprint": self._stage3_fingerprint(
                [str(r.get("record_id", idx)) for idx, r in enumerate(protection_targets)]
            ),
        }

    @staticmethod
    def _stage3_config_hash(payload: Dict) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _stage3_fingerprint(values: List[str]) -> str:
        h = hashlib.sha256()
        for value in values:
            h.update(str(value).encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _stage3_backup_path(self) -> Optional[str]:
        explicit = getattr(self.args, "stage3_backup_path", None)
        if explicit:
            return explicit
        backup_root = os.path.join(os.getcwd(), "backups")
        try:
            names = [
                name for name in os.listdir(backup_root)
                if name.startswith("stage3_prune_backup_") and
                os.path.isdir(os.path.join(backup_root, name))
            ]
        except OSError:
            return None
        if not names:
            return None
        return os.path.join("backups", sorted(names)[-1])

    def _stage3_block_schedule(self, target_count: int) -> List[int]:
        target_count = max(0, int(target_count))
        if target_count <= 0:
            return []
        raw = str(getattr(self.args, "prune_block_schedule", "10,10,5") or "10,10,5")
        parts = []
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            value = int(chunk)
            if value > 0:
                parts.append(value)
        if not parts:
            parts = [target_count]
        remaining = target_count
        schedule = []
        for value in parts:
            if remaining <= 0:
                break
            take = min(int(value), int(remaining))
            if take > 0:
                schedule.append(take)
                remaining -= take
        if remaining > 0:
            schedule.append(int(remaining))
        return schedule

    def _load_resume_json(self, path: str, label: str):
        if not bool(getattr(self.args, "stage3_resume", True)):
            return None
        if (
            bool(getattr(self.args, "stage3_force_recompute", False)) and
            not bool(getattr(self.args, "resume_stage3_from_cached_stage12", False))
        ):
            return None
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as exc:
            print(f"[{label}][Cache] failed_to_load path={path} error={exc}")
            return None

    def _load_stage3_cache(self, filename: str, config_hash: str):
        if not bool(getattr(self.args, "stage3_resume", True)):
            return None
        if bool(getattr(self.args, "stage3_force_recompute", False)):
            return None
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[Stage3][Cache] failed_to_load file={filename} error={exc}")
            return None
        if payload.get("config_hash") != config_hash:
            print(f"[Stage3][Cache] config_mismatch file={filename}; recomputing")
            return None
        return payload

    def _save_stage3_cache(self, filename: str, payload: Dict):
        atomic_save_json(os.path.join(self.output_dir, filename), payload)

    def _load_completed_stage3(self, layers: List[LoRALayerHandle], config_hash: str):
        summary = self._load_stage3_cache("pruning_summary.json", config_hash)
        decisions_payload = self._load_stage3_cache("pruning_decisions.json", config_hash)
        if not summary or not decisions_payload:
            return None
        if summary.get("status") != "completed":
            return None
        decisions = decisions_payload.get("decisions", [])
        selected = [
            f"{d['module_name']}:{int(d['rank_id'])}"
            for d in decisions
            if d.get("route") == "hard_prune"
        ]
        restore_rank_masks(layers, {key: 1 for key in expected_lora_rank_mask_keys(layers)})
        apply_permanent_rank_masks(layers, selected)
        print("[Stage3][Completed][Cache] pruning_summary.json")
        return {"decisions": decisions, "summary": summary}

    def _get_or_build_stage3_calibration(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
        config_hash: str,
    ) -> Dict:
        cached = self._load_stage3_cache("calibration_targets.json", config_hash)
        if cached is not None:
            print("[Stage3][Calibration][Cache] calibration_targets.json")
            return cached

        forget_size = max(0, int(getattr(self.args, "forget_calibration_size", 128) or 128))
        protection_size = max(0, int(getattr(self.args, "protection_calibration_size", 512) or 512))
        forget_selected, forget_comp = self._select_forget_calibration(residual_targets, forget_size)
        protection_selected, protection_comp = self._select_protection_calibration(
            protection_targets,
            protection_size,
        )
        payload = {
            "config_hash": config_hash,
            "forget_total": int(len(residual_targets)),
            "forget_selected": int(len(forget_selected)),
            "protection_total": int(len(protection_targets)),
            "protection_selected": int(len(protection_selected)),
            "forget_sampling_composition": forget_comp,
            "protection_sampling_composition": protection_comp,
            "seed": int(getattr(self.args, "seed", 42) or 42),
            "forget_record_ids": [str(r["record_id"]) for r in forget_selected],
            "protection_record_ids": [str(r["record_id"]) for r in protection_selected],
        }
        self._save_stage3_cache("calibration_targets.json", payload)
        return payload

    def _select_forget_calibration(self, records: List[Dict], requested: int) -> Tuple[List[Dict], Dict]:
        requested = min(max(0, int(requested)), len(records))
        if requested <= 0:
            return [], {"high_residual": 0, "boundary_sensitive": 0, "stratified_random": 0}
        high_n = min(requested, int(round(requested * 0.50)))
        boundary_n = min(requested - high_n, int(round(requested * 0.30)))
        random_n = requested - high_n - boundary_n
        selected = []
        seen = set()
        high_residual = sorted(
            records,
            key=lambda r: (
                float(r.get("W_unl", r.get("forget_weight", 0.0))),
                float(r.get("residual_z", 0.0)),
                abs(float(r.get("residual_raw", r.get("residual_score", 0.0)))),
            ),
            reverse=True,
        )
        selected.extend(self._stage3_take_unique(high_residual, seen, high_n))
        boundary = sorted(
            records,
            key=lambda r: (
                abs(float(r.get("boundary_gap_original", 0.0))),
                -float(r.get("residual_z", 0.0)),
            ),
        )
        selected.extend(self._stage3_take_unique(boundary, seen, boundary_n))
        selected.extend(self._stage3_take_unique(
            self._stage3_stratified_order(records, salt="forget-calibration"),
            seen,
            random_n,
        ))
        if len(selected) < requested:
            selected.extend(self._stage3_take_unique(high_residual, seen, requested - len(selected)))
        comp = {
            "high_residual": min(high_n, len(selected)),
            "boundary_sensitive": max(0, min(boundary_n, len(selected) - high_n)),
            "stratified_random": max(0, len(selected) - high_n - boundary_n),
        }
        return selected, comp

    def _select_protection_calibration(self, records: List[Dict], requested: int) -> Tuple[List[Dict], Dict]:
        requested = min(max(0, int(requested)), len(records))
        if requested <= 0:
            return [], {"high_protection": 0, "boundary_sensitive": 0, "coverage_stratified": 0}
        high_n = min(requested, int(round(requested * 0.60)))
        boundary_n = min(requested - high_n, int(round(requested * 0.25)))
        coverage_n = requested - high_n - boundary_n
        selected = []
        seen = set()
        high_protection = sorted(
            records,
            key=lambda r: (
                float(r.get("W_prot", r.get("retain_weight", 0.0))),
                float(r.get("S_ret", r.get("retain_support", 0.0))),
                float(r.get("I_unl", r.get("normalized_residual", 0.0))),
            ),
            reverse=True,
        )
        selected.extend(self._stage3_take_unique(high_protection, seen, high_n))
        boundary = sorted(
            records,
            key=lambda r: (
                abs(float(r.get("boundary_gap_original", 0.0))),
                -float(r.get("W_prot", r.get("retain_weight", 0.0))),
            ),
        )
        selected.extend(self._stage3_take_unique(boundary, seen, boundary_n))
        selected.extend(self._stage3_take_unique(
            self._stage3_stratified_order(records, salt="protection-calibration"),
            seen,
            coverage_n,
        ))
        if len(selected) < requested:
            selected.extend(self._stage3_take_unique(high_protection, seen, requested - len(selected)))
        comp = {
            "high_protection": min(high_n, len(selected)),
            "boundary_sensitive": max(0, min(boundary_n, len(selected) - high_n)),
            "coverage_stratified": max(0, len(selected) - high_n - boundary_n),
        }
        return selected, comp

    @staticmethod
    def _stage3_take_unique(records: List[Dict], seen: set, limit: int) -> List[Dict]:
        out = []
        for record in records:
            rid = str(record.get("record_id"))
            if rid in seen:
                continue
            seen.add(rid)
            out.append(record)
            if len(out) >= int(limit):
                break
        return out

    def _stage3_stratified_order(self, records: List[Dict], salt: str) -> List[Dict]:
        seed = int(getattr(self.args, "seed", 42) or 42)
        buckets = defaultdict(list)
        for record in records:
            uid = int(record.get("uid", record.get("user_id", 0)) or 0)
            iid = int(record.get("candidate_iid", record.get("candidate_item_id", 0)) or 0)
            buckets[(uid % 32, iid % 32)].append(record)
        for key, bucket in buckets.items():
            bucket.sort(key=lambda r: hashlib.sha256(
                f"{seed}|{salt}|{r.get('record_id')}".encode("utf-8")
            ).hexdigest())
        ordered = []
        keys = sorted(buckets.keys())
        offset = 0
        while True:
            added = False
            for key in keys:
                bucket = buckets[key]
                if offset < len(bucket):
                    ordered.append(bucket[offset])
                    added = True
            if not added:
                break
            offset += 1
        return ordered

    @staticmethod
    def _stage3_records_by_ids(records: List[Dict], record_ids: List[str]) -> List[Dict]:
        by_id = {str(r.get("record_id")): r for r in records}
        return [by_id[str(rid)] for rid in record_ids if str(rid) in by_id]

    def _get_or_build_stage3_probe_scores(
        self,
        layers: List[LoRALayerHandle],
        forget_calibration: List[Dict],
        protection_calibration: List[Dict],
        config_hash: str,
    ) -> Dict:
        cached = self._load_stage3_cache("rank_prune_probe_scores.json", config_hash)
        if cached is not None:
            print("[Stage3][Probe][Cache] rank_prune_probe_scores.json")
            return cached
        baseline = self._stage3_measure_metrics(
            forget_calibration,
            protection_calibration,
            protection_reference_scores=None,
        )
        records = []
        for rank_id in expected_lora_rank_mask_keys(layers):
            if not self._stage3_rank_is_active(layers, rank_id):
                continue
            records.append(self._stage3_probe_rank(
                layers=layers,
                rank_id=rank_id,
                forget_targets=forget_calibration,
                protection_targets=protection_calibration,
                baseline_forget_metric=baseline["forget_metric"],
                baseline_protection_metric=baseline["protection_metric"],
                baseline_protection_scores=baseline["protection_scores"],
            ))
        self._stage3_add_normalized_scores(records)
        payload = {
            "config_hash": config_hash,
            "mode": "no_gradient_mask_ablation",
            "total_ranks": int(sum(layer.rank for layer in layers)),
            "calibration_target_count": int(len(forget_calibration) + len(protection_calibration)),
            "baseline_forget_metric": float(baseline["forget_metric"]),
            "baseline_protection_metric": float(baseline["protection_metric"]),
            "records": records,
        }
        self._save_stage3_cache("rank_prune_probe_scores.json", payload)
        return payload

    def _stage3_measure_metrics(
        self,
        forget_targets: List[Dict],
        protection_targets: List[Dict],
        protection_reference_scores: Optional[Dict[str, Dict[str, float]]],
    ) -> Dict:
        self.model.eval()
        with torch.inference_mode():
            forget_scores = self._score_calibrated_records(forget_targets) if forget_targets else {}
            protection_scores = self._score_calibrated_records(protection_targets) if protection_targets else {}
        forget_metric = self._weighted_residual_energy(forget_targets, forget_scores)
        if protection_reference_scores is None:
            protection_metric = 0.0
        else:
            protection_metric = self._weighted_protection_perturbation(
                protection_targets,
                protection_reference_scores,
                protection_scores,
            )
        return {
            "forget_metric": float(forget_metric),
            "protection_metric": float(protection_metric),
            "forget_scores": forget_scores,
            "protection_scores": protection_scores,
        }

    def _stage3_probe_rank(
        self,
        layers: List[LoRALayerHandle],
        rank_id: str,
        forget_targets: List[Dict],
        protection_targets: List[Dict],
        baseline_forget_metric: float,
        baseline_protection_metric: float,
        baseline_protection_scores: Dict[str, Dict[str, float]],
    ) -> Dict:
        with temporary_rank_mask(layers, rank_id):
            masked = self._stage3_measure_metrics(
                forget_targets,
                protection_targets,
                protection_reference_scores=baseline_protection_scores,
            )
        module_name, rank_s = str(rank_id).rsplit(":", 1)
        forget_gain = float(baseline_forget_metric - masked["forget_metric"])
        protection_damage = float(masked["protection_metric"] - baseline_protection_metric)
        return {
            "rank_id": str(rank_id),
            "parameter_key": str(rank_id),
            "module_name": module_name,
            "rank_index": int(rank_s),
            "rank_id_local": int(rank_s),
            "forget_gain": forget_gain,
            "protection_damage": protection_damage,
            "masked_forget_metric": float(masked["forget_metric"]),
            "masked_protection_metric": float(masked["protection_metric"]),
            "forget_gain_normalized": 0.0,
            "protection_damage_normalized": 0.0,
            "passes_protection_constraint": True,
        }

    def _stage3_add_normalized_scores(self, records: List[Dict]):
        if not records:
            return
        forget_norm = self._stage3_robust_unit_values([float(r.get("forget_gain", 0.0)) for r in records])
        damage_norm = self._stage3_robust_unit_values([float(r.get("protection_damage", 0.0)) for r in records])
        q = float(getattr(self.args, "protection_risk_quantile", 0.70) or 0.70)
        damage_threshold = _safe_quantile([r.get("protection_damage", 0.0) for r in records], q, default=0.0)
        for idx, record in enumerate(records):
            record["forget_gain_normalized"] = float(forget_norm[idx])
            record["protection_damage_normalized"] = float(damage_norm[idx])
            record["passes_protection_constraint"] = bool(
                float(record.get("protection_damage", 0.0)) <= damage_threshold + 1e-12
            )

    @staticmethod
    def _stage3_robust_unit_values(values: List[float]) -> List[float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return []
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad <= 1e-12:
            q75, q25 = np.percentile(arr, [75, 25])
            scale = float(q75 - q25)
        else:
            scale = 1.4826 * mad
        if scale <= 1e-12:
            return [0.0 for _ in arr.tolist()]
        z = np.clip((arr - median) / scale, -5.0, 5.0)
        lo = float(np.min(z))
        hi = float(np.max(z))
        if hi <= lo + 1e-12:
            return [0.0 for _ in z.tolist()]
        return [float(v) for v in ((z - lo) / (hi - lo)).tolist()]

    def _get_or_build_stage3_shortlist(self, records: List[Dict], config_hash: str) -> Tuple[Dict, Dict]:
        shortlist = self._load_stage3_cache("rank_shortlist.json", config_hash)
        reserve = self._load_stage3_cache("rank_reserve_pool.json", config_hash)
        if shortlist is not None and reserve is not None:
            print("[Stage3][Shortlist][Cache] rank_shortlist.json rank_reserve_pool.json")
            return shortlist, reserve

        shortlist_size = max(0, int(getattr(self.args, "rank_shortlist_size", 64) or 64))
        reserve_size = max(shortlist_size, int(getattr(self.args, "rank_reserve_size", 128) or 128))
        q = float(getattr(self.args, "protection_risk_quantile", 0.70) or 0.70)
        damage_threshold = _safe_quantile([r.get("protection_damage", 0.0) for r in records], q, default=0.0)
        eligible = [
            r for r in records
            if float(r.get("protection_damage", 0.0)) <= damage_threshold + 1e-12
        ]
        fallback_used = False
        if not eligible:
            eligible = list(records)
            fallback_used = True
        ordered = sorted(
            eligible,
            key=lambda r: (
                -float(r.get("forget_gain", 0.0)),
                float(r.get("protection_damage", 0.0)),
                str(r.get("rank_id")),
            ),
        )
        reserve_ordered = sorted(
            records,
            key=lambda r: (
                r not in eligible,
                -float(r.get("forget_gain", 0.0)),
                float(r.get("protection_damage", 0.0)),
                str(r.get("rank_id")),
            ),
        )
        shortlist_records = ordered[:shortlist_size]
        reserve_records = reserve_ordered[:reserve_size]
        shortlist = {
            "config_hash": config_hash,
            "protection_damage_threshold": float(damage_threshold),
            "fallback_used": bool(fallback_used),
            "rank_ids": [str(r["rank_id"]) for r in shortlist_records],
            "records": shortlist_records,
        }
        reserve = {
            "config_hash": config_hash,
            "protection_damage_threshold": float(damage_threshold),
            "rank_ids": [str(r["rank_id"]) for r in reserve_records],
            "records": reserve_records,
        }
        self._save_stage3_cache("rank_shortlist.json", shortlist)
        self._save_stage3_cache("rank_reserve_pool.json", reserve)
        return shortlist, reserve

    def _stage3_select_block(
        self,
        records: List[Dict],
        block_size: int,
        min_forget_gain: float,
    ) -> Tuple[List[Dict], bool, float]:
        q = float(getattr(self.args, "protection_risk_quantile", 0.70) or 0.70)
        damage_threshold = _safe_quantile([r.get("protection_damage", 0.0) for r in records], q, default=0.0)
        eligible = [
            r for r in records
            if float(r.get("forget_gain", 0.0)) > max(0.0, float(min_forget_gain)) and
            float(r.get("protection_damage", 0.0)) <= damage_threshold + 1e-12
        ]
        fallback_used = False
        if len(eligible) < block_size:
            fallback_used = True
            eligible = [
                r for r in records
                if float(r.get("forget_gain", 0.0)) > max(0.0, float(min_forget_gain))
            ]
        if not eligible:
            return [], bool(fallback_used), float(damage_threshold)
        ordered = sorted(
            eligible,
            key=lambda r: (
                -float(r.get("forget_gain", 0.0)),
                float(r.get("protection_damage", 0.0)),
                str(r.get("rank_id")),
            ),
        )
        return [dict(r) for r in ordered[:block_size]], bool(fallback_used), float(damage_threshold)

    def _stage3_rank_is_active(self, layers: List[LoRALayerHandle], rank_id: str) -> bool:
        index = _lora_layer_index(layers)
        if rank_id not in index:
            return False
        layer, local_rank = index[rank_id]
        return float(layer.rank_mask[int(local_rank)].detach().float().cpu()) > 0.0

    def _stage3_resume_selected(
        self,
        layers: List[LoRALayerHandle],
        config_hash: str,
    ) -> Tuple[List[str], List[Dict], Dict[int, Dict]]:
        selected_ids = []
        selected_records = []
        completed_rounds = {}
        if not bool(getattr(self.args, "stage3_resume", True)):
            return selected_ids, selected_records, completed_rounds
        if bool(getattr(self.args, "stage3_force_recompute", False)):
            return selected_ids, selected_records, completed_rounds

        for round_index in range(1, 100):
            payload = self._load_stage3_cache(f"rank_exact_scores_round{round_index}.json", config_hash)
            if payload is None:
                continue
            completed_rounds[int(round_index)] = payload
            for record in payload.get("selected", []):
                rank_id = str(record.get("rank_id"))
                if rank_id and rank_id not in selected_ids:
                    selected_ids.append(rank_id)
                    selected_records.append(dict(record))

        payload = self._load_stage3_cache("selected_rank_masks.json", config_hash)
        if payload is not None:
            for record in payload.get("selected", []):
                rank_id = str(record.get("rank_id"))
                if rank_id and rank_id not in selected_ids:
                    selected_ids.append(rank_id)
                    selected_records.append(dict(record))
        apply_permanent_rank_masks(layers, selected_ids)
        if selected_ids:
            print(f"[Stage3][Resume] restored_selected_rank_masks={len(selected_ids)}")
        return selected_ids, selected_records, completed_rounds

    def _save_stage3_selected_masks(
        self,
        layers: List[LoRALayerHandle],
        selected_rank_ids: List[str],
        selected_records: List[Dict],
        config_hash: str,
    ):
        payload = {
            "config_hash": config_hash,
            "selected_rank_ids": [str(x) for x in selected_rank_ids],
            "selected": selected_records,
            "mask": get_current_rank_mask_state(layers),
            "selected_count": int(len(selected_rank_ids)),
        }
        self._save_stage3_cache("selected_rank_masks.json", payload)

    def _stage3_decisions_from_selected(
        self,
        layers: List[LoRALayerHandle],
        selected_rank_ids: List[str],
        selected_score_by_id: Dict[str, Dict],
        reason: str,
    ) -> List[Dict]:
        selected_set = set(str(x) for x in selected_rank_ids)
        decisions = []
        for layer in layers:
            for local_rank in range(int(layer.rank)):
                rank_id = lora_rank_key(layer, local_rank)
                selected = rank_id in selected_set
                score_record = dict(selected_score_by_id.get(rank_id, {}))
                decision = {
                    **score_record,
                    "rank_id": int(local_rank),
                    "rank_index": int(local_rank),
                    "module_name": layer.key,
                    "parameter_key": rank_id,
                    "route": "hard_prune" if selected else "protect",
                    "route_reason": (
                        score_record.get("route_reason", "coarse_to_fine_block_selected")
                        if selected else reason
                    ),
                    "rank_mask_value": 0.0 if selected else 1.0,
                    "pruned_by_rank_mask": bool(selected),
                }
                decisions.append(decision)
        return decisions

    def _coarse_pruning_summary(
        self,
        layers: List[LoRALayerHandle],
        decisions: List[Dict],
        rounds: List[Dict],
        reason: str,
        global_budget: int,
        max_prune_ratio: float,
        min_forget_gain: float,
        tau_info: Dict,
        block_schedule: List[int],
        full_before: Optional[Dict],
        full_after: Optional[Dict],
        config_hash: str,
        elapsed_seconds: float,
    ) -> Dict:
        total_ranks = int(sum(layer.rank for layer in layers))
        hard = int(sum(1 for d in decisions if d.get("route") == "hard_prune"))
        per_layer_actual = defaultdict(list)
        for decision in decisions:
            if decision.get("route") == "hard_prune":
                per_layer_actual[decision["module_name"]].append(int(decision["rank_id"]))
        selected_rank_ids = [
            f"{d['module_name']}:{int(d['rank_id'])}"
            for d in decisions
            if d.get("route") == "hard_prune"
        ]
        full_validation = {}
        if full_before is not None and full_after is not None:
            full_validation = {
                "baseline_forget_metric": float(full_before.get("forget_metric", 0.0)),
                "final_forget_metric": float(full_after.get("forget_metric", 0.0)),
                "forget_gain": float(full_before.get("forget_metric", 0.0) - full_after.get("forget_metric", 0.0)),
                "baseline_protection_metric": float(full_before.get("protection_metric", 0.0)),
                "final_protection_metric": float(full_after.get("protection_metric", 0.0)),
            }
        return {
            "status": "completed",
            "stage": "Stage 3: coarse-to-fine no-gradient LoRA rank pruning",
            "stage3_mode": "coarse_to_fine_pruning",
            "forget_method_type": "pruning_unlearning",
            "enabled": True,
            "reason": reason,
            "rank_mask_pruning": True,
            "greedy_pruning": False,
            "block_pruning": True,
            "uses_gradient": False,
            "uses_optimizer_step": False,
            "full_model_parameter_updates": False,
            "gradient_updates": False,
            "selection_rule": "protection-constrained coarse shortlist, then block mask ablation on calibration targets",
            "total_lora_layers": int(len(layers)),
            "total_ranks": total_ranks,
            "total_rank_count": total_ranks,
            "target_prune_count": int(global_budget),
            "actual_prune_count": int(hard),
            "rank_shortlist_size": int(getattr(self.args, "rank_shortlist_size", 64) or 64),
            "rank_reserve_size": int(getattr(self.args, "rank_reserve_size", 128) or 128),
            "forget_calibration_size": int(getattr(self.args, "forget_calibration_size", 128) or 128),
            "protection_calibration_size": int(getattr(self.args, "protection_calibration_size", 512) or 512),
            "protection_risk_quantile": float(getattr(self.args, "protection_risk_quantile", 0.70) or 0.70),
            "block_schedule": [int(x) for x in block_schedule],
            "full_forget_target_count": int(len(full_before.get("forget_scores", {}))) if full_before else 0,
            "full_protection_target_count": int(len(full_before.get("protection_scores", {}))) if full_before else 0,
            "selected_rank_ids": selected_rank_ids,
            "backup_path": self._stage3_backup_path(),
            "config_hash": config_hash,
            "per_layer_lora_rank_number": {layer.key: int(layer.rank) for layer in layers},
            "per_layer_actual_pruned_ranks": {
                key: sorted([int(x) for x in ranks]) for key, ranks in per_layer_actual.items()
            },
            "actual_pruned_ranks": [
                {"module_name": key, "rank_id": int(rank_id)}
                for key, ranks in per_layer_actual.items()
                for rank_id in sorted(ranks)
            ],
            "hard_prune": hard,
            "soft_suppress": 0,
            "protect": int(total_ranks - hard),
            "final_prune_ratio": float(hard) / float(total_ranks) if total_ranks else 0.0,
            "actual_intervention_ratio": float(hard) / float(total_ranks) if total_ranks else 0.0,
            "global_prune_budget": int(global_budget),
            "max_prune_ratio": float(max_prune_ratio),
            "min_forget_gain": float(min_forget_gain),
            "tau_prot": float(tau_info.get("tau_prot", 0.0)),
            "null_control_thresholds": tau_info,
            "rounds": rounds,
            "full_validation": full_validation,
            "residual_energy_before": float(full_validation.get("baseline_forget_metric", 0.0)),
            "residual_energy_after": float(full_validation.get("final_forget_metric", 0.0)),
            "residual_energy_drop": float(full_validation.get("forget_gain", 0.0)),
            "protection_perturbation_before": float(full_validation.get("baseline_protection_metric", 0.0)),
            "protection_perturbation_after": float(full_validation.get("final_protection_metric", 0.0)),
            "rank_mask_checksum": lora_rank_mask_checksums(layers),
            "elapsed_seconds": float(elapsed_seconds),
        }

    def compute_forget_gain(
        self,
        layer: LoRALayerHandle,
        rank_id: int,
        residual_targets: List[Dict],
        masked_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        before_loss = self._forget_distance_loss(residual_targets, score_overrides=None)
        if masked_scores is None:
            temporary_mask_rank(layer, rank_id)
            try:
                masked_scores = self._score_residual_targets(residual_targets)
            finally:
                restore_rank(layer, rank_id)
        after_loss = self._forget_distance_loss(residual_targets, score_overrides=masked_scores)
        return {
            "before_loss": float(before_loss),
            "after_loss": float(after_loss),
            "forget_gain": float(before_loss - after_loss),
        }

    def compute_retain_risk(
        self,
        layer: LoRALayerHandle,
        rank_id: int,
        residual_targets: List[Dict],
        masked_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        before_loss = self._retain_consistency_loss(residual_targets, score_overrides=None)
        if masked_scores is None:
            temporary_mask_rank(layer, rank_id)
            try:
                masked_scores = self._score_residual_targets(residual_targets)
            finally:
                restore_rank(layer, rank_id)
        after_loss = self._retain_consistency_loss(residual_targets, score_overrides=masked_scores)
        return {
            "before_loss": float(before_loss),
            "after_loss": float(after_loss),
            "retain_risk": float(after_loss - before_loss),
        }

    # ------------------------------------------------------------------
    # Scoring, losses, and summaries
    # ------------------------------------------------------------------

    @staticmethod
    def compute_raw_counterfactual_residual(score_original: float, score_counterfactual: float) -> float:
        """Project-local S_cf fallback: true-ranker score delta."""
        return float(score_original) - float(score_counterfactual)

    def aggregate_retain_support(self, contributions: List[float], mode: str) -> float:
        values = [max(0.0, float(v)) for v in contributions]
        if not values:
            return 0.0
        if mode == "max_positive_delta":
            return float(max(values))
        if mode == "sum_positive_delta":
            return float(sum(values))
        if mode != "mean_positive_delta":
            raise ValueError(f"Unsupported rp_cf_retain_aggregation={mode!r}")
        return float(np.mean(values))

    @staticmethod
    def _normalize_global_nonnegative(values: List[float]) -> List[float]:
        arr = np.asarray([max(0.0, float(v)) for v in values], dtype=np.float64)
        if arr.size == 0:
            return []
        positive = arr[arr > 0.0]
        if positive.size == 0:
            return [0.0 for _ in arr.tolist()]
        lo = float(np.min(positive))
        hi = float(np.max(positive))
        if hi <= lo + 1e-12:
            return [1.0 if float(v) > 0.0 else 0.0 for v in arr.tolist()]
        out = []
        for value in arr.tolist():
            if value <= 0.0:
                out.append(0.0)
            else:
                out.append(float((value - lo) / (hi - lo)))
        return out

    def _robust_distance_name(self) -> str:
        return str(getattr(self.args, "rp_cf_robust_distance", "smooth_l1") or "smooth_l1")

    def _robust_distance(self, value: float, reference: float) -> float:
        delta = float(value) - float(reference)
        name = self._robust_distance_name()
        if name == "l1":
            return float(abs(delta))
        if name == "l2":
            return float(delta * delta)
        if name != "smooth_l1":
            raise ValueError(f"Unsupported rp_cf_robust_distance={name!r}")
        beta = max(float(getattr(self.args, "rp_cf_robust_beta", 1.0) or 1.0), 1e-12)
        abs_delta = abs(delta)
        if abs_delta < beta:
            return float(0.5 * delta * delta / beta)
        return float(abs_delta - 0.5 * beta)

    def _boundary_top_k_for_gap(self, candidate_count: int) -> int:
        value = getattr(self.args, "boundary_top_k", None)
        if value is None:
            value = getattr(self.args, "topk_boundary", None)
        if value is None:
            value = max(getattr(self.args, "rerank_metric_ks", [10]))
        return max(1, min(int(value), int(candidate_count)))

    def _boundary_gap_definition(self) -> Dict:
        return {
            "formula": "score(candidate) - kth_score(candidate_pool)",
            "top_k_source": "boundary_top_k if set else topk_boundary if set else max(rerank_metric_ks)",
            "sign": "positive means candidate is above the selected Top-K boundary",
        }

    def _boundary_gap_from_scores(
        self,
        score_map: Dict[str, float],
        candidate_items: List[int],
        candidate_iid: int,
    ) -> float:
        candidates = [int(i) for i in candidate_items]
        if not candidates:
            return 0.0
        scores = [float(score_map.get(str(int(iid)), 0.0)) for iid in candidates]
        order = np.argsort(-np.asarray(scores)).tolist()
        k = self._boundary_top_k_for_gap(len(candidates))
        boundary_idx = order[k - 1]
        boundary_score = float(scores[boundary_idx])
        target_score = float(score_map.get(str(int(candidate_iid)), 0.0))
        return float(target_score - boundary_score)

    def _boundary_gap_from_rank_info(self, rank_info: Dict, candidate_items: List[int], candidate_iid: int) -> float:
        return self._boundary_gap_from_scores(
            rank_info.get("scores", {}),
            candidate_items,
            int(candidate_iid),
        )

    def _score_calibrated_records(self, records: List[Dict]) -> Dict[str, Dict[str, float]]:
        grouped = defaultdict(list)
        for record in records:
            key = (
                tuple(int(x) for x in record.get("history", [])),
                tuple(int(x) for x in record.get("candidate_items", [])),
                int(record.get("scoring_label_iid", record.get("forget_item_id", record.get("target_iid", 0)))),
            )
            grouped[key].append(record)

        scores_by_id: Dict[str, Dict[str, float]] = {}
        for (history, candidates, scoring_label_iid), group in grouped.items():
            if not candidates:
                continue
            scored = self._score_candidate_list(
                list(history),
                list(candidates),
                int(scoring_label_iid),
                grad=False,
            )
            score_map = scored.get("scores", {})
            for record in group:
                candidate_iid = int(record["candidate_item_id"])
                scores_by_id[record["record_id"]] = {
                    "score": float(score_map.get(str(candidate_iid), 0.0)),
                    "boundary_gap": self._boundary_gap_from_rank_info(
                        scored,
                        list(candidates),
                        candidate_iid,
                    ),
                }
        return scores_by_id

    def _weighted_residual_energy(self, records: List[Dict], scores_by_id: Dict[str, Dict[str, float]]) -> float:
        values = []
        weights = []
        for record in records:
            weight = float(record.get("W_unl", record.get("forget_weight", 0.0)))
            if weight <= 0.0:
                continue
            current = scores_by_id.get(record["record_id"], {})
            current_gap = float(current.get("boundary_gap", record.get("boundary_gap_original", 0.0)))
            cf_gap = float(record.get("boundary_gap_cf", 0.0))
            values.append(self._robust_distance(current_gap, cf_gap))
            weights.append(weight)
        return self._weighted_mean(values, weights)

    def _weighted_protection_perturbation(
        self,
        records: List[Dict],
        reference_scores: Dict[str, Dict[str, float]],
        current_scores: Dict[str, Dict[str, float]],
    ) -> float:
        values = []
        weights = []
        for record in records:
            weight = float(record.get("W_prot", record.get("retain_weight", 0.0)))
            if weight <= 0.0:
                continue
            ref = reference_scores.get(record["record_id"], {})
            cur = current_scores.get(record["record_id"], {})
            ref_gap = float(ref.get("boundary_gap", record.get("boundary_gap_original", 0.0)))
            cur_gap = float(cur.get("boundary_gap", ref_gap))
            values.append(self._robust_distance(cur_gap, ref_gap))
            weights.append(weight)
        return self._weighted_mean(values, weights)

    def _derive_null_control_thresholds(
        self,
        residual_targets: List[Dict],
        protection_targets: List[Dict],
    ) -> Dict:
        values = []
        for record in list(residual_targets) + list(protection_targets):
            for delta in record.get("null_control_deltas", []):
                values.append(self._robust_distance(float(delta), 0.0))
        q_prot = float(getattr(self.args, "rp_cf_protection_quantile", 0.95) or 0.95)
        q_stop = getattr(self.args, "rp_cf_residual_stop_quantile", None)
        if q_stop is None:
            q_stop = q_prot
        q_stop = float(q_stop)
        fallback = float(getattr(self.args, "retain_drop_tolerance", 0.05) or 0.05)
        tau_prot = _safe_quantile(values, q_prot, default=fallback)
        residual_stop = _safe_quantile(values, q_stop, default=fallback)
        if tau_prot <= 0.0:
            tau_prot = fallback
        return {
            "null_control_distribution_source": "same-user retain-removal score deltas from Stage 1",
            "null_control_value_count": int(len(values)),
            "null_control_stats": self._value_stats(values),
            "protection_quantile": float(q_prot),
            "residual_stop_quantile": float(q_stop),
            "tau_prot": float(tau_prot),
            "residual_stop_threshold": float(residual_stop),
            "fallback_used": bool(len(values) == 0),
            "fallback_value": float(fallback),
        }

    def _greedy_pruning_summary(
        self,
        decisions: List[Dict],
        layers: List[LoRALayerHandle],
        iterations: List[Dict],
        reason: str,
        global_budget: int,
        max_prune_ratio: float,
        min_forget_gain: float,
        tau_info: Dict,
        residual_before: float,
        residual_after: float,
        protection_after: float,
    ) -> Dict:
        total_ranks = int(sum(layer.rank for layer in layers))
        hard = int(sum(1 for d in decisions if d.get("route") == "hard_prune"))
        protect = int(total_ranks - hard)
        per_layer_actual = defaultdict(list)
        for decision in decisions:
            if decision.get("route") == "hard_prune":
                per_layer_actual[decision["module_name"]].append(int(decision["rank_id"]))
        return {
            "stage": "Stage 3: retention-first counterfactual-calibrated LoRA rank pruning",
            "enabled": True,
            "reason": reason,
            "rank_mask_pruning": True,
            "greedy_pruning": True,
            "full_model_parameter_updates": False,
            "gradient_updates": False,
            "importance_rule": "I_lk=G_lk if G_lk>0 and P_lk<=tau_prot else -inf",
            "selection_rule": "argmax feasible G_lk; accepted only after global residual/protection validation",
            "total_lora_layers": int(len(layers)),
            "total_ranks": total_ranks,
            "per_layer_lora_rank_number": {layer.key: int(layer.rank) for layer in layers},
            "per_layer_actual_pruned_ranks": {
                key: sorted([int(x) for x in ranks]) for key, ranks in per_layer_actual.items()
            },
            "actual_pruned_ranks": [
                {"module_name": key, "rank_id": int(rank_id)}
                for key, ranks in per_layer_actual.items()
                for rank_id in sorted(ranks)
            ],
            "hard_prune": hard,
            "soft_suppress": 0,
            "protect": protect,
            "final_prune_ratio": float(hard) / float(total_ranks) if total_ranks else 0.0,
            "actual_intervention_ratio": float(hard) / float(total_ranks) if total_ranks else 0.0,
            "global_prune_budget": int(global_budget),
            "max_prune_ratio": float(max_prune_ratio),
            "min_forget_gain": float(min_forget_gain),
            "tau_prot": float(tau_info.get("tau_prot", 0.0)),
            "residual_stop_threshold": float(tau_info.get("residual_stop_threshold", 0.0)),
            "null_control_thresholds": tau_info,
            "robust_distance": self._robust_distance_name(),
            "robust_distance_formula_status": (
                "no project rho definition found; using explicit configured rho"
            ),
            "boundary_gap_definition": self._boundary_gap_definition(),
            "residual_energy_before": float(residual_before),
            "residual_energy_after": float(residual_after),
            "residual_energy_drop": float(residual_before - residual_after),
            "protection_perturbation_before": 0.0,
            "protection_perturbation_after": float(protection_after),
            "iterations": iterations,
            "top_pruned_rank_scores": [
                {
                    "module_name": d["module_name"],
                    "rank_id": int(d["rank_id"]),
                    "G_lk": float(d.get("G_lk", 0.0)),
                    "P_lk": float(d.get("P_lk", 0.0)),
                    "I_lk": float(d.get("G_lk", 0.0)),
                    "accepted_iteration": int(d.get("accepted_iteration", 0) or 0),
                }
                for d in decisions
                if d.get("route") == "hard_prune"
            ][:20],
            "rank_mask_checksum": lora_rank_mask_checksums(layers),
        }

    def _score_residual_targets(self, residual_targets: List[Dict]) -> Dict[str, float]:
        grouped = defaultdict(list)
        for record in residual_targets:
            key = (
                tuple(int(x) for x in record.get("history", [])),
                tuple(int(x) for x in record.get("candidate_items", [])),
                int(record.get("scoring_label_iid", record.get("forget_item_id"))),
            )
            grouped[key].append(record)

        scores_by_id: Dict[str, float] = {}
        for (history, candidates, scoring_label_iid), records in grouped.items():
            if not candidates:
                continue
            scored = self._score_candidate_list(
                list(history),
                list(candidates),
                int(scoring_label_iid),
                grad=False,
            )
            for record in records:
                item_key = str(int(record["candidate_item_id"]))
                scores_by_id[record["record_id"]] = float(scored["scores"].get(item_key, 0.0))
        return scores_by_id

    def _forget_distance_loss(
        self,
        records: List[Dict],
        score_overrides: Optional[Dict[str, float]],
    ) -> float:
        values = []
        weights = []
        for record in records:
            current = self._record_current_score(record, score_overrides)
            cf_score = float(record.get("score_cf_forget", record.get("score_counterfactual", 0.0)))
            weight = max(float(record.get("forget_weight", record.get("normalized_residual", 0.0))), 1e-8)
            values.append((current - cf_score) ** 2)
            weights.append(weight)
        return self._weighted_mean(values, weights)

    def _retain_consistency_loss(
        self,
        records: List[Dict],
        score_overrides: Optional[Dict[str, float]],
    ) -> float:
        values = []
        weights = []
        for record in records:
            weight = float(record.get("retain_weight", 0.0))
            if weight <= 0.0:
                continue
            current = self._record_current_score(record, score_overrides)
            original = float(record.get("score_original", 0.0))
            values.append((current - original) ** 2)
            weights.append(weight)
        return self._weighted_mean(values, weights)

    @staticmethod
    def _record_current_score(record: Dict, score_overrides: Optional[Dict[str, float]]) -> float:
        if score_overrides and record.get("record_id") in score_overrides:
            return float(score_overrides[record["record_id"]])
        return float(record.get("score_original", 0.0))

    @staticmethod
    def _weighted_mean(values, weights) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        denom = float(np.sum(w))
        if denom <= 1e-12:
            return float(np.mean(arr))
        return float(np.sum(arr * w) / denom)

    def _forget_residual_summary(
        self,
        residual_targets: List[Dict],
        score_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict:
        residuals = []
        weighted_abs = []
        weights = []
        for record in residual_targets:
            current = self._record_current_score(record, score_overrides)
            cf_score = float(record.get("score_cf_forget", record.get("score_counterfactual", 0.0)))
            residual = float(current - cf_score)
            residuals.append(residual)
            weight = max(float(record.get("forget_weight", 0.0)), 1e-8)
            weights.append(weight)
            weighted_abs.append(abs(residual))
        return {
            "num_targets": int(len(residual_targets)),
            "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
            "mean_abs_residual": float(np.mean(np.abs(residuals))) if residuals else 0.0,
            "weighted_mean_abs_residual": self._weighted_mean(weighted_abs, weights),
        }

    def _save_rank_mask(self, pruning: Dict):
        mask = {}
        for decision in pruning.get("decisions", []):
            key = f"{decision['module_name']}:{int(decision['rank_id'])}"
            mask[key] = 0 if decision.get("route") == "hard_prune" else 1
        torch.save(mask, os.path.join(self.output_dir, "prune_mask.pt"))

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _freeze_all_parameters(self):
        trainable_before = 0
        total_params = 0
        for param in self.model.parameters():
            total_params += int(param.numel())
            trainable_before += int(param.numel()) if param.requires_grad else 0
            param.requires_grad_(False)
        self.logs["trainable_parameters_before_freeze"] = int(trainable_before)
        self.logs["total_parameters_seen"] = int(total_params)
        self.logs["parameters_frozen_for_mask_pruning"] = True

    def _residual_top_m(self) -> int:
        value = getattr(self.args, "rp_cf_top_m", None)
        if value is None:
            value = getattr(self.args, "residual_top_m", 64)
        return max(0, int(value or 0))

    def _rank_probe_targets(self, residual_targets: List[Dict]) -> List[Dict]:
        limit = int(getattr(self.args, "probe_top_m", 64) or 64)
        ordered = sorted(
            residual_targets,
            key=lambda r: float(r.get("residual_z", 0.0)),
            reverse=True,
        )
        if limit > 0:
            ordered = ordered[:limit]
        self.logs["rank_probe_target_count"] = int(len(ordered))
        return ordered

    def _sample_retain_interactions(
        self,
        uid: int,
        history: List[int],
        forget_iid: int,
        requested: int,
        salt: str,
    ) -> List[int]:
        requested = max(0, int(requested or 0))
        if requested <= 0:
            return []
        retain_history = remove_forget_item_from_history(history, forget_iid)
        retain_train = self.dataset_data.get("retain_train", self.dataset_data.get("train", {})).get(int(uid), [])
        retain_train_set = {int(x) for x in retain_train if int(x) != int(forget_iid)}
        pool = []
        seen = set()
        for iid in retain_history:
            iid = int(iid)
            if iid == int(forget_iid) or iid in seen:
                continue
            if retain_train_set and iid not in retain_train_set:
                continue
            pool.append(iid)
            seen.add(iid)
        if not pool:
            return []
        seed_payload = f"{getattr(self.args, 'seed', 42)}|{salt}|{int(uid)}|{int(forget_iid)}"
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        if len(pool) <= requested:
            return [int(x) for x in pool]
        chosen = rng.choice(pool, size=requested, replace=False)
        return [int(x) for x in chosen.tolist()]

    @staticmethod
    def _normalize_user_values(values: List[float]) -> List[float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return []
        arr = np.maximum(arr, 0.0)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi <= lo + 1e-12:
            return [1.0 if hi > 0.0 else 0.0 for _ in arr.tolist()]
        return [float(v) for v in ((arr - lo) / (hi - lo)).tolist()]

    @staticmethod
    def _value_stats(values) -> Dict:
        arr = np.asarray([float(v) for v in values], dtype=np.float64)
        if arr.size == 0:
            return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    @staticmethod
    def _top_examples(records: List[Dict], limit: int = 5) -> List[Dict]:
        examples = []
        for record in records[:limit]:
            examples.append({
                "uid": int(record.get("uid", record.get("user_id", 0))),
                "forget_iid": int(record.get("forget_iid", record.get("forget_item_id", 0))),
                "candidate_iid": int(record.get("candidate_iid", record.get("candidate_item_id", 0))),
                "residual_z": float(record.get("residual_z", 0.0)),
                "residual_raw": float(record.get("residual_raw", record.get("residual_score", 0.0))),
                "retain_support": float(record.get("retain_support", 0.0)),
                "forget_weight": float(record.get("forget_weight", 0.0)),
                "retain_weight": float(record.get("retain_weight", 0.0)),
            })
        return examples
