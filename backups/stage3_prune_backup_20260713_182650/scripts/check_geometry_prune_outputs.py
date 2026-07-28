#!/usr/bin/env python3
"""Sanity check geometry_prune output artifacts.

This script is intentionally read-only. It validates that a geometry_prune run
produced the common interaction-level unlearning artifacts, that pruning made
non-empty decisions, and that before/after predictions changed.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REQUIRED_FILES = [
    "residual_candidates.json",
    "logit_correction.json",
    "path_loss_summary.json",
    "retain_protection_summary.json",
    "parameter_importance.json",
    "pruning_decisions.json",
    "pruning_summary.json",
    "method_logs.json",
    "predictions_before.json",
    "predictions_after.json",
    "metrics_unlearning.json",
    "rollback_log.json",
]


def _load_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    if not path.exists():
        return None, "missing"
    if path.stat().st_size == 0:
        return None, "empty"
    try:
        with path.open("r") as f:
            return json.load(f), None
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"
    except OSError as exc:
        return None, f"read error: {exc}"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
        decisions = payload.get("decisions")
        if isinstance(decisions, list):
            return [r for r in decisions if isinstance(r, dict)]
    return []


def _metric_value(metrics: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _drop_values(drop_payload: Any) -> List[float]:
    if drop_payload is None:
        return []
    if isinstance(drop_payload, (int, float)):
        value = _as_float(drop_payload)
        return [] if value is None else [value]
    if not isinstance(drop_payload, dict):
        return []

    if isinstance(drop_payload.get("drop"), dict):
        source = drop_payload["drop"]
    else:
        source = drop_payload

    values = []
    for value in source.values():
        numeric = _as_float(value)
        if numeric is not None:
            values.append(numeric)
    return values


def _extract_exposure(metrics: Dict[str, Any], after: bool) -> Any:
    if after:
        return _metric_value(
            metrics,
            "forget_item_residual_exposure_after",
            "forget_item_residual_exposure",
        )
    return _metric_value(metrics, "forget_item_residual_exposure_before")


def _exposure_declined(before: Any, after: Any) -> Optional[bool]:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        before_value = _as_float(before)
        after_value = _as_float(after)
        if before_value is None or after_value is None:
            return None
        return after_value < before_value

    if not isinstance(before, dict) or not isinstance(after, dict):
        return None

    comparable = []
    for key, before_value in before.items():
        after_value = after.get(key)
        before_float = _as_float(before_value)
        after_float = _as_float(after_value)
        if before_float is None or after_float is None:
            continue
        comparable.append(after_float < before_float)

    if not comparable:
        return None
    return any(comparable)


def _pruning_summary(pruning: Any) -> Dict[str, Any]:
    if not isinstance(pruning, dict):
        return {
            "decisions": [],
            "summary": {},
            "total_ranks": 0,
            "hard_prune": 0,
            "soft_suppress": 0,
            "protect": 0,
            "hard_prune_ratio": None,
            "soft_suppress_ratio": None,
            "protect_ratio": None,
            "max_prune_ratio": None,
            "intervened": 0,
            "intervention_ratio": None,
            "max_prune_ratio_used_fraction": None,
            "exceeds_max_prune_ratio": None,
            "avg_rank_unlearn_score": None,
            "avg_forget_influence_score": None,
            "avg_residual_boundary_score": None,
            "avg_retain_protection_score": None,
            "avg_semantic_protection_score": None,
            "global_scores": {},
        }

    decisions = _records(pruning)
    summary = pruning.get("summary") if isinstance(pruning.get("summary"), dict) else {}

    route_counts = {"hard_prune": 0, "soft_suppress": 0, "protect": 0}
    for decision in decisions:
        route = decision.get("route")
        if route in route_counts:
            route_counts[route] += 1

    hard_prune = int(summary.get("hard_prune", route_counts["hard_prune"]) or 0)
    soft_suppress = int(summary.get("soft_suppress", route_counts["soft_suppress"]) or 0)
    protect = int(summary.get("protect", route_counts["protect"]) or 0)
    total_ranks = int(summary.get("total_ranks", len(decisions)) or 0)
    if total_ranks <= 0 and decisions:
        total_ranks = len(decisions)

    max_prune_ratio = _as_float(summary.get("max_prune_ratio"))
    intervened = hard_prune + soft_suppress
    intervention_ratio = (
        float(intervened) / float(total_ranks)
        if total_ranks > 0 else None
    )
    hard_prune_ratio = float(hard_prune) / float(total_ranks) if total_ranks else None
    soft_suppress_ratio = float(soft_suppress) / float(total_ranks) if total_ranks else None
    protect_ratio = float(protect) / float(total_ranks) if total_ranks else None
    used_fraction = None
    if max_prune_ratio is not None and max_prune_ratio > 0 and intervention_ratio is not None:
        used_fraction = intervention_ratio / max_prune_ratio
    exceeds = None
    if max_prune_ratio is not None and intervention_ratio is not None:
        exceeds = intervention_ratio > max_prune_ratio + 1e-9

    def mean_field(field: str) -> Optional[float]:
        values = []
        for decision in decisions:
            value = _as_float(decision.get(field))
            if value is not None:
                values.append(value)
        return (sum(values) / len(values)) if values else None

    return {
        "decisions": decisions,
        "summary": summary,
        "total_ranks": total_ranks,
        "hard_prune": hard_prune,
        "soft_suppress": soft_suppress,
        "protect": protect,
        "hard_prune_ratio": hard_prune_ratio,
        "soft_suppress_ratio": soft_suppress_ratio,
        "protect_ratio": protect_ratio,
        "max_prune_ratio": max_prune_ratio,
        "intervened": intervened,
        "intervention_ratio": intervention_ratio,
        "max_prune_ratio_used_fraction": used_fraction,
        "exceeds_max_prune_ratio": exceeds,
        "avg_rank_unlearn_score": mean_field("rank_unlearn_score"),
        "avg_forget_influence_score": mean_field("forget_influence_score"),
        "avg_residual_boundary_score": mean_field("residual_boundary_score"),
        "avg_retain_protection_score": mean_field("retain_protection_score"),
        "avg_semantic_protection_score": mean_field("semantic_protection_score"),
        "global_scores": summary.get("global_scores", {}),
    }


def _prediction_id(record: Dict[str, Any], index: int) -> str:
    prediction_id = record.get("prediction_id")
    if prediction_id is not None:
        return str(prediction_id)
    uid = record.get("uid")
    iid = record.get("target_iid")
    tag = record.get("split_tag")
    position = record.get("position")
    return f"fallback:{tag}:{uid}:{iid}:{position}:{index}"


def _topk_changed(before: Any, after: Any) -> bool:
    if before is None and after is None:
        return False
    return list(before or []) != list(after or [])


def _value_changed(before: Any, after: Any, tolerance: float = 1e-12) -> bool:
    before_float = _as_float(before)
    after_float = _as_float(after)
    if before_float is None or after_float is None:
        return before != after
    return abs(before_float - after_float) > tolerance


def _prediction_changes(before: Any, after: Any) -> Dict[str, Any]:
    before_records = _records(before)
    after_records = _records(after)
    before_by_id = {
        _prediction_id(record, idx): record
        for idx, record in enumerate(before_records)
    }

    paired = 0
    score_changed = 0
    rank_changed = 0
    topk_changed = 0
    changed_records = 0

    for idx, after_record in enumerate(after_records):
        record_id = _prediction_id(after_record, idx)
        before_record = before_by_id.get(record_id)
        if before_record is None:
            continue
        paired += 1
        score_diff = _value_changed(
            before_record.get("target_score"),
            after_record.get("target_score"),
        )
        rank_diff = _value_changed(
            before_record.get("target_rank"),
            after_record.get("target_rank"),
            tolerance=0.0,
        )
        topk_diff = _topk_changed(
            before_record.get("topk_items"),
            after_record.get("topk_items"),
        )
        score_changed += int(score_diff)
        rank_changed += int(rank_diff)
        topk_changed += int(topk_diff)
        changed_records += int(score_diff or rank_diff or topk_diff)

    changed_ratio = float(changed_records) / float(paired) if paired else 0.0
    return {
        "before_records": len(before_records),
        "after_records": len(after_records),
        "paired_records": paired,
        "score_changed": score_changed,
        "rank_changed": rank_changed,
        "topk_changed": topk_changed,
        "changed_records": changed_records,
        "changed_records_ratio": changed_ratio,
    }


def _probe_summary(probe_payload: Any) -> Dict[str, Any]:
    records = _records(probe_payload)
    return {
        "num_probe_ranks": len(records),
        "num_helpful_ranks": sum(1 for r in records if r.get("direction") == "helpful"),
        "num_harmful_ranks": sum(1 for r in records if r.get("direction") == "harmful"),
        "num_neutral_ranks": sum(1 for r in records if r.get("direction") == "neutral"),
    }


def _importance_summary(payload: Any) -> Dict[str, Any]:
    records = _records(payload)
    positive_final = [
        r for r in records
        if (_as_float(r.get("final_importance")) or 0.0) > 0.0
    ]
    positive_forget = [
        r for r in records
        if (_as_float(r.get("forget_path_importance")) or 0.0) > 0.0
    ]
    return {
        "num_ranks": len(records),
        "positive_final_importance": len(positive_final),
        "positive_forget_path_importance": len(positive_forget),
    }


def _mean_prediction_delta(before: Any, after: Any, split_tag: str, field: str) -> Optional[float]:
    before_records = [
        r for r in _records(before) if r.get("split_tag") == split_tag
    ]
    after_by_id = {
        _prediction_id(r, idx): r
        for idx, r in enumerate(_records(after))
        if r.get("split_tag") == split_tag
    }
    deltas = []
    for idx, before_record in enumerate(before_records):
        after_record = after_by_id.get(_prediction_id(before_record, idx))
        if not after_record:
            continue
        before_value = _as_float(before_record.get(field))
        after_value = _as_float(after_record.get(field))
        if before_value is None or after_value is None:
            continue
        deltas.append(after_value - before_value)
    return sum(deltas) / len(deltas) if deltas else None


def _format_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _format_float(value: Any) -> str:
    number = _as_float(value)
    return "null" if number is None else f"{number:.6f}"


def _print_list(header: str, items: Iterable[str]) -> None:
    print(header)
    for item in items:
        print(f"  - {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check geometry_prune output artifacts for sanity.",
    )
    parser.add_argument(
        "--method_dir",
        required=True,
        help="Path to a geometry_prune output directory.",
    )
    parser.add_argument(
        "--retain_drop_warning_threshold",
        type=float,
        default=0.05,
        help="Warn when the largest positive retain drop exceeds this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    method_dir = Path(args.method_dir)
    failures: List[str] = []
    warnings: List[str] = []

    loaded: Dict[str, Any] = {}
    file_errors = []
    for filename in REQUIRED_FILES:
        path = method_dir / filename
        payload, error = _load_json(path)
        if error:
            file_errors.append(f"{filename}: {error}")
        else:
            loaded[filename] = payload

    if file_errors:
        failures.extend(file_errors)

    pruning_info = _pruning_summary(loaded.get("pruning_decisions.json", {}))
    if not pruning_info["decisions"]:
        if pruning_info["summary"]:
            warnings.append("pruning_decisions has no per-rank decisions; summary-only pruning result")
        else:
            failures.append("pruning_decisions is empty")
    if pruning_info["exceeds_max_prune_ratio"] is True:
        failures.append(
            "intervened rank ratio exceeds max_prune_ratio "
            f"({_format_float(pruning_info['intervention_ratio'])} > "
            f"{_format_float(pruning_info['max_prune_ratio'])})"
        )
    if pruning_info["decisions"] and pruning_info["hard_prune"] == 0:
        warnings.append("hard_prune count is 0; forgetting may rely only on soft suppression")
    if (
        pruning_info["summary"].get("reason") == "no_positive_importance_skip_hard_prune" and
        pruning_info["hard_prune"] > 0
    ):
        failures.append("hard_prune is non-zero even though pruning summary reports no positive importance")
    if (
        pruning_info["protect_ratio"] is not None and
        pruning_info["protect_ratio"] > 0.9
    ):
        warnings.append(
            "protect route dominates pruning decisions "
            f"(protect_ratio={pruning_info['protect_ratio']:.6f})"
        )
    if (
        pruning_info["max_prune_ratio_used_fraction"] is not None and
        pruning_info["max_prune_ratio_used_fraction"] < 0.8
    ):
        warnings.append(
            "max_prune_ratio is not close to being used "
            f"(used_fraction={pruning_info['max_prune_ratio_used_fraction']:.6f})"
        )
    global_scores = pruning_info.get("global_scores", {})
    if (
        isinstance(global_scores, dict) and
        global_scores.get("has_strong_protection") is True and
        pruning_info["intervened"] == 0
    ):
        warnings.append(
            "strong protection is present and no ranks were intervened; "
            "check whether protection thresholds are too aggressive"
        )

    prediction_info = _prediction_changes(
        loaded.get("predictions_before.json", {}),
        loaded.get("predictions_after.json", {}),
    )
    target_score_delta = _mean_prediction_delta(
        loaded.get("predictions_before.json", {}),
        loaded.get("predictions_after.json", {}),
        "forget",
        "target_score",
    )
    target_rank_delta = _mean_prediction_delta(
        loaded.get("predictions_before.json", {}),
        loaded.get("predictions_after.json", {}),
        "forget",
        "target_rank",
    )
    margin_delta = _mean_prediction_delta(
        loaded.get("predictions_before.json", {}),
        loaded.get("predictions_after.json", {}),
        "forget",
        "margin_to_topk_boundary",
    )
    if prediction_info["paired_records"] <= 0:
        failures.append("predictions_before and predictions_after have no paired records")
    elif prediction_info["changed_records_ratio"] <= 0.0:
        warnings.append("predictions_before and predictions_after are completely unchanged")
    if target_score_delta is not None and target_score_delta > 0:
        warnings.append("forget target_score increased after pruning")
        if pruning_info["decisions"]:
            warnings.append("pruning_decisions non-empty but forget_score increased; direction attribution failed")

    residual_info = _records(loaded.get("residual_candidates.json", {}))
    if not residual_info:
        warnings.append("residual_candidates.json has no records")

    importance_info = _importance_summary(loaded.get("parameter_importance.json", {}))
    if importance_info["num_ranks"] <= 0:
        failures.append("parameter_importance.json has no rank records")

    path_summary = loaded.get("path_loss_summary.json", {})
    if not isinstance(path_summary, dict) or not path_summary:
        failures.append("path_loss_summary.json is empty or not an object")
        path_summary = {}
    retain_summary = loaded.get("retain_protection_summary.json", {})
    if not isinstance(retain_summary, dict) or not retain_summary:
        failures.append("retain_protection_summary.json is empty or not an object")
        retain_summary = {}

    rollback_log = loaded.get("rollback_log.json", {})
    if not isinstance(rollback_log, dict) or not rollback_log:
        failures.append("rollback_log.json is empty or not an object")
        rollback_log = {}

    metrics = loaded.get("metrics_unlearning.json", {})
    if not isinstance(metrics, dict):
        metrics = {}
        failures.append("metrics_unlearning.json is not a JSON object")

    exposure_before = _extract_exposure(metrics, after=False)
    exposure_after = _extract_exposure(metrics, after=True)
    exposure_declined = _exposure_declined(exposure_before, exposure_after)
    if exposure_declined is False:
        warnings.append("forget exposure did not decline")
    elif exposure_declined is None:
        warnings.append("forget exposure before/after could not be compared")

    retain_drop = _metric_value(metrics, "retain_utility_drop")
    retain_drop_values = _drop_values(retain_drop)
    max_retain_drop = max(retain_drop_values) if retain_drop_values else None
    if max_retain_drop is not None and max_retain_drop > args.retain_drop_warning_threshold:
        warnings.append(
            "retain_utility_drop is large "
            f"(max positive drop={max_retain_drop:.6f}, "
            f"threshold={args.retain_drop_warning_threshold:.6f})"
        )

    method_logs = loaded.get("method_logs.json", {})
    if isinstance(method_logs, dict):
        if method_logs.get("is_effective_unlearning_baseline") is False:
            failures.append("method_logs marks is_effective_unlearning_baseline=false")
    else:
        failures.append("method_logs.json is not a JSON object")

    status = "FAIL" if failures else ("WARNING" if warnings else "PASS")

    print("Geometry prune output sanity check")
    print(f"method_dir: {method_dir}")
    print(f"Required files: {'FAIL' if file_errors else 'PASS'}")
    if file_errors:
        _print_list("File issues:", file_errors)

    print("")
    print("Pruning decisions:")
    print(f"  total_ranks: {pruning_info['total_ranks']}")
    print(f"  hard_prune: {pruning_info['hard_prune']}")
    print(f"  soft_suppress: {pruning_info['soft_suppress']}")
    print(f"  protect: {pruning_info['protect']}")
    print(f"  hard_prune_ratio: {_format_float(pruning_info['hard_prune_ratio'])}")
    print(f"  soft_suppress_ratio: {_format_float(pruning_info['soft_suppress_ratio'])}")
    print(f"  protect_ratio: {_format_float(pruning_info['protect_ratio'])}")
    print(f"  intervened_ranks: {pruning_info['intervened']}")
    print(f"  max_prune_ratio: {_format_float(pruning_info['max_prune_ratio'])}")
    print(f"  intervention_ratio: {_format_float(pruning_info['intervention_ratio'])}")
    print(
        "  max_prune_ratio_used_fraction: "
        f"{_format_float(pruning_info['max_prune_ratio_used_fraction'])}"
    )
    print(f"  exceeds_max_prune_ratio: {pruning_info['exceeds_max_prune_ratio']}")
    print(f"  avg_rank_unlearn_score: {_format_float(pruning_info['avg_rank_unlearn_score'])}")
    print(
        "  avg_forget_influence_score: "
        f"{_format_float(pruning_info['avg_forget_influence_score'])}"
    )
    print(
        "  avg_residual_boundary_score: "
        f"{_format_float(pruning_info['avg_residual_boundary_score'])}"
    )
    print(
        "  avg_retain_protection_score: "
        f"{_format_float(pruning_info['avg_retain_protection_score'])}"
    )
    print(
        "  avg_semantic_protection_score: "
        f"{_format_float(pruning_info['avg_semantic_protection_score'])}"
    )
    print(f"  global_scores: {_format_json(pruning_info['global_scores'])}")

    print("")
    print("Prediction changes:")
    print(f"  before_records: {prediction_info['before_records']}")
    print(f"  after_records: {prediction_info['after_records']}")
    print(f"  paired_records: {prediction_info['paired_records']}")
    print(f"  target_score_changed: {prediction_info['score_changed']}")
    print(f"  target_rank_changed: {prediction_info['rank_changed']}")
    print(f"  topk_items_changed: {prediction_info['topk_changed']}")
    print(f"  changed_records_ratio: {prediction_info['changed_records_ratio']:.6f}")
    print(f"  forget_target_score_delta_mean: {_format_float(target_score_delta)}")
    print(f"  forget_target_rank_delta_mean: {_format_float(target_rank_delta)}")
    print(f"  forget_margin_delta_mean: {_format_float(margin_delta)}")

    print("")
    print("Importance:")
    print(f"  residual_candidates: {len(residual_info)}")
    print(f"  parameter_ranks: {importance_info['num_ranks']}")
    print(f"  positive_final_importance: {importance_info['positive_final_importance']}")
    print(f"  positive_forget_path_importance: {importance_info['positive_forget_path_importance']}")
    print(f"  path_loss_num_candidates: {path_summary.get('num_candidates')}")
    print(f"  path_loss: {_format_float(path_summary.get('loss'))}")
    print(f"  retain_protection_num_samples: {retain_summary.get('num_samples')}")
    print(f"  retain_protection_loss: {_format_float(retain_summary.get('loss'))}")

    print("")
    print("Rollback:")
    print(f"  enabled: {rollback_log.get('enabled')}")
    print(f"  rollback_applied: {rollback_log.get('rollback_applied')}")
    print(f"  retain_drop: {_format_float(rollback_log.get('retain_drop'))}")
    print(f"  retain_drop_tolerance: {_format_float(rollback_log.get('retain_drop_tolerance'))}")
    print(f"  num_pruned_before_rollback: {rollback_log.get('num_pruned_before_rollback')}")

    print("")
    print("Metrics:")
    print(f"  forget_item_residual_exposure_before: {_format_json(exposure_before)}")
    print(f"  forget_item_residual_exposure_after: {_format_json(exposure_after)}")
    print(
        "  forget_item_rank_delta: "
        f"{_format_json(_metric_value(metrics, 'forget_item_rank_delta'))}"
    )
    print(f"  retain_utility_drop: {_format_json(retain_drop)}")
    print(
        "  overlap_retain_protection_drop: "
        f"{_format_json(_metric_value(metrics, 'overlap_retain_protection_drop'))}"
    )
    print(
        "  marginal_residual_margin_delta: "
        f"{_format_json(_metric_value(metrics, 'marginal_residual_margin_delta'))}"
    )

    print("")
    print(f"Status: {status}")
    if failures:
        _print_list("Failures:", failures)
    if warnings:
        _print_list("Warnings:", warnings)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
