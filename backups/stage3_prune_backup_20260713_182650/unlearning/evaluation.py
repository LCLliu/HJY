import hashlib
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from dataloader.llm import seq_to_token_ids
from trainer.verb import ManualVerbalizer


def _save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _row_uid(row: Dict) -> int:
    return int(row.get("uid", row.get("user_id")))


def _row_iid(row: Dict) -> int:
    return int(row.get("iid", row.get("item_id")))


def _row_position(row: Dict):
    pos = row.get("position", row.get("sequence_index"))
    if pos is None or pos == "" or pos == "null":
        return None
    return int(pos)


def _stable_int(*parts) -> int:
    payload = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _limit_records(records: List[Dict], max_eval_samples: int) -> List[Dict]:
    if max_eval_samples and max_eval_samples > 0:
        return records[:max_eval_samples]
    return records


def _forget_positions(split_data: Dict) -> Dict[int, set]:
    positions = {}
    for row in split_data.get("forget_interactions", []):
        uid = _row_uid(row)
        pos = _row_position(row)
        if pos is not None:
            positions.setdefault(uid, set()).add(pos)
    return positions


def _context_items_for_record(
    row: Dict,
    split_tag: str,
    dataset_data: Dict,
    split_data: Dict,
    max_history: int,
) -> List[int]:
    uid = _row_uid(row)
    if split_tag == "test":
        context = dataset_data.get("retain_train", {}).get(uid, []) + dataset_data.get("val", {}).get(uid, [])
        return context[-max_history:]

    train_seq = dataset_data.get("train", {}).get(uid, [])
    pos = _row_position(row)
    forget_pos = _forget_positions(split_data).get(uid, set())

    if pos is None or not train_seq:
        context = dataset_data.get("retain_train", {}).get(uid, [])
    else:
        context = [
            iid for idx, iid in enumerate(train_seq[:pos])
            if idx not in forget_pos
        ]
    return context[-max_history:]


def _candidate_items(args, uid: int, target_iid: int, context_items: List[int], split_tag: str, num_items: int):
    candidate_size = args.llm_negative_sample_size + 1
    rng = np.random.RandomState(_stable_int(args.seed, uid, target_iid, split_tag))
    candidates = [int(target_iid)]
    blocked = set(context_items) | {int(target_iid), 0}
    while len(candidates) < candidate_size:
        iid = int(rng.randint(1, num_items + 1))
        if iid not in blocked and iid not in candidates:
            candidates.append(iid)
    rng.shuffle(candidates)
    return candidates


def _build_prediction_requests(split_data: Dict, dataset_data: Dict, args) -> List[Dict]:
    max_eval_samples = getattr(args, "max_eval_samples", 0)
    requests = []

    split_groups = [
        ("forget", split_data.get("forget_interactions", [])),
        ("retain", split_data.get("retain_interactions", [])),
        ("overlap", split_data.get("overlap_retain_interactions", [])),
        ("semantic_neighbor", split_data.get("semantic_neighbor_retain", [])),
        ("collaborative_neighbor", split_data.get("collaborative_neighbor_retain", [])),
    ]

    for split_tag, rows in split_groups:
        for local_idx, row in enumerate(_limit_records(rows, max_eval_samples)):
            uid = _row_uid(row)
            target_iid = _row_iid(row)
            pos = _row_position(row)
            requests.append({
                "prediction_id": f"{split_tag}:{uid}:{target_iid}:{pos}:{local_idx}",
                "source_row": row,
                "uid": uid,
                "target_iid": target_iid,
                "position": pos,
                "split_tag": split_tag,
            })

    test_rows = []
    for uid in sorted(dataset_data.get("test", {}).keys()):
        test_items = dataset_data["test"].get(uid, [])
        if test_items:
            test_rows.append({
                "uid": int(uid),
                "iid": int(test_items[0]),
                "user_id": int(uid),
                "item_id": int(test_items[0]),
                "position": None,
                "split_name": "test",
            })

    for local_idx, row in enumerate(_limit_records(test_rows, max_eval_samples)):
        uid = _row_uid(row)
        target_iid = _row_iid(row)
        requests.append({
            "prediction_id": f"test:{uid}:{target_iid}:none:{local_idx}",
            "source_row": row,
            "uid": uid,
            "target_iid": target_iid,
            "position": None,
            "split_tag": "test",
        })

    return requests


def _verbalizer(args, tokenizer):
    return ManualVerbalizer(
        tokenizer=tokenizer,
        prefix="",
        post_log_softmax=False,
        classes=list(range(args.llm_negative_sample_size + 1)),
        label_words={
            i: chr(ord("A") + i)
            for i in range(args.llm_negative_sample_size + 1)
        },
    )


def _rank_record(scores: List[float], candidate_items: List[int], target_iid: int, metric_ks: List[int]) -> Dict:
    order = np.argsort(-np.array(scores)).tolist()
    ranked_items = [int(candidate_items[idx]) for idx in order]
    ranks = {str(int(candidate_items[idx])): rank + 1 for rank, idx in enumerate(order)}
    score_map = {
        str(int(iid)): float(score)
        for iid, score in zip(candidate_items, scores)
    }
    target_rank = int(ranks[str(int(target_iid))])
    target_score = float(score_map[str(int(target_iid))])
    max_k = min(max(metric_ks), len(ranked_items))
    boundary_item = ranked_items[max_k - 1]
    boundary_score = float(score_map[str(boundary_item)])
    margin = float(target_score - boundary_score)

    return {
        "scores": score_map,
        "ranks": ranks,
        "topk_items": ranked_items[:max_k],
        "target_rank": target_rank,
        "target_score": target_score,
        "topk_boundary_score": boundary_score,
        "margin_to_topk_boundary": margin,
    }


def _prediction_key(req: Dict) -> str:
    position = req.get("position")
    position_key = "none" if position is None else str(int(position))
    return "|".join([
        str(req["split_tag"]),
        str(int(req["uid"])),
        str(int(req["target_iid"])),
        position_key,
    ])


def _prediction_cache_path(args, stage: str) -> str:
    dtype_tag = "4bit" if bool(getattr(args, "llm_load_in_4bit", True)) else "fp16"
    filename = (
        f"predictions_{stage}_unique_cache_"
        f"{dtype_tag}_seed{getattr(args, 'seed', 0)}_"
        f"neg{getattr(args, 'llm_negative_sample_size', 0)}.jsonl"
    )
    return os.path.join(
        getattr(args, "output_dir", "experiments/unlearning"),
        getattr(args, "unlearn_method", "unknown"),
        filename,
    )


def _load_prediction_cache_with_stats(path: str):
    cached: Dict[str, Dict] = {}
    stats = {
        "path": path,
        "exists": os.path.exists(path),
        "valid_lines": 0,
        "invalid_lines": 0,
        "duplicate_keys": 0,
        "schema_errors": 0,
    }
    if not os.path.exists(path):
        return cached, stats
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_lines"] += 1
                continue
            key = payload.get("unique_key")
            prediction = payload.get("prediction")
            if isinstance(key, str) and isinstance(prediction, dict):
                stats["valid_lines"] += 1
                if key in cached:
                    stats["duplicate_keys"] += 1
                cached[key] = prediction
            else:
                stats["schema_errors"] += 1
    stats["unique_keys"] = len(cached)
    return cached, stats


def _load_prediction_cache(path: str) -> Dict[str, Dict]:
    cached, _ = _load_prediction_cache_with_stats(path)
    return cached


def _append_prediction_cache(handle, unique_key: str, prediction: Dict):
    handle.write(json.dumps({
        "unique_key": unique_key,
        "prediction": prediction,
    }, default=str) + "\n")


def collect_predictions(model, tokenizer, split_data: Dict, dataset_data: Dict, args, stage: str) -> Dict:
    """Collect candidate-level prediction records for all protocol splits."""
    from dataloader.utils import Prompter

    model.eval()
    device = next(model.parameters()).device
    metric_ks = getattr(args, "rerank_metric_ks", [1, 5, 10])
    prompter = Prompter()
    verbalizer = _verbalizer(args, tokenizer)
    meta = dataset_data["meta"]
    num_items = len(meta)
    correction_map = _load_logit_correction(args, stage)

    requests = _build_prediction_requests(split_data, dataset_data, args)
    request_keys = []
    unique_requests = []
    seen_keys = set()
    for req in requests:
        key = _prediction_key(req)
        request_keys.append(key)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_requests.append(req)

    cache_path = _prediction_cache_path(args, stage)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    predictions_by_key, cache_stats = _load_prediction_cache_with_stats(cache_path)
    remaining_requests = [
        req for req in unique_requests
        if _prediction_key(req) not in predictions_by_key
    ]
    cache_summary = {
        **cache_stats,
        "stage": stage,
        "total_requests": len(requests),
        "expected_unique_requests": len(unique_requests),
        "cached_unique_matching_or_extra": len(predictions_by_key),
        "remaining_unique_requests": len(remaining_requests),
        "will_resume_from_cache": bool(predictions_by_key),
    }
    print(
        "[prediction_cache] "
        f"stage={stage} path={cache_path} "
        f"expected_unique={len(unique_requests)} "
        f"cached_unique={len(predictions_by_key)} "
        f"remaining={len(remaining_requests)} "
        f"invalid_lines={cache_stats.get('invalid_lines', 0)} "
        f"duplicates={cache_stats.get('duplicate_keys', 0)}",
        flush=True,
    )
    _save_json(
        os.path.join(
            os.path.dirname(cache_path),
            f"prediction_cache_{stage}_summary.json",
        ),
        cache_summary,
    )
    batch_size = max(1, int(getattr(args, "val_batch_size", 1) or 1))
    cache_handle = open(cache_path, "a")
    try:
        with tqdm(
            total=len(unique_requests),
            initial=len(unique_requests) - len(remaining_requests),
            desc=f"Collecting predictions ({stage}, unique)",
            leave=False,
            mininterval=30.0,
            miniters=max(1, batch_size * 250),
        ) as progress:
            for start in range(0, len(remaining_requests), batch_size):
                batch_requests = remaining_requests[start:start + batch_size]
                batch_inputs = []
                prepared = []
                for req in batch_requests:
                    uid = req["uid"]
                    target_iid = req["target_iid"]
                    split_tag = req["split_tag"]
                    context_items = _context_items_for_record(
                        req["source_row"],
                        split_tag,
                        dataset_data,
                        split_data,
                        args.llm_max_history,
                    )
                    candidate_items = _candidate_items(
                        args, uid, target_iid, context_items, split_tag, num_items
                    )

                    tokenized = seq_to_token_ids(
                        args,
                        context_items,
                        candidate_items,
                        target_iid,
                        meta,
                        tokenizer,
                        prompter,
                        eval=True,
                    )
                    batch_inputs.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                    })
                    prepared.append({
                        "request": req,
                        "uid": uid,
                        "target_iid": target_iid,
                        "split_tag": split_tag,
                        "context_items": context_items,
                        "candidate_items": candidate_items,
                    })

                batch = tokenizer.pad(batch_inputs, padding=True, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**batch)
                    class_scores_batch = verbalizer.process_logits(outputs.logits.float().cpu())

                for item, class_scores in zip(prepared, class_scores_batch):
                    uid = item["uid"]
                    target_iid = item["target_iid"]
                    candidate_items = item["candidate_items"]
                    scores = class_scores[:len(candidate_items)].tolist()
                    if correction_map:
                        scores = _apply_logit_correction_to_scores(
                            scores=scores,
                            candidate_items=candidate_items,
                            uid=uid,
                            correction_map=correction_map,
                        )

                    rank_info = _rank_record(scores, candidate_items, target_iid, metric_ks)
                    req = item["request"]
                    key = _prediction_key(req)
                    prediction = {
                        "uid": uid,
                        "target_iid": target_iid,
                        "position": req.get("position"),
                        "split_tag": item["split_tag"],
                        "context_items": [int(i) for i in item["context_items"]],
                        "candidate_items": [int(i) for i in candidate_items],
                        **rank_info,
                    }
                    predictions_by_key[key] = prediction
                    _append_prediction_cache(cache_handle, key, prediction)
                cache_handle.flush()
                progress.update(len(batch_requests))
    finally:
        cache_handle.close()

    records = []
    missing_keys = []
    for req, key in zip(requests, request_keys):
        prediction = predictions_by_key.get(key)
        if prediction is None:
            missing_keys.append(key)
            continue
        records.append({
            "prediction_id": req["prediction_id"],
            **prediction,
        })
    if missing_keys:
        preview = ", ".join(missing_keys[:10])
        raise RuntimeError(
            f"Prediction cache for stage={stage} is incomplete after collection: "
            f"missing {len(missing_keys)} request keys. Examples: {preview}"
        )

    return {
        "stage": stage,
        "prediction_schema": "candidate_level_v1",
        "metric_ks": metric_ks,
        "num_records": len(records),
        "num_inference_records": len(unique_requests),
        "num_deduplicated_records": len(requests) - len(unique_requests),
        "sample_limit_per_split": getattr(args, "max_eval_samples", 0),
        "records": records,
    }


def _load_logit_correction(args, stage: str) -> Dict:
    if stage != "after" or not bool(getattr(args, "enable_logit_correction", False)):
        return {}
    path = getattr(args, "logit_correction_path", None)
    if not path:
        path = os.path.join(
            getattr(args, "output_dir", "experiments/unlearning"),
            getattr(args, "unlearn_method", "geometry_prune"),
            "logit_correction.json",
        )
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("corrections"), dict):
        return payload["corrections"]
    return payload if isinstance(payload, dict) else {}


def _apply_logit_correction_to_scores(
    scores: List[float],
    candidate_items: List[int],
    uid: int,
    correction_map: Dict,
) -> List[float]:
    updated = list(scores)
    global_map = correction_map.get("*", {}) if isinstance(correction_map, dict) else {}
    user_map = correction_map.get(str(int(uid)), {}) if isinstance(correction_map, dict) else {}
    for idx, iid in enumerate(candidate_items):
        penalty = 0.0
        for source in (global_map, user_map):
            if isinstance(source, dict):
                try:
                    penalty += float(source.get(str(int(iid)), 0.0))
                except (TypeError, ValueError):
                    pass
        if penalty:
            updated[idx] = float(updated[idx]) - penalty
    return updated


def _records_by_id(predictions: Dict) -> Dict[str, Dict]:
    return {r["prediction_id"]: r for r in predictions.get("records", [])}


def _records_by_tag(predictions: Dict, split_tag: str) -> List[Dict]:
    return [r for r in predictions.get("records", []) if r.get("split_tag") == split_tag]


def _ranking_metrics(records: List[Dict], ks: List[int]) -> Dict:
    if not records:
        return {f"Recall@{k}": None for k in ks} | \
            {f"NDCG@{k}": None for k in ks} | \
            {f"MRR@{k}": None for k in ks}

    metrics = {}
    ranks = [r["target_rank"] for r in records]
    for k in ks:
        hits = [1.0 if rank <= k else 0.0 for rank in ranks]
        metrics[f"Recall@{k}"] = float(np.mean(hits))
        metrics[f"MRR@{k}"] = float(np.mean([
            (1.0 / rank) if rank <= k else 0.0 for rank in ranks
        ]))
        metrics[f"NDCG@{k}"] = float(np.mean([
            (1.0 / math.log2(rank + 1)) if rank <= k else 0.0
            for rank in ranks
        ]))
    return metrics


def _metric_drop(before_records: List[Dict], after_records: List[Dict], ks: List[int]) -> Optional[Dict]:
    if not before_records or not after_records:
        return None
    before = _ranking_metrics(before_records, ks)
    after = _ranking_metrics(after_records, ks)
    drop = {}
    for key, before_value in before.items():
        after_value = after.get(key)
        drop[key] = None if before_value is None or after_value is None else float(before_value - after_value)
    return {
        "before": before,
        "after": after,
        "drop": drop,
    }


def _paired_values(pred_before: Dict, pred_after: Dict, split_tags: List[str], field: str) -> List[float]:
    before_by_id = _records_by_id(pred_before)
    values = []
    for after in pred_after.get("records", []):
        if after.get("split_tag") not in split_tags:
            continue
        before = before_by_id.get(after["prediction_id"])
        if not before:
            continue
        values.append(float(after[field]) - float(before[field]))
    return values


def _exposure(records: List[Dict], ks: List[int]) -> Dict:
    if not records:
        return {f"@{k}": None for k in ks}
    return {
        f"@{k}": float(np.mean([1.0 if r["target_rank"] <= k else 0.0 for r in records]))
        for k in ks
    }


def evaluate_unlearning(
    predictions_before: Dict,
    predictions_after: Dict,
    split_data: Dict,
    args,
    output_dir: str,
) -> Dict:
    """Compute common metrics from before/after prediction dumps."""
    metric_ks = getattr(args, "rerank_metric_ks", [1, 5, 10])

    before_forget = _records_by_tag(predictions_before, "forget")
    after_forget = _records_by_tag(predictions_after, "forget")
    before_retain = _records_by_tag(predictions_before, "retain")
    after_retain = _records_by_tag(predictions_after, "retain")
    before_overlap = _records_by_tag(predictions_before, "overlap")
    after_overlap = _records_by_tag(predictions_after, "overlap")
    before_semantic = _records_by_tag(predictions_before, "semantic_neighbor")
    after_semantic = _records_by_tag(predictions_after, "semantic_neighbor")
    before_collab = _records_by_tag(predictions_before, "collaborative_neighbor")
    after_collab = _records_by_tag(predictions_after, "collaborative_neighbor")
    after_test = _records_by_tag(predictions_after, "test")

    rank_deltas = _paired_values(predictions_before, predictions_after, ["forget"], "target_rank")
    margin_deltas = _paired_values(
        predictions_before, predictions_after, ["forget", "overlap"], "margin_to_topk_boundary"
    )

    metrics = {
        "metric_protocol": "interaction_level_unlearning_v2",
        "temporary_candidate_ranking_note": (
            "Temporary candidate ranking metrics rank the target interaction item "
            "inside a fixed sampled candidate set. True forget exposure metrics "
            "are reported separately as forget_item_residual_exposure@K."
        ),
        "temporary_forget_candidate_ranking_before": _ranking_metrics(before_forget, metric_ks),
        "temporary_forget_candidate_ranking_after": _ranking_metrics(after_forget, metric_ks),
        "forget_item_residual_exposure": _exposure(after_forget, metric_ks),
        "forget_item_residual_exposure_before": _exposure(before_forget, metric_ks),
        "forget_item_rank_delta": (
            float(np.mean(rank_deltas)) if rank_deltas else None
        ),
        "retain_utility_drop": _metric_drop(before_retain, after_retain, metric_ks),
        "overlap_retain_protection_drop": _metric_drop(before_overlap, after_overlap, metric_ks),
        "semantic_neighbor_preservation": _metric_drop(before_semantic, after_semantic, metric_ks),
        "collaborative_neighbor_suppression": _metric_drop(before_collab, after_collab, metric_ks),
        "marginal_residual_margin_delta": (
            float(np.mean(margin_deltas)) if margin_deltas else None
        ),
        "test_ranking_after": _ranking_metrics(after_test, metric_ks),
        "unlearning_tradeoff_score": None,
        "_meta": {
            "metric_ks": metric_ks,
            "num_forget_interactions": len(split_data.get("forget_interactions", [])),
            "num_retain_interactions": len(split_data.get("retain_interactions", [])),
            "num_overlap_retain_interactions": len(split_data.get("overlap_retain_interactions", [])),
            "num_semantic_neighbor_retain": len(split_data.get("semantic_neighbor_retain", [])),
            "num_collaborative_neighbor_retain": len(split_data.get("collaborative_neighbor_retain", [])),
            "num_prediction_records_before": predictions_before.get("num_records", 0),
            "num_prediction_records_after": predictions_after.get("num_records", 0),
            "sample_limit_per_split": getattr(args, "max_eval_samples", 0),
        },
        "_todo": {
            "unlearning_tradeoff_score": "Define weighting after core metrics are validated.",
            "full_counterfactual_boundary": (
                "Current margin uses sampled candidate top-k boundary. Future work "
                "should add controlled counterfactual candidate pools."
            ),
        },
    }

    _save_json(os.path.join(output_dir, "metrics_unlearning.json"), metrics)
    if not os.path.exists(os.path.join(output_dir, "unlearning_metrics.json")):
        _save_json(os.path.join(output_dir, "unlearning_metrics.json"), metrics)
    _save_json(os.path.join(output_dir, "predictions_before.json"), predictions_before)
    _save_json(os.path.join(output_dir, "predictions_after.json"), predictions_after)
    return metrics
