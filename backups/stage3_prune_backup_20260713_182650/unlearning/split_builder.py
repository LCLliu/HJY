#!/usr/bin/env python3
"""
Build fixed interaction-level unlearning splits.

This script creates a reusable forget/retain protocol from the training
interactions. The output files are intended to be consumed by every
unlearning method, so methods never sample their own forget set.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# config.py parses argv at import time. Hide split-builder specific arguments
# while importing dataset classes, then restore them for our own parser.
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from datasets import DATASETS  # noqa: E402
sys.argv = _ORIG_ARGV


BASE_COLUMNS = [
    "interaction_id",
    "forget_id",
    "uid",
    "iid",
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "position",
    "sequence_index",
    "raw_index",
    "split_name",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fixed interaction-level unlearning split."
    )
    parser.add_argument("--dataset_code", type=str, default="ml-100k",
                        choices=sorted(DATASETS.keys()))
    parser.add_argument("--output_split_dir", type=str, required=True)
    parser.add_argument("--forget_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_rating", type=int, default=0)
    parser.add_argument("--min_uc", type=int, default=5)
    parser.add_argument("--min_sc", type=int, default=5)
    parser.add_argument("--overlap_window", type=int, default=2)
    parser.add_argument("--semantic_topk", type=int, default=5)
    parser.add_argument("--semantic_threshold", type=float, default=0.1)
    parser.add_argument("--collaborative_topk", type=int, default=5)
    parser.add_argument("--collaborative_threshold", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing split directory.")
    return parser.parse_args()


def _load_full_train_dataset(args) -> Dict:
    """Load the base dataset without applying an unlearning split."""
    dataset_args = argparse.Namespace(
        dataset_code=args.dataset_code,
        min_rating=args.min_rating,
        min_uc=args.min_uc,
        min_sc=args.min_sc,
        forget_ratio=0.0,
        forget_seed=args.seed,
        forget_interactions_path=None,
        retain_interactions_path=None,
        split_metadata_path=None,
    )
    dataset = DATASETS[args.dataset_code](dataset_args)
    return dataset.load_dataset()


def _build_train_interaction_records(args, data: Dict) -> Dict[int, List[Dict]]:
    """Recover dense train interactions with raw rating/timestamp metadata."""
    dataset = DATASETS[args.dataset_code](argparse.Namespace(
        dataset_code=args.dataset_code,
        min_rating=args.min_rating,
        min_uc=args.min_uc,
        min_sc=args.min_sc,
        forget_ratio=0.0,
        forget_seed=args.seed,
        forget_interactions_path=None,
        retain_interactions_path=None,
        split_metadata_path=None,
    ))
    df = dataset.load_ratings_df().copy()
    df["raw_index"] = df.index
    meta_raw = dataset.load_meta_dict()
    df = df[df["sid"].isin(meta_raw)]
    df = dataset.filter_triplets(df)

    umap = data["umap"]
    smap = data["smap"]
    df = df[df["uid"].isin(umap) & df["sid"].isin(smap)].copy()
    df["dense_uid"] = df["uid"].map(umap)
    df["dense_iid"] = df["sid"].map(smap)

    records_by_user = {}
    null_reasons = set()
    for uid, user_df in df.groupby("dense_uid"):
        user_df = user_df.sort_values(by=["timestamp", "dense_iid"])
        rows = []
        for pos, row in enumerate(user_df.itertuples(index=False)):
            rows.append({
                "uid": int(row.dense_uid),
                "iid": int(row.dense_iid),
                "user_id": int(row.dense_uid),
                "item_id": int(row.dense_iid),
                "rating": float(row.rating) if hasattr(row, "rating") else None,
                "timestamp": int(row.timestamp) if hasattr(row, "timestamp") else None,
                "position": int(pos),
                "sequence_index": int(pos),
                "raw_index": int(row.raw_index),
            })

        train_len = len(data["train"][int(uid)])
        train_rows = rows[:train_len]
        expected_items = data["train"][int(uid)]
        recovered_items = [r["iid"] for r in train_rows]
        if recovered_items != expected_items:
            raise ValueError(
                f"Recovered raw train sequence does not match cached train sequence "
                f"for uid={uid}."
            )
        records_by_user[int(uid)] = train_rows

    if not null_reasons:
        null_reasons.add("none")
    return records_by_user, sorted(null_reasons)


def _build_user_stratified_split(
    records_by_user: Dict[int, List[Dict]],
    forget_ratio: float,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.RandomState(seed)
    forget_rows = []
    retain_rows = []

    for uid in sorted(records_by_user.keys()):
        records = records_by_user[uid]
        if not records:
            continue

        if len(records) <= 1 or forget_ratio <= 0:
            forget_positions = set()
        else:
            n_forget = max(1, int(len(records) * forget_ratio))
            n_forget = min(n_forget, len(records) - 1)
            positions = np.arange(len(records))
            rng.shuffle(positions)
            forget_positions = set(int(p) for p in positions[:n_forget])

        for pos, record in enumerate(records):
            row = dict(record)
            row["interaction_id"] = f"u{uid}_p{pos}_i{row['iid']}"
            if pos in forget_positions:
                row["forget_id"] = f"f_u{uid}_p{pos}_i{row['iid']}"
                row["split_name"] = "forget"
                forget_rows.append(row)
            else:
                row["forget_id"] = ""
                row["split_name"] = "retain"
                retain_rows.append(row)

    return forget_rows, retain_rows


def _tokenize_title(title: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (title or "").lower()))


def _semantic_score(item_a: int, item_b: int, title_tokens: Dict[int, set]) -> float:
    ta = title_tokens.get(item_a, set())
    tb = title_tokens.get(item_b, set())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _build_item_user_sets(train: Dict[int, List[int]]) -> Dict[int, set]:
    item_users = defaultdict(set)
    for uid, seq in train.items():
        for iid in set(seq):
            item_users[int(iid)].add(int(uid))
    return item_users


def _collaborative_score(item_a: int, item_b: int, item_users: Dict[int, set]) -> float:
    ua = item_users.get(item_a, set())
    ub = item_users.get(item_b, set())
    if not ua or not ub:
        return 0.0
    return len(ua & ub) / len(ua | ub)


def _with_source(row: Dict, split_name: str, source: Dict, score_name: str, score: float) -> Dict:
    out = {
        "interaction_id": row.get("interaction_id", ""),
        "forget_id": row.get("forget_id", ""),
        "uid": row["user_id"],
        "iid": row["item_id"],
        "user_id": row["user_id"],
        "item_id": row["item_id"],
        "rating": row.get("rating", ""),
        "timestamp": row.get("timestamp", ""),
        "position": row.get("position", ""),
        "sequence_index": row.get("sequence_index", row.get("position", "")),
        "raw_index": row.get("raw_index", ""),
        "split_name": split_name,
        "source_forget_user_id": source["user_id"],
        "source_forget_item_id": source["item_id"],
        "source_forget_position": source["position"],
        "source_forget_id": source.get("forget_id", ""),
        score_name: float(score),
    }
    return out


def _build_neighbor_splits(
    train: Dict[int, List[int]],
    meta: Dict[int, str],
    forget_rows: List[Dict],
    retain_rows: List[Dict],
    overlap_window: int,
    semantic_topk: int,
    semantic_threshold: float,
    collaborative_topk: int,
    collaborative_threshold: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    retain_by_user = defaultdict(list)
    for row in retain_rows:
        retain_by_user[int(row["user_id"])].append(row)

    overlap_rows = {}
    semantic_rows = {}
    collaborative_rows = {}

    title_tokens = {int(iid): _tokenize_title(title) for iid, title in meta.items()}
    item_users = _build_item_user_sets(train)

    for forget in forget_rows:
        uid = int(forget["user_id"])
        fid = int(forget["item_id"])
        fpos = int(forget["position"])
        user_retain = retain_by_user.get(uid, [])

        for retain in user_retain:
            rpos = int(retain["position"])
            if abs(rpos - fpos) <= overlap_window:
                key = (uid, int(retain["item_id"]), rpos, fid, fpos)
                overlap_rows[key] = _with_source(
                    retain, "overlap_retain", forget, "overlap_distance",
                    abs(rpos - fpos),
                )

        semantic_candidates = []
        collaborative_candidates = []
        for retain in user_retain:
            rid = int(retain["item_id"])
            s_score = _semantic_score(fid, rid, title_tokens)
            if s_score >= semantic_threshold:
                semantic_candidates.append((s_score, retain))
            c_score = _collaborative_score(fid, rid, item_users)
            if c_score >= collaborative_threshold:
                collaborative_candidates.append((c_score, retain))

        semantic_candidates.sort(key=lambda x: x[0], reverse=True)
        collaborative_candidates.sort(key=lambda x: x[0], reverse=True)

        for score, retain in semantic_candidates[:semantic_topk]:
            key = (uid, int(retain["item_id"]), int(retain["position"]), fid, fpos)
            semantic_rows[key] = _with_source(
                retain, "semantic_neighbor_retain", forget,
                "semantic_score", score,
            )

        for score, retain in collaborative_candidates[:collaborative_topk]:
            key = (uid, int(retain["item_id"]), int(retain["position"]), fid, fpos)
            collaborative_rows[key] = _with_source(
                retain, "collaborative_neighbor_retain", forget,
                "collaborative_score", score,
            )

    return (
        list(overlap_rows.values()),
        list(semantic_rows.values()),
        list(collaborative_rows.values()),
    )


def _write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = []
    for row in rows:
        for key in row.keys():
            if key not in BASE_COLUMNS and key not in extra_columns:
                extra_columns.append(key)
    fieldnames = BASE_COLUMNS + extra_columns

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _ensure_output_is_new(output_dir: Path, files: Dict[str, str], overwrite: bool):
    existing = [
        str(output_dir / filename)
        for filename in files.values()
        if (output_dir / filename).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Split files already exist. Refusing to overwrite fixed protocol "
            "artifacts without --overwrite: " + ", ".join(existing)
        )


def _stable_rows_digest(*row_groups: List[Dict]) -> str:
    payload = json.dumps(row_groups, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    args = parse_args()
    if args.dataset_code != "ml-100k":
        raise ValueError("This first-stage split builder is intended for ML-100K.")
    if not 0.0 < args.forget_ratio < 1.0:
        raise ValueError("--forget_ratio must be in (0, 1).")

    data = _load_full_train_dataset(args)
    train = data["train"]
    meta = data.get("meta", {})
    records_by_user, null_reasons = _build_train_interaction_records(args, data)

    forget_rows, retain_rows = _build_user_stratified_split(
        records_by_user, args.forget_ratio, args.seed
    )
    overlap_rows, semantic_rows, collaborative_rows = _build_neighbor_splits(
        train=train,
        meta=meta,
        forget_rows=forget_rows,
        retain_rows=retain_rows,
        overlap_window=args.overlap_window,
        semantic_topk=args.semantic_topk,
        semantic_threshold=args.semantic_threshold,
        collaborative_topk=args.collaborative_topk,
        collaborative_threshold=args.collaborative_threshold,
    )

    output_dir = Path(args.output_split_dir)
    files = {
        "forget_interactions": "forget_interactions.csv",
        "retain_interactions": "retain_interactions.csv",
        "overlap_retain_interactions": "overlap_retain_interactions.csv",
        "semantic_neighbor_retain": "semantic_neighbor_retain.csv",
        "collaborative_neighbor_retain": "collaborative_neighbor_retain.csv",
        "split_metadata": "split_metadata.json",
    }
    _ensure_output_is_new(output_dir, files, args.overwrite)

    _write_csv(output_dir / files["forget_interactions"], forget_rows)
    _write_csv(output_dir / files["retain_interactions"], retain_rows)
    _write_csv(output_dir / files["overlap_retain_interactions"], overlap_rows)
    _write_csv(output_dir / files["semantic_neighbor_retain"], semantic_rows)
    _write_csv(output_dir / files["collaborative_neighbor_retain"], collaborative_rows)

    total_train = sum(len(v) for v in train.values())
    metadata = {
        "schema_version": "interaction_split_v1",
        "dataset_code": args.dataset_code,
        "split_strategy": "user_stratified",
        "forget_ratio": args.forget_ratio,
        "retain_ratio": 1.0 - args.forget_ratio,
        "seed": args.seed,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_users": len(train),
        "num_items": len(data.get("smap", {})),
        "num_train_interactions": total_train,
        "num_forget_interactions": len(forget_rows),
        "num_retain_interactions": len(retain_rows),
        "num_overlap_retain_interactions": len(overlap_rows),
        "num_semantic_neighbor_retain": len(semantic_rows),
        "num_collaborative_neighbor_retain": len(collaborative_rows),
        "split_fingerprint": _stable_rows_digest(
            forget_rows, retain_rows, overlap_rows, semantic_rows, collaborative_rows
        ),
        "schema": {
            "required_columns": BASE_COLUMNS,
            "id_fields": ["interaction_id", "forget_id", "uid", "iid", "position"],
            "null_field_reasons": {
                "rating": null_reasons,
                "timestamp": null_reasons,
                "position": null_reasons,
                "sequence_index": null_reasons,
                "raw_index": null_reasons,
            },
        },
        "params": {
            "overlap_window": args.overlap_window,
            "semantic_topk": args.semantic_topk,
            "semantic_threshold": args.semantic_threshold,
            "collaborative_topk": args.collaborative_topk,
            "collaborative_threshold": args.collaborative_threshold,
        },
        "files": files,
        "notes": [
            "forget_interactions.csv is the fixed unlearning target list.",
            "Unlearning methods must not resample forget interactions.",
        ],
    }
    with (output_dir / files["split_metadata"]).open("w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
