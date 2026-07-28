#!/usr/bin/env python3
"""
Unified interaction-level machine unlearning entrypoint for LlamaRec.

The unlearning target is a fixed interaction split artifact. Methods must read
the same forget/retain files and must not resample forget interactions.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# config.py parses argv at import time. Hide unlearning-specific arguments while
# importing constants/model classes, then restore argv for this script parser.
_ORIG_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
from config import *  # noqa: E402,F403
from model import LlamaForCausalLM  # noqa: E402
sys.argv = _ORIG_ARGV

from datasets import DATASETS  # noqa: E402
from unlearning.evaluation import collect_predictions, evaluate_unlearning  # noqa: E402
from unlearning.methods import METHOD_REGISTRY  # noqa: E402
from unlearning.methods.retain_prior_cf_lora_prune import (  # noqa: E402
    load_and_apply_lora_rank_mask,
    materialize_lora_rank_mask,
)

from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed-split interaction-level unlearning."
    )

    parser.add_argument("--dataset_code", type=str, default="ml-100k")
    parser.add_argument("--min_rating", type=int, default=0)
    parser.add_argument("--min_uc", type=int, default=5)
    parser.add_argument("--min_sc", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--llm_base_model", type=str,
                        default="/data/users/hjy/models/Llama-2-7b-hf")
    parser.add_argument("--llm_base_tokenizer", type=str,
                        default="/data/users/hjy/models/Llama-2-7b-hf")
    parser.add_argument("--llm_cache_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str,
                        default="experiments/Llama-2-7b-hf/ml-100k/checkpoint-4400")
    parser.add_argument("--llm_retrieved_path", type=str,
                        default="experiments/lru/ml-100k")

    parser.add_argument(
        "--unlearn_method",
        type=str,
        default="none",
        choices=sorted(METHOD_REGISTRY.keys()),
    )
    parser.add_argument("--interaction_level_unlearning", type=str2bool, default=True)
    parser.add_argument("--forget_interactions_path", type=str, default=None)
    parser.add_argument("--retain_interactions_path", type=str, default=None)
    parser.add_argument("--split_metadata_path", type=str, default=None)
    parser.add_argument("--forget_ratio", type=float, default=0.2,
                        help="Fallback only when explicit split files are absent.")
    parser.add_argument("--forget_seed", type=int, default=42,
                        help="Fallback only when explicit split files are absent.")
    parser.add_argument("--output_dir", type=str, default="experiments/unlearning")
    parser.add_argument("--overwrite_output", type=str2bool, default=False)
    parser.add_argument("--random_prune_ratio", type=float, default=0.05)

    # Geometry-aware pruning diagnostics and routing.
    parser.add_argument("--residual_alpha", type=float, default=1.0)
    parser.add_argument("--residual_beta", type=float, default=1.0)
    parser.add_argument("--residual_gamma", type=float, default=1.0)
    parser.add_argument("--topk_boundary", type=int, default=10)
    parser.add_argument("--boundary_window", type=int, default=10)
    parser.add_argument("--counterfactual_mode", type=str, default="mask",
                        choices=["mask", "remove"])
    parser.add_argument("--path_user_weight", type=float, default=1.0)
    parser.add_argument("--path_item_weight", type=float, default=1.0)
    parser.add_argument("--path_retriever_weight", type=float, default=0.5)
    parser.add_argument("--enable_retriever_diagnosis", type=str2bool, default=False)
    parser.add_argument("--hard_prune_threshold", type=float, default=0.7)
    parser.add_argument("--soft_suppress_threshold", type=float, default=0.25)
    parser.add_argument("--protect_threshold", type=float, default=0.8)
    parser.add_argument("--max_prune_ratio", type=float, default=0.05)
    parser.add_argument("--suppression_strength", type=float, default=0.7)
    parser.add_argument("--lambda_forget", type=float, default=2.0)
    parser.add_argument("--lambda_residual", type=float, default=2.0)
    parser.add_argument("--lambda_collab", type=float, default=1.5)
    parser.add_argument("--lambda_retain", type=float, default=1.0)
    parser.add_argument("--lambda_semantic", type=float, default=1.0)
    parser.add_argument("--enable_boundary_calibration", type=str2bool, default=False)
    parser.add_argument("--calibration_steps", type=int, default=1)
    parser.add_argument("--forget_residual_target", type=float, default=0.2)
    parser.add_argument("--retain_drop_tolerance", type=float, default=0.05)
    parser.add_argument("--overlap_drop_tolerance", type=float, default=0.05)
    parser.add_argument("--target_forget_exposure_drop", type=float, default=0.01)
    parser.add_argument("--enable_directional_probe", type=str2bool, default=True)
    parser.add_argument("--probe_top_m", type=int, default=64)
    parser.add_argument("--probe_forget_samples", type=int, default=8)
    parser.add_argument("--probe_retain_samples", type=int, default=8)
    parser.add_argument("--probe_score_tolerance", type=float, default=1e-4)
    parser.add_argument("--retain_probe_drop_tolerance", type=float, default=0.05)
    parser.add_argument("--enable_prune_rollback", type=str2bool, default=True)
    parser.add_argument("--prefer_soft_when_uncertain", type=str2bool, default=True)

    # Semantic geometry prune controls. These names are kept separate from the
    # older geometry diagnostics flags so existing runs remain parse-compatible.
    parser.add_argument("--alpha_residual", type=float, default=1.0)
    parser.add_argument("--beta_boundary", type=float, default=0.5)
    parser.add_argument("--gamma_collab", type=float, default=0.5)
    parser.add_argument("--lambda_sem_protect", type=float, default=0.7)
    parser.add_argument("--boundary_tau", type=float, default=5.0)
    parser.add_argument("--residual_top_m", type=int, default=64)
    parser.add_argument("--tau_residual_z", type=float, default=1.0)
    parser.add_argument("--num_null_removals", type=int, default=5)
    parser.add_argument("--null_eps", type=float, default=1e-8)
    parser.add_argument("--boundary_top_k", type=int, default=None)
    parser.add_argument("--boundary_score_margin", type=float, default=None)
    parser.add_argument("--collab_top_q", type=float, default=0.2)
    parser.add_argument("--collab_top_n", type=int, default=None)
    parser.add_argument("--tau_semantic_protect", type=float, default=0.5)
    parser.add_argument("--z_clip", type=float, default=5.0)
    parser.add_argument("--prune_ratio", type=float, default=0.01)
    parser.add_argument("--prune_budget_ratio", type=float, default=None)
    parser.add_argument("--retain_protection_threshold", type=float, default=0.3)
    parser.add_argument("--prune_threshold", type=float, default=None)
    parser.add_argument("--enable_logit_correction", type=str2bool, default=True)
    parser.add_argument("--enable_lora_suppression", type=str2bool, default=True)
    parser.add_argument("--enable_semantic_protected_prune", type=str2bool, default=True)
    parser.add_argument("--eta_logit", type=float, default=0.1)
    parser.add_argument("--lora_suppression_rho", type=float, default=0.05)
    parser.add_argument("--tau_rank_protect", type=float, default=0.7)
    parser.add_argument("--rank_score_delta", type=float, default=1e-8)
    parser.add_argument("--epsilon_retain", type=float, default=None)
    parser.add_argument("--prune_batch_size", type=int, default=1)
    parser.add_argument("--path_margin", type=float, default=0.0)

    # Retain-Prioritized Collaborative-Boundary Unlearning controls.
    parser.add_argument("--cbr_tau_z", type=float, default=1.0)
    parser.add_argument("--cbr_tau_c", type=float, default=0.0)
    parser.add_argument("--cbr_collab_max_hops", type=int, default=4)
    parser.add_argument("--cbr_boundary_top_k", type=int, default=None)
    parser.add_argument("--cbr_boundary_window", type=int, default=10)
    parser.add_argument("--cbr_tau_p", type=float, default=0.3)
    parser.add_argument("--cbr_alpha_sem", type=float, default=1.0)
    parser.add_argument("--cbr_beta_collab", type=float, default=1.0)
    parser.add_argument("--cbr_gamma_drop", type=float, default=1.0)
    parser.add_argument("--cbr_loss_max_groups", type=int, default=64)
    parser.add_argument("--cbr_distill_temperature", type=float, default=1.0)
    parser.add_argument("--cbr_retain_loss_samples", type=int, default=8)
    parser.add_argument("--cbr_rank_high_quantile", type=float, default=0.75)
    parser.add_argument("--cbr_rank_low_quantile", type=float, default=0.25)
    parser.add_argument("--cbr_overlap_quantile", type=float, default=0.75)
    parser.add_argument("--cbr_forget_intervention", type=str, default="suppress",
                        choices=["suppress", "prune"])
    parser.add_argument("--cbr_pruning_budget_ratio", type=float, default=0.01)
    parser.add_argument("--cbr_suppression_strength", type=float, default=0.7)
    parser.add_argument("--cbr_projection_gate", type=float, default=1.0)
    parser.add_argument("--cbr_update_lr", type=float, default=1e-4)
    parser.add_argument("--cbr_max_update_attempts", type=int, default=3)
    parser.add_argument("--cbr_soften_factor", type=float, default=0.5)

    # Retain-prior counterfactual LoRA rank pruning controls.
    parser.add_argument("--rp_cf_top_m", type=int, default=None,
                        help="Per forget interaction residual target count. Defaults to --residual_top_m.")
    parser.add_argument("--rp_cf_residual_selection_mode", type=str, default="topk",
                        choices=["topk", "threshold"],
                        help="Residual set selection mode for retain_prior_cf_lora_prune.")
    parser.add_argument("--rp_cf_residual_threshold", type=float, default=None,
                        help="Threshold used when --rp_cf_residual_selection_mode threshold. Defaults to --tau_residual_z.")
    parser.add_argument("--rp_cf_gamma", type=float, default=1.0,
                        help="Retain-support damping coefficient for W_unl.")
    parser.add_argument("--rp_cf_protection_quantile", type=float, default=0.95,
                        help="Null-control quantile used to derive tau_prot.")
    parser.add_argument("--rp_cf_residual_stop_quantile", type=float, default=None,
                        help="Null-control quantile used as residual natural-range stop threshold. Defaults to protection quantile.")
    parser.add_argument("--rp_cf_retain_aggregation", type=str, default="mean_positive_delta",
                        choices=["mean_positive_delta", "max_positive_delta", "sum_positive_delta"],
                        help="Explicit fallback aggregation for S_ret when no project formula is available.")
    parser.add_argument("--rp_cf_robust_distance", type=str, default="smooth_l1",
                        choices=["smooth_l1", "l1", "l2"],
                        help="Configurable rho for boundary-gap distances when no project rho is available.")
    parser.add_argument("--rp_cf_robust_beta", type=float, default=1.0,
                        help="Beta for smooth_l1 robust distance.")
    parser.add_argument("--rp_cf_accept_tol", type=float, default=1e-12,
                        help="Minimum numerical decrease required to accept a rank prune.")
    parser.add_argument("--retain_support_samples", type=int, default=None,
                        help="Retain history removals used for retain-support calibration. Defaults to --num_null_removals.")
    parser.add_argument("--top_prune_per_layer", type=int, default=1,
                        help="Safety cap on LoRA ranks pruned per layer for retain_prior_cf_lora_prune.")
    parser.add_argument("--min_forget_gain", type=float, default=0.0,
                        help="Minimum raw forget gain required before a rank can be pruned.")
    parser.add_argument("--resume_after_from_prune_mask", type=str2bool, default=False,
                        help=(
                            "Resume retain_prior_cf_lora_prune at Stage 3/after by loading "
                            "an existing prune_mask.pt and skipping Stage 1/Stage 2/decision generation."
                        ))
    parser.add_argument("--prune_mask_path", type=str, default=None,
                        help="Optional explicit prune_mask.pt path for --resume_after_from_prune_mask.")
    parser.add_argument("--predictions_before_path", type=str, default=None,
                        help="Optional explicit predictions_before.json path for after-only resume metrics.")
    parser.add_argument("--resume_pruned_adapter_dir", type=str, default=None,
                        help="Directory to save the recovered materialized pruned LoRA adapter.")

    parser.add_argument("--llm_max_title_len", type=int, default=32)
    parser.add_argument("--llm_max_text_len", type=int, default=1536)
    parser.add_argument("--llm_max_history", type=int, default=20)
    parser.add_argument("--llm_negative_sample_size", type=int, default=19)
    parser.add_argument("--llm_system_template", type=str,
                        default="Given user history in chronological order, recommend an item from the candidate pool with its index letter.")
    parser.add_argument("--llm_input_template", type=str,
                        default="User history: {}; \n Candidate pool: {}")
    parser.add_argument("--llm_train_on_inputs", type=bool, default=False)
    parser.add_argument("--llm_load_in_4bit", type=str2bool, default=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lora_micro_batch_size", type=int, default=16)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument("--rerank_metric_ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--max_eval_samples", type=int, default=0,
                        help="Per-split prediction dump limit. 0 means evaluate all.")
    parser.add_argument(
        "--debug_split_sample_limit",
        type=int,
        default=None,
        help=(
            "Debug-only per-split cap for method-internal unlearning "
            "diagnosis/evaluation records. None means formal run with no "
            "method split truncation."
        ),
    )
    parser.add_argument("--model_code", type=str, default="llm")
    parser.add_argument("--sliding_window_size", type=float, default=1.0)
    parser.add_argument("--negative_sample_size", type=int, default=10)
    parser.add_argument("--bert_max_len", type=int, default=200)
    parser.add_argument("--best_metric", type=str, default="Recall@10")
    parser.add_argument("--rerank_best_metric", type=str, default="NDCG@10")
    parser.add_argument("--enable_lr_schedule", type=bool, default=False)
    parser.add_argument("--enable_lr_warmup", type=bool, default=False)
    parser.add_argument("--early_stopping", type=bool, default=False)

    return parser.parse_args()


def _configure_run_mode(args):
    debug_limit = getattr(args, "debug_split_sample_limit", None)
    if debug_limit is not None and debug_limit <= 0:
        raise ValueError("--debug_split_sample_limit must be a positive integer or omitted.")

    args.formal_run = debug_limit is None
    args.run_is_formal = args.formal_run
    args.non_formal_reason = None if args.formal_run else "debug_split_sample_limit"
    return args


def _print_run_mode(args):
    print("=" * 60)
    print("Run mode")
    print("=" * 60)
    print(f"formal_run: {bool(getattr(args, 'formal_run', True))}")
    print(f"debug_split_sample_limit: {getattr(args, 'debug_split_sample_limit', None)}")
    if not bool(getattr(args, "formal_run", True)):
        print("WARNING: debug_split_sample_limit is enabled; this is NOT a formal run.")


def _read_interactions(path: str) -> List[Dict]:
    if path is None:
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, newline="") as f:
            return [_normalize_row(row) for row in csv.DictReader(f)]
    if ext == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ["interactions", "forget_interactions", "retain_interactions"]:
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"JSON interaction file must contain a list: {path}")
        return [_normalize_row(row) for row in data]
    raise ValueError(f"Unsupported interaction file format: {path}")


def _normalize_row(row: Dict) -> Dict:
    def optional_int(value):
        if value is None or value == "":
            return None
        return int(value)

    def first_nonempty(*values):
        for value in values:
            if value is not None and value != "" and value != "null":
                return value
        return None

    uid = first_nonempty(row.get("uid"), row.get("user_id"))
    iid = first_nonempty(row.get("iid"), row.get("item_id"))
    position = first_nonempty(row.get("position"), row.get("sequence_index"))

    out = {
        "uid": int(uid),
        "iid": int(iid),
        "user_id": int(uid),
        "item_id": int(iid),
        "rating": row.get("rating", ""),
        "timestamp": row.get("timestamp", ""),
        "position": optional_int(position),
        "sequence_index": optional_int(first_nonempty(row.get("sequence_index"), position)),
        "raw_index": optional_int(row.get("raw_index")),
        "split_name": row.get("split_name", ""),
    }
    for key, value in row.items():
        if key not in out:
            out[key] = value
    return out


def _load_split_data(args) -> Dict:
    metadata = {}
    base_dir = None
    if args.split_metadata_path:
        with open(args.split_metadata_path, "r") as f:
            metadata = json.load(f)
        base_dir = os.path.dirname(os.path.abspath(args.split_metadata_path))

    files = metadata.get("files", {})
    split_aliases = {
        "forget_interactions": ["forget_interactions", "forget", "forget_set"],
        "retain_interactions": ["retain_interactions", "retain", "retain_set"],
        "overlap_retain_interactions": ["overlap_retain_interactions", "overlap_retain", "overlap"],
        "semantic_neighbor_retain": ["semantic_neighbor_retain", "semantic_retain", "semantic_neighbors"],
        "collaborative_neighbor_retain": [
            "collaborative_neighbor_retain",
            "collaborative_retain",
            "collaborative_neighbors",
        ],
    }

    def metadata_value(key):
        for alias in split_aliases.get(key, [key]):
            if alias in metadata:
                return metadata[alias]
            if alias in files:
                return files[alias]
        return None

    def rows_from_metadata(key):
        value = metadata_value(key)
        if isinstance(value, list):
            return [_normalize_row(row) for row in value]
        if isinstance(value, dict):
            for inner_key in ["interactions", key, *split_aliases.get(key, [])]:
                if isinstance(value.get(inner_key), list):
                    return [_normalize_row(row) for row in value[inner_key]]
        return None

    def resolve(path, key):
        if path:
            return path
        filename = metadata_value(key)
        if isinstance(filename, (list, dict)):
            return None
        if filename:
            if os.path.isabs(filename):
                return filename
            if base_dir:
                return os.path.join(base_dir, filename)
            return filename
        return None

    forget_path = resolve(args.forget_interactions_path, "forget_interactions")
    retain_path = resolve(args.retain_interactions_path, "retain_interactions")
    forget_rows = rows_from_metadata("forget_interactions")
    retain_rows = rows_from_metadata("retain_interactions")

    if args.interaction_level_unlearning and not forget_path and forget_rows is None:
        raise ValueError(
            "Interaction-level unlearning requires --forget_interactions_path "
            "or --split_metadata_path with a forget/retain split entry."
        )

    split_data = {
        "forget_interactions": (
            forget_rows if forget_rows is not None
            else (_read_interactions(forget_path) if forget_path else [])
        ),
        "retain_interactions": (
            retain_rows if retain_rows is not None
            else (_read_interactions(retain_path) if retain_path else [])
        ),
        "metadata": metadata,
        "paths": {
            "forget_interactions_path": forget_path,
            "retain_interactions_path": retain_path,
            "split_metadata_path": args.split_metadata_path,
        },
    }

    optional_keys = [
        "overlap_retain_interactions",
        "semantic_neighbor_retain",
        "collaborative_neighbor_retain",
    ]
    for key in optional_keys:
        rows = rows_from_metadata(key)
        if rows is not None:
            split_data[key] = rows
            continue
        path = resolve(None, key)
        split_data[key] = _read_interactions(path) if path and os.path.exists(path) else []

    return split_data


def _split_fingerprint(split_data: Dict) -> str:
    payload = {
        "forget_interactions": split_data.get("forget_interactions", []),
        "retain_interactions": split_data.get("retain_interactions", []),
        "overlap_retain_interactions": split_data.get("overlap_retain_interactions", []),
        "semantic_neighbor_retain": split_data.get("semantic_neighbor_retain", []),
        "collaborative_neighbor_retain": split_data.get("collaborative_neighbor_retain", []),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _split_from_dataset_fallback(dataset_data: Dict) -> Dict:
    def rows_from_sequences(sequences, split_name):
        rows = []
        for uid, seq in sequences.items():
            for pos, iid in enumerate(seq):
                rows.append({
                    "user_id": int(uid),
                    "item_id": int(iid),
                    "rating": "",
                    "timestamp": "",
                    "position": int(pos),
                    "split_name": split_name,
                })
        return rows

    return {
        "forget_interactions": rows_from_sequences(
            dataset_data.get("forget_train", {}), "forget"
        ),
        "retain_interactions": rows_from_sequences(
            dataset_data.get("retain_train", dataset_data.get("train", {})), "retain"
        ),
        "overlap_retain_interactions": [],
        "semantic_neighbor_retain": [],
        "collaborative_neighbor_retain": [],
        "metadata": {
            "schema_version": "fallback_forget_ratio_split",
            "warning": (
                "This fallback is for backward compatibility only. "
                "Fixed interaction-level experiments should use split_builder.py."
            ),
        },
        "paths": {},
    }


def _resolve_split_paths(args):
    if not args.split_metadata_path:
        return args.forget_interactions_path, args.retain_interactions_path
    with open(args.split_metadata_path, "r") as f:
        metadata = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(args.split_metadata_path))
    files = metadata.get("files", {})
    split_aliases = {
        "forget_interactions": ["forget_interactions", "forget", "forget_set"],
        "retain_interactions": ["retain_interactions", "retain", "retain_set"],
    }

    def resolve(current, key):
        if current:
            return current
        filename = None
        for alias in split_aliases.get(key, [key]):
            candidate = files.get(alias, metadata.get(alias))
            if isinstance(candidate, str):
                filename = candidate
                break
        if not filename:
            return None
        if os.path.isabs(filename):
            return filename
        if base_dir:
            return os.path.join(base_dir, filename)
        return filename

    return (
        resolve(args.forget_interactions_path, "forget_interactions"),
        resolve(args.retain_interactions_path, "retain_interactions"),
    )


def _prepare_dataset_args(args):
    dataset_args = argparse.Namespace(**vars(args))
    dataset_args.model_code = "llm"
    forget_path, retain_path = _resolve_split_paths(args)
    dataset_args.forget_interactions_path = forget_path
    dataset_args.retain_interactions_path = retain_path
    if forget_path or args.split_metadata_path:
        dataset_args.forget_ratio = 0.0
    return dataset_args


def load_dataset_data(args) -> Dict:
    dataset_class = DATASETS[args.dataset_code]
    dataset = dataset_class(_prepare_dataset_args(args))
    data = dataset.load_dataset()
    if not data.get("retain_train"):
        data["retain_train"] = data["train"]
    if "forget_train" not in data:
        data["forget_train"] = {uid: [] for uid in data["train"].keys()}
    return data


def load_ranker_model(args):
    print("=" * 60)
    print("Loading trained LLM Ranker")
    print("=" * 60)

    if args.llm_load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None
    torch_dtype = (
        torch.float16
        if args.device == "cuda" and not args.llm_load_in_4bit
        else None
    )

    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map={"": "cuda:0"} if args.device == "cuda" else None,
        cache_dir=args.llm_cache_dir,
    )
    if args.llm_load_in_4bit:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    print(f"Loading LoRA adapter from: {args.checkpoint_dir}")
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_base_tokenizer, cache_dir=args.llm_cache_dir
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    return model, tokenizer


def _method_output_dir(args):
    return os.path.join(args.output_dir, args.unlearn_method)


def _prepare_method_output_dir(args):
    method_dir = _method_output_dir(args)
    if bool(getattr(args, "resume_after_from_prune_mask", False)):
        os.makedirs(method_dir, exist_ok=True)
        return method_dir
    if os.path.isdir(method_dir) and os.listdir(method_dir) and not args.overwrite_output:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {method_dir}. "
            "Use --overwrite_output true or choose a new --output_dir."
        )
    if os.path.isdir(method_dir) and os.listdir(method_dir) and args.overwrite_output:
        shutil.rmtree(method_dir)
    os.makedirs(method_dir, exist_ok=True)
    return method_dir


def _save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _merge_method_log_fields(method_dir: str, fields: Dict):
    method_logs_path = os.path.join(method_dir, "method_logs.json")
    if os.path.exists(method_logs_path):
        method_logs = _load_json(method_logs_path)
        if not isinstance(method_logs, dict):
            method_logs = {}
    else:
        method_logs = {}
    method_logs.update(fields)
    _save_json(method_logs_path, method_logs)


def _mean_exposure_delta(before, after):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    deltas = []
    for key, before_value in before.items():
        if key not in after:
            continue
        try:
            deltas.append(float(before_value) - float(after[key]))
        except (TypeError, ValueError):
            continue
    return float(np.mean(deltas)) if deltas else None


def _max_positive_metric_drop(metric_drop):
    if not isinstance(metric_drop, dict):
        return None
    drop = metric_drop.get("drop")
    if not isinstance(drop, dict):
        return None
    values = []
    for value in drop.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _prediction_changed_ratio(predictions_before, predictions_after):
    before_by_id = {
        r.get("prediction_id"): r
        for r in predictions_before.get("records", [])
        if r.get("prediction_id") is not None
    }
    paired = 0
    changed = 0
    for after in predictions_after.get("records", []):
        before = before_by_id.get(after.get("prediction_id"))
        if not before:
            continue
        paired += 1
        score_changed = before.get("target_score") != after.get("target_score")
        rank_changed = before.get("target_rank") != after.get("target_rank")
        topk_changed = before.get("topk_items") != after.get("topk_items")
        changed += int(score_changed or rank_changed or topk_changed)
    return (float(changed) / float(paired)) if paired else None


def _append_method_diagnostics(
    method_dir,
    args,
    metrics,
    predictions_before,
    predictions_after,
    split_data=None,
    dataset_data=None,
):
    method_logs_path = os.path.join(method_dir, "method_logs.json")
    if os.path.exists(method_logs_path):
        with open(method_logs_path, "r") as f:
            method_logs = json.load(f)
    else:
        method_logs = {}

    pruning_path = os.path.join(method_dir, "pruning_decisions.json")
    pruning = {}
    if os.path.exists(pruning_path):
        with open(pruning_path, "r") as f:
            pruning = json.load(f)
    summary = pruning.get("summary", {}) if isinstance(pruning, dict) else {}

    total_ranks = int(summary.get("total_ranks", 0) or 0)
    hard = int(summary.get("hard_prune", 0) or 0)
    soft = int(summary.get("soft_suppress", 0) or 0)
    protect = int(summary.get("protect", 0) or 0)
    exposure_before = metrics.get("forget_item_residual_exposure_before")
    exposure_after = metrics.get(
        "forget_item_residual_exposure_after",
        metrics.get("forget_item_residual_exposure"),
    )
    exposure_drop = _mean_exposure_delta(exposure_before, exposure_after)
    exposure_improved = (
        exposure_drop is not None and
        exposure_drop >= float(getattr(args, "target_forget_exposure_drop", 0.0))
    )
    retain_drop_max = _max_positive_metric_drop(metrics.get("retain_utility_drop"))
    retain_drop_warning = (
        retain_drop_max is not None and
        retain_drop_max > float(getattr(args, "retain_drop_tolerance", 0.0))
    )
    changed_ratio = _prediction_changed_ratio(predictions_before, predictions_after)

    warnings = list(method_logs.get("warnings", []))
    if (
        changed_ratio is not None and changed_ratio > 0 and
        exposure_drop is not None and exposure_drop <= 0
    ):
        message = "geometry_prune changed scores but did not reduce forget exposure"
        if message not in warnings:
            warnings.append(message)

    method_logs.update({
        "loaded_split_fingerprint": ((split_data or {}).get("metadata") or {}).get("split_fingerprint"),
        "num_forget_interactions": len((split_data or {}).get("forget_interactions", [])),
        "num_retain_interactions": len((split_data or {}).get("retain_interactions", [])),
        "num_overlap_retain_interactions": len((split_data or {}).get("overlap_retain_interactions", [])),
        "num_semantic_neighbor_retain": len((split_data or {}).get("semantic_neighbor_retain", [])),
        "num_collaborative_neighbor_retain": len((split_data or {}).get("collaborative_neighbor_retain", [])),
        "retain_train_loaded_from_split": (
            (dataset_data or {}).get("unlearning_split_diagnostics", {}).get(
                "retain_train_loaded_from_split"
            )
        ),
        "retain_train_excludes_forget_interactions": (
            (dataset_data or {}).get("unlearning_split_diagnostics", {}).get(
                "retain_train_excludes_forget_interactions"
            )
        ),
        "forgotten_interactions_in_retain_train": (
            (dataset_data or {}).get("unlearning_split_diagnostics", {}).get(
                "forgotten_interactions_in_retain_train"
            )
        ),
        "unlearning_split_diagnostics": (dataset_data or {}).get("unlearning_split_diagnostics", {}),
        "max_eval_samples_applied_to_split": False,
        "debug_split_sample_limit": getattr(args, "debug_split_sample_limit", None),
        "debug_split_sample_limit_applied": getattr(args, "debug_split_sample_limit", None) is not None,
        "formal_run": bool(getattr(args, "formal_run", True)),
        "run_is_formal": bool(getattr(args, "run_is_formal", True)),
        "non_formal_reason": getattr(args, "non_formal_reason", None),
        "num_total_ranks": total_ranks,
        "num_hard_prune": hard,
        "num_soft_suppress": soft,
        "num_protect": protect,
        "actual_prune_ratio": float(hard) / float(total_ranks) if total_ranks else None,
        "actual_suppress_ratio": float(soft) / float(total_ranks) if total_ranks else None,
        "protected_rank_ratio": float(protect) / float(total_ranks) if total_ranks else None,
        "forget_exposure_before": exposure_before,
        "forget_exposure_after": exposure_after,
        "target_forget_exposure_drop": float(getattr(args, "target_forget_exposure_drop", 0.0)),
        "forget_exposure_drop_mean": exposure_drop,
        "exposure_improved": bool(exposure_improved),
        "retain_drop_max": retain_drop_max,
        "retain_drop_warning": bool(retain_drop_warning),
        "candidate_level_exposure": True,
        "not_full_item_exposure": True,
        "prediction_changed_records_ratio": changed_ratio,
        "warnings": warnings,
    })
    _save_json(method_logs_path, method_logs)


def _write_run_artifacts(args, split_data, method_dir):
    config = vars(args).copy()
    metadata = split_data.get("metadata") or {}
    split_fingerprint = metadata.get("split_fingerprint") or _split_fingerprint(split_data)
    config["method_output_dir"] = method_dir
    config["protocol"] = "interaction_level_unlearning_v1"
    config["split_fingerprint"] = split_fingerprint
    config["loaded_split_fingerprint"] = split_fingerprint
    config["num_forget_interactions"] = len(split_data.get("forget_interactions", []))
    config["num_retain_interactions"] = len(split_data.get("retain_interactions", []))
    config["num_overlap_retain_interactions"] = len(split_data.get("overlap_retain_interactions", []))
    config["num_semantic_neighbor_retain"] = len(split_data.get("semantic_neighbor_retain", []))
    config["num_collaborative_neighbor_retain"] = len(split_data.get("collaborative_neighbor_retain", []))
    config["max_eval_samples_applied_to_split"] = False
    config["debug_split_sample_limit"] = getattr(args, "debug_split_sample_limit", None)
    config["debug_split_sample_limit_applied"] = getattr(args, "debug_split_sample_limit", None) is not None
    config["formal_run"] = bool(getattr(args, "formal_run", True))
    config["run_is_formal"] = bool(getattr(args, "run_is_formal", True))
    config["non_formal_reason"] = getattr(args, "non_formal_reason", None)
    _save_json(os.path.join(method_dir, "config.json"), config)
    _save_json(os.path.join(method_dir, "run_config.json"), config)

    metadata = metadata or {
        "schema_version": "ad_hoc_explicit_split",
        "files": split_data.get("paths", {}),
        "num_forget_interactions": len(split_data.get("forget_interactions", [])),
        "num_retain_interactions": len(split_data.get("retain_interactions", [])),
    }
    metadata.setdefault("split_fingerprint", split_fingerprint)
    _save_json(os.path.join(method_dir, "split_metadata.json"), metadata)


def _write_resume_run_artifacts(args, split_data, method_dir):
    config = vars(args).copy()
    metadata = split_data.get("metadata") or {}
    split_fingerprint = metadata.get("split_fingerprint") or _split_fingerprint(split_data)
    config["method_output_dir"] = method_dir
    config["protocol"] = "interaction_level_unlearning_v1"
    config["resume_mode"] = "after_from_prune_mask"
    config["split_fingerprint"] = split_fingerprint
    config["loaded_split_fingerprint"] = split_fingerprint
    config["num_forget_interactions"] = len(split_data.get("forget_interactions", []))
    config["num_retain_interactions"] = len(split_data.get("retain_interactions", []))
    config["num_overlap_retain_interactions"] = len(split_data.get("overlap_retain_interactions", []))
    config["num_semantic_neighbor_retain"] = len(split_data.get("semantic_neighbor_retain", []))
    config["num_collaborative_neighbor_retain"] = len(split_data.get("collaborative_neighbor_retain", []))
    config["formal_run"] = bool(getattr(args, "formal_run", True))
    config["run_is_formal"] = bool(getattr(args, "run_is_formal", True))
    _save_json(os.path.join(method_dir, "resume_run_config.json"), config)


def _resolve_required_resume_file(method_dir: str, explicit_path: str, filename: str) -> str:
    path = explicit_path or os.path.join(method_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required resume artifact not found: {path}")
    return path


def _validate_pruning_decisions_for_resume(method_dir: str, mask_summary: Dict) -> Dict:
    pruning_path = os.path.join(method_dir, "pruning_decisions.json")
    if not os.path.exists(pruning_path):
        return {
            "pruning_decisions_path": pruning_path,
            "pruning_decisions_found": False,
        }

    pruning = _load_json(pruning_path)
    summary = pruning.get("summary", {}) if isinstance(pruning, dict) else {}
    validation = {
        "pruning_decisions_path": pruning_path,
        "pruning_decisions_found": True,
        "decisions_count": len(pruning.get("decisions", [])) if isinstance(pruning, dict) else None,
        "summary_total_lora_layers": summary.get("total_lora_layers"),
        "summary_total_ranks": summary.get("total_ranks"),
        "summary_hard_prune": summary.get("hard_prune"),
    }

    expected_total = summary.get("total_ranks")
    if expected_total is not None and int(expected_total) != int(mask_summary["applied_rank_mask_count"]):
        raise RuntimeError(
            "pruning_decisions.json total_ranks does not match applied prune_mask.pt: "
            f"{expected_total} != {mask_summary['applied_rank_mask_count']}"
        )
    expected_hard = summary.get("hard_prune")
    if expected_hard is not None and int(expected_hard) != int(mask_summary["applied_pruned_rank_count"]):
        raise RuntimeError(
            "pruning_decisions.json hard_prune does not match applied prune_mask.pt: "
            f"{expected_hard} != {mask_summary['applied_pruned_rank_count']}"
        )
    expected_layers = summary.get("total_lora_layers")
    if expected_layers is not None and int(expected_layers) != int(mask_summary["matched_lora_module_count"]):
        raise RuntimeError(
            "pruning_decisions.json total_lora_layers does not match current model: "
            f"{expected_layers} != {mask_summary['matched_lora_module_count']}"
        )
    return validation


def _save_recovered_pruned_adapter(
    model,
    layers,
    adapter_dir: str,
    mask_path: str,
    logit_correction_path: str,
) -> Dict:
    os.makedirs(adapter_dir, exist_ok=True)
    materialize_summary = materialize_lora_rank_mask(layers)
    model.save_pretrained(adapter_dir)
    shutil.copyfile(mask_path, os.path.join(adapter_dir, "prune_mask.pt"))
    if logit_correction_path and os.path.exists(logit_correction_path):
        shutil.copyfile(logit_correction_path, os.path.join(adapter_dir, "logit_correction.json"))
    materialize_summary.update({
        "recovered_pruned_adapter_dir": adapter_dir,
        "adapter_saved_with_materialized_lora_B_mask": True,
    })
    _save_json(os.path.join(adapter_dir, "recovery_summary.json"), materialize_summary)
    return materialize_summary


def _run_after_resume_from_prune_mask(args, split_data, dataset_data, method_dir):
    if args.unlearn_method != "retain_prior_cf_lora_prune":
        raise ValueError(
            "--resume_after_from_prune_mask is only supported for "
            "--unlearn_method retain_prior_cf_lora_prune."
        )

    mask_path = _resolve_required_resume_file(method_dir, args.prune_mask_path, "prune_mask.pt")
    before_path = _resolve_required_resume_file(
        method_dir,
        args.predictions_before_path,
        "predictions_before.json",
    )
    logit_correction_path = os.path.join(method_dir, "logit_correction.json")
    if os.path.exists(logit_correction_path):
        args.logit_correction_path = logit_correction_path

    print("=" * 60)
    print("Resume retain_prior_cf_lora_prune at Stage 3/after")
    print("=" * 60)
    print("Skipping Stage 1 residual localization")
    print("Skipping Stage 2 retain-aware calibration")
    print("Skipping pruning decision generation")
    print(f"Loading existing predictions_before from: {before_path}")
    print(f"Loading existing prune mask from: {mask_path}")
    if os.path.exists(logit_correction_path):
        print(f"Restoring logit correction state from: {logit_correction_path}")
    else:
        print("No logit_correction.json found; after inference will run without correction state.")

    predictions_before = _load_json(before_path)
    model, tokenizer = load_ranker_model(args)

    layers, mask_summary = load_and_apply_lora_rank_mask(
        model=model,
        mask_path=mask_path,
        strict=True,
    )
    pruning_validation = _validate_pruning_decisions_for_resume(method_dir, mask_summary)

    adapter_dir = (
        args.resume_pruned_adapter_dir
        or os.path.join(method_dir, "recovered_pruned_adapter")
    )
    adapter_summary = _save_recovered_pruned_adapter(
        model=model,
        layers=layers,
        adapter_dir=adapter_dir,
        mask_path=mask_path,
        logit_correction_path=logit_correction_path,
    )
    recovery_summary = {
        "resume_after_from_prune_mask": True,
        "stage1_skipped": True,
        "stage2_skipped": True,
        "pruning_decision_generation_skipped": True,
        "checkpoint_dir_reloaded": args.checkpoint_dir,
        "original_lora_adapter_reloaded": args.checkpoint_dir,
        "mask_application": mask_summary,
        "pruning_decisions_validation": pruning_validation,
        "adapter_recovery": adapter_summary,
        "logit_correction_path": logit_correction_path if os.path.exists(logit_correction_path) else None,
        "predictions_before_path": before_path,
    }
    _save_json(os.path.join(method_dir, "resume_after_recovery_summary.json"), recovery_summary)
    _merge_method_log_fields(method_dir, recovery_summary)

    print("[resume_after] matched_lora_module_count:", mask_summary["matched_lora_module_count"])
    print("[resume_after] applied_rank_mask_count:", mask_summary["applied_rank_mask_count"])
    print("[resume_after] applied_pruned_rank_count:", mask_summary["applied_pruned_rank_count"])
    print("[resume_after] checksum_before:", json.dumps(mask_summary["checksum_before"], sort_keys=True))
    print("[resume_after] checksum_after:", json.dumps(mask_summary["checksum_after"], sort_keys=True))
    print("[resume_after] materialize_checksum_before:", json.dumps(
        adapter_summary["checksum_before_materialize"],
        sort_keys=True,
    ))
    print("[resume_after] materialize_checksum_after:", json.dumps(
        adapter_summary["checksum_after_materialize"],
        sort_keys=True,
    ))
    print(f"[resume_after] recovered_pruned_adapter_dir: {adapter_dir}")

    print("=" * 60)
    print("Collecting predictions after unlearning")
    print("=" * 60)
    predictions_after = collect_predictions(
        model=model,
        tokenizer=tokenizer,
        split_data=split_data,
        dataset_data=dataset_data,
        args=args,
        stage="after",
    )

    print("=" * 60)
    print("Unified unlearning evaluation")
    print("=" * 60)
    metrics = evaluate_unlearning(
        predictions_before=predictions_before,
        predictions_after=predictions_after,
        split_data=split_data,
        args=args,
        output_dir=method_dir,
    )
    _append_method_diagnostics(
        method_dir=method_dir,
        args=args,
        metrics=metrics,
        predictions_before=predictions_before,
        predictions_after=predictions_after,
        split_data=split_data,
        dataset_data=dataset_data,
    )
    _merge_method_log_fields(method_dir, {
        "status": "completed",
        "resume_after_from_prune_mask": True,
        "stage1_skipped": True,
        "stage2_skipped": True,
        "pruning_decision_generation_skipped": True,
        "recovered_pruned_adapter_dir": adapter_dir,
        "metrics_file": os.path.join(method_dir, "metrics_unlearning.json"),
        "predictions_after_file": os.path.join(method_dir, "predictions_after.json"),
    })

    print(json.dumps(metrics, indent=2, default=str))
    print(f"Results saved to {method_dir}")


def _set_reproducibility(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    args = _configure_run_mode(args)
    _print_run_mode(args)
    _set_reproducibility(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    split_data = _load_split_data(args)
    dataset_data = load_dataset_data(args)
    if not split_data.get("forget_interactions") and not args.interaction_level_unlearning:
        split_data = _split_from_dataset_fallback(dataset_data)

    method_dir = _prepare_method_output_dir(args)
    if bool(getattr(args, "resume_after_from_prune_mask", False)):
        _write_resume_run_artifacts(args, split_data, method_dir)
        _run_after_resume_from_prune_mask(args, split_data, dataset_data, method_dir)
        return

    _write_run_artifacts(args, split_data, method_dir)

    model, tokenizer = load_ranker_model(args)

    print("=" * 60)
    print("Collecting predictions before unlearning")
    print("=" * 60)
    predictions_before = collect_predictions(
        model=model,
        tokenizer=tokenizer,
        split_data=split_data,
        dataset_data=dataset_data,
        args=args,
        stage="before",
    )
    _save_json(os.path.join(method_dir, "predictions_before.json"), predictions_before)

    method_cls = METHOD_REGISTRY[args.unlearn_method]
    method = method_cls(
        model=model,
        tokenizer=tokenizer,
        split_data=split_data,
        args=args,
        output_dir=method_dir,
        dataset_data=dataset_data,
        predictions_before=predictions_before,
    )

    print("=" * 60)
    print(f"Running unlearning method: {args.unlearn_method}")
    print("=" * 60)
    method.run()

    print("=" * 60)
    print("Collecting predictions after unlearning")
    print("=" * 60)
    predictions_after = collect_predictions(
        model=model,
        tokenizer=tokenizer,
        split_data=split_data,
        dataset_data=dataset_data,
        args=args,
        stage="after",
    )

    print("=" * 60)
    print("Unified unlearning evaluation")
    print("=" * 60)
    metrics = evaluate_unlearning(
        predictions_before=predictions_before,
        predictions_after=predictions_after,
        split_data=split_data,
        args=args,
        output_dir=method_dir,
    )
    _append_method_diagnostics(
        method_dir=method_dir,
        args=args,
        metrics=metrics,
        predictions_before=predictions_before,
        predictions_after=predictions_after,
        split_data=split_data,
        dataset_data=dataset_data,
    )

    print(json.dumps(metrics, indent=2, default=str))
    print(f"Results saved to {method_dir}")


if __name__ == "__main__":
    main()
