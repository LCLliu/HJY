#!/usr/bin/env python
"""Smoke tests for retain_prior_cf_lora_prune.

This uses a deterministic toy scorer and PEFT-like LoRA modules so the pruning
state machine can be tested without loading a full LLM checkpoint.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

KEEP_OUTPUT_ARG = "--keep-output" in sys.argv
if KEEP_OUTPUT_ARG:
    sys.argv = [arg for arg in sys.argv if arg != "--keep-output"]

from unlearning.methods.retain_prior_cf_lora_prune import (
    RetainPriorCFLoraPruneMethod,
    collect_lora_layers,
    load_and_apply_lora_rank_mask,
    temporary_mask_rank,
    restore_rank,
)


class FakeLoRAModule(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.lora_A = torch.nn.Linear(2, rank, bias=False)
        self.lora_B = torch.nn.Linear(rank, 2, bias=False)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class FakeRanker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(1, 1, bias=False)
        self.block1 = FakeLoRAModule(rank=2)
        self.block2 = FakeLoRAModule(rank=3)

    def forward(self, *args, **kwargs):
        return None


class ToyRetainPriorCF(RetainPriorCFLoraPruneMethod):
    def _score_candidate_list(
        self,
        user_history,
        candidates,
        target_iid,
        grad=False,
        score_batch_size=None,
        candidate_chunk_size=None,
    ):
        layers = collect_lora_layers(self.model)
        masks = {
            f"{layer.key}:{rid}": float(layer.rank_mask[rid].detach().cpu())
            for layer in layers
            for rid in range(int(layer.rank))
        }
        candidates = [int(x) for x in candidates]
        target_iid = int(target_iid)
        if target_iid not in candidates:
            candidates = [target_iid] + candidates
        forget_active = 1.0 if any(int(i) in {5, 7} for i in user_history) else 0.0
        retain_count = sum(1 for i in user_history if int(i) in {1, 2, 3, 4, 8, 9})
        scores = []
        for iid in candidates:
            family = int(iid) % 10
            base = {
                0: 0.20,
                1: 0.18,
                2: 0.12,
                3: 0.08,
            }.get(family, 0.04)
            residual = 0.0
            if iid in {10, 11}:
                residual += 0.80 * forget_active * masks.get("block1:0", 1.0)
                residual += 0.05 * forget_active * masks.get("block2:2", 1.0)
            if iid in {20, 21}:
                residual += 0.15 * retain_count * masks.get("block1:1", 1.0)
            if iid in {30, 31}:
                residual += 0.10 * retain_count * masks.get("block2:1", 1.0)
            scores.append(float(base + residual))
        return self._rank_scores(
            scores,
            candidates,
            target_iid,
            getattr(self.args, "rerank_metric_ks", [1, 2]),
        )


def make_args(output_dir, **overrides):
    values = {
        "seed": 123,
        "formal_run": False,
        "run_is_formal": False,
        "debug_split_sample_limit": None,
        "llm_max_history": 20,
        "llm_negative_sample_size": 2,
        "rerank_metric_ks": [1, 2],
        "topk_boundary": 2,
        "boundary_top_k": 2,
        "num_null_removals": 2,
        "null_eps": 1e-8,
        "residual_top_m": 2,
        "rp_cf_top_m": 1,
        "rp_cf_residual_selection_mode": "topk",
        "rp_cf_residual_threshold": None,
        "rp_cf_gamma": 1.0,
        "rp_cf_protection_quantile": 0.95,
        "rp_cf_residual_stop_quantile": 0.95,
        "rp_cf_retain_aggregation": "mean_positive_delta",
        "rp_cf_robust_distance": "smooth_l1",
        "rp_cf_robust_beta": 1.0,
        "rp_cf_accept_tol": 1e-12,
        "retain_support_samples": 2,
        "max_prune_ratio": 0.20,
        "min_forget_gain": 0.0,
        "retain_drop_tolerance": 0.05,
        "probe_retain_samples": 2,
        "output_dir": output_dir,
        "unlearn_method": "retain_prior_cf_lora_prune",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_split():
    return {
        "forget_interactions": [
            {"uid": 1, "iid": 5, "position": 1, "history": [1, 5, 2], "candidate_items": [10, 20, 30]},
            {"uid": 2, "iid": 7, "position": 1, "history": [3, 7, 4], "candidate_items": [11, 21, 31]},
        ],
        "retain_interactions": [
            {"uid": 1, "iid": 2, "position": 2, "history": [1, 2], "candidate_items": [20, 10, 30]},
            {"uid": 2, "iid": 4, "position": 2, "history": [3, 4], "candidate_items": [21, 11, 31]},
        ],
        "overlap_retain_interactions": [],
        "semantic_neighbor_retain": [],
        "collaborative_neighbor_retain": [],
    }


def make_dataset():
    return {
        "train": {1: [1, 5, 2], 2: [3, 7, 4]},
        "retain_train": {1: [1, 2], 2: [3, 4]},
        "val": {},
        "test": {},
        "meta": {i: f"Item {i}" for i in range(1, 40)},
        "unlearning_split_diagnostics": {
            "retain_train_loaded_from_split": True,
            "retain_train_excludes_forget_interactions": True,
        },
    }


def make_method(tmpdir, **arg_overrides):
    model = FakeRanker()
    args = make_args(tmpdir, **arg_overrides)
    return ToyRetainPriorCF(
        model=model,
        tokenizer=None,
        split_data=make_split(),
        args=args,
        output_dir=tmpdir,
        dataset_data=make_dataset(),
        predictions_before={},
    )


def assert_close(a, b, tol=1e-9, msg="values differ"):
    if abs(float(a) - float(b)) > tol:
        raise AssertionError(f"{msg}: {a} vs {b}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-output", action="store_true", default=KEEP_OUTPUT_ARG)
    parsed = parser.parse_args()

    torch.manual_seed(123)
    np.random.seed(123)
    tmpdir = tempfile.mkdtemp(prefix="rp_cf_smoke_")
    try:
        method = make_method(tmpdir)
        layers = collect_lora_layers(method.model)
        assert len(layers) == 2, "expected two LoRA modules"
        assert [layer.rank for layer in layers] == [2, 3], "different rank sizes not detected"

        baseline = method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        all_one = method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        assert_close(all_one, baseline, msg="all-one mask changed output")

        temporary_mask_rank(layers[0], 0)
        masked = method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        restore_rank(layers[0], 0)
        restored = method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        if not masked < baseline:
            raise AssertionError("temporary mask did not affect target score")
        assert_close(restored, baseline, msg="temporary mask restore failed")

        base_before = method.model.base.weight.detach().clone()
        logs = method.run()
        summary = logs["stage3_rank_pruning"]
        hard = int(summary["hard_prune"])
        if hard != 1:
            raise AssertionError(f"expected exactly one accepted prune at max ratio, got {hard}")
        if summary["reason"] != "max_prune_ratio_reached":
            raise AssertionError(f"expected max ratio stop, got {summary['reason']}")
        if summary["residual_energy_after"] >= summary["residual_energy_before"]:
            raise AssertionError("accepted prune did not reduce residual energy")
        if summary["protection_perturbation_after"] > summary["tau_prot"] + 1e-12:
            raise AssertionError("accepted prune violates protection threshold")
        if not torch.equal(base_before, method.model.base.weight.detach()):
            raise AssertionError("base model parameter changed")
        if logs["stage1_residual_localization"]["processed_forget_interactions"] != 2:
            raise AssertionError("multi-interaction forget set was not processed separately")

        after_score = method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        reload_method = make_method(os.path.join(tmpdir, "reload"))
        load_and_apply_lora_rank_mask(
            reload_method.model,
            os.path.join(tmpdir, "prune_mask.pt"),
            strict=True,
        )
        reloaded_score = reload_method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        assert_close(reloaded_score, after_score, msg="saved/reloaded mask output mismatch")

        rollback_dir = os.path.join(tmpdir, "rollback")
        rollback_method = make_method(rollback_dir, rp_cf_accept_tol=100.0, max_prune_ratio=0.20)
        rollback_logs = rollback_method.run()
        rollback_summary = rollback_logs["stage3_rank_pruning"]
        if int(rollback_summary["hard_prune"]) != 0:
            raise AssertionError("rollback run should not keep a rejected rank pruned")
        rollback_score = rollback_method._score_candidate_list([1, 5, 2], [10, 20, 30], 5)["scores"]["10"]
        assert_close(rollback_score, baseline, msg="rejected rank rollback polluted model state")

        no_valid_dir = os.path.join(tmpdir, "no_valid")
        no_valid_method = make_method(no_valid_dir, min_forget_gain=999.0)
        no_valid_logs = no_valid_method.run()
        no_valid_summary = no_valid_logs["stage3_rank_pruning"]
        if no_valid_summary["reason"] != "completed_no_valid_rank":
            raise AssertionError(f"expected completed_no_valid_rank, got {no_valid_summary['reason']}")
        if int(no_valid_summary["hard_prune"]) != 0:
            raise AssertionError("no-valid run pruned a rank")

        repeat_dir = os.path.join(tmpdir, "repeat")
        repeat_method = make_method(repeat_dir)
        repeat_logs = repeat_method.run()
        if repeat_logs["stage3_rank_pruning"]["actual_pruned_ranks"] != summary["actual_pruned_ranks"]:
            raise AssertionError("same seed did not reproduce final mask")

        result = {
            "output_dir": tmpdir,
            "hard_prune": hard,
            "final_prune_ratio": summary["final_prune_ratio"],
            "residual_energy_before": summary["residual_energy_before"],
            "residual_energy_after": summary["residual_energy_after"],
            "protection_perturbation_after": summary["protection_perturbation_after"],
            "tau_prot": summary["tau_prot"],
            "mask_path": os.path.join(tmpdir, "prune_mask.pt"),
            "log_path": os.path.join(tmpdir, "method_logs.json"),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if parsed.keep_output:
            print(f"kept smoke output: {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
