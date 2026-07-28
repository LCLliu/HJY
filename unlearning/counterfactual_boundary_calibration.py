import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataloader.llm import seq_to_token_ids
from trainer.verb import ManualVerbalizer


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return [0.0 for _ in values]
    return ((arr - vmin) / (vmax - vmin)).tolist()


def row_uid(row: Dict) -> int:
    return int(row.get("uid", row.get("user_id")))


def row_iid(row: Dict) -> int:
    return int(row.get("iid", row.get("item_id")))


def record_by_uid_iid(predictions_before: Dict, split_tag: str) -> Dict[Tuple[int, int], Dict]:
    result = {}
    for record in predictions_before.get("records", []):
        if record.get("split_tag") == split_tag:
            result[(int(record["uid"]), int(record["target_iid"]))] = record
    return result


class CounterfactualBoundaryCalibrator:
    """Candidate-level counterfactual residual and boundary sensitivity.

    This is a candidate-level fallback, not full causal intervention over the
    training process. It removes or masks the forget item from the prompt
    context and recomputes the same candidate set.
    """

    def __init__(self, model, tokenizer, dataset_data: Dict, args):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_data = dataset_data
        self.args = args
        self.logs = {
            "counterfactual_mode": args.counterfactual_mode,
            "fallback": "candidate_level_context_intervention",
            "notes": [
                "This is not full retraining counterfactual.",
                "Scores are recomputed for the same candidate list after context masking/removal.",
            ],
        }

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

    def _context_without_forget(self, record: Dict) -> List[int]:
        fid = int(record["target_iid"])
        context = list(record.get("context_items", []))
        if self.args.counterfactual_mode == "remove":
            return [int(i) for i in context if int(i) != fid]
        masked = [int(i) for i in context]
        for idx, iid in enumerate(masked):
            if iid == fid:
                masked[idx] = 0
        return [iid for iid in masked if iid != 0]

    def recompute_scores(self, record: Dict) -> Tuple[Dict[str, float], bool, str]:
        from dataloader.utils import Prompter

        try:
            context_items = self._context_without_forget(record)
            candidate_items = [int(i) for i in record["candidate_items"]]
            target_iid = int(record["target_iid"])
            meta = self.dataset_data["meta"]
            prompter = Prompter()
            tokenized = seq_to_token_ids(
                self.args,
                context_items,
                candidate_items,
                target_iid,
                meta,
                self.tokenizer,
                prompter,
                eval=True,
            )
            device = next(self.model.parameters()).device
            batch = {
                "input_ids": torch.tensor([tokenized["input_ids"]], dtype=torch.long).to(device),
                "attention_mask": torch.tensor([tokenized["attention_mask"]], dtype=torch.long).to(device),
            }
            with torch.no_grad():
                outputs = self.model(**batch)
                class_scores = self._verbalizer().process_logits(outputs.logits.float().cpu())[0]
                scores = class_scores[:len(candidate_items)].tolist()
            return {
                str(iid): float(score)
                for iid, score in zip(candidate_items, scores)
            }, True, ""
        except Exception as exc:
            return dict(record.get("scores", {})), False, str(exc)

    def boundary_sensitivity(self, record: Dict) -> float:
        rank = int(record.get("target_rank", 10**6))
        topk = int(getattr(self.args, "topk_boundary", 10))
        window = max(1, int(getattr(self.args, "boundary_window", 2)))
        margin = float(record.get("margin_to_topk_boundary", 0.0))
        rank_component = max(0.0, 1.0 - abs(rank - topk) / max(window, 1))
        margin_component = 1.0 / (1.0 + math.exp(abs(margin)))
        if rank <= topk:
            rank_component = max(rank_component, 0.7)
        return float(np.clip(0.6 * rank_component + 0.4 * margin_component, 0.0, 1.0))

    def counterfactual_residual(self, record: Dict, scores_cf: Dict[str, float]) -> float:
        target = str(int(record["target_iid"]))
        before = float(record.get("scores", {}).get(target, record.get("target_score", 0.0)))
        after = float(scores_cf.get(target, before))
        return float(max(0.0, before - after))

    def rank_from_scores(self, candidate_iid: int, scores: Dict[str, float]) -> int:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for idx, (iid, _) in enumerate(ranked, start=1):
            if int(iid) == int(candidate_iid):
                return idx
        return len(ranked) + 1
