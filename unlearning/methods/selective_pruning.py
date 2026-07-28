"""Traditional parameter-level selective pruning unlearning strategy.

This module is a strategy module only. It is invoked through
`run_unlearning.py --unlearn_method selective_pruning` via `METHOD_REGISTRY`.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils import data as data_utils
from tqdm import tqdm

from dataloader.llm import LLMTrainDataset, worker_init_fn
from dataloader.utils import Prompter
from trainer.llm import llama_collate_fn_w_truncation

from .retain_ft import RetainFT


class SelectivePruning(RetainFT):
    """Forget-gradient Taylor pruning for downstream LLM ranker parameters.

    This is traditional static neural-network pruning: compute a parameter-level
    importance score from forget batches, choose a fixed fraction of weights,
    and hard-zero those weights. It does not learn masks, localize knowledge
    modules, prune LoRA ranks, or use retain/semantic/residual protection terms.
    """

    method_name = "selective_pruning"

    def run(self) -> Dict:
        forget_loader = self._build_forget_loader()
        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        prunable_named = self._configure_ranker_trainable_parameters()
        if not prunable_named:
            raise RuntimeError(
                "Selective Pruning found no downstream ranker parameters to prune. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        trainable_before = self._parameter_digest(prunable_named)
        trainable_before_by_name = self._parameter_digest_by_name(prunable_named)
        pruning_ratio = self._pruning_ratio()
        max_batches = int(getattr(self.args, "selective_pruning_max_batches", 0) or 0)
        device = self._training_device(prunable_named)

        self._accumulate_forget_gradients(
            forget_loader,
            prunable_named,
            device,
            max_batches=max_batches,
        )
        importance_named = self._taylor_importance(prunable_named)
        total_parameters = int(sum(score.numel() for _, score in importance_named))
        pruned_parameters, per_tensor_pruned = self._hard_prune_by_importance(
            prunable_named,
            importance_named,
            pruning_ratio,
        )

        self.model.zero_grad(set_to_none=True)
        for _, param in prunable_named:
            param.requires_grad_(False)
        self.model.eval()

        trainable_after = self._parameter_digest(prunable_named)
        trainable_after_by_name = self._parameter_digest_by_name(prunable_named)
        retriever_after = self._parameter_digest(self._named_retriever_parameters())
        changed_parameter_names = sorted(
            name
            for name, before_hash in trainable_before_by_name.items()
            if trainable_after_by_name.get(name) != before_hash
        )
        adapter_dir = self._save_pruned_ranker()
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})

        self.logs.update({
            "status": "completed",
            "method": self.method_name,
            "action": "traditional_parameter_level_selective_pruning",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "train_uses_forget_loader": True,
            "train_uses_retain_loader": False,
            "forget_loader_used_for_importance": True,
            "retain_loader_used_for_importance": False,
            "full_train_loader_used_for_importance": False,
            "optimizer_step_used": False,
            "importance_method": "taylor",
            "importance_formula": "abs(parameter * gradient_from_forget_loss)",
            "importance_source": "forget_interactions_only",
            "prune_direction": "highest_forget_taylor_importance",
            "pruning_granularity": "parameter_level_unstructured_zero_pruning",
            "rank_pruning_used": False,
            "learned_mask_used": False,
            "llm_eraser_logic_used": False,
            "semantic_or_residual_protection_used": False,
            "pruned_parameters": int(pruned_parameters),
            "total_parameters": int(total_parameters),
            "pruning_ratio": float(pruning_ratio),
            "actual_pruning_ratio": (
                float(pruned_parameters) / float(total_parameters)
                if total_parameters else 0.0
            ),
            "importance_batches": int(self.logs.get("importance_batches", 0)),
            "final_forget_loss": self.logs.get("final_forget_loss"),
            "mean_forget_loss": self.logs.get("mean_forget_loss"),
            "forget_loader_num_batches": int(len(forget_loader)),
            "forget_train_num_sequences": int(len(forget_loader.dataset)),
            "forget_train_num_users": int(len(self.dataset_data.get("forget_train", {}))),
            "max_importance_batches": int(max_batches),
            "split_counts": self._split_counts(),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get(
                "retain_train_excludes_forget_interactions"
            ),
            "forgotten_interactions_in_retain_train": split_diag.get(
                "forgotten_interactions_in_retain_train"
            ),
            "optimizer_parameter_scope": "downstream_llm_ranker_lora_or_ranker_parameters_only",
            "prunable_parameter_count": int(sum(param.numel() for _, param in prunable_named)),
            "prunable_parameter_tensors": int(len(prunable_named)),
            "prunable_parameter_name_sample": [name for name, _ in prunable_named[:20]],
            "per_tensor_pruned_parameter_sample": per_tensor_pruned[:20],
            "changed_parameter_tensors": int(len(changed_parameter_names)),
            "changed_parameter_name_sample": changed_parameter_names[:20],
            "trainable_checksum_before": trainable_before,
            "trainable_checksum_after": trainable_after,
            "checkpoint_changed_after_pruning": (
                trainable_before.get("sha256") != trainable_after.get("sha256")
            ),
            "retriever_parameter_tensors": int(retriever_before.get("num_tensors", 0)),
            "retriever_trainable_parameters_after_freeze": int(
                self._count_trainable(self._named_retriever_parameters())
            ),
            "retriever_checksum_before": retriever_before,
            "retriever_checksum_after": retriever_after,
            "retriever_unchanged": (
                retriever_before.get("sha256") == retriever_after.get("sha256")
            ),
            "updated_adapter_dir": adapter_dir,
            "notes": [
                "Importance is computed only from forget batches by backpropagating forget loss.",
                "No optimizer is created or stepped; gradients are used only for Taylor importance.",
                "Hard pruning is unstructured parameter-level zeroing, not LoRA rank pruning.",
                "No retain loss, semantic similarity, boundary residuals, learned soft masks, or LLM-Eraser localization are used.",
                "Retriever/candidate parameters remain frozen and are excluded from importance scoring and pruning.",
            ],
        })
        self.save_logs()
        return self.logs

    def _build_forget_loader(self):
        forget_train = self.dataset_data.get("forget_train")
        if not forget_train:
            raise ValueError("Selective Pruning requires dataset_data['forget_train'] to build forget_loader.")

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("Selective Pruning requires dataset_data['meta'] for LLM ranker prompts.")

        self._ensure_num_items()
        rng = np.random.RandomState(int(getattr(self.args, "seed", 42)))
        dataset = LLMTrainDataset(
            self.args,
            forget_train,
            int(getattr(self.args, "llm_max_history", 20)),
            rng,
            text_dict,
            self.tokenizer,
            Prompter(),
        )
        if len(dataset) == 0:
            raise ValueError(
                "Selective Pruning forget_loader is empty. "
                "The fixed forget split must contain enough per-user sequence context "
                "to form LLM ranker training samples."
            )
        batch_size = int(getattr(self.args, "lora_micro_batch_size", 16) or 16)
        loader = data_utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=(str(getattr(self.args, "device", "cpu")) == "cuda"),
            num_workers=int(getattr(self.args, "num_workers", 0) or 0),
            worker_init_fn=worker_init_fn,
        )
        loader.collate_fn = llama_collate_fn_w_truncation(
            int(getattr(self.args, "llm_max_text_len", 1536)),
            eval=False,
        )
        return loader

    def _accumulate_forget_gradients(
        self,
        forget_loader,
        prunable_named: List[Tuple[str, torch.nn.Parameter]],
        device: torch.device,
        *,
        max_batches: int = 0,
    ):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        loss_count = 0

        progress = tqdm(forget_loader, desc="Selective Pruning importance")
        for batch_idx, batch in enumerate(progress):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            batch = self._move_batch_to_device(batch, device)
            outputs = self.model(**batch)
            loss = self._extract_loss(outputs)
            if loss is None:
                raise RuntimeError("Ranker forward did not return a forget loss for pruning importance.")

            raw_loss = float(loss.detach().cpu())
            loss_sum += raw_loss
            loss_count += 1
            loss.backward()
            progress.set_postfix(loss=f"{raw_loss:.4f}")

        if loss_count == 0:
            raise RuntimeError("Selective Pruning did not process any forget batches.")

        self.logs["importance_batches"] = int(loss_count)
        self.logs["final_forget_loss"] = raw_loss
        self.logs["mean_forget_loss"] = float(loss_sum / loss_count)
        self.logs["gradient_parameter_tensors"] = int(
            sum(1 for _, param in prunable_named if param.grad is not None)
        )

    @staticmethod
    def _taylor_importance(prunable_named: List[Tuple[str, torch.nn.Parameter]]):
        importance_named = []
        for name, param in prunable_named:
            if param.grad is None:
                score = torch.zeros(param.numel(), dtype=torch.float32)
            else:
                score = (param.detach().float() * param.grad.detach().float()).abs().flatten().cpu()
            importance_named.append((name, score))
        return importance_named

    @staticmethod
    def _hard_prune_by_importance(
        prunable_named: List[Tuple[str, torch.nn.Parameter]],
        importance_named,
        pruning_ratio: float,
    ):
        all_scores = torch.cat([score for _, score in importance_named], dim=0)
        total_parameters = int(all_scores.numel())
        num_to_prune = int(total_parameters * pruning_ratio)
        if num_to_prune <= 0:
            return 0, []

        num_to_prune = min(num_to_prune, total_parameters)
        prune_indices = torch.topk(all_scores, k=num_to_prune, largest=True).indices
        global_prune_mask = torch.zeros(total_parameters, dtype=torch.bool)
        global_prune_mask[prune_indices] = True

        per_tensor_pruned = []
        offset = 0
        with torch.no_grad():
            for name, param in prunable_named:
                numel = param.numel()
                prune_mask = global_prune_mask[offset:offset + numel].view_as(param)
                pruned_here = int(prune_mask.sum().item())
                if pruned_here:
                    param.masked_fill_(prune_mask.to(device=param.device), 0.0)
                per_tensor_pruned.append({
                    "name": name,
                    "pruned_parameters": pruned_here,
                    "total_parameters": int(numel),
                })
                offset += numel

        return int(num_to_prune), per_tensor_pruned

    def _save_pruned_ranker(self):
        adapter_dir = os.path.join(self.output_dir, "selective_pruning_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None

    def _pruning_ratio(self) -> float:
        ratio = getattr(
            self.args,
            "selective_pruning_ratio",
            getattr(self.args, "random_prune_ratio", 0.05),
        )
        ratio = float(ratio or 0.0)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("Selective Pruning requires pruning ratio in [0, 1].")
        return ratio


class SelectivePruningMethod(SelectivePruning):
    """Compatibility class used by METHOD_REGISTRY."""
