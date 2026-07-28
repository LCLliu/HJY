"""LLM-Eraser style knowledge-localization and soft-mask pruning.

This module is a strategy module only. It is invoked through
`run_unlearning.py --unlearn_method llm_eraser` via `METHOD_REGISTRY`.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.utils import data as data_utils
from tqdm import tqdm

from dataloader.llm import LLMTrainDataset, worker_init_fn
from dataloader.utils import Prompter
from trainer.llm import llama_collate_fn_w_truncation

from .retain_ft import RetainFT


class _LocalizedSoftMask(nn.Module):
    """Elementwise soft mask applied only to localized parameter entries."""

    def __init__(self, localization_mask: torch.Tensor, init_logit: float):
        super().__init__()
        self.register_buffer("localization_mask", localization_mask.bool())
        self.mask_logits = nn.Parameter(
            torch.full(
                localization_mask.shape,
                float(init_logit),
                dtype=torch.float32,
                device=localization_mask.device,
            )
        )

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        soft_mask = torch.sigmoid(self.mask_logits).to(device=original.device, dtype=original.dtype)
        localization_mask = self.localization_mask.to(device=original.device)
        effective_mask = torch.where(
            localization_mask,
            soft_mask,
            torch.ones_like(soft_mask, dtype=original.dtype, device=original.device),
        )
        return original * effective_mask

    def localized_values(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)[self.localization_mask]


class LLMEraserMethod(RetainFT):
    """LLM-Eraser baseline for the downstream LLM ranker.

    The method simulates LLM-Eraser's pipeline:
    1. localize forget-related ranker parameters with forget-gradient Taylor scores;
    2. attach trainable soft masks to localized parameter entries while freezing
       original model weights;
    3. optimize only the masks to suppress forget interactions;
    4. convert low mask scores into hard unstructured parameter pruning.
    """

    method_name = "llm_eraser"

    def run(self) -> Dict:
        forget_loader = self._build_forget_loader()
        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        ranker_named = self._configure_ranker_trainable_parameters()
        if not ranker_named:
            raise RuntimeError(
                "LLM-Eraser found no downstream ranker parameters to process. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        ranker_before = self._parameter_digest(ranker_named)
        ranker_before_by_name = self._parameter_digest_by_name(ranker_named)
        pruning_ratio = self._pruning_ratio()
        localization_ratio = self._localization_ratio(pruning_ratio)
        total_parameters = int(sum(param.numel() for _, param in ranker_named))
        max_importance_batches = int(getattr(self.args, "llm_eraser_importance_max_batches", 0) or 0)
        device = self._training_device(ranker_named)

        self._accumulate_forget_gradients(
            forget_loader,
            ranker_named,
            device,
            max_batches=max_importance_batches,
        )
        importance_named = self._taylor_importance(ranker_named)
        localization_masks, localized_parameters = self._localize_parameters(
            ranker_named,
            importance_named,
            total_parameters,
            pruning_ratio,
            localization_ratio,
        )

        self.model.zero_grad(set_to_none=True)
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)

        mask_specs = self._register_soft_masks(ranker_named, localization_masks)
        mask_optimizer, mask_parameter_count = self._create_mask_optimizer(mask_specs)
        mask_summary = self._optimize_soft_masks(
            forget_loader,
            mask_specs,
            mask_optimizer,
            device,
        )
        mask_sparsity = self._mask_sparsity(mask_specs)
        pruned_parameters, per_tensor_pruned = self._hard_prune_from_masks(
            mask_specs,
            pruning_ratio,
            total_parameters,
        )
        self._remove_soft_masks(mask_specs)

        self.model.zero_grad(set_to_none=True)
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
        self.model.eval()

        ranker_after_named = self._refresh_named_parameters([name for name, _ in ranker_named])
        ranker_after = self._parameter_digest(ranker_after_named)
        ranker_after_by_name = self._parameter_digest_by_name(ranker_after_named)
        retriever_after = self._parameter_digest(self._named_retriever_parameters())
        changed_parameter_names = sorted(
            name
            for name, before_hash in ranker_before_by_name.items()
            if ranker_after_by_name.get(name) != before_hash
        )
        adapter_dir = self._save_erased_ranker()
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})

        self.logs.update({
            "status": "completed",
            "method": self.method_name,
            "action": "knowledge_localization_soft_mask_selective_pruning",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "train_uses_forget_loader": True,
            "train_uses_retain_loader": False,
            "forget_loader_used_for_localization": True,
            "retain_loader_used_for_localization": False,
            "full_train_loader_used_for_localization": False,
            "localization_method": "forget_gradient_taylor_importance",
            "importance_estimation": "abs(parameter * gradient_from_forget_loss)",
            "localized_parameters": int(localized_parameters),
            "total_parameters": int(total_parameters),
            "localization_ratio": float(localization_ratio),
            "soft_mask_used": True,
            "mask_parameter_count": int(mask_parameter_count),
            "mask_sparsity": float(mask_sparsity),
            "mask_optimization_steps": int(mask_summary["steps"]),
            "mask_final_loss": mask_summary["final_loss"],
            "mask_mean_loss": mask_summary["mean_loss"],
            "mask_mean_value": mask_summary["mean_mask_value"],
            "mask_min_value": mask_summary["min_mask_value"],
            "optimization_target": "soft_mask_only",
            "original_ranker_parameters_updated_by_optimizer": False,
            "pruned_parameters": int(pruned_parameters),
            "pruning_ratio": float(pruning_ratio),
            "actual_pruning_ratio": (
                float(pruned_parameters) / float(total_parameters)
                if total_parameters else 0.0
            ),
            "pruning_granularity": "parameter_level_unstructured_zero_pruning_after_soft_mask",
            "per_tensor_pruned_parameter_sample": per_tensor_pruned[:20],
            "rank_pruning_used": False,
            "semantic_or_residual_protection_used": False,
            "boundary_sensitivity_used": False,
            "retain_overlap_analysis_used": False,
            "forget_loader_num_batches": int(len(forget_loader)),
            "forget_train_num_sequences": int(len(forget_loader.dataset)),
            "forget_train_num_users": int(len(self.dataset_data.get("forget_train", {}))),
            "importance_batches": int(self.logs.get("importance_batches", 0)),
            "importance_final_forget_loss": self.logs.get("importance_final_forget_loss"),
            "importance_mean_forget_loss": self.logs.get("importance_mean_forget_loss"),
            "split_counts": self._split_counts(),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get(
                "retain_train_excludes_forget_interactions"
            ),
            "forgotten_interactions_in_retain_train": split_diag.get(
                "forgotten_interactions_in_retain_train"
            ),
            "optimizer_parameter_scope": "trainable_soft_masks_only_over_downstream_llm_ranker_parameters",
            "ranker_parameter_tensors": int(len(ranker_named)),
            "ranker_parameter_name_sample": [name for name, _ in ranker_named[:20]],
            "changed_parameter_tensors": int(len(changed_parameter_names)),
            "changed_parameter_name_sample": changed_parameter_names[:20],
            "ranker_checksum_before": ranker_before,
            "ranker_checksum_after": ranker_after,
            "checkpoint_changed_after_pruning": (
                ranker_before.get("sha256") != ranker_after.get("sha256")
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
                "Forget-gradient Taylor scores localize forget-related ranker parameters before masks are trained.",
                "Original LLM/ranker parameters are frozen during soft-mask optimization; only mask logits are optimized.",
                "Hard pruning is applied after mask optimization by zeroing localized parameters with the lowest learned mask scores.",
                "No retain loss, residual localization, semantic protection, boundary sensitivity, overlap analysis, or LoRA rank geometry pruning is used.",
                "Retriever/candidate parameters remain frozen and are excluded from localization, mask optimization, and pruning.",
            ],
        })
        self.save_logs()
        return self.logs

    def _build_forget_loader(self):
        forget_train = self.dataset_data.get("forget_train")
        if not forget_train:
            raise ValueError("LLM-Eraser requires dataset_data['forget_train'] to build forget_loader.")

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("LLM-Eraser requires dataset_data['meta'] for LLM ranker prompts.")

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
                "LLM-Eraser forget_loader is empty. "
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
        ranker_named: List[Tuple[str, torch.nn.Parameter]],
        device: torch.device,
        *,
        max_batches: int = 0,
    ):
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0.0
        loss_count = 0

        progress = tqdm(forget_loader, desc="LLM-Eraser localization")
        for batch_idx, batch in enumerate(progress):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            batch = self._move_batch_to_device(batch, device)
            outputs = self.model(**batch)
            loss = self._extract_loss(outputs)
            if loss is None:
                raise RuntimeError("Ranker forward did not return a forget loss for LLM-Eraser localization.")

            raw_loss = float(loss.detach().cpu())
            loss_sum += raw_loss
            loss_count += 1
            loss.backward()
            progress.set_postfix(loss=f"{raw_loss:.4f}")

        if loss_count == 0:
            raise RuntimeError("LLM-Eraser did not process any forget batches for localization.")

        self.logs["importance_batches"] = int(loss_count)
        self.logs["importance_final_forget_loss"] = raw_loss
        self.logs["importance_mean_forget_loss"] = float(loss_sum / loss_count)
        self.logs["gradient_parameter_tensors"] = int(
            sum(1 for _, param in ranker_named if param.grad is not None)
        )

    @staticmethod
    def _taylor_importance(ranker_named: List[Tuple[str, torch.nn.Parameter]]):
        importance_named = []
        for name, param in ranker_named:
            if param.grad is None:
                score = torch.zeros(param.numel(), dtype=torch.float32)
            else:
                score = (param.detach().float() * param.grad.detach().float()).abs().flatten().cpu()
            importance_named.append((name, score))
        return importance_named

    @staticmethod
    def _localize_parameters(
        ranker_named: List[Tuple[str, torch.nn.Parameter]],
        importance_named,
        total_parameters: int,
        pruning_ratio: float,
        localization_ratio: float,
    ):
        all_scores = torch.cat([score for _, score in importance_named], dim=0)
        requested_localized = int(total_parameters * localization_ratio)
        requested_pruned = int(total_parameters * pruning_ratio)
        num_to_localize = min(total_parameters, max(requested_localized, requested_pruned))
        if num_to_localize <= 0:
            return {
                name: torch.zeros_like(param, dtype=torch.bool, device="cpu")
                for name, param in ranker_named
            }, 0

        localized_indices = torch.topk(all_scores, k=num_to_localize, largest=True).indices
        global_localization_mask = torch.zeros(total_parameters, dtype=torch.bool)
        global_localization_mask[localized_indices] = True

        masks = {}
        offset = 0
        for name, param in ranker_named:
            numel = param.numel()
            masks[name] = global_localization_mask[offset:offset + numel].view_as(param).clone()
            offset += numel
        return masks, int(num_to_localize)

    def _register_soft_masks(self, ranker_named, localization_masks):
        modules = dict(self.model.named_modules())
        init_logit = float(getattr(self.args, "llm_eraser_mask_init_logit", 2.0) or 2.0)
        specs = []

        for name, param in ranker_named:
            localization_mask = localization_masks.get(name)
            if localization_mask is None or not bool(localization_mask.any()):
                continue
            if "." not in name:
                raise ValueError(f"Cannot register LLM-Eraser mask for top-level parameter: {name}")

            module_name, parameter_name = name.rsplit(".", 1)
            module = modules.get(module_name)
            if module is None:
                raise ValueError(f"Cannot find module for parameter: {name}")
            if parametrize.is_parametrized(module, parameter_name):
                raise RuntimeError(f"Parameter is already parametrized before LLM-Eraser masking: {name}")

            mask_module = _LocalizedSoftMask(
                localization_mask.to(device=param.device),
                init_logit=init_logit,
            )
            parametrize.register_parametrization(module, parameter_name, mask_module)
            module.parametrizations[parameter_name].original.requires_grad_(False)
            specs.append({
                "name": name,
                "module": module,
                "parameter_name": parameter_name,
                "mask_module": mask_module,
                "localized_parameters": int(localization_mask.sum().item()),
            })

        if not specs and sum(int(mask.sum().item()) for mask in localization_masks.values()) > 0:
            raise RuntimeError("LLM-Eraser localization selected parameters, but no soft masks were registered.")
        return specs

    def _create_mask_optimizer(self, mask_specs):
        mask_parameters = [spec["mask_module"].mask_logits for spec in mask_specs]
        if not mask_parameters:
            return None, 0
        lr = float(getattr(self.args, "llm_eraser_mask_lr", 1e-2) or 1e-2)
        optimizer = torch.optim.Adam(mask_parameters, lr=lr)
        self.logs["mask_optimizer"] = "adam"
        self.logs["mask_learning_rate"] = lr
        return optimizer, int(sum(param.numel() for param in mask_parameters))

    def _optimize_soft_masks(self, forget_loader, mask_specs, optimizer, device):
        if optimizer is None:
            return {
                "steps": 0,
                "final_loss": None,
                "mean_loss": None,
                "mean_mask_value": 0.0,
                "min_mask_value": 0.0,
            }

        epochs = int(getattr(self.args, "llm_eraser_mask_epochs", 1) or 1)
        max_steps = int(getattr(self.args, "llm_eraser_mask_max_steps", 0) or 0)
        sparsity_lambda = float(getattr(self.args, "llm_eraser_mask_l1", 1e-4) or 1e-4)
        grad_accum_steps = self._gradient_accumulation_steps()
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 5.0) or 5.0)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        steps = 0
        final_loss = None
        loss_sum = 0.0
        loss_count = 0

        for epoch in range(epochs):
            progress = tqdm(forget_loader, desc=f"LLM-Eraser mask epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(progress):
                batch = self._move_batch_to_device(batch, device)
                outputs = self.model(**batch)
                forget_loss = self._extract_loss(outputs)
                if forget_loss is None:
                    raise RuntimeError("Ranker forward did not return a forget loss for soft-mask optimization.")

                mask_mean = self._localized_mask_mean(mask_specs)
                loss = -forget_loss + sparsity_lambda * mask_mean
                raw_loss = float(loss.detach().cpu())
                final_loss = raw_loss
                loss_sum += raw_loss
                loss_count += 1
                (loss / grad_accum_steps).backward()

                is_update_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(forget_loader)
                if is_update_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(
                        [spec["mask_module"].mask_logits for spec in mask_specs],
                        max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    progress.set_postfix(loss=f"{raw_loss:.4f}", mask=f"{float(mask_mean.detach().cpu()):.4f}")

                    if max_steps > 0 and steps >= max_steps:
                        break

            if max_steps > 0 and steps >= max_steps:
                break

        mask_values = self._all_localized_mask_values(mask_specs)
        return {
            "steps": int(steps),
            "final_loss": final_loss,
            "mean_loss": float(loss_sum / loss_count) if loss_count else None,
            "mean_mask_value": float(mask_values.mean().cpu()) if mask_values.numel() else 0.0,
            "min_mask_value": float(mask_values.min().cpu()) if mask_values.numel() else 0.0,
        }

    @staticmethod
    def _localized_mask_mean(mask_specs):
        values = [
            spec["mask_module"].localized_values()
            for spec in mask_specs
            if spec["mask_module"].localized_values().numel()
        ]
        if not values:
            return torch.tensor(0.0)
        return torch.cat(values, dim=0).mean()

    @staticmethod
    def _all_localized_mask_values(mask_specs):
        values = [
            spec["mask_module"].localized_values().detach().flatten().cpu()
            for spec in mask_specs
            if spec["mask_module"].localized_values().numel()
        ]
        return torch.cat(values, dim=0) if values else torch.empty(0)

    def _mask_sparsity(self, mask_specs):
        threshold = float(getattr(self.args, "llm_eraser_mask_sparsity_threshold", 0.5) or 0.5)
        values = self._all_localized_mask_values(mask_specs)
        if not values.numel():
            return 0.0
        return float((values <= threshold).float().mean().item())

    @staticmethod
    def _hard_prune_from_masks(mask_specs, pruning_ratio: float, total_parameters: int):
        target_pruned = int(total_parameters * pruning_ratio)
        if target_pruned <= 0 or not mask_specs:
            return 0, []

        values_by_spec = []
        all_values = []
        for spec in mask_specs:
            mask_module = spec["mask_module"]
            flat_values = torch.sigmoid(mask_module.mask_logits.detach()).flatten().cpu()
            flat_localized = mask_module.localization_mask.detach().flatten().cpu()
            localized_indices = torch.nonzero(flat_localized, as_tuple=False).flatten()
            localized_values = flat_values[localized_indices]
            values_by_spec.append((spec, localized_indices, localized_values))
            all_values.append(localized_values)

        if not all_values:
            return 0, []

        all_values = torch.cat(all_values, dim=0)
        num_to_prune = min(int(target_pruned), int(all_values.numel()))
        prune_indices = torch.topk(all_values, k=num_to_prune, largest=False).indices
        global_prune_mask = torch.zeros(all_values.numel(), dtype=torch.bool)
        global_prune_mask[prune_indices] = True

        per_tensor_pruned = []
        offset = 0
        with torch.no_grad():
            for spec, localized_indices, localized_values in values_by_spec:
                local_count = int(localized_values.numel())
                local_prune = global_prune_mask[offset:offset + local_count]
                full_prune_flat = torch.zeros_like(
                    spec["mask_module"].localization_mask.flatten().cpu(),
                    dtype=torch.bool,
                )
                if local_count:
                    full_prune_flat[localized_indices[local_prune]] = True
                prune_mask = full_prune_flat.view_as(spec["mask_module"].localization_mask)
                original = spec["module"].parametrizations[spec["parameter_name"]].original
                pruned_here = int(prune_mask.sum().item())
                if pruned_here:
                    original.masked_fill_(prune_mask.to(device=original.device), 0.0)
                per_tensor_pruned.append({
                    "name": spec["name"],
                    "pruned_parameters": pruned_here,
                    "localized_parameters": int(spec["localized_parameters"]),
                    "total_parameters": int(original.numel()),
                })
                offset += local_count

        return int(num_to_prune), per_tensor_pruned

    @staticmethod
    def _remove_soft_masks(mask_specs):
        for spec in mask_specs:
            parametrize.remove_parametrizations(
                spec["module"],
                spec["parameter_name"],
                leave_parametrized=False,
            )

    def _save_erased_ranker(self):
        adapter_dir = os.path.join(self.output_dir, "llm_eraser_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None

    def _refresh_named_parameters(self, names: List[str]):
        current = dict(self.model.named_parameters())
        return [(name, current[name]) for name in names if name in current]

    def _pruning_ratio(self) -> float:
        ratio = getattr(
            self.args,
            "llm_eraser_pruning_ratio",
            getattr(self.args, "random_prune_ratio", 0.05),
        )
        ratio = float(ratio or 0.0)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("LLM-Eraser requires pruning ratio in [0, 1].")
        return ratio

    def _localization_ratio(self, pruning_ratio: float) -> float:
        default_ratio = min(1.0, max(pruning_ratio, 2.0 * pruning_ratio))
        ratio = float(getattr(self.args, "llm_eraser_localization_ratio", default_ratio) or 0.0)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("LLM-Eraser requires localization ratio in [0, 1].")
        return max(ratio, pruning_ratio)
