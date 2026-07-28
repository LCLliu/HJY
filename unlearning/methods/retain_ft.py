"""Retain-set fine-tuning strategy.

This module is a strategy module only. It is invoked through
`run_unlearning.py --unlearn_method retain_ft` via `METHOD_REGISTRY`.
"""

import hashlib
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils import data as data_utils
from tqdm import tqdm

from dataloader.llm import LLMTrainDataset, worker_init_fn
from dataloader.utils import Prompter
from trainer.llm import llama_collate_fn_w_truncation

from .base import BaseUnlearningMethod


class RetainFT(BaseUnlearningMethod):
    """Retain-only fine-tuning for the downstream LLM ranker.

    The method starts from the checkpoint already loaded by `run_unlearning.py`
    and updates only downstream ranker trainable parameters. In the current
    LlamaRec ranker this means PEFT/LoRA adapter parameters; retriever and
    candidate-generation artifacts are not part of the optimizer.
    """

    method_name = "retain_ft"

    _RETRIEVER_OR_CANDIDATE_TOKENS = (
        "retriever",
        "retrieval",
        "candidate",
        "generator",
        "generation",
    )

    def run(self) -> Dict:
        retain_loader = self._build_retain_loader()
        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        trainable_named = self._configure_ranker_trainable_parameters()
        if not trainable_named:
            raise RuntimeError(
                "Retain-FT found no downstream ranker parameters to optimize. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        trainable_before = self._parameter_digest(trainable_named)
        trainable_before_by_name = self._parameter_digest_by_name(trainable_named)
        optimizer = self._create_optimizer(trainable_named)

        device = self._training_device(trainable_named)
        epochs = int(getattr(self.args, "retain_ft_epochs", getattr(self.args, "lora_num_epochs", 1)) or 1)
        max_steps = int(getattr(self.args, "retain_ft_max_steps", 0) or 0)
        grad_accum_steps = self._gradient_accumulation_steps()
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 5.0) or 5.0)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        num_steps = 0
        final_loss = None
        loss_sum = 0.0
        loss_count = 0

        for epoch in range(epochs):
            progress = tqdm(
                retain_loader,
                desc=f"Retain-FT epoch {epoch + 1}/{epochs}",
            )
            for batch_idx, batch in enumerate(progress):
                batch = self._move_batch_to_device(batch, device)
                outputs = self.model(**batch)
                loss = self._extract_loss(outputs)
                if loss is None:
                    raise RuntimeError("Ranker forward did not return a training loss.")

                raw_loss = float(loss.detach().cpu())
                final_loss = raw_loss
                loss_sum += raw_loss
                loss_count += 1
                (loss / grad_accum_steps).backward()

                is_update_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(retain_loader)
                if is_update_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(
                        [param for _, param in trainable_named],
                        max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    num_steps += 1
                    progress.set_postfix(loss=f"{raw_loss:.4f}", steps=num_steps)

                    if max_steps > 0 and num_steps >= max_steps:
                        break

            if max_steps > 0 and num_steps >= max_steps:
                break

        self.model.eval()
        trainable_after = self._parameter_digest(trainable_named)
        trainable_after_by_name = self._parameter_digest_by_name(trainable_named)
        retriever_after = self._parameter_digest(self._named_retriever_parameters())
        changed_parameter_names = sorted(
            name
            for name, before_hash in trainable_before_by_name.items()
            if trainable_after_by_name.get(name) != before_hash
        )
        adapter_dir = self._save_updated_ranker()
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})

        self.logs.update({
            "status": "completed",
            "method": self.method_name,
            "action": "retain_only_fine_tuning",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "train_uses_retain_loader": True,
            "forget_loader_used_for_training": False,
            "full_train_loader_used_for_training": False,
            "num_steps": int(num_steps),
            "num_epochs": int(epochs),
            "max_steps": int(max_steps),
            "gradient_accumulation_steps": int(grad_accum_steps),
            "final_loss": final_loss,
            "mean_loss": float(loss_sum / loss_count) if loss_count else None,
            "retain_loader_num_batches": int(len(retain_loader)),
            "retain_train_num_sequences": int(len(retain_loader.dataset)),
            "retain_train_num_users": int(len(self.dataset_data.get("retain_train", {}))),
            "split_counts": self._split_counts(),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get(
                "retain_train_excludes_forget_interactions"
            ),
            "forgotten_interactions_in_retain_train": split_diag.get(
                "forgotten_interactions_in_retain_train"
            ),
            "optimizer": self.logs.get("optimizer"),
            "optimizer_parameter_scope": "downstream_llm_ranker_lora_or_ranker_parameters_only",
            "updated_parameter_count": int(sum(param.numel() for _, param in trainable_named)),
            "updated_parameter_tensors": int(len(trainable_named)),
            "updated_parameter_name_sample": [name for name, _ in trainable_named[:20]],
            "changed_parameter_tensors": int(len(changed_parameter_names)),
            "changed_parameter_name_sample": changed_parameter_names[:20],
            "trainable_checksum_before": trainable_before,
            "trainable_checksum_after": trainable_after,
            "checkpoint_changed_after_training": (
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
                "Training data is built from dataset_data['retain_train']; forget interactions are not used in the training loop.",
                "All parameters are frozen first, then only LoRA/ranker parameters are re-enabled for the optimizer.",
                "If a retriever module exists in the model, its parameters remain requires_grad=False and are excluded from optimizer groups.",
            ],
        })
        self.save_logs()
        return self.logs

    def _build_retain_loader(self):
        retain_train = self.dataset_data.get("retain_train")
        if not retain_train:
            raise ValueError("Retain-FT requires dataset_data['retain_train'] to build retain_loader.")

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("Retain-FT requires dataset_data['meta'] for LLM ranker prompts.")

        self._ensure_num_items()
        rng = np.random.RandomState(int(getattr(self.args, "seed", 42)))
        dataset = LLMTrainDataset(
            self.args,
            retain_train,
            int(getattr(self.args, "llm_max_history", 20)),
            rng,
            text_dict,
            self.tokenizer,
            Prompter(),
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

    def _ensure_num_items(self):
        if getattr(self.args, "num_items", None):
            return
        smap = self.dataset_data.get("smap") or {}
        if smap:
            self.args.num_items = len(smap)
            return

        max_item = 0
        for key in ("train", "retain_train", "forget_train", "val", "test"):
            for seq in (self.dataset_data.get(key) or {}).values():
                if seq:
                    max_item = max(max_item, max(int(item) for item in seq))
        if max_item <= 0:
            raise ValueError("Cannot infer args.num_items for Retain-FT negative sampling.")
        self.args.num_items = max_item

    def _configure_ranker_trainable_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()

        named = list(self.model.named_parameters())
        for _, param in named:
            param.requires_grad_(False)

        lora_names = {
            name
            for name, param in named
            if self._is_lora_or_adapter_parameter(name) and not self._is_retriever_or_candidate_name(name)
        }
        ranker_names = {
            name
            for name, param in named
            if self._is_explicit_ranker_parameter(name) and not self._is_retriever_or_candidate_name(name)
        }

        if lora_names:
            selected_names = lora_names
            scope = "peft_lora_adapter_parameters"
        elif ranker_names:
            selected_names = ranker_names
            scope = "explicit_ranker_module_parameters"
        else:
            selected_names = {
                name
                for name, param in named
                if not self._is_retriever_or_candidate_name(name)
            }
            scope = "all_non_retriever_model_parameters"
            self.logs.setdefault("warnings", []).append(
                "No LoRA or explicit ranker parameter names were found; "
                "falling back to all non-retriever model parameters."
            )

        trainable = []
        for name, param in named:
            should_train = name in selected_names
            param.requires_grad_(should_train)
            if should_train:
                trainable.append((name, param))

        self.logs["ranker_update_scope"] = scope
        self.logs["all_model_parameter_tensors"] = int(len(named))
        self.logs["blocked_retriever_or_candidate_parameter_tensors"] = int(
            sum(1 for name, _ in named if self._is_retriever_or_candidate_name(name))
        )
        return trainable

    def _create_optimizer(self, trainable_named: List[Tuple[str, torch.nn.Parameter]]):
        no_decay = ("bias", "layer_norm")
        weight_decay = float(getattr(self.args, "weight_decay", 0.0) or 0.0)
        lr = float(getattr(self.args, "retain_ft_lr", getattr(self.args, "lora_lr", 1e-4)) or 1e-4)
        eps = float(getattr(self.args, "adam_epsilon", 1e-8) or 1e-8)
        grouped = [
            {
                "params": [
                    param
                    for name, param in trainable_named
                    if not any(token in name.lower() for token in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in trainable_named
                    if any(token in name.lower() for token in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        grouped = [group for group in grouped if group["params"]]

        optimizer_name = str(
            getattr(
                self.args,
                "retain_ft_optimizer",
                "paged_adamw_32bit" if bool(getattr(self.args, "llm_load_in_4bit", True)) else "adamw",
            )
        ).lower()

        if optimizer_name == "paged_adamw_32bit":
            try:
                import bitsandbytes as bnb

                optimizer = bnb.optim.PagedAdamW32bit(grouped, lr=lr, eps=eps)
            except Exception as exc:  # pragma: no cover - depends on local bnb install/CUDA.
                self.logs.setdefault("warnings", []).append(
                    f"bitsandbytes PagedAdamW32bit unavailable ({exc}); falling back to torch AdamW."
                )
                optimizer_name = "adamw"
                optimizer = torch.optim.AdamW(grouped, lr=lr, eps=eps)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(grouped, lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(grouped, lr=lr, eps=eps)
        else:
            raise ValueError(f"Unsupported Retain-FT optimizer: {optimizer_name}")

        self.logs["optimizer"] = optimizer_name
        self.logs["learning_rate"] = lr
        self.logs["weight_decay"] = weight_decay
        return optimizer

    def _gradient_accumulation_steps(self) -> int:
        train_batch_size = int(getattr(self.args, "train_batch_size", 16) or 16)
        micro_batch_size = int(getattr(self.args, "lora_micro_batch_size", train_batch_size) or train_batch_size)
        return max(1, train_batch_size // micro_batch_size)

    def _disable_cache_for_training(self):
        configs = [getattr(self.model, "config", None)]
        base_model = getattr(self.model, "base_model", None)
        configs.append(getattr(base_model, "config", None))
        for config in configs:
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

    def _freeze_named_modules(self, name_tokens: Iterable[str]):
        lowered_tokens = tuple(token.lower() for token in name_tokens)
        for module_name, module in self.model.named_modules():
            lowered = module_name.lower()
            if any(token in lowered for token in lowered_tokens):
                module.eval()
                for param in module.parameters(recurse=True):
                    param.requires_grad_(False)

    def _save_updated_ranker(self):
        adapter_dir = os.path.join(self.output_dir, "retain_ft_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None

    def _training_device(self, trainable_named: List[Tuple[str, torch.nn.Parameter]]) -> torch.device:
        for _, param in trainable_named:
            return param.device
        for param in self.model.parameters():
            return param.device
        requested = str(getattr(self.args, "device", "cpu"))
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _move_batch_to_device(batch, device):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    @staticmethod
    def _extract_loss(outputs):
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        if isinstance(outputs, dict):
            return outputs.get("loss")
        return None

    def _named_retriever_parameters(self) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in self.model.named_parameters()
            if self._is_retriever_or_candidate_name(name)
        ]

    def _is_lora_or_adapter_parameter(self, name: str) -> bool:
        lowered = name.lower()
        return "lora_" in lowered or "modules_to_save" in lowered

    def _is_explicit_ranker_parameter(self, name: str) -> bool:
        lowered = name.lower()
        return (
            lowered.startswith("ranker.")
            or ".ranker." in lowered
            or lowered.startswith("llm_ranker.")
            or ".llm_ranker." in lowered
        )

    def _is_retriever_or_candidate_name(self, name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in self._RETRIEVER_OR_CANDIDATE_TOKENS)

    @staticmethod
    def _count_trainable(named_params: List[Tuple[str, torch.nn.Parameter]]) -> int:
        return int(sum(param.numel() for _, param in named_params if param.requires_grad))

    def _parameter_digest(self, named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict:
        digest_by_name = self._parameter_digest_by_name(named_params)
        sha = hashlib.sha256()
        total_numel = 0
        total_abs = 0.0
        for name, param in named_params:
            sha.update(name.encode("utf-8"))
            sha.update(digest_by_name[name].encode("ascii"))
            total_numel += int(param.numel())
            total_abs += float(param.detach().float().abs().sum().cpu())
        return {
            "sha256": sha.hexdigest() if digest_by_name else None,
            "num_tensors": int(len(named_params)),
            "numel": int(total_numel),
            "abs_sum": float(total_abs),
        }

    @staticmethod
    def _parameter_digest_by_name(named_params: List[Tuple[str, torch.nn.Parameter]]) -> Dict[str, str]:
        digests = {}
        for name, param in named_params:
            tensor = param.detach().float().cpu().contiguous()
            sha = hashlib.sha256()
            sha.update(str(tuple(tensor.shape)).encode("ascii"))
            sha.update(tensor.numpy().tobytes())
            digests[name] = sha.hexdigest()
        return digests


class RetainFTMethod(RetainFT):
    """Compatibility class used by METHOD_REGISTRY."""

