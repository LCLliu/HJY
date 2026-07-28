"""Forget-set gradient-ascent unlearning strategy.

This module is a strategy module only. It is invoked through
`run_unlearning.py --unlearn_method gradient_ascent` via `METHOD_REGISTRY`.
"""

import os
from typing import Dict

import numpy as np
import torch
from torch.utils import data as data_utils
from tqdm import tqdm

from dataloader.llm import LLMTrainDataset, worker_init_fn
from dataloader.utils import Prompter
from trainer.llm import llama_collate_fn_w_truncation

from .retain_ft import RetainFT


class GradientAscentMethod(RetainFT):
    """Maximize forget-set ranker loss for downstream LLM unlearning.

    The implementation reuses Retain-FT's ranker training utilities, but trains
    only on the fixed forget split and reverses the optimization direction by
    backpropagating `-forget_loss`.
    """

    method_name = "gradient_ascent"

    def run(self) -> Dict:
        forget_loader = self._build_forget_loader()
        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        trainable_named = self._configure_ranker_trainable_parameters()
        if not trainable_named:
            raise RuntimeError(
                "Gradient Ascent found no downstream ranker parameters to optimize. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        trainable_before = self._parameter_digest(trainable_named)
        trainable_before_by_name = self._parameter_digest_by_name(trainable_named)
        optimizer = self._create_optimizer(trainable_named)

        device = self._training_device(trainable_named)
        epochs = int(
            getattr(
                self.args,
                "gradient_ascent_epochs",
                getattr(self.args, "retain_ft_epochs", getattr(self.args, "lora_num_epochs", 1)),
            )
            or 1
        )
        max_steps = int(
            getattr(
                self.args,
                "gradient_ascent_max_steps",
                getattr(self.args, "retain_ft_max_steps", 0),
            )
            or 0
        )
        grad_accum_steps = self._gradient_accumulation_steps()
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 5.0) or 5.0)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        num_steps = 0
        final_loss = None
        final_ascent_loss = None
        loss_sum = 0.0
        loss_count = 0

        for epoch in range(epochs):
            progress = tqdm(
                forget_loader,
                desc=f"Gradient Ascent epoch {epoch + 1}/{epochs}",
            )
            for batch_idx, batch in enumerate(progress):
                batch = self._move_batch_to_device(batch, device)
                outputs = self.model(**batch)
                forget_loss = self._extract_loss(outputs)
                if forget_loss is None:
                    raise RuntimeError("Ranker forward did not return a training loss.")

                raw_loss = float(forget_loss.detach().cpu())
                final_loss = raw_loss
                final_ascent_loss = -raw_loss
                loss_sum += raw_loss
                loss_count += 1

                ascent_loss = -forget_loss
                (ascent_loss / grad_accum_steps).backward()

                is_update_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(forget_loader)
                if is_update_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(
                        [param for _, param in trainable_named],
                        max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    num_steps += 1
                    progress.set_postfix(forget_loss=f"{raw_loss:.4f}", steps=num_steps)

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
            "action": "forget_set_gradient_ascent",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "train_uses_forget_loader": True,
            "train_uses_retain_loader": False,
            "forget_loader_used_for_training": True,
            "retain_loader_used_for_training": False,
            "full_train_loader_used_for_training": False,
            "optimization_objective": "maximize_forget_rank_loss",
            "gradient_direction_reversal": "ascent_loss = -forget_loss before backward",
            "num_steps": int(num_steps),
            "num_epochs": int(epochs),
            "max_steps": int(max_steps),
            "gradient_accumulation_steps": int(grad_accum_steps),
            "final_loss": final_loss,
            "final_forget_loss": final_loss,
            "final_ascent_loss": final_ascent_loss,
            "mean_forget_loss": float(loss_sum / loss_count) if loss_count else None,
            "forget_loader_num_batches": int(len(forget_loader)),
            "forget_train_num_sequences": int(len(forget_loader.dataset)),
            "forget_train_num_users": int(len(self.dataset_data.get("forget_train", {}))),
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
                "Training data is built from dataset_data['forget_train']; retain interactions are not used in the training loop.",
                "PyTorch still performs gradient descent, so the method backpropagates -forget_loss to ascend the forget loss.",
                "All parameters are frozen first, then only LoRA/ranker parameters are re-enabled for the optimizer.",
                "If a retriever module exists in the model, its parameters remain requires_grad=False and are excluded from optimizer groups.",
            ],
        })
        self.save_logs()
        return self.logs

    def _build_forget_loader(self):
        forget_train = self.dataset_data.get("forget_train")
        if not forget_train:
            raise ValueError("Gradient Ascent requires dataset_data['forget_train'] to build forget_loader.")

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("Gradient Ascent requires dataset_data['meta'] for LLM ranker prompts.")

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
                "Gradient Ascent forget_loader is empty. "
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

    def _save_updated_ranker(self):
        adapter_dir = os.path.join(self.output_dir, "gradient_ascent_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None
