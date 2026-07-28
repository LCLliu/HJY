"""Negative Preference Optimization unlearning strategy.

This module is a strategy module only. It is invoked through
`run_unlearning.py --unlearn_method npo` via `METHOD_REGISTRY`.
"""

import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data as data_utils
from tqdm import tqdm

from dataloader.llm import LLMTrainDataset, worker_init_fn
from dataloader.utils import Prompter
from trainer.llm import llama_collate_fn_w_truncation

from .retain_ft import RetainFT


class NPOMethod(RetainFT):
    """Negative Preference Optimization for forget interactions.

    NPO treats each forget interaction as a negative preference target. It
    compares the current ranker's log-probability for the forget target against
    a frozen reference snapshot from the original checkpoint, then optimizes a
    preference loss that pushes the current probability below the reference.
    """

    method_name = "npo"

    def run(self) -> Dict:
        forget_loader = self._build_forget_loader()
        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        trainable_named = self._configure_ranker_trainable_parameters()
        if not trainable_named:
            raise RuntimeError(
                "NPO found no downstream ranker parameters to optimize. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        trainable_before = self._parameter_digest(trainable_named)
        trainable_before_by_name = self._parameter_digest_by_name(trainable_named)
        reference_state = self._capture_reference_state(trainable_named)
        optimizer = self._create_optimizer(trainable_named)

        device = self._training_device(trainable_named)
        epochs = int(
            getattr(
                self.args,
                "npo_epochs",
                getattr(self.args, "retain_ft_epochs", getattr(self.args, "lora_num_epochs", 1)),
            )
            or 1
        )
        max_steps = int(
            getattr(
                self.args,
                "npo_max_steps",
                getattr(self.args, "retain_ft_max_steps", 0),
            )
            or 0
        )
        beta = float(getattr(self.args, "npo_beta", 0.1) or 0.1)
        if beta <= 0:
            raise ValueError("NPO requires npo_beta > 0.")
        grad_accum_steps = self._gradient_accumulation_steps()
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 5.0) or 5.0)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        num_steps = 0
        final_loss = None
        final_log_prob_current = None
        final_log_prob_reference = None
        final_log_prob_delta = None
        loss_sum = 0.0
        loss_count = 0

        for epoch in range(epochs):
            progress = tqdm(
                forget_loader,
                desc=f"NPO epoch {epoch + 1}/{epochs}",
            )
            for batch_idx, batch in enumerate(progress):
                batch = self._move_batch_to_device(batch, device)

                log_prob_reference = self._reference_log_probability(
                    batch,
                    trainable_named,
                    reference_state,
                )

                self.model.train()
                outputs = self.model(**batch)
                log_prob_current = self._log_probability_from_outputs(outputs)
                log_prob_delta = log_prob_current - log_prob_reference.detach()
                loss = self._npo_loss(log_prob_delta, beta)

                raw_loss = float(loss.detach().cpu())
                final_loss = raw_loss
                final_log_prob_current = float(log_prob_current.detach().cpu())
                final_log_prob_reference = float(log_prob_reference.detach().cpu())
                final_log_prob_delta = float(log_prob_delta.detach().cpu())
                loss_sum += raw_loss
                loss_count += 1

                (loss / grad_accum_steps).backward()

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
            "action": "negative_preference_optimization",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "train_uses_forget_loader": True,
            "train_uses_retain_loader": False,
            "forget_loader_used_for_training": True,
            "retain_loader_used_for_training": False,
            "full_train_loader_used_for_training": False,
            "optimization_objective": "negative_preference_against_frozen_reference",
            "npo_loss_formula": "-2/beta * logsigmoid(-beta/2 * (logp_current - logp_reference))",
            "npo_beta": beta,
            "log_probability_scope": "batch_mean_labeled_token_log_probability",
            "reference_model": "frozen_original_checkpoint_ranker_parameter_snapshot",
            "reference_requires_grad": False,
            "num_steps": int(num_steps),
            "num_epochs": int(epochs),
            "max_steps": int(max_steps),
            "gradient_accumulation_steps": int(grad_accum_steps),
            "final_loss": final_loss,
            "mean_loss": float(loss_sum / loss_count) if loss_count else None,
            "final_log_prob_current": final_log_prob_current,
            "final_log_prob_reference": final_log_prob_reference,
            "final_log_prob_delta": final_log_prob_delta,
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
            "reference_checksum": trainable_before,
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
                "Forget interactions are used as negative preference targets; retain interactions are not optimized.",
                "The reference ranker is the original checkpoint state captured before any optimizer step.",
                "Only the current model receives gradients; reference log-probability is computed under no_grad with trainable parameters temporarily restored to the frozen snapshot.",
                "All parameters are frozen first, then only LoRA/ranker parameters are re-enabled for the optimizer.",
                "If a retriever module exists in the model, its parameters remain requires_grad=False and are excluded from optimizer groups.",
            ],
        })
        self.save_logs()
        return self.logs

    def _build_forget_loader(self):
        forget_train = self.dataset_data.get("forget_train")
        if not forget_train:
            raise ValueError("NPO requires dataset_data['forget_train'] to build forget_loader.")

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("NPO requires dataset_data['meta'] for LLM ranker prompts.")

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
                "NPO forget_loader is empty. "
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
        adapter_dir = os.path.join(self.output_dir, "npo_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None

    @staticmethod
    def _capture_reference_state(trainable_named):
        return {
            name: param.detach().cpu().clone()
            for name, param in trainable_named
        }

    def _reference_log_probability(self, batch, trainable_named, reference_state):
        current_values = [
            (param, param.detach().clone())
            for _, param in trainable_named
        ]
        requires_grad_flags = [
            (param, param.requires_grad)
            for _, param in trainable_named
        ]
        module_training_flags = [
            (module, module.training)
            for module in self.model.modules()
        ]

        try:
            with torch.no_grad():
                for name, param in trainable_named:
                    param.requires_grad_(False)
                    param.copy_(reference_state[name].to(device=param.device, dtype=param.dtype))

                self._set_reference_eval_mode_with_loss_branch()
                outputs = self.model(**batch)
                return self._log_probability_from_outputs(outputs).detach()
        finally:
            with torch.no_grad():
                for param, current_value in current_values:
                    param.copy_(current_value)
            for param, requires_grad in requires_grad_flags:
                param.requires_grad_(requires_grad)
            for module, was_training in module_training_flags:
                module.training = was_training

    def _set_reference_eval_mode_with_loss_branch(self):
        self.model.eval()
        toggled = 0
        for module in self.model.modules():
            if self._is_causal_lm_loss_module(module):
                module.training = True
                toggled += 1
        if toggled == 0:
            self.model.training = True
            self.logs.setdefault("warnings", []).append(
                "Could not identify the causal LM loss module for reference forward; "
                "set only the top-level model.training flag to compute ranker loss."
            )

    @staticmethod
    def _is_causal_lm_loss_module(module) -> bool:
        class_name = module.__class__.__name__.lower()
        return (
            "causallm" in class_name
            and hasattr(module, "lm_head")
            and hasattr(module, "config")
        )

    def _log_probability_from_outputs(self, outputs):
        loss = self._extract_loss(outputs)
        if loss is None:
            raise RuntimeError("Ranker forward did not return a training loss for NPO log-probability.")
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss)
        raw_loss = float(loss.detach().cpu())
        if raw_loss < 0:
            raise RuntimeError(
                "Ranker forward returned a negative sentinel loss; "
                "NPO requires a real labeled-token loss to compute log-probability."
            )
        return -loss

    @staticmethod
    def _npo_loss(log_prob_delta, beta: float):
        return -(2.0 / beta) * F.logsigmoid(-0.5 * beta * log_prob_delta).mean()
