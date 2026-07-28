"""RecEraser-style shard retraining baseline for LlamaRec unlearning.

This module is a strategy module only. It follows the BaseUnlearningMethod
class interface and does not modify the framework registry by itself.
"""

import hashlib
import json
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


class RecEraserMethod(RetainFT):
    """RecEraser adaptation for downstream LLM ranker unlearning.

    The baseline partitions interaction rows into shards. A forget request only
    retrains shard ranker states whose shard contains a forget interaction; all
    other shard states remain equal to the original checkpoint. The final ranker
    is the parameter average of unchanged shard states and retrained affected
    shard states.
    """

    method_name = "receraser"

    def run(self) -> Dict:
        num_shards = self._num_shards()
        retain_rows = self._interaction_rows("retain")
        forget_rows = self._interaction_rows("forget")
        if not retain_rows:
            raise ValueError("RecEraser requires retain interactions to retrain affected shards.")
        if not forget_rows:
            raise ValueError("RecEraser requires forget interactions to locate affected shards.")

        shard_assignments, affected_shards = self._assign_interactions_to_shards(
            retain_rows,
            forget_rows,
            num_shards,
        )
        shard_retain_sequences = self._build_retain_sequences_by_shard(retain_rows, num_shards)
        assignment_path = self._save_shard_assignments(shard_assignments, num_shards)

        self._disable_cache_for_training()
        self._freeze_named_modules(self._RETRIEVER_OR_CANDIDATE_TOKENS)

        retriever_before = self._parameter_digest(self._named_retriever_parameters())
        ranker_named = self._configure_ranker_trainable_parameters()
        if not ranker_named:
            raise RuntimeError(
                "RecEraser found no downstream ranker parameters to train. "
                "Expected LoRA adapter parameters or an explicit ranker module."
            )

        original_ranker = self._capture_parameter_state(ranker_named)
        original_digest = self._parameter_digest(ranker_named)
        original_digest_by_name = self._parameter_digest_by_name(ranker_named)

        device = self._training_device(ranker_named)
        shard_states = {}
        shard_training_logs = []
        retrained_steps = 0

        for shard_id in affected_shards:
            self._load_parameter_state(ranker_named, original_ranker)
            for _, param in ranker_named:
                param.requires_grad_(True)

            train_log = self._retrain_affected_shard(
                shard_id,
                shard_retain_sequences.get(shard_id, {}),
                ranker_named,
                device,
            )
            retrained_steps += int(train_log["num_steps"])
            shard_training_logs.append(train_log)
            shard_states[shard_id] = self._capture_parameter_state(ranker_named)

        self._aggregate_shard_states(
            ranker_named,
            original_ranker,
            shard_states,
            num_shards,
        )

        for _, param in ranker_named:
            param.requires_grad_(False)
        self.model.eval()

        ranker_after = self._parameter_digest(ranker_named)
        ranker_after_by_name = self._parameter_digest_by_name(ranker_named)
        retriever_after = self._parameter_digest(self._named_retriever_parameters())
        changed_parameter_names = sorted(
            name
            for name, before_hash in original_digest_by_name.items()
            if ranker_after_by_name.get(name) != before_hash
        )
        adapter_dir = self._save_receraser_ranker()
        unaffected_shards = [idx for idx in range(num_shards) if idx not in set(affected_shards)]
        split_diag = self.dataset_data.get("unlearning_split_diagnostics", {})

        self.logs.update({
            "status": "completed",
            "method": self.method_name,
            "action": "receraser_shard_retraining_and_ranker_aggregation",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "num_shards": int(num_shards),
            "affected_shards": [int(idx) for idx in affected_shards],
            "num_affected_shards": int(len(affected_shards)),
            "unaffected_shards": [int(idx) for idx in unaffected_shards],
            "num_unaffected_shards": int(len(unaffected_shards)),
            "retrained_steps": int(retrained_steps),
            "optimizer_parameter_scope": "downstream_llm_ranker_only",
            "shard_partition_method": "stable_hash_user_position_item_mod_num_shards",
            "shard_assignment_path": assignment_path,
            "interaction_to_shard_recorded": True,
            "affected_shard_training_data": "retain_interactions_assigned_to_affected_shards_only",
            "unaffected_shards_kept_unchanged": True,
            "aggregation_method": "uniform_average_of_shard_ranker_parameters",
            "direct_full_ranker_finetuning_used": False,
            "gradient_ascent_used": False,
            "preference_optimization_used": False,
            "parameter_pruning_used": False,
            "mask_optimization_used": False,
            "ranker_parameter_tensors": int(len(ranker_named)),
            "ranker_parameter_count": int(sum(param.numel() for _, param in ranker_named)),
            "ranker_parameter_name_sample": [name for name, _ in ranker_named[:20]],
            "changed_parameter_tensors": int(len(changed_parameter_names)),
            "changed_parameter_name_sample": changed_parameter_names[:20],
            "ranker_checksum_before": original_digest,
            "ranker_checksum_after": ranker_after,
            "checkpoint_changed_after_receraser": (
                original_digest.get("sha256") != ranker_after.get("sha256")
            ),
            "unaffected_shard_reference_checksum": original_digest,
            "unaffected_shards_unchanged": True,
            "shard_training_logs": shard_training_logs,
            "split_counts": self._split_counts(),
            "retain_train_loaded_from_split": split_diag.get("retain_train_loaded_from_split"),
            "retain_train_excludes_forget_interactions": split_diag.get(
                "retain_train_excludes_forget_interactions"
            ),
            "forgotten_interactions_in_retain_train": split_diag.get(
                "forgotten_interactions_in_retain_train"
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
                "Interactions are assigned to deterministic shards using user, position, and item identifiers.",
                "Forget rows only determine affected shard ids; they are not used as optimization targets.",
                "Only affected shard ranker states are retrained on retained interactions assigned to those shards.",
                "Unaffected shard states are represented by the original checkpoint snapshot and are never trained.",
                "The final downstream ranker is the uniform average of retrained affected shard states and unchanged original shard states.",
                "Retriever/candidate parameters remain frozen and are excluded from shard retraining.",
            ],
        })
        self.save_logs()
        return self.logs

    def _interaction_rows(self, split_name: str) -> List[Dict]:
        key = f"{split_name}_interactions"
        rows = self.split_data.get(key) or []
        if rows:
            return [self._normalize_row(row, split_name) for row in rows]

        data_key = "retain_train" if split_name == "retain" else "forget_train"
        sequences = self.dataset_data.get(data_key, {})
        fallback_rows = []
        for uid, seq in sequences.items():
            for pos, iid in enumerate(seq):
                fallback_rows.append({
                    "user_id": int(uid),
                    "item_id": int(iid),
                    "position": int(pos),
                    "sequence_index": int(pos),
                    "split_name": split_name,
                })
        return fallback_rows

    @staticmethod
    def _normalize_row(row: Dict, split_name: str) -> Dict:
        uid = row.get("user_id", row.get("uid"))
        iid = row.get("item_id", row.get("iid"))
        position = row.get("position", row.get("sequence_index"))
        return {
            **row,
            "user_id": int(uid),
            "item_id": int(iid),
            "position": None if position in (None, "", "null") else int(position),
            "split_name": row.get("split_name", split_name) or split_name,
        }

    def _assign_interactions_to_shards(self, retain_rows, forget_rows, num_shards: int):
        assignments = []
        affected = set()
        for split_name, rows in (("retain", retain_rows), ("forget", forget_rows)):
            for row in rows:
                shard = self._shard_for_row(row, num_shards)
                record = {
                    "interaction_key": self._interaction_key(row),
                    "split_name": split_name,
                    "user_id": int(row["user_id"]),
                    "item_id": int(row["item_id"]),
                    "position": row.get("position"),
                    "shard": int(shard),
                }
                assignments.append(record)
                if split_name == "forget":
                    affected.add(int(shard))
        return assignments, sorted(affected)

    def _build_retain_sequences_by_shard(self, retain_rows, num_shards: int):
        shard_items = {idx: {} for idx in range(num_shards)}
        for row_idx, row in enumerate(retain_rows):
            shard = self._shard_for_row(row, num_shards)
            uid = int(row["user_id"])
            position = row.get("position")
            order = int(position) if position is not None else int(row_idx)
            shard_items[shard].setdefault(uid, []).append((order, int(row["item_id"])))

        shard_sequences = {}
        for shard, user_items in shard_items.items():
            shard_sequences[shard] = {
                uid: [iid for _, iid in sorted(items, key=lambda pair: pair[0])]
                for uid, items in user_items.items()
            }
        return shard_sequences

    def _retrain_affected_shard(self, shard_id, shard_train, ranker_named, device):
        loader = self._build_shard_loader(shard_train)
        if loader is None:
            return {
                "shard": int(shard_id),
                "num_steps": 0,
                "num_sequences": 0,
                "num_users": int(len(shard_train)),
                "final_loss": None,
                "mean_loss": None,
                "skipped": True,
                "skip_reason": "shard has no ranker training sequences after forgetting",
            }

        optimizer = self._create_optimizer(ranker_named)
        epochs = int(getattr(self.args, "receraser_epochs", getattr(self.args, "lora_num_epochs", 1)) or 1)
        max_steps = int(getattr(self.args, "receraser_max_steps", 0) or 0)
        grad_accum_steps = self._gradient_accumulation_steps()
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 5.0) or 5.0)

        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        num_steps = 0
        final_loss = None
        loss_sum = 0.0
        loss_count = 0

        for epoch in range(epochs):
            progress = tqdm(loader, desc=f"RecEraser shard {shard_id} epoch {epoch + 1}/{epochs}")
            for batch_idx, batch in enumerate(progress):
                batch = self._move_batch_to_device(batch, device)
                outputs = self.model(**batch)
                loss = self._extract_loss(outputs)
                if loss is None:
                    raise RuntimeError("Ranker forward did not return a shard retraining loss.")

                raw_loss = float(loss.detach().cpu())
                final_loss = raw_loss
                loss_sum += raw_loss
                loss_count += 1
                (loss / grad_accum_steps).backward()

                is_update_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(loader)
                if is_update_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(
                        [param for _, param in ranker_named],
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

        return {
            "shard": int(shard_id),
            "num_steps": int(num_steps),
            "num_sequences": int(len(loader.dataset)),
            "num_users": int(len(shard_train)),
            "final_loss": final_loss,
            "mean_loss": float(loss_sum / loss_count) if loss_count else None,
            "skipped": False,
        }

    def _build_shard_loader(self, shard_train):
        if not shard_train:
            return None

        text_dict = self.dataset_data.get("meta")
        if not text_dict:
            raise ValueError("RecEraser requires dataset_data['meta'] for LLM ranker prompts.")

        self._ensure_num_items()
        rng = np.random.RandomState(int(getattr(self.args, "seed", 42)))
        dataset = LLMTrainDataset(
            self.args,
            shard_train,
            int(getattr(self.args, "llm_max_history", 20)),
            rng,
            text_dict,
            self.tokenizer,
            Prompter(),
        )
        if len(dataset) == 0:
            return None

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

    def _save_shard_assignments(self, assignments, num_shards: int):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "receraser_shard_assignments.json")
        with open(path, "w") as f:
            json.dump({
                "num_shards": int(num_shards),
                "assignments": assignments,
            }, f, indent=2, default=str)
        return path

    def _save_receraser_ranker(self):
        adapter_dir = os.path.join(self.output_dir, "receraser_adapter")
        if hasattr(self.model, "save_pretrained"):
            os.makedirs(adapter_dir, exist_ok=True)
            self.model.save_pretrained(adapter_dir)
            return adapter_dir
        return None

    @staticmethod
    def _capture_parameter_state(named_params):
        return {
            name: param.detach().cpu().clone()
            for name, param in named_params
        }

    @staticmethod
    def _load_parameter_state(named_params, state):
        with torch.no_grad():
            for name, param in named_params:
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))

    @staticmethod
    def _aggregate_shard_states(named_params, original_state, affected_states, num_shards: int):
        unaffected_count = int(num_shards) - int(len(affected_states))
        with torch.no_grad():
            for name, param in named_params:
                aggregate = original_state[name].float() * (float(unaffected_count) / float(num_shards))
                for shard_state in affected_states.values():
                    aggregate += shard_state[name].float() / float(num_shards)
                param.copy_(aggregate.to(device=param.device, dtype=param.dtype))

    def _num_shards(self) -> int:
        num_shards = int(getattr(self.args, "receraser_num_shards", 4) or 4)
        if num_shards <= 0:
            raise ValueError("RecEraser requires receraser_num_shards > 0.")
        return num_shards

    def _shard_for_row(self, row: Dict, num_shards: int) -> int:
        digest = hashlib.sha256(self._interaction_key(row).encode("utf-8")).hexdigest()
        return int(digest, 16) % int(num_shards)

    @staticmethod
    def _interaction_key(row: Dict) -> str:
        position = row.get("position")
        position_part = "none" if position is None else str(position)
        return f"u{int(row['user_id'])}:p{position_part}:i{int(row['item_id'])}"
