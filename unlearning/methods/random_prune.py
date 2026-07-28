import random

import torch

from .base import BaseUnlearningMethod


class RandomPruneMethod(BaseUnlearningMethod):
    method_name = "random_prune"

    def run(self):
        prune_ratio = float(getattr(self.args, "random_prune_ratio", 0.05))
        seed = int(getattr(self.args, "seed", 42))
        rng = random.Random(seed)

        lora_pairs = self._collect_lora_pairs()
        pruned = []

        with torch.no_grad():
            for key, pair in lora_pairs.items():
                rank = pair["A"].shape[0]
                n_prune = max(1, int(rank * prune_ratio)) if rank > 0 and prune_ratio > 0 else 0
                chosen = sorted(rng.sample(range(rank), min(n_prune, rank)))
                for ridx in chosen:
                    pair["A"][ridx, :] = 0.0
                    pair["B"][:, ridx] = 0.0
                pruned.append({
                    "module": key,
                    "rank": rank,
                    "pruned_ranks": chosen,
                })

        self.logs.update({
            "status": "completed",
            "action": "random_lora_rank_prune",
            "is_effective_unlearning_baseline": True,
            "uses_fixed_split": True,
            "warning": (
                "Random pruning is a method baseline. It reads the fixed split "
                "for protocol consistency but does not use split contents to "
                "choose ranks."
            ),
            "random_prune_ratio": prune_ratio,
            "seed": seed,
            "split_counts": self._split_counts(),
            "pruned_modules": pruned,
        })
        self.save_logs()
        return self.logs

    def _collect_lora_pairs(self):
        pairs = {}
        for name, param in self.model.named_parameters():
            if "lora_A" in name:
                key = name.replace("lora_A", "lora")
                pairs.setdefault(key, {})["A"] = param.data
            elif "lora_B" in name:
                key = name.replace("lora_B", "lora")
                pairs.setdefault(key, {})["B"] = param.data
        return {k: v for k, v in pairs.items() if "A" in v and "B" in v}
