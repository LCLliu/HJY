import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseUnlearningMethod(ABC):
    """Common interface for all interaction-level unlearning strategies.

    `run_unlearning.py` instantiates one registered strategy class with this
    constructor and then calls `run()`. Method modules must not define their
    own training entrypoints.
    """

    method_name = "base"

    def __init__(
        self,
        model,
        tokenizer,
        split_data: Dict,
        args,
        output_dir: str,
        dataset_data: Dict = None,
        predictions_before: Dict = None,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.split_data = split_data
        self.args = args
        self.output_dir = output_dir
        self.dataset_data = dataset_data or {}
        self.predictions_before = predictions_before or {}
        self.extra_context = kwargs
        self.logs: Dict[str, Any] = {
            "method_name": self.method_name,
            "status": "initialized",
            "notes": [],
        }

    @abstractmethod
    def run(self) -> Dict:
        """Apply unlearning and return a method-specific log dictionary."""
        raise NotImplementedError

    def todo_noop(self, *, action: str, todo: str, warning: str = None) -> Dict:
        """Standard placeholder response for algorithms not implemented yet."""
        self.logs.update({
            "status": "todo_fallback_noop",
            "action": action,
            "is_effective_unlearning_baseline": False,
            "warning": warning or "TODO fallback only; do not report as an implemented baseline.",
            "split_counts": self._split_counts(),
            "todo": todo,
        })
        self.save_logs()
        return self.logs

    def save_logs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "method_logs.json"), "w") as f:
            json.dump(self.logs, f, indent=2, default=str)

    def _split_counts(self) -> Dict[str, int]:
        return {
            "forget": len(self.split_data.get("forget_interactions", [])),
            "retain": len(self.split_data.get("retain_interactions", [])),
            "overlap_retain": len(self.split_data.get("overlap_retain_interactions", [])),
            "semantic_neighbor_retain": len(self.split_data.get("semantic_neighbor_retain", [])),
            "collaborative_neighbor_retain": len(self.split_data.get("collaborative_neighbor_retain", [])),
        }
