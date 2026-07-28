"""No-op unlearning baseline."""

from .base import BaseUnlearningMethod


class NoneUnlearningMethod(BaseUnlearningMethod):
    """Leave the loaded downstream ranker unchanged."""

    method_name = "none"

    def run(self):
        self.logs.update({
            "status": "completed",
            "method": self.method_name,
            "action": "no_unlearning",
            "is_effective_unlearning_baseline": False,
            "split_counts": self._split_counts(),
            "notes": [
                "No model parameters were modified.",
            ],
        })
        self.save_logs()
        return self.logs
