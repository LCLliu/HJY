"""Backward-compatible alias for retain-set fine-tuning."""

from .retain_ft import RetainFTMethod


class FinetuneMethod(RetainFTMethod):
    """Legacy method name kept for existing commands."""

    method_name = "finetune"
