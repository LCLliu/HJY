"""LLM-Eraser-like compatibility strategy."""

from .llm_eraser import LLMEraserMethod


class LLMEraserLikeMethod(LLMEraserMethod):
    """Compatibility alias kept for existing `llm_eraser_like` commands."""

    method_name = "llm_eraser_like"
