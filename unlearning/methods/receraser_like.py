"""RecEraser-like compatibility strategy."""

from .receraser import RecEraserMethod


class RecEraserLikeMethod(RecEraserMethod):
    """Compatibility alias kept for existing `receraser_like` commands."""

    method_name = "receraser_like"
