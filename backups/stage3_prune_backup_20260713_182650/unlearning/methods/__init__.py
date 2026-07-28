from .finetune import FinetuneMethod
from .geometry_prune import GeometryPruneMethod, SemanticGeometryPruneMethod
from .gradient_ascent import GradientAscentMethod
from .llm_eraser import LLMEraserMethod
from .llm_eraser_like import LLMEraserLikeMethod
from .none import NoneUnlearningMethod
from .npo import NPOMethod
from .random_prune import RandomPruneMethod
from .receraser_like import RecEraserLikeMethod
from .retain_ft import RetainFTMethod
from .retain_prior_cf_lora_prune import RetainPriorCFLoraPruneMethod
from .retain_prioritized_cbr_unlearning import RetainPrioritizedCBRUnlearningMethod
from .selective_pruning import SelectivePruningMethod


METHOD_REGISTRY = {
    "none": NoneUnlearningMethod,
    "random_prune": RandomPruneMethod,
    "retain_ft": RetainFTMethod,
    "gradient_ascent": GradientAscentMethod,
    "npo": NPOMethod,
    "selective_pruning": SelectivePruningMethod,
    "llm_eraser": LLMEraserMethod,
    "finetune": FinetuneMethod,
    "receraser_like": RecEraserLikeMethod,
    "llm_eraser_like": LLMEraserLikeMethod,
    "geometry_prune": GeometryPruneMethod,
    "semantic_geometry_prune": SemanticGeometryPruneMethod,
    "retain_prioritized_cbr": RetainPrioritizedCBRUnlearningMethod,
    "retain_prior_cf_lora_prune": RetainPriorCFLoraPruneMethod,
}
