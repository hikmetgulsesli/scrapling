__author__ = "Karim Shoair (karim.shoair@pm.me)"
__version__ = "0.4"
__copyright__ = "Copyright (c) 2024 Karim Shoair"

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scrapling.parser import Selector, Selectors
    from scrapling.core.custom_types import AttributesHandler, TextHandler
    from scrapling.fetchers import Fetcher, AsyncFetcher, StealthyFetcher, DynamicFetcher


# Lazy import mapping
_LAZY_IMPORTS = {
    "Fetcher": ("scrapling.fetchers", "Fetcher"),
    "Selector": ("scrapling.parser", "Selector"),
    "Selectors": ("scrapling.parser", "Selectors"),
    "AttributesHandler": ("scrapling.core.custom_types", "AttributesHandler"),
    "TextHandler": ("scrapling.core.custom_types", "TextHandler"),
    "AsyncFetcher": ("scrapling.fetchers", "AsyncFetcher"),
    "StealthyFetcher": ("scrapling.fetchers", "StealthyFetcher"),
    "DynamicFetcher": ("scrapling.fetchers", "DynamicFetcher"),
    # Pipeline
    "Pipeline": ("scrapling.pipeline", "Pipeline"),
    "PipelineStage": ("scrapling.pipeline", "PipelineStage"),
    "FetchStage": ("scrapling.pipeline", "FetchStage"),
    "ParseStage": ("scrapling.pipeline", "ParseStage"),
    "SaveStage": ("scrapling.pipeline", "SaveStage"),
    "PipelineResult": ("scrapling.pipeline", "PipelineResult"),
    "PipelineMetadata": ("scrapling.pipeline", "PipelineMetadata"),
    "PipelineError": ("scrapling.pipeline", "PipelineError"),
    "PipelineContext": ("scrapling.pipeline", "PipelineContext"),
    "PipelineStageError": ("scrapling.pipeline", "PipelineStageError"),
    "StageStatus": ("scrapling.pipeline", "StageStatus"),
}
__all__ = [
    "Selector",
    "Fetcher",
    "AsyncFetcher",
    "StealthyFetcher",
    "DynamicFetcher",
    # Pipeline
    "Pipeline",
    "PipelineStage",
    "FetchStage",
    "ParseStage",
    "SaveStage",
    "PipelineResult",
    "PipelineMetadata",
    "PipelineError",
    "PipelineContext",
    "PipelineStageError",
    "StageStatus",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Support for dir() and autocomplete."""
    return sorted(__all__ + ["fetchers", "parser", "cli", "core", "pipeline", "__author__", "__version__", "__copyright__"])
