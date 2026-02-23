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
    # Retry strategies
    "RetryConfig": ("scrapling.retry", "RetryConfig"),
    "ExponentialBackoff": ("scrapling.retry", "ExponentialBackoff"),
    "LinearBackoff": ("scrapling.retry", "LinearBackoff"),
    "FixedBackoff": ("scrapling.retry", "FixedBackoff"),
    "RateLimiter": ("scrapling.retry", "RateLimiter"),
    "AdaptiveRateLimiter": ("scrapling.retry", "AdaptiveRateLimiter"),
    "CircuitBreaker": ("scrapling.retry", "CircuitBreaker"),
    "CircuitState": ("scrapling.retry", "CircuitState"),
    "CircuitOpenError": ("scrapling.retry", "CircuitOpenError"),
    "with_retry": ("scrapling.retry", "with_retry"),
    "RetryExhaustedError": ("scrapling.retry", "RetryExhaustedError"),
}
__all__ = [
    "Selector",
    "Fetcher",
    "AsyncFetcher",
    "StealthyFetcher",
    "DynamicFetcher",
    # Retry strategies
    "RetryConfig",
    "ExponentialBackoff",
    "LinearBackoff",
    "FixedBackoff",
    "RateLimiter",
    "AdaptiveRateLimiter",
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "with_retry",
    "RetryExhaustedError",
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
    return sorted(__all__ + ["fetchers", "parser", "cli", "core", "retry", "__author__", "__version__", "__copyright__"])
