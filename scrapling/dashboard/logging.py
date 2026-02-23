"""Logging utilities for scraper events."""

from datetime import datetime
from functools import wraps
from typing import Any, Callable, Literal

from .database import get_database


def log_event(
    event_type: Literal["fetch", "parse", "save", "error"],
    domain: str,
    status: str,
    message: str | None = None,
    duration_ms: float | None = None,
    cached: bool = False,
) -> None:
    """Log a scraper event to the metrics database."""
    try:
        db = get_database()
        db.log_event(
            event_type=event_type,
            domain=domain,
            status=status,
            message=message,
            duration_ms=duration_ms,
            cached=cached,
        )
    except Exception:
        # Don't fail the scraper if logging fails
        pass


def log_fetch(
    domain: str,
    status: str,
    duration_ms: float | None = None,
    cached: bool = False,
    message: str | None = None,
) -> None:
    """Log a fetch event."""
    log_event("fetch", domain, status, message, duration_ms, cached)


def log_parse(
    domain: str,
    status: str,
    duration_ms: float | None = None,
    message: str | None = None,
) -> None:
    """Log a parse event."""
    log_event("parse", domain, status, message, duration_ms)


def log_save(
    domain: str,
    status: str,
    duration_ms: float | None = None,
    message: str | None = None,
) -> None:
    """Log a save event."""
    log_event("save", domain, status, message, duration_ms)


def log_error(
    domain: str,
    message: str,
    duration_ms: float | None = None,
) -> None:
    """Log an error event."""
    log_event("error", domain, "error", message, duration_ms)


def timed_operation(event_type: Literal["fetch", "parse", "save"]) -> Callable:
    """Decorator to automatically log timing for operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start).total_seconds() * 1000

                # Try to extract domain from args
                domain = "unknown"
                if args:
                    url = args[0] if args else kwargs.get("url", "")
                    if url:
                        # Simple domain extraction
                        domain = url.split("/")[2] if "://" in url else url.split("/")[0]

                log_event(event_type, domain, "success", None, duration, False)
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start).total_seconds() * 1000
                domain = "unknown"
                if args:
                    url = args[0] if args else kwargs.get("url", "")
                    if url:
                        domain = url.split("/")[2] if "://" in url else url.split("/")[0]

                log_error(domain, str(e), duration)
                raise

        return wrapper

    return decorator
