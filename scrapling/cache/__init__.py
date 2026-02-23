"""Response Caching with ETag Support for Scrapling.

This module provides HTTP response caching with ETag/Last-Modified validation,
disk and memory cache backends, and cache statistics.

Example:
    >>> from scrapling.cache import DiskCache, CacheManager
    >>> cache = DiskCache(ttl=3600)
    >>> result = cache.get("https://example.com")
    >>> if result is None:
    ...     result = fetch(url)
    ...     cache.set(url, result)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".scrapling" / "cache"


@dataclass
class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        key: Cache key (usually URL)
        value: Cached value
        created_at: When the entry was created
        expires_at: When the entry expires (None for no expiration)
        etag: ETag header value
        last_modified: Last-Modified header value
        headers: Original response headers
    """
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    etag: str | None = None
    last_modified: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "headers": self.headers,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            etag=data.get("etag"),
            last_modified=data.get("last_modified"),
            headers=data.get("headers", {}),
        )


@dataclass
class CacheStats:
    """Cache statistics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        size: Current cache size in bytes
        entries: Number of cached entries
    """
    hits: int = 0
    misses: int = 0
    size: int = 0
    entries: int = 0

    @property
    def hit_ratio(self) -> float:
        """Calculate hit ratio (0.0 to 1.0)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.size = 0
        self.entries = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "size": self.size,
            "entries": self.entries,
            "hit_ratio": self.hit_ratio,
        }


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None:
        """Get a cached entry.

        Args:
            key: Cache key

        Returns:
            Cache entry if found and not expired, None otherwise
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None, **kwargs: Any) -> None:
        """Set a cache entry.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            **kwargs: Additional metadata (etag, last_modified, headers)
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a cache entry.

        Args:
            key: Cache key
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend.

    Example:
        >>> cache = MemoryCache(ttl=3600)
        >>> cache.set("key", "value", ttl=60)
        >>> entry = cache.get("key")
    """

    def __init__(self, ttl: int | None = None, max_size: int | None = None) -> None:
        """Initialize memory cache.

        Args:
            ttl: Default time to live in seconds
            max_size: Maximum number of entries
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    def get(self, key: str) -> CacheEntry | None:
        """Get a cached entry."""
        if key not in self._cache:
            self.stats.record_miss()
            return None

        entry = self._cache[key]

        # Check expiration
        if entry.is_expired():
            self.delete(key)
            self.stats.record_miss()
            return None

        self.stats.record_hit()
        return entry

    def set(self, key: str, value: Any, ttl: int | None = None, **kwargs: Any) -> None:
        """Set a cache entry."""
        # Calculate expiration time
        expires_at: float | None = None
        effective_ttl = ttl if ttl is not None else self.ttl
        if effective_ttl is not None:
            expires_at = time.time() + effective_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            etag=kwargs.get("etag"),
            last_modified=kwargs.get("last_modified"),
            headers=kwargs.get("headers", {}),
        )

        # Evict if at max size
        if self.max_size and len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            self.delete(oldest_key)

        self._cache[key] = entry

    def delete(self, key: str) -> None:
        """Delete a cache entry."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self.stats.reset()

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            self.delete(key)
            return False
        return True


class DiskCache(CacheBackend):
    """Disk-based cache backend.

    Example:
        >>> cache = DiskCache(ttl=3600, cache_dir=~/.scrapling/cache)
        >>> cache.set("https://example.com", response_content, etag="abc123")
        >>> entry = cache.get("https://example.com")
    """

    def __init__(
        self,
        ttl: int | None = None,
        cache_dir: Path | str | None = None,
        max_size_mb: float | None = None
    ) -> None:
        """Initialize disk cache.

        Args:
            ttl: Default time to live in seconds
            cache_dir: Cache directory path
            max_size_mb: Maximum cache size in MB
        """
        self.ttl = ttl
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.max_size_mb = max_size_mb
        self.stats = CacheStats()

        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """Get filesystem path for a cache key."""
        # Hash the key to create a safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> CacheEntry | None:
        """Get a cached entry from disk."""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self.stats.record_miss()
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entry = CacheEntry.from_dict(data)

            # Check expiration
            if entry.is_expired():
                self.delete(key)
                self.stats.record_miss()
                return None

            self.stats.record_hit()
            self.stats.size = self._get_cache_size()
            self.stats.entries = len(list(self.cache_dir.glob("*.json")))
            return entry

        except (json.JSONDecodeError, KeyError, IOError):
            self.stats.record_miss()
            return None

    def set(self, key: str, value: Any, ttl: int | None = None, **kwargs: Any) -> None:
        """Set a cache entry on disk."""
        # Calculate expiration time
        expires_at: float | None = None
        effective_ttl = ttl if ttl is not None else self.ttl
        if effective_ttl is not None:
            expires_at = time.time() + effective_ttl

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
            etag=kwargs.get("etag"),
            last_modified=kwargs.get("last_modified"),
            headers=kwargs.get("headers", {}),
        )

        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)

            # Update stats
            self.stats.size = self._get_cache_size()
            self.stats.entries = len(list(self.cache_dir.glob("*.json")))

            # Check size limit
            if self.max_size_mb:
                self._enforce_size_limit()

        except IOError:
            pass  # Silently fail on write errors

    def delete(self, key: str) -> None:
        """Delete a cache entry from disk."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            self.stats.size = self._get_cache_size()
            self.stats.entries = len(list(self.cache_dir.glob("*.json")))

    def clear(self) -> None:
        """Clear all cache entries from disk."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        self.stats.reset()

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def _get_cache_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                total_size += cache_file.stat().st_size
        return total_size

    def _enforce_size_limit(self) -> None:
        """Remove oldest entries to stay under size limit."""
        if not self.max_size_mb:
            return

        max_bytes = self.max_size_mb * 1024 * 1024

        while self.stats.size > max_bytes:
            # Find oldest entry
            oldest_file = min(
                self.cache_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                default=None
            )
            if oldest_file is None:
                break
            oldest_file.unlink()
            self.stats.size = self._get_cache_size()


class CacheManager:
    """Manages caching with ETag/Last-Modified support.

    Example:
        >>> manager = CacheManager(backend=DiskCache(ttl=3600))
        >>> # Check if we have a cached response
        >>> cached = manager.get_cached("https://example.com")
        >>> if cached:
        ...     # Use cached response
        ...     print(cached.value)
        >>> else:
        ...     # Fetch new response
        ...     response = fetcher.get("https://example.com")
        ...     # Cache with ETag
        ...     manager.cache_response("https://example.com", response, response.headers)
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        ttl: int | None = 3600,
    ) -> None:
        """Initialize cache manager.

        Args:
            backend: Cache backend (defaults to MemoryCache)
            ttl: Default time to live in seconds
        """
        self.backend = backend or MemoryCache(ttl=ttl)
        self.ttl = ttl

    def get_cached(self, url: str) -> CacheEntry | None:
        """Get cached response for URL.

        Args:
            url: URL to get cached response for

        Returns:
            Cache entry if found, None otherwise
        """
        return self.backend.get(url)

    def cache_response(
        self,
        url: str,
        content: Any,
        headers: dict[str, str] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Cache an HTTP response.

        Args:
            url: URL of the response
            content: Response content
            headers: Response headers (for ETag/Last-Modified)
            ttl: Time to live in seconds
        """
        headers = headers or {}

        self.backend.set(
            url,
            content,
            ttl=ttl,
            etag=headers.get("ETag") or headers.get("etag"),
            last_modified=headers.get("Last-Modified") or headers.get("last-modified"),
            headers=dict(headers),
        )

    def get_conditional_headers(self, url: str) -> dict[str, str]:
        """Get conditional request headers for a cached URL.

        Args:
            url: URL to get conditional headers for

        Returns:
            Dictionary with If-None-Match and/or If-Modified-Since headers
        """
        entry = self.backend.get(url)
        if entry is None:
            return {}

        headers = {}
        if entry.etag:
            headers["If-None-Match"] = entry.etag
        if entry.last_modified:
            headers["If-Modified-Since"] = entry.last_modified

        return headers

    def is_cache_valid(
        self,
        url: str,
        status: int,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Check if cached response is still valid based on response.

        Args:
            url: URL that was requested
            status: HTTP status code of response
            headers: Response headers

        Returns:
            True if cached response is valid (304 Not Modified)
        """
        if status == 304:
            return True

        headers = headers or {}
        if status == 200:
            # Check ETag match
            etag = headers.get("ETag") or headers.get("etag")
            cached = self.backend.get(url)
            if cached and etag and cached.etag == etag:
                return True

        return False

    def invalidate(self, url: str) -> None:
        """Invalidate cached response for URL.

        Args:
            url: URL to invalidate
        """
        self.backend.delete(url)

    def clear(self) -> None:
        """Clear all cached responses."""
        self.backend.clear()

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        return self.backend.stats


def get_cache_key(url: str, method: str = "GET") -> str:
    """Generate a cache key from URL and method.

    Args:
        url: URL to generate key for
        method: HTTP method

    Returns:
        Cache key
    """
    return f"{method}:{url}"


def parse_cache_control(header: str) -> dict[str, Any]:
    """Parse Cache-Control header value.

    Args:
        header: Cache-Control header value

    Returns:
        Dictionary of cache control directives
    """
    directives: dict[str, Any] = {}

    if not header:
        return directives

    for part in header.split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            directives[key.strip().lower()] = value.strip()
        else:
            directives[part.lower()] = True

    return directives


# Global cache manager instance
_default_cache_manager: CacheManager | None = None


def get_default_cache() -> CacheManager:
    """Get the default cache manager instance.

    Returns:
        Default CacheManager instance
    """
    global _default_cache_manager
    if _default_cache_manager is None:
        _default_cache_manager = CacheManager(backend=DiskCache(ttl=3600))
    return _default_cache_manager


__all__ = [
    # Data classes
    "CacheEntry",
    "CacheStats",
    # Backends
    "CacheBackend",
    "MemoryCache",
    "DiskCache",
    # Manager
    "CacheManager",
    # Utilities
    "get_cache_key",
    "parse_cache_control",
    "get_default_cache",
    # Constants
    "DEFAULT_CACHE_DIR",
]
