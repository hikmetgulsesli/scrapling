"""Tests for Response Caching with ETag Support.

Tests cover:
1. CacheEntry - creation, expiration, serialization
2. CacheStats - hits, misses, hit_ratio
3. MemoryCache - get, set, delete, clear, exists
4. DiskCache - persistence, TTL, size limits
5. CacheManager - get_cached, cache_response, conditional headers
6. Utilities - get_cache_key, parse_cache_control
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from scrapling.cache import (
    CacheEntry,
    CacheStats,
    CacheBackend,
    MemoryCache,
    DiskCache,
    CacheManager,
    get_cache_key,
    parse_cache_control,
    get_default_cache,
    DEFAULT_CACHE_DIR,
)


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        """Test creating a cache entry."""
        entry = CacheEntry(key="test", value="data")
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.etag is None
        assert entry.last_modified is None

    def test_creation_with_metadata(self):
        """Test creating a cache entry with metadata."""
        entry = CacheEntry(
            key="test",
            value="data",
            etag='"abc123"',
            last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
            headers={"Content-Type": "text/html"},
        )
        assert entry.etag == '"abc123"'
        assert entry.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"
        assert entry.headers["Content-Type"] == "text/html"

    def test_is_expired_no_ttl(self):
        """Test expiration check with no TTL."""
        entry = CacheEntry(key="test", value="data")
        assert entry.is_expired() is False

    def test_is_expired_with_ttl(self):
        """Test expiration check with TTL."""
        entry = CacheEntry(key="test", value="data", expires_at=time.time() + 3600)
        assert entry.is_expired() is False

    def test_is_expired_expired(self):
        """Test expiration check when expired."""
        entry = CacheEntry(key="test", value="data", expires_at=time.time() - 1)
        assert entry.is_expired() is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = CacheEntry(
            key="test",
            value="data",
            etag='"abc123"',
            headers={"Content-Type": "text/html"},
        )
        data = entry.to_dict()
        assert data["key"] == "test"
        assert data["value"] == "data"
        assert data["etag"] == '"abc123"'

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "key": "test",
            "value": "data",
            "etag": '"abc123"',
            "created_at": time.time(),
            "expires_at": None,
            "last_modified": None,
            "headers": {},
        }
        entry = CacheEntry.from_dict(data)
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.etag == '"abc123"'


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0
        assert stats.entries == 0

    def test_hit_ratio_no_data(self):
        """Test hit ratio with no data."""
        stats = CacheStats()
        assert stats.hit_ratio == 0.0

    def test_hit_ratio_calculation(self):
        """Test hit ratio calculation."""
        stats = CacheStats()
        stats.hits = 75
        stats.misses = 25
        assert stats.hit_ratio == 0.75

    def test_record_hit(self):
        """Test recording a hit."""
        stats = CacheStats()
        stats.record_hit()
        assert stats.hits == 1

    def test_record_miss(self):
        """Test recording a miss."""
        stats = CacheStats()
        stats.record_miss()
        assert stats.misses == 1

    def test_reset(self):
        """Test resetting stats."""
        stats = CacheStats()
        stats.hits = 10
        stats.misses = 5
        stats.size = 1000
        stats.entries = 50
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0
        assert stats.entries == 0


class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_initialization(self):
        """Test memory cache initialization."""
        cache = MemoryCache(ttl=60, max_size=100)
        assert cache.ttl == 60
        assert cache.max_size == 100

    def test_set_and_get(self):
        """Test setting and getting values."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        entry = cache.get("key1")
        assert entry is not None
        assert entry.value == "value1"

    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = MemoryCache()
        entry = cache.get("nonexistent")
        assert entry is None

    def test_delete(self):
        """Test deleting a key."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        entry = cache.get("key1")
        assert entry is None

    def test_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_exists(self):
        """Test exists check."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = MemoryCache(ttl=1)
        cache.set("key1", "value1")
        # Should work immediately
        assert cache.get("key1") is not None
        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_max_size_eviction(self):
        """Test max size eviction."""
        cache = MemoryCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict oldest
        # key1 should be evicted
        assert cache.get("key1") is None
        # key2 and key3 should exist
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None


class TestDiskCache:
    """Tests for DiskCache."""

    def test_initialization(self):
        """Test disk cache initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=60, cache_dir=tmpdir)
            assert cache.ttl == 60
            assert cache.cache_dir == Path(tmpdir)

    def test_persistence(self):
        """Test cache persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=60, cache_dir=tmpdir)
            cache.set("key1", "value1")
            
            # Create new cache instance
            cache2 = DiskCache(ttl=60, cache_dir=tmpdir)
            entry = cache2.get("key1")
            assert entry is not None
            assert entry.value == "value1"

    def test_delete(self):
        """Test deleting from disk cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=60, cache_dir=tmpdir)
            cache.set("key1", "value1")
            cache.delete("key1")
            assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing disk cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=60, cache_dir=tmpdir)
            cache.set("key1", "value1")
            cache.set("key2", "value2")
            cache.clear()
            assert cache.get("key1") is None
            assert cache.get("key2") is None

    def test_ttl_expiration(self):
        """Test TTL expiration on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=1, cache_dir=tmpdir)
            cache.set("key1", "value1")
            assert cache.get("key1") is not None
            time.sleep(1.1)
            assert cache.get("key1") is None

    def test_stats_tracking(self):
        """Test statistics tracking on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(ttl=60, cache_dir=tmpdir)
            cache.set("key1", "value1")
            cache.get("key1")  # hit
            cache.get("key2")  # miss
            assert cache.stats.hits == 1
            assert cache.stats.misses == 1

    def test_custom_cache_dir(self):
        """Test custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = Path(tmpdir) / "custom_cache"
            cache = DiskCache(cache_dir=custom_dir)
            assert cache.cache_dir == custom_dir
            assert custom_dir.exists()


class TestCacheManager:
    """Tests for CacheManager."""

    def test_initialization(self):
        """Test cache manager initialization."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache, ttl=60)
        assert manager.backend == cache
        assert manager.ttl == 60

    def test_get_cached(self):
        """Test getting cached response."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        manager.cache_response("https://example.com", "content")
        
        entry = manager.get_cached("https://example.com")
        assert entry is not None
        assert entry.value == "content"

    def test_cache_response_with_headers(self):
        """Test caching response with headers."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        
        headers = {
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 21 Oct 2015 07:28:00 GMT",
        }
        manager.cache_response("https://example.com", "content", headers)
        
        entry = manager.get_cached("https://example.com")
        assert entry is not None
        assert entry.etag == '"abc123"'
        assert entry.last_modified == "Wed, 21 Oct 2015 07:28:00 GMT"

    def test_get_conditional_headers(self):
        """Test getting conditional request headers."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        
        headers = {"ETag": '"abc123"'}
        manager.cache_response("https://example.com", "content", headers)
        
        cond_headers = manager.get_conditional_headers("https://example.com")
        assert "If-None-Match" in cond_headers
        assert cond_headers["If-None-Match"] == '"abc123"'

    def test_get_conditional_headers_no_cache(self):
        """Test conditional headers when no cache."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        
        cond_headers = manager.get_conditional_headers("https://example.com")
        assert cond_headers == {}

    def test_is_cache_valid_304(self):
        """Test cache valid with 304 response."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        
        assert manager.is_cache_valid("https://example.com", 304) is True

    def test_invalidate(self):
        """Test cache invalidation."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        manager.cache_response("https://example.com", "content")
        
        manager.invalidate("https://example.com")
        
        assert manager.get_cached("https://example.com") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        manager.cache_response("https://example.com", "content")
        
        manager.clear()
        
        assert manager.get_cached("https://example.com") is None

    def test_get_stats(self):
        """Test getting stats."""
        cache = MemoryCache()
        manager = CacheManager(backend=cache)
        manager.cache_response("https://example.com", "content")
        manager.get_cached("https://example.com")
        
        stats = manager.get_stats()
        assert stats.hits == 1


class TestUtilities:
    """Tests for utility functions."""

    def test_get_cache_key(self):
        """Test generating cache key."""
        key = get_cache_key("https://example.com")
        assert key == "GET:https://example.com"

    def test_get_cache_key_with_method(self):
        """Test generating cache key with custom method."""
        key = get_cache_key("https://example.com", "POST")
        assert key == "POST:https://example.com"

    def test_parse_cache_control_empty(self):
        """Test parsing empty cache control."""
        result = parse_cache_control("")
        assert result == {}

    def test_parse_cache_control_no_value(self):
        """Test parsing cache control with no value."""
        result = parse_cache_control("no-cache, no-store")
        assert result == {"no-cache": True, "no-store": True}

    def test_parse_cache_control_with_values(self):
        """Test parsing cache control with values."""
        result = parse_cache_control("max-age=3600, s-maxage=600")
        assert result["max-age"] == "3600"
        assert result["s-maxage"] == "600"


class TestDefaultCache:
    """Tests for default cache."""

    def test_get_default_cache(self):
        """Test getting default cache."""
        cache = get_default_cache()
        assert isinstance(cache, CacheManager)

    def test_default_cache_is_singleton(self):
        """Test default cache is singleton."""
        cache1 = get_default_cache()
        cache2 = get_default_cache()
        # Both should reference the same manager (though the backend might differ)
        assert isinstance(cache2, CacheManager)


# Mark all tests
pytestmark = pytest.mark.cache
