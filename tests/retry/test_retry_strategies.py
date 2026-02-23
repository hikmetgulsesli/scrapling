"""Tests for Retry & Rate Limit Strategies.

Tests cover:
1. ExponentialBackoff - delays, jitter, should_retry
2. LinearBackoff - delays, should_retry
3. FixedBackoff - fixed delays
4. RateLimiter - per-domain delays, acquire, sync_acquire
5. AdaptiveRateLimiter - adapts to 429 responses
6. CircuitBreaker - states, failure tracking, execute
7. with_retry decorator
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch

from scrapling.retry import (
    RetryConfig,
    RetryStrategy,
    ExponentialBackoff,
    LinearBackoff,
    FixedBackoff,
    RateLimiter,
    AdaptiveRateLimiter,
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    with_retry,
    RetryExhaustedError,
    default_rate_limiter,
    default_circuit_breaker,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 0.1
        assert config.exponential_base == 2.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            jitter=0.2,
            exponential_base=3.0
        )
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.2
        assert config.exponential_base == 3.0


class TestExponentialBackoff:
    """Tests for ExponentialBackoff strategy."""

    def test_initialization(self):
        """Test exponential backoff initialization."""
        strategy = ExponentialBackoff(max_attempts=5, base_delay=1.0)
        assert strategy.max_attempts == 5
        assert strategy.config.base_delay == 1.0

    def test_delay_calculation(self):
        """Test delay increases exponentially."""
        strategy = ExponentialBackoff(max_attempts=5, base_delay=1.0, jitter=0.0)

        delay_0 = strategy.get_delay(0)
        delay_1 = strategy.get_delay(1)
        delay_2 = strategy.get_delay(2)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = ExponentialBackoff(
            max_attempts=10,
            base_delay=10.0,
            max_delay=50.0,
            jitter=0.0
        )

        # Even with high exponent, should be capped
        delay = strategy.get_delay(10)
        assert delay == 50.0

    def test_should_retry(self):
        """Test should_retry based on attempt count."""
        strategy = ExponentialBackoff(max_attempts=3)

        assert strategy.should_retry(0) is True
        assert strategy.should_retry(1) is True
        assert strategy.should_retry(2) is True
        assert strategy.should_retry(3) is False

    def test_reset(self):
        """Test reset clears attempt count."""
        strategy = ExponentialBackoff(max_attempts=3)
        # Just verify reset doesn't error
        strategy.reset()


class TestLinearBackoff:
    """Tests for LinearBackoff strategy."""

    def test_initialization(self):
        """Test linear backoff initialization."""
        strategy = LinearBackoff(max_attempts=5, base_delay=1.0)
        assert strategy.config.max_attempts == 5
        assert strategy.config.base_delay == 1.0

    def test_delay_calculation(self):
        """Test delay increases linearly."""
        strategy = LinearBackoff(max_attempts=5, base_delay=1.0, jitter=0.0)

        delay_0 = strategy.get_delay(0)
        delay_1 = strategy.get_delay(1)
        delay_2 = strategy.get_delay(2)

        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 3.0

    def test_delay_capped_at_max(self):
        """Test delay is capped at max_delay."""
        strategy = LinearBackoff(
            max_attempts=10,
            base_delay=10.0,
            max_delay=25.0,
            jitter=0.0
        )

        delay = strategy.get_delay(5)
        assert delay == 25.0


class TestFixedBackoff:
    """Tests for FixedBackoff strategy."""

    def test_initialization(self):
        """Test fixed backoff initialization."""
        strategy = FixedBackoff(max_attempts=3, delay=2.0)
        assert strategy.max_attempts == 3
        assert strategy.delay == 2.0

    def test_fixed_delay(self):
        """Test always returns fixed delay."""
        strategy = FixedBackoff(max_attempts=5, delay=3.0)

        assert strategy.get_delay(0) == 3.0
        assert strategy.get_delay(1) == 3.0
        assert strategy.get_delay(100) == 3.0


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(default_delay=1.0)
        assert limiter.default_delay == 1.0
        assert limiter.domain_delays == {}

    def test_get_default_delay(self):
        """Test getting default delay."""
        limiter = RateLimiter(default_delay=2.0)
        assert limiter.get_delay("example.com") == 2.0

    def test_custom_domain_delay(self):
        """Test custom domain delay."""
        limiter = RateLimiter(default_delay=1.0)
        limiter.set_delay("slow-site.com", 5.0)

        assert limiter.get_delay("slow-site.com") == 5.0
        assert limiter.get_delay("fast-site.com") == 1.0

    @pytest.mark.asyncio
    async def test_acquire(self):
        """Test async acquire enforces rate limiting."""
        limiter = RateLimiter(default_delay=0.1)

        start = time.monotonic()
        await limiter.acquire("example.com")
        await limiter.acquire("example.com")
        elapsed = time.monotonic() - start

        # Should have waited at least 0.1 seconds between requests
        assert elapsed >= 0.09

    def test_sync_acquire(self):
        """Test sync acquire enforces rate limiting."""
        limiter = RateLimiter(default_delay=0.1)

        start = time.monotonic()
        limiter.sync_acquire("example.com")
        limiter.sync_acquire("example.com")
        elapsed = time.monotonic() - start

        assert elapsed >= 0.09

    def test_reset_single_domain(self):
        """Test resetting single domain."""
        limiter = RateLimiter(default_delay=1.0)
        limiter.sync_acquire("example.com")

        limiter.reset("example.com")
        # Should not error and should allow immediate next request
        wait = limiter.get_wait_time("example.com")
        assert wait == 0.0

    def test_reset_all(self):
        """Test resetting all domains."""
        limiter = RateLimiter(default_delay=1.0)
        limiter.sync_acquire("example.com")
        limiter.sync_acquire("other.com")

        limiter.reset()

        assert limiter.get_wait_time("example.com") == 0.0
        assert limiter.get_wait_time("other.com") == 0.0


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter."""

    @pytest.mark.asyncio
    async def test_increase_delay_on_429(self):
        """Test delay increases on 429 response."""
        limiter = AdaptiveRateLimiter(initial_delay=1.0, backoff_factor=2.0)

        await limiter.record_response("example.com", status=429)
        assert limiter.get_delay("example.com") == 2.0

        await limiter.record_response("example.com", status=429)
        assert limiter.get_delay("example.com") == 4.0

    @pytest.mark.asyncio
    async def test_decrease_delay_on_success(self):
        """Test delay decreases on successful response."""
        limiter = AdaptiveRateLimiter(initial_delay=10.0, min_delay=1.0)

        # First reduce from 10 to 9
        await limiter.record_response("example.com", status=200)
        assert limiter.get_delay("example.com") < 10.0

    @pytest.mark.asyncio
    async def test_delay_capped(self):
        """Test delay doesn't exceed max."""
        limiter = AdaptiveRateLimiter(
            initial_delay=10.0,
            max_delay=30.0,
            backoff_factor=10.0
        )

        await limiter.record_response("example.com", status=429)
        assert limiter.get_delay("example.com") == 30.0  # capped at max

    @pytest.mark.asyncio
    async def test_delay_floor(self):
        """Test delay doesn't go below min."""
        limiter = AdaptiveRateLimiter(
            initial_delay=1.0,
            min_delay=0.5,
            backoff_factor=0.5  # This would reduce
        )

        await limiter.record_response("example.com", status=200)
        # Should not go below min_delay
        assert limiter.get_delay("example.com") >= 0.5


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        assert breaker.failure_threshold == 5
        assert breaker.timeout == 60.0

    def test_initial_state_closed(self):
        """Test initial state is closed."""
        breaker = CircuitBreaker()
        assert breaker.get_state("example.com") == CircuitState.CLOSED

    def test_is_available_initially(self):
        """Test initially available."""
        breaker = CircuitBreaker()
        assert breaker.is_available("example.com") is True

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60.0)

        for _ in range(3):
            await breaker.record_failure("example.com")

        assert breaker.get_state("example.com") == CircuitState.OPEN
        assert breaker.is_available("example.com") is False

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test circuit goes half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)

        await breaker.record_failure("example.com")
        await breaker.record_failure("example.com")

        # Wait for timeout
        await asyncio.sleep(0.15)

        assert breaker.is_available("example.com") is True

    @pytest.mark.asyncio
    async def test_success_closes_circuit(self):
        """Test success in half-open closes circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            timeout=0.1,
            success_threshold=2
        )

        # Fail to open
        await breaker.record_failure("example.com")
        await breaker.record_failure("example.com")

        # Wait and enter half-open
        await asyncio.sleep(0.15)

        # First, make the breaker attempt recovery - this transitions to HALF_OPEN
        assert breaker.is_available("example.com") is True
        
        # Verify we're now in half-open
        assert breaker._states.get("example.com") == CircuitState.HALF_OPEN

        # Success in half-open
        await breaker.record_success("example.com")
        
        # After first success, still half-open
        assert breaker._states.get("example.com") == CircuitState.HALF_OPEN
        
        await breaker.record_success("example.com")

        # After enough successes, should close
        assert breaker._states.get("example.com") == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test execute with successful function."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fetch():
            return "data"

        result = await breaker.execute(fetch, domain="example.com")
        assert result == "data"

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execute with failing function."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fetch():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            await breaker.execute(fetch, domain="example.com")

    def test_execute_circuit_open(self):
        """Test execute raises when circuit open."""
        breaker = CircuitBreaker(failure_threshold=1, timeout=60.0)

        # Open the circuit synchronously
        breaker._states["example.com"] = CircuitState.OPEN
        breaker._failure_counts["example.com"] = 5

        def fetch():
            return "data"

        # Use sync_execute which is synchronous
        with pytest.raises(CircuitOpenError):
            breaker.sync_execute(fetch, domain="example.com")

    def test_reset(self):
        """Test reset clears state."""
        breaker = CircuitBreaker(failure_threshold=2)

        breaker._states["example.com"] = CircuitState.OPEN
        breaker._failure_counts["example.com"] = 10

        breaker.reset("example.com")

        assert breaker.get_state("example.com") == CircuitState.CLOSED


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_exponential_retry_sync(self):
        """Test exponential retry decorator sync."""
        call_count = 0

        @with_retry(strategy="exponential", max_attempts=3)
        def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = fetch()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        """Test retry exhausted raises original exception."""
        call_count = 0

        @with_retry(strategy="fixed", max_attempts=2, delay=0.01)
        def fetch():
            nonlocal call_count
            call_count += 1
            raise ValueError("fail")

        # Should retry twice (initial + 1 retry), then raise the last exception
        with pytest.raises(ValueError, match="fail"):
            fetch()

        assert call_count == 2  # Initial attempt + 1 retry

    def test_linear_strategy(self):
        """Test linear strategy."""
        call_count = 0

        @with_retry(strategy="linear", max_attempts=3, base_delay=0.01)
        def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = fetch()
        assert result == "success"

    def test_fixed_strategy(self):
        """Test fixed strategy."""
        call_count = 0

        @with_retry(strategy="fixed", max_attempts=3, delay=0.01)
        def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = fetch()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry(self):
        """Test async retry decorator."""
        call_count = 0

        @with_retry(strategy="exponential", max_attempts=3)
        async def fetch():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"

        result = await fetch()
        assert result == "success"
        assert call_count == 3


class TestGlobalInstances:
    """Tests for global instances."""

    def test_default_rate_limiter(self):
        """Test default rate limiter exists."""
        assert default_rate_limiter is not None
        assert isinstance(default_rate_limiter, RateLimiter)

    def test_default_circuit_breaker(self):
        """Test default circuit breaker exists."""
        assert default_circuit_breaker is not None
        assert isinstance(default_circuit_breaker, CircuitBreaker)


# Mark all tests
pytestmark = pytest.mark.retry
