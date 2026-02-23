"""Retry & Rate Limit Strategies for Scrapling.

This module provides retry mechanisms with exponential backoff,
rate limiting per domain, and circuit breaker patterns.

Example:
    >>> from scrapling.retry import ExponentialBackoff, RateLimiter, CircuitBreaker
    >>> retry = ExponentialBackoff(max_attempts=3, base_delay=1.0, jitter=0.1)
    >>> result = await retry.execute(fetch_url, "https://example.com")
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar, ParamSpec

from typing_extensions import Self

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Random jitter factor (0.0 to 1.0)
        exponential_base: Base for exponential backoff
    """
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1
    exponential_base: float = 2.0


class RetryStrategy(ABC):
    """Abstract base class for retry strategies."""

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """Get delay for the given attempt number.

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        pass

    @abstractmethod
    def should_retry(self, attempt: int, exception: Exception | None = None) -> bool:
        """Determine if we should retry.

        Args:
            attempt: Current attempt number
            exception: The exception that caused the failure

        Returns:
            True if we should retry
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the retry strategy state."""
        pass


class ExponentialBackoff(RetryStrategy):
    """Exponential backoff retry strategy with jitter.

    Example:
        >>> strategy = ExponentialBackoff(max_attempts=5, base_delay=1.0)
        >>> for attempt in range(5):
        ...     delay = strategy.get_delay(attempt)
        ...     print(f"Attempt {attempt}: wait {delay:.2f}s")
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1,
        exponential_base: float = 2.0
    ) -> None:
        """Initialize exponential backoff.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Random jitter factor (0.0 to 1.0)
            exponential_base: Base for exponential backoff
        """
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            exponential_base=exponential_base
        )
        self._attempt_count = 0

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter.

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Calculate exponential delay
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)

        # Cap at max_delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter > 0:
            jitter_amount = delay * self.config.jitter
            delay = delay + random.uniform(-jitter_amount, jitter_amount)

        # Ensure non-negative
        return max(0, delay)

    def should_retry(self, attempt: int, exception: Exception | None = None) -> bool:
        """Check if we should retry based on attempt count.

        Args:
            attempt: Current attempt number
            exception: The exception that caused the failure (unused)

        Returns:
            True if we should retry
        """
        return attempt < self.config.max_attempts

    def reset(self) -> None:
        """Reset the retry counter."""
        self._attempt_count = 0

    @property
    def max_attempts(self) -> int:
        """Maximum retry attempts."""
        return self.config.max_attempts


class LinearBackoff(RetryStrategy):
    """Linear backoff retry strategy with jitter.

    Example:
        >>> strategy = LinearBackoff(max_attempts=3, base_delay=1.0)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.1
    ) -> None:
        """Initialize linear backoff.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            jitter: Random jitter factor (0.0 to 1.0)
        """
        self.config = RetryConfig(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            exponential_base=1.0
        )

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with linear backoff and jitter.

        Args:
            attempt: The attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.config.base_delay * (attempt + 1)
        delay = min(delay, self.config.max_delay)

        if self.config.jitter > 0:
            jitter_amount = delay * self.config.jitter
            delay = delay + random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def should_retry(self, attempt: int, exception: Exception | None = None) -> bool:
        """Check if we should retry."""
        return attempt < self.config.max_attempts

    def reset(self) -> None:
        """Reset (no-op for linear)."""
        pass


class FixedBackoff(RetryStrategy):
    """Fixed delay retry strategy.

    Example:
        >>> strategy = FixedBackoff(max_attempts=3, delay=2.0)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0
    ) -> None:
        """Initialize fixed backoff.

        Args:
            max_attempts: Maximum number of retry attempts
            delay: Fixed delay in seconds
        """
        self.max_attempts = max_attempts
        self.delay = delay

    def get_delay(self, attempt: int) -> float:
        """Return fixed delay.

        Args:
            attempt: The attempt number (unused)

        Returns:
            Fixed delay in seconds
        """
        return self.delay

    def should_retry(self, attempt: int, exception: Exception | None = None) -> bool:
        """Check if we should retry."""
        return attempt < self.max_attempts

    def reset(self) -> None:
        """Reset (no-op for fixed)."""
        pass


class RateLimiter:
    """Rate limiter with per-domain delays.

    Example:
        >>> limiter = RateLimiter(default_delay=1.0)
        >>> await limiter.acquire("example.com")
        >>> # ... make request ...
    """

    def __init__(
        self,
        default_delay: float = 1.0,
        domain_delays: dict[str, float] | None = None
    ) -> None:
        """Initialize rate limiter.

        Args:
            default_delay: Default delay between requests in seconds
            domain_delays: Per-domain delay overrides
        """
        self.default_delay = default_delay
        self.domain_delays = domain_delays or {}
        self._last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def get_delay(self, domain: str) -> float:
        """Get delay for a domain.

        Args:
            domain: Domain name

        Returns:
            Delay in seconds
        """
        return self.domain_delays.get(domain, self.default_delay)

    def set_delay(self, domain: str, delay: float) -> None:
        """Set custom delay for a domain.

        Args:
            domain: Domain name
            delay: Delay in seconds
        """
        self.domain_delays[domain] = delay

    async def acquire(self, domain: str) -> None:
        """Acquire the rate limit for a domain.

        Args:
            domain: Domain name
        """
        async with self._lock:
            now = time.monotonic()
            delay = self.get_delay(domain)

            if domain in self._last_request:
                elapsed = now - self._last_request[domain]
                wait_time = delay - elapsed
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self._last_request[domain] = time.monotonic()

    def sync_acquire(self, domain: str) -> None:
        """Synchronous acquire the rate limit for a domain.

        Args:
            domain: Domain name
        """
        now = time.monotonic()
        delay = self.get_delay(domain)

        if domain in self._last_request:
            elapsed = now - self._last_request[domain]
            wait_time = delay - elapsed
            if wait_time > 0:
                time.sleep(wait_time)

        self._last_request[domain] = time.monotonic()

    def reset(self, domain: str | None = None) -> None:
        """Reset rate limit state.

        Args:
            domain: Domain to reset, or None for all
        """
        if domain:
            self._last_request.pop(domain, None)
        else:
            self._last_request.clear()

    def get_wait_time(self, domain: str) -> float:
        """Get time until next request is allowed.

        Args:
            domain: Domain name

        Returns:
            Seconds until next request, or 0 if allowed now
        """
        delay = self.get_delay(domain)
        if domain in self._last_request:
            elapsed = time.monotonic() - self._last_request[domain]
            return max(0, delay - elapsed)
        return 0


class CircuitBreaker:
    """Circuit breaker for handling failing domains.

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        >>> try:
        ...     result = await breaker.execute(fetch_url, "https://example.com")
        ... except CircuitOpenError:
        ...     print("Circuit is open, skipping request")
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2,
        half_open_max_calls: int = 3
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit from half-open
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.half_open_max_calls = half_open_max_calls

        self._states: dict[str, CircuitState] = {}
        self._failure_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}
        self._last_failure_time: dict[str, float] = {}
        self._half_open_calls: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _get_state(self, domain: str) -> CircuitState:
        """Get current circuit state for domain."""
        if domain not in self._states:
            return CircuitState.CLOSED
        return self._states[domain]

    def _should_attempt_recovery(self, domain: str) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if domain not in self._last_failure_time:
            return False
        elapsed = time.monotonic() - self._last_failure_time[domain]
        return elapsed >= self.timeout

    def is_available(self, domain: str) -> bool:
        """Check if circuit allows requests for domain.

        Args:
            domain: Domain name

        Returns:
            True if requests are allowed
        """
        state = self._get_state(domain)

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if self._should_attempt_recovery(domain):
                self._states[domain] = CircuitState.HALF_OPEN
                self._half_open_calls[domain] = 0
                return True
            return False

        # Half-open: allow limited calls
        if state == CircuitState.HALF_OPEN:
            calls = self._half_open_calls.get(domain, 0)
            return calls < self.half_open_max_calls

        return False

    async def record_success(self, domain: str) -> None:
        """Record a successful request.

        Args:
            domain: Domain name
        """
        async with self._lock:
            state = self._get_state(domain)

            if state == CircuitState.HALF_OPEN:
                self._success_counts[domain] = self._success_counts.get(domain, 0) + 1
                if self._success_counts[domain] >= self.success_threshold:
                    self._states[domain] = CircuitState.CLOSED
                    self._failure_counts[domain] = 0
                    self._success_counts[domain] = 0
            elif state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_counts[domain] = 0

    async def record_failure(self, domain: str) -> None:
        """Record a failed request.

        Args:
            domain: Domain name
        """
        async with self._lock:
            self._failure_counts[domain] = self._failure_counts.get(domain, 0) + 1
            self._last_failure_time[domain] = time.monotonic()

            state = self._get_state(domain)

            if state == CircuitState.CLOSED:
                if self._failure_counts[domain] >= self.failure_threshold:
                    self._states[domain] = CircuitState.OPEN

            elif state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._states[domain] = CircuitState.OPEN

    async def execute(
        self,
        func: Callable[P, T],
        *args: P.args,
        domain: str | None = None,
        **kwargs: P.kwargs
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            domain: Domain for circuit tracking
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        domain = domain or "default"

        if not self.is_available(domain):
            raise CircuitOpenError(f"Circuit is open for {domain}")

        # Handle half-open state
        if self._get_state(domain) == CircuitState.HALF_OPEN:
            async with self._lock:
                self._half_open_calls[domain] = self._half_open_calls.get(domain, 0) + 1

        try:
            result = await func(*args, **kwargs)
            await self.record_success(domain)
            return result
        except Exception:
            await self.record_failure(domain)
            raise

    def sync_execute(
        self,
        func: Callable[P, T],
        *args: P.args,
        domain: str | None = None,
        **kwargs: P.kwargs
    ) -> T:
        """Synchronous execute with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments
            domain: Domain for circuit tracking
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        domain = domain or "default"

        if not self.is_available(domain):
            raise CircuitOpenError(f"Circuit is open for {domain}")

        # Handle half-open state
        if self._get_state(domain) == CircuitState.HALF_OPEN:
            self._half_open_calls[domain] = self._half_open_calls.get(domain, 0) + 1

        try:
            result = func(*args, **kwargs)
            # Use internal sync methods
            self._internal_record_success(domain)
            return result
        except Exception:
            # Use internal sync methods
            self._internal_record_failure(domain)
            raise

    def reset(self, domain: str | None = None) -> None:
        """Reset circuit breaker state.

        Args:
            domain: Domain to reset, or None for all
        """
        if domain:
            self._states.pop(domain, None)
            self._failure_counts.pop(domain, None)
            self._success_counts.pop(domain, None)
            self._last_failure_time.pop(domain, None)
            self._half_open_calls.pop(domain, None)
        else:
            self._states.clear()
            self._failure_counts.clear()
            self._success_counts.clear()
            self._last_failure_time.clear()
            self._half_open_calls.clear()

    def get_state(self, domain: str) -> CircuitState:
        """Get current circuit state for domain.

        Args:
            domain: Domain name

        Returns:
            Current circuit state
        """
        state = self._get_state(domain)

        # Check if we should transition from open to half-open
        if state == CircuitState.OPEN and self._should_attempt_recovery(domain):
            return CircuitState.HALF_OPEN

        return state

    def _internal_record_success(self, domain: str) -> None:
        """Internal sync version of record_success for sync_execute."""
        state = self._get_state(domain)

        if state == CircuitState.HALF_OPEN:
            self._success_counts[domain] = self._success_counts.get(domain, 0) + 1
            if self._success_counts[domain] >= self.success_threshold:
                self._states[domain] = CircuitState.CLOSED
                self._failure_counts[domain] = 0
                self._success_counts[domain] = 0
        elif state == CircuitState.CLOSED:
            self._failure_counts[domain] = 0

    def _internal_record_failure(self, domain: str) -> None:
        """Internal sync version of record_failure for sync_execute."""
        self._failure_counts[domain] = self._failure_counts.get(domain, 0) + 1
        self._last_failure_time[domain] = time.monotonic()
        state = self._get_state(domain)
        if state == CircuitState.CLOSED:
            if self._failure_counts[domain] >= self.failure_threshold:
                self._states[domain] = CircuitState.OPEN
        elif state == CircuitState.HALF_OPEN:
            self._states[domain] = CircuitState.OPEN


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on 429 responses.

    Example:
        >>> limiter = AdaptiveRateLimiter(initial_delay=1.0)
        >>> await limiter.acquire("example.com")
        >>> # On 429 response:
        >>> await limiter.record_response("example.com", status=429)
    """

    def __init__(
        self,
        initial_delay: float = 1.0,
        min_delay: float = 0.5,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0
    ) -> None:
        """Initialize adaptive rate limiter.

        Args:
            initial_delay: Initial delay between requests
            min_delay: Minimum delay floor
            max_delay: Maximum delay ceiling
            backoff_factor: Multiplier on 429 response
        """
        super().__init__(default_delay=initial_delay)
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def record_response(self, domain: str, status: int) -> None:
        """Record a response to adapt rate limiting.

        Args:
            domain: Domain name
            status: HTTP status code
        """
        async with self._lock:
            current_delay = self.get_delay(domain)

            if status == 429:
                # Increase delay
                new_delay = min(current_delay * self.backoff_factor, self.max_delay)
                self.set_delay(domain, new_delay)
            elif 200 <= status < 300:
                # Success - could decrease delay slowly
                if current_delay > self.min_delay:
                    new_delay = max(current_delay * 0.9, self.min_delay)
                    self.set_delay(domain, new_delay)


def with_retry(
    strategy: RetryStrategy | str = "exponential",
    max_attempts: int = 3,
    **kwargs: Any
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add retry logic to any function.

    Example:
        >>> @with_retry(strategy='exponential', max_attempts=3)
        ... def fetch_url(url: str) -> Response:
        ...     return requests.get(url)

        >>> @with_retry('linear', max_attempts=5, base_delay=2.0)
        ... async def fetch_async(url: str) -> Response:
        ...     return await async_get(url)
    """
    # Convert string strategy to strategy object
    if isinstance(strategy, str):
        if strategy == "exponential":
            strategy = ExponentialBackoff(max_attempts=max_attempts, **kwargs)
        elif strategy == "linear":
            strategy = LinearBackoff(max_attempts=max_attempts, **kwargs)
        elif strategy == "fixed":
            strategy = FixedBackoff(max_attempts=max_attempts, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Check if function is async
        import asyncio
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            async def wrapper(*args: P.args, **kwkwargs: P.kwargs) -> T:
                attempt = 0
                last_exception = None

                while strategy.should_retry(attempt, last_exception):
                    try:
                        return await func(*args, **kwkwargs)
                    except Exception as e:
                        last_exception = e
                        if strategy.should_retry(attempt + 1, e):
                            delay = strategy.get_delay(attempt)
                            await asyncio.sleep(delay)
                        attempt += 1

                # Exhausted retries
                if last_exception:
                    raise last_exception
                raise RetryExhaustedError(f"Retry exhausted after {max_attempts} attempts")

            return wrapper  # type: ignore[return-value]
        else:
            def wrapper(*args: P.args, **kwkwargs: P.kwargs) -> T:
                attempt = 0
                last_exception = None

                while strategy.should_retry(attempt, last_exception):
                    try:
                        return func(*args, **kwkwargs)
                    except Exception as e:
                        last_exception = e
                        if strategy.should_retry(attempt + 1, e):
                            delay = strategy.get_delay(attempt)
                            time.sleep(delay)
                        attempt += 1

                if last_exception:
                    raise last_exception
                raise RetryExhaustedError(f"Retry exhausted after {max_attempts} attempts")

            return wrapper  # type: ignore[return-value]

    return decorator


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


# Global instances for convenience
default_rate_limiter = RateLimiter()
default_circuit_breaker = CircuitBreaker()


__all__ = [
    # Config
    "RetryConfig",
    # Strategies
    "RetryStrategy",
    "ExponentialBackoff",
    "LinearBackoff",
    "FixedBackoff",
    # Rate limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    # Decorator
    "with_retry",
    "RetryExhaustedError",
    # Global instances
    "default_rate_limiter",
    "default_circuit_breaker",
]
