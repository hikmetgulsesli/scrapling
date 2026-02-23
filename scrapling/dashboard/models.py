"""Database models for dashboard metrics."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class StatsSummary(TypedDict):
    """Stats summary type."""

    total_requests: int
    total_errors: int
    cache_hits: int
    cache_misses: int


class DomainStats(TypedDict):
    """Per-domain stats type."""

    domain: str
    requests: int
    errors: int
    cache_hits: int
    cache_misses: int


class ScraperEvent(BaseModel):
    """Scraper event model."""

    id: int | None = None
    timestamp: datetime
    event_type: Literal["fetch", "parse", "save", "error"]
    domain: str
    status: str
    message: str | None = None
    duration_ms: float | None = None
    cached: bool = False


class StatsResponse(BaseModel):
    """Stats API response."""

    total_requests: int
    total_errors: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float


class DomainsResponse(BaseModel):
    """Domains API response."""

    domains: list[DomainStats]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: str
    timestamp: datetime


class EventLogResponse(BaseModel):
    """Event log response."""

    events: list[ScraperEvent]
    total: int
    page: int
    limit: int
