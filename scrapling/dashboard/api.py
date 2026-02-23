"""FastAPI application for scraper health monitoring dashboard."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Generator

from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware

from .database import MetricsDatabase, get_database
from .models import (
    DomainsResponse,
    EventLogResponse,
    HealthResponse,
    StatsResponse,
)


def create_app(db_instance: MetricsDatabase | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Scrapling Dashboard API",
        description="Backend API for scraper health monitoring",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Database instance - use provided instance or global
    _db: MetricsDatabase | None = db_instance

    def get_db() -> Generator[MetricsDatabase, None, None]:
        """Get database instance."""
        if _db is not None:
            yield _db
        else:
            yield get_database()

    @app.get("/api/stats", response_model=StatsResponse)
    async def get_stats(db: MetricsDatabase = Depends(get_db)):
        """Get overall statistics."""
        stats = db.get_stats()
        total = stats["total_requests"]
        cache_hits = stats["cache_hits"]
        cache_hit_rate = cache_hits / total if total > 0 else 0.0

        return StatsResponse(
            total_requests=stats["total_requests"],
            total_errors=stats["total_errors"],
            cache_hits=stats["cache_hits"],
            cache_misses=stats["cache_misses"],
            cache_hit_rate=round(cache_hit_rate, 3),
        )

    @app.get("/api/domains", response_model=DomainsResponse)
    async def get_domains(db: MetricsDatabase = Depends(get_db)):
        """Get per-domain statistics."""
        domain_stats = db.get_domain_stats()
        return DomainsResponse(domains=domain_stats)

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check(db: MetricsDatabase = Depends(get_db)):
        """Service health check."""
        try:
            # Simple query to check DB
            db.get_stats()
            db_status = "healthy"
        except Exception:
            db_status = "unhealthy"

        return HealthResponse(
            status="healthy" if db_status == "healthy" else "degraded",
            database=db_status,
            timestamp=datetime.now(timezone.utc),
        )

    @app.get("/api/events", response_model=EventLogResponse)
    async def get_events(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
        event_type: Annotated[
            Literal["fetch", "parse", "save", "error"] | None, Query()
        ] = None,
        db: MetricsDatabase = Depends(get_db),
    ):
        """Get paginated event log."""
        events, total = db.get_events(limit=limit, offset=offset, event_type=event_type)
        page = (offset // limit) + 1

        return EventLogResponse(
            events=events,
            total=total,
            page=page,
            limit=limit,
        )

    return app


# Default app instance
app = create_app()
