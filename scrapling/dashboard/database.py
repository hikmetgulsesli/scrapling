"""Database module for dashboard metrics storage."""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import DomainStats, ScraperEvent, StatsSummary


class MetricsDatabase:
    """SQLite-backed metrics storage for scraper monitoring."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the database."""
        if db_path is None:
            db_path = Path.home() / ".scrapling" / "metrics.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False  # Allow multi-threaded access
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scraper_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                duration_ms REAL,
                cached INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON scraper_events(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_domain ON scraper_events(domain)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_type ON scraper_events(event_type)"
        )
        conn.commit()

    def log_event(
        self,
        event_type: str,
        domain: str,
        status: str,
        message: str | None = None,
        duration_ms: float | None = None,
        cached: bool = False,
    ) -> int:
        """Log a scraper event."""
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO scraper_events
            (timestamp, event_type, domain, status, message, duration_ms, cached)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                event_type,
                domain,
                status,
                message,
                duration_ms,
                1 if cached else 0,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    def get_stats(self) -> StatsSummary:
        """Get overall statistics."""
        conn = self._get_conn()

        # Total requests (fetch events)
        total_requests = conn.execute(
            "SELECT COUNT(*) FROM scraper_events WHERE event_type = ?",
            ("fetch",),
        ).fetchone()[0]

        # Total errors (event_type = error)
        total_errors = conn.execute(
            "SELECT COUNT(*) FROM scraper_events WHERE event_type = ?",
            ("error",),
        ).fetchone()[0]

        # Cache hits (cached=true for fetch events)
        cache_hits = conn.execute(
            "SELECT COUNT(*) FROM scraper_events WHERE event_type = ? AND cached = ?",
            ("fetch", 1),
        ).fetchone()[0]

        # Cache misses
        cache_misses = conn.execute(
            "SELECT COUNT(*) FROM scraper_events WHERE event_type = ? AND cached = ?",
            ("fetch", 0),
        ).fetchone()[0]

        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }

    def get_domain_stats(self) -> list[DomainStats]:
        """Get per-domain statistics."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT
                domain,
                COUNT(*) as requests,
                SUM(CASE WHEN event_type = 'error' THEN 1 ELSE 0 END) as errors,
                SUM(CASE WHEN cached = 1 AND event_type = 'fetch' THEN 1 ELSE 0 END) as cache_hits,
                SUM(CASE WHEN cached = 0 AND event_type = 'fetch' THEN 1 ELSE 0 END) as cache_misses
            FROM scraper_events
            GROUP BY domain
            ORDER BY requests DESC
            """
        ).fetchall()

        return [
            {
                "domain": row["domain"],
                "requests": row["requests"],
                "errors": row["errors"] or 0,
                "cache_hits": row["cache_hits"] or 0,
                "cache_misses": row["cache_misses"] or 0,
            }
            for row in rows
        ]

    def get_events(
        self, limit: int = 100, offset: int = 0, event_type: str | None = None
    ) -> tuple[list[ScraperEvent], int]:
        """Get paginated event log."""
        conn = self._get_conn()
        where_clause = ""
        params: list[Any] = []
        if event_type:
            where_clause = "WHERE event_type = ?"
            params.append(event_type)

        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM scraper_events {where_clause}", params
        ).fetchone()[0]

        # Get events
        rows = conn.execute(
            f"""
            SELECT * FROM scraper_events
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()

        events = [
            ScraperEvent(
                id=row["id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type=row["event_type"],
                domain=row["domain"],
                status=row["status"],
                message=row["message"],
                duration_ms=row["duration_ms"],
                cached=bool(row["cached"]),
            )
            for row in rows
        ]

        return events, total

    def clear_events(self) -> None:
        """Clear all events (for testing)."""
        conn = self._get_conn()
        conn.execute("DELETE FROM scraper_events")
        conn.commit()


# Global database instance
_db: MetricsDatabase | None = None


def get_database(db_path: str | Path | None = None) -> MetricsDatabase:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = MetricsDatabase(db_path)
    return _db
