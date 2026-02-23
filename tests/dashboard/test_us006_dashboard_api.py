"""Tests for US-006: Monitoring Dashboard Backend API."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scrapling.dashboard.api import create_app
from scrapling.dashboard.database import MetricsDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = MetricsDatabase(db_path)
    yield db

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def client(temp_db):
    """Create a test client with temp database."""
    app = create_app(db_instance=temp_db)
    return TestClient(app)


class TestStatsEndpoint:
    """Tests for /api/stats endpoint."""

    def test_stats_empty(self, client):
        """Test stats returns zeros when no data."""
        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_requests" in data
        assert "total_errors" in data
        assert "cache_hits" in data
        assert "cache_misses" in data
        assert "cache_hit_rate" in data
        assert data["total_requests"] == 0

    def test_stats_with_data(self, client, temp_db):
        """Test stats with logged events."""
        # Log some events
        temp_db.log_event("fetch", "example.com", "success", cached=True)
        temp_db.log_event("fetch", "example.com", "success", cached=False)
        temp_db.log_event("error", "example.com", "error", "Test error")

        response = client.get("/api/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 2
        assert data["total_errors"] == 1
        assert data["cache_hits"] == 1
        assert data["cache_misses"] == 1


class TestDomainsEndpoint:
    """Tests for /api/domains endpoint."""

    def test_domains_empty(self, client):
        """Test domains returns empty list when no data."""
        response = client.get("/api/domains")
        assert response.status_code == 200

        data = response.json()
        assert "domains" in data
        assert data["domains"] == []

    def test_domains_with_data(self, client, temp_db):
        """Test domains with logged events."""
        temp_db.log_event("fetch", "example.com", "success")
        temp_db.log_event("fetch", "example.com", "success")
        temp_db.log_event("fetch", "test.org", "success")
        temp_db.log_event("error", "example.com", "error", "Test")

        response = client.get("/api/domains")
        assert response.status_code == 200

        data = response.json()
        assert len(data["domains"]) == 2

        # Check example.com
        example = next(d for d in data["domains"] if d["domain"] == "example.com")
        assert example["requests"] == 3
        assert example["errors"] == 1

        # Check test.org
        test = next(d for d in data["domains"] if d["domain"] == "test.org")
        assert test["requests"] == 1
        assert test["errors"] == 0


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""

    def test_health_healthy(self, client):
        """Test health check when database is healthy."""
        response = client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert data["database"] in ["healthy", "unhealthy"]
        assert "timestamp" in data

    def test_health_timestamp_format(self, client):
        """Test health timestamp is valid ISO format."""
        response = client.get("/api/health")
        data = response.json()

        # Should parse without error
        timestamp = datetime.fromisoformat(data["timestamp"])
        assert timestamp is not None


class TestEventsEndpoint:
    """Tests for /api/events endpoint."""

    def test_events_empty(self, client):
        """Test events returns empty when no data."""
        response = client.get("/api/events")
        assert response.status_code == 200

        data = response.json()
        assert "events" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert data["events"] == []
        assert data["total"] == 0

    def test_events_with_data(self, client, temp_db):
        """Test events with logged events."""
        temp_db.log_event("fetch", "example.com", "success")
        temp_db.log_event("parse", "example.com", "success")
        temp_db.log_event("save", "example.com", "success")
        temp_db.log_event("error", "example.com", "error", "Test error")

        response = client.get("/api/events")
        assert response.status_code == 200

        data = response.json()
        assert len(data["events"]) == 4
        assert data["total"] == 4
        assert data["events"][0]["event_type"] == "error"  # Most recent first

    def test_events_pagination(self, client, temp_db):
        """Test events pagination."""
        for i in range(15):
            temp_db.log_event("fetch", f"domain{i}.com", "success")

        # First page
        response = client.get("/api/events?limit=10&offset=0")
        data = response.json()
        assert len(data["events"]) == 10
        assert data["total"] == 15
        assert data["page"] == 1
        assert data["limit"] == 10

        # Second page
        response = client.get("/api/events?limit=10&offset=10")
        data = response.json()
        assert len(data["events"]) == 5
        assert data["page"] == 2

    def test_events_filter_by_type(self, client, temp_db):
        """Test filtering events by type."""
        temp_db.log_event("fetch", "example.com", "success")
        temp_db.log_event("error", "example.com", "error")
        temp_db.log_event("error", "test.org", "error")

        # Filter by fetch
        response = client.get("/api/events?event_type=fetch")
        data = response.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["event_type"] == "fetch"

        # Filter by error
        response = client.get("/api/events?event_type=error")
        data = response.json()
        assert len(data["events"]) == 2
        assert all(e["event_type"] == "error" for e in data["events"])


class TestEventLogging:
    """Tests for event logging functionality."""

    def test_log_fetch_event(self, temp_db):
        """Test logging a fetch event."""
        event_id = temp_db.log_event(
            "fetch", "example.com", "success", duration_ms=150.5, cached=True
        )
        assert event_id > 0

        stats = temp_db.get_stats()
        assert stats["total_requests"] == 1
        assert stats["cache_hits"] == 1

    def test_log_error_event(self, temp_db):
        """Test logging an error event."""
        event_id = temp_db.log_event(
            "error", "example.com", "error", "Connection timeout"
        )
        assert event_id > 0

        stats = temp_db.get_stats()
        assert stats["total_errors"] == 1

    def test_log_all_event_types(self, temp_db):
        """Test logging all event types."""
        temp_db.log_event("fetch", "example.com", "success")
        temp_db.log_event("parse", "example.com", "success")
        temp_db.log_event("save", "example.com", "success")
        temp_db.log_event("error", "example.com", "error")

        events, total = temp_db.get_events()
        assert total == 4

        event_types = {e.event_type for e in events}
        assert event_types == {"fetch", "parse", "save", "error"}


class TestDatabasePersistence:
    """Tests for database persistence."""

    def test_stats_persistence(self, temp_db):
        """Test that stats persist across get_stats calls."""
        temp_db.log_event("fetch", "example.com", "success")

        stats1 = temp_db.get_stats()
        stats2 = temp_db.get_stats()

        assert stats1 == stats2

    def test_domain_stats_ordering(self, temp_db):
        """Test domain stats are ordered by requests descending."""
        temp_db.log_event("fetch", "aaa.com", "success")
        temp_db.log_event("fetch", "aaa.com", "success")
        temp_db.log_event("fetch", "zzz.com", "success")

        domains = temp_db.get_domain_stats()
        assert domains[0]["domain"] == "aaa.com"
        assert domains[1]["domain"] == "zzz.com"
