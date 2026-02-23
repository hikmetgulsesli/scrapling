"""Tests for Monitoring Dashboard Frontend (US-007).

Tests cover:
1. Dashboard config and design tokens
2. Frontend integration points
"""

import pytest

from scrapling.dashboard import get_dashboard_config


class TestDashboardConfig:
    """Tests for dashboard configuration."""

    def test_get_dashboard_config(self):
        """Test getting dashboard config."""
        config = get_dashboard_config()

        assert config["theme"] == "dark"
        assert "palette" in config
        assert config["palette"]["primary"] == "#22d3ee"
        assert config["palette"]["accent"] == "#a3e635"

    def test_palette_colors(self):
        """Test palette colors match DevTool."""
        config = get_dashboard_config()
        palette = config["palette"]

        # Verify DevTool palette
        assert palette["primary"] == "#22d3ee"  # cyan-400
        assert palette["accent"] == "#a3e635"   # lime-400
        assert palette["surface"] == "#18181b"  # zinc-900
        assert palette["background"] == "#09090b"  # zinc-950

    def test_refresh_interval(self):
        """Test refresh interval configuration."""
        config = get_dashboard_config()

        assert config["refreshInterval"] == 5000  # 5 seconds


class TestDesignTokens:
    """Tests for design tokens CSS."""

    def test_css_file_exists(self):
        """Test that design tokens CSS file exists."""
        import os
        from scrapling.dashboard import __file__ as dashboard_init

        dashboard_dir = os.path.dirname(dashboard_init)
        css_path = os.path.join(dashboard_dir, "design-tokens.css")

        assert os.path.exists(css_path), "design-tokens.css should exist"

    def test_css_contains_variables(self):
        """Test that CSS contains design token variables."""
        import os
        from scrapling.dashboard import __file__ as dashboard_init

        dashboard_dir = os.path.dirname(dashboard_init)
        css_path = os.path.join(dashboard_dir, "design-tokens.css")

        with open(css_path) as f:
            content = f.read()

        # Check for key design tokens
        assert "--primary-400: #22d3ee" in content
        assert "--accent-400: #a3e635" in content
        assert "--surface-900: #18181b" in content


# Mark tests
pytestmark = pytest.mark.dashboard
