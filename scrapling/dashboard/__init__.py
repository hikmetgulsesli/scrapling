"""Monitoring Dashboard - Frontend components for scraper health visualization.

This module provides React components for the monitoring dashboard.
For the actual React components, see the dashboard frontend package.
"""

from typing import Any

# This module serves as a placeholder/interface for the frontend dashboard.
# The React frontend is served separately via the monitoring dashboard.

__all__ = []


def get_dashboard_config() -> dict[str, Any]:
    """Get dashboard configuration."""
    return {
        "theme": "dark",
        "palette": {
            "primary": "#22d3ee",  # cyan-400
            "accent": "#a3e635",   # lime-400
            "surface": "#18181b",  # zinc-900
            "background": "#09090b", # zinc-950
            "text": "#fafafa",     # zinc-50
            "textMuted": "#a1a1aa", # zinc-400
        },
        "refreshInterval": 5000,  # ms
    }
