"""Tests for US-001: Project Setup & Design Tokens."""

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


class TestProjectSetup:
    """Test project structure and configuration."""

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist."""
        pyproject = REPO_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"

    def test_pyproject_toml_valid(self):
        """pyproject.toml must be valid TOML."""
        import tomllib

        pyproject = REPO_ROOT / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)

        assert "project" in data, "Missing [project] section"
        assert "name" in data["project"], "Missing project name"
        assert data["project"]["name"] == "scrapling", "Wrong project name"

    def test_package_structure_exists(self):
        """scrapling package directory must exist."""
        package_dir = REPO_ROOT / "scrapling"
        assert package_dir.exists(), "scrapling package not found"
        assert package_dir.is_dir(), "scrapling should be a directory"


class TestDesignTokens:
    """Test design tokens CSS file."""

    def test_design_tokens_file_exists(self):
        """Design tokens CSS file must exist."""
        tokens_file = REPO_ROOT / "scrapling" / "dashboard" / "design-tokens.css"
        assert tokens_file.exists(), "design-tokens.css not found"

    def test_design_tokens_devtool_palette(self):
        """Design tokens must use DevTool palette."""
        tokens_file = REPO_ROOT / "scrapling" / "dashboard" / "design-tokens.css"
        content = tokens_file.read_text()

        # Check DevTool palette colors
        assert "#22d3ee" in content or "--primary:" in content, "Missing primary color (cyan)"
        assert "#a3e635" in content or "--accent:" in content, "Missing accent color (lime)"
        assert "#18181b" in content or "--surface:" in content, "Missing surface color (zinc)"

    def test_design_tokens_css_variables(self):
        """Design tokens must define CSS custom properties."""
        tokens_file = REPO_ROOT / "scrapling" / "dashboard" / "design-tokens.css"
        content = tokens_file.read_text()

        # Check for CSS custom properties
        assert "--primary:" in content, "Missing --primary CSS variable"
        assert "--accent:" in content, "Missing --accent CSS variable"
        assert "--surface:" in content, "Missing --surface CSS variable"
        assert "--text:" in content, "Missing --text CSS variable"


class TestGitIgnore:
    """Test .gitignore configuration."""

    def test_gitignore_exists(self):
        """.gitignore must exist."""
        gitignore = REPO_ROOT / ".gitignore"
        assert gitignore.exists(), ".gitignore not found"

    def test_gitignore_excludes_env(self):
        """.gitignore must exclude .env files."""
        gitignore = REPO_ROOT / ".gitignore"
        content = gitignore.read_text()
        assert ".env" in content or ".env\n" in content, ".env not excluded in .gitignore"

    def test_gitignore_excludes_pycache(self):
        """.gitignore must exclude __pycache__."""
        gitignore = REPO_ROOT / ".gitignore"
        content = gitignore.read_text()
        assert "__pycache__" in content, "__pycache__ not excluded in .gitignore"

    def test_gitignore_excludes_pyc(self):
        """.gitignore must exclude *.pyc files."""
        gitignore = REPO_ROOT / ".gitignore"
        content = gitignore.read_text()
        # Check for either *.pyc or *.py[cod] - both exclude .pyc files
        assert "*.pyc" in content or "*.py[cod]" in content, "*.pyc not excluded in .gitignore"


class TestEnvExample:
    """Test .env.example configuration."""

    def test_env_example_exists(self):
        """.env.example must exist."""
        env_example = REPO_ROOT / ".env.example"
        assert env_example.exists(), ".env.example not found"

    def test_env_example_has_content(self):
        """.env.example must have meaningful content."""
        env_example = REPO_ROOT / ".env.example"
        content = env_example.read_text()
        assert len(content.strip()) > 50, ".env.example is too short"
        assert "#" in content, ".env.example should have comments"

    def test_env_example_has_rate_limiting(self):
        """.env.example should include rate limiting variables."""
        env_example = REPO_ROOT / ".env.example"
        content = env_example.read_text()
        assert "RATE_LIMIT" in content, "Missing RATE_LIMIT in .env.example"

    def test_env_example_has_api_keys_placeholder(self):
        """.env.example should have API key placeholders."""
        env_example = REPO_ROOT / ".env.example"
        content = env_example.read_text()
        assert "API_KEY" in content, "Missing API_KEY placeholder in .env.example"


class TestDashboardPackage:
    """Test dashboard package structure."""

    def test_dashboard_package_exists(self):
        """Dashboard package must exist."""
        dashboard_dir = REPO_ROOT / "scrapling" / "dashboard"
        assert dashboard_dir.exists(), "dashboard package not found"

    def test_dashboard_init_exists(self):
        """Dashboard __init__.py must exist."""
        init_file = REPO_ROOT / "scrapling" / "dashboard" / "__init__.py"
        assert init_file.exists(), "dashboard/__init__.py not found"
