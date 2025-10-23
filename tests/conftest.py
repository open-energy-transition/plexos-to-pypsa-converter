"""Shared pytest fixtures for all tests."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def minimal_model_dir(fixtures_dir):
    """Return path to minimal test model directory."""
    return fixtures_dir / "minimal_model"


@pytest.fixture
def minimal_model_csv_dir(minimal_model_dir):
    """Return path to minimal model CSV exports."""
    return minimal_model_dir / "csvs_from_xml" / "Test System"


@pytest.fixture
def timeseries_dir(fixtures_dir):
    """Return path to timeseries data directory."""
    return fixtures_dir / "timeseries"


@pytest.fixture
def sample_snapshots():
    """Return a sample datetime index for testing (24 hours)."""
    return pd.date_range("2023-01-01", periods=24, freq="H")


@pytest.fixture
def sample_snapshots_year():
    """Return a full year of hourly snapshots for testing."""
    return pd.date_range("2023-01-01", periods=8760, freq="H")
