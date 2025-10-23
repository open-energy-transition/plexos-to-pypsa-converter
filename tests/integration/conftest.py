"""Pytest fixtures for integration tests."""

import pytest


@pytest.fixture(scope="session")
def sem_model_id():
    """Return the SEM model ID."""
    return "sem-2024-2032"


@pytest.fixture(scope="session")
def aemo_model_id():
    """Return the AEMO model ID."""
    return "aemo-2024-isp-progressive-change"


@pytest.fixture(scope="session")
def snapshot_limit():
    """Return the default snapshot limit for fast testing."""
    return 50
