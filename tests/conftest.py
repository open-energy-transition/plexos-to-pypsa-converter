"""Shared pytest fixtures and configuration for all tests."""

from pathlib import Path
from typing import Any

import pandas as pd
import pypsa
import pytest

# ============================================================================
# Path Fixtures
# ============================================================================


@pytest.fixture
def fixtures_path() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_data_path(fixtures_path) -> Path:
    """Return path to sample data files."""
    return fixtures_path / "sample_data"


# ============================================================================
# PyPSA Network Fixtures
# ============================================================================


@pytest.fixture
def empty_network() -> pypsa.Network:
    """Create an empty PyPSA network for testing."""
    return pypsa.Network()


@pytest.fixture
def network_with_snapshots() -> pypsa.Network:
    """Create PyPSA network with hourly snapshots for 24 hours."""
    n = pypsa.Network()
    snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
    n.set_snapshots(snapshots)
    return n


@pytest.fixture
def network_with_bus(empty_network) -> pypsa.Network:
    """Create PyPSA network with a single bus."""
    network = empty_network
    network.add("Bus", name="Bus1", carrier="AC", v_nom=110)
    return network


@pytest.fixture
def network_with_generator(network_with_bus) -> pypsa.Network:
    """Create PyPSA network with a bus and generator."""
    network = network_with_bus
    network.add(
        "Generator",
        name="Gen1",
        bus="Bus1",
        p_nom=100,
        marginal_cost=50,
        carrier="Gas",
    )
    return network


@pytest.fixture
def network_with_generators(network_with_bus) -> pypsa.Network:
    """Create PyPSA network with a bus and 3 generators."""
    network = network_with_bus
    network.add("Generator", "Gen1", bus="Bus1", p_nom=100, carrier="Gas")
    network.add("Generator", "Gen2", bus="Bus1", p_nom=200, carrier="Coal")
    network.add("Generator", "Gen3", bus="Bus1", p_nom=150, carrier="Solar")
    return network


@pytest.fixture
def network_with_multiple_buses(empty_network) -> pypsa.Network:
    """Create PyPSA network with multiple buses."""
    network = empty_network
    network.add("Bus", "Bus1", carrier="AC", v_nom=110)
    network.add("Bus", "Bus2", carrier="AC", v_nom=110)
    network.add("Bus", "Bus3", carrier="AC", v_nom=220)
    return network


# ============================================================================
# Mock PlexosDB Fixtures
# ============================================================================


@pytest.fixture
def mock_plexos_db(mocker):
    """Create a mock PlexosDB instance with common methods."""
    mock_db = mocker.MagicMock()

    # Default return values for common methods
    mock_db.list_objects_by_class.return_value = []
    mock_db.get_object_properties.return_value = []
    mock_db.get_memberships_system.return_value = []

    return mock_db


@pytest.fixture
def mock_plexos_db_with_nodes(mock_plexos_db):
    """Mock PlexosDB with sample nodes."""
    from plexosdb.enums import ClassEnum

    # Return sample nodes when list_objects_by_class is called with ClassEnum.Node
    def side_effect_list(class_enum):
        if class_enum == ClassEnum.Node:
            return ["Bus1", "Bus2", "Bus3"]
        return []

    mock_plexos_db.list_objects_by_class.side_effect = side_effect_list
    return mock_plexos_db


@pytest.fixture
def mock_plexos_db_with_generator(mock_plexos_db):
    """Mock PlexosDB with a sample generator."""
    from plexosdb.enums import ClassEnum

    def side_effect_list(class_enum):
        if class_enum == ClassEnum.Generator:
            return ["Gen1"]
        elif class_enum == ClassEnum.Node:
            return ["Bus1"]
        return []

    def side_effect_props(class_enum, name):
        if class_enum == ClassEnum.Generator and name == "Gen1":
            return [
                {"property": "Max Capacity", "value": "100"},
                {"property": "Min Capacity", "value": "0"},
                {"property": "Heat Rate Base", "value": "10"},
                {"property": "VO&M Charge", "value": "5"},
            ]
        return []

    def side_effect_memberships(name, object_class):
        if name == "Gen1":
            return [
                {
                    "name": "Gen1",
                    "collection_name": "Node",
                    "child_class_name": "Node",
                    "child_object_name": "Bus1",
                }
            ]
        return []

    mock_plexos_db.list_objects_by_class.side_effect = side_effect_list
    mock_plexos_db.get_object_properties.side_effect = side_effect_props
    mock_plexos_db.get_memberships_system.side_effect = side_effect_memberships

    return mock_plexos_db


# ============================================================================
# Temporary File Fixtures
# ============================================================================


@pytest.fixture
def temp_csv_file(tmp_path) -> Path:
    """Create a temporary CSV file."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_demand_csv(tmp_path) -> Path:
    """Create sample demand CSV file with hourly data for 24 hours."""
    csv_file = tmp_path / "demand.csv"

    # Create hourly demand data for one day
    data = {
        "Year": [2024] * 24,
        "Month": [1] * 24,
        "Day": [1] * 24,
    }

    # Add 24 hourly columns
    for hour in range(1, 25):
        data[str(hour)] = [100 + hour * 5] * 1  # Simple pattern

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_demand_csv_with_period(tmp_path) -> Path:
    """Create sample demand CSV with Period column (CAISO/SEM format)."""
    csv_file = tmp_path / "demand_period.csv"

    # Create data for 2 days, 24 periods each
    rows = []
    for day in range(1, 3):
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": day,
                    "Period": period,
                    "Zone1": 100 + period * 2,
                    "Zone2": 150 + period * 3,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_vre_profile(tmp_path) -> Path:
    """Create sample VRE (renewable) generation profile CSV."""
    csv_file = tmp_path / "solar_profile.csv"

    data = {
        "Year": [2024] * 24,
        "Month": [1] * 24,
        "Day": [1] * 24,
    }

    # Add 24 hourly capacity factor columns (0-1 range)
    for hour in range(1, 25):
        # Simple solar pattern: 0 at night, peak at noon
        if 6 <= hour <= 18:
            cf = 0.1 + 0.7 * (1 - abs(12 - hour) / 6)
        else:
            cf = 0.0
        data[str(hour)] = [cf]

    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_demand_directory(tmp_path) -> Path:
    """Create directory with multiple demand CSV files."""
    demand_dir = tmp_path / "demand"
    demand_dir.mkdir()

    # Create Bus1_demand.csv, Bus2_demand.csv
    for bus in ["Bus1", "Bus2"]:
        csv_file = demand_dir / f"{bus}_demand.csv"
        data = {
            "Year": [2024] * 2,
            "Month": [1] * 2,
            "Day": [1, 2],
        }
        # Add 24 hourly columns
        for h in range(1, 25):
            data[str(h)] = [100 + h * 5] * 2

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

    return demand_dir


@pytest.fixture
def sample_vre_profiles_dir(tmp_path) -> Path:
    """Create directory with solar/wind profile CSVs."""
    traces_dir = tmp_path / "Traces"
    (traces_dir / "solar").mkdir(parents=True)
    (traces_dir / "wind").mkdir(parents=True)

    # Solar profile
    solar_csv = traces_dir / "solar" / "Gen_Solar.csv"
    data = {
        "Year": [2024] * 2,
        "Month": [1] * 2,
        "Day": [1, 2],
    }
    for h in range(1, 25):
        # Simple solar pattern
        if 6 <= h <= 18:
            cf = 0.8
        else:
            cf = 0.0
        data[str(h)] = [cf] * 2
    df = pd.DataFrame(data)
    df.to_csv(solar_csv, index=False)

    # Wind profile
    wind_csv = traces_dir / "wind" / "Gen_Wind.csv"
    data = {
        "Year": [2024] * 2,
        "Month": [1] * 2,
        "Day": [1, 2],
    }
    for h in range(1, 25):
        data[str(h)] = [0.6] * 2  # Constant wind
    df = pd.DataFrame(data)
    df.to_csv(wind_csv, index=False)

    return tmp_path


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def sample_properties() -> list[dict[str, Any]]:
    """Return sample PLEXOS properties list."""
    return [
        {"property": "Max Capacity", "value": "100"},
        {"property": "Min Capacity", "value": "0"},
        {"property": "Heat Rate Base", "value": "10"},
        {"property": "Heat Rate Incr1", "value": "9.5"},
        {"property": "VO&M Charge", "value": "5"},
        {"property": "Build Cost", "value": "1000"},
        {"property": "WACC", "value": "0.08"},
        {"property": "Economic Life", "value": "30"},
    ]


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (slower)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (>1 second)")
    config.addinivalue_line(
        "markers", "requires_solver: mark test as requiring an optimization solver"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data files"
    )
