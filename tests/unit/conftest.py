"""Pytest fixtures for unit tests."""

import pandas as pd
import pypsa
import pytest


@pytest.fixture
def empty_network():
    """Create an empty PyPSA network for testing."""
    return pypsa.Network()


@pytest.fixture
def network_with_buses():
    """Create a network with 2 buses."""
    network = pypsa.Network()
    network.add("Bus", "Bus1")
    network.add("Bus", "Bus2")
    return network


@pytest.fixture
def network_with_generators(network_with_buses):
    """Create a network with buses and 4 generators (2 thermal, 2 VRE)."""
    network = network_with_buses

    # Add carriers
    network.add("Carrier", "gas")
    network.add("Carrier", "coal")

    # Thermal generators
    network.add(
        "Generator",
        "Gas1",
        bus="Bus1",
        carrier="gas",
        p_nom=100,
        marginal_cost=50,
        ramp_limit_up=0.05,
        ramp_limit_down=0.05,
    )
    network.add(
        "Generator",
        "Coal1",
        bus="Bus1",
        carrier="coal",
        p_nom=200,
        marginal_cost=30,
        ramp_limit_up=0.02,
        ramp_limit_down=0.02,
    )

    # VRE generators (empty carrier indicates VRE in test setup)
    network.add(
        "Generator",
        "Wind1",
        bus="Bus2",
        carrier="",  # Empty carrier = VRE
        p_nom=50,
        marginal_cost=0,
    )
    network.add(
        "Generator",
        "Solar1",
        bus="Bus2",
        carrier="",  # Empty carrier = VRE
        p_nom=30,
        marginal_cost=0,
    )

    return network


@pytest.fixture
def network_with_snapshots(network_with_buses):
    """Create a network with 24 hourly snapshots."""
    network = network_with_buses
    network.set_snapshots(pd.date_range("2023-01-01", periods=24, freq="H"))
    return network


@pytest.fixture
def network_with_p_min_pu(network_with_generators):
    """Create a network with generators that have p_min_pu time series."""
    network = network_with_generators
    snapshots = pd.date_range("2023-01-01", periods=24, freq="H")
    network.set_snapshots(snapshots)

    # Add p_min_pu for thermal generators
    network.generators_t.p_min_pu["Gas1"] = [0.4] * len(snapshots)
    network.generators_t.p_min_pu["Coal1"] = [0.6] * len(snapshots)

    return network


@pytest.fixture
def network_with_storage(network_with_buses):
    """Create a network with a storage unit."""
    network = network_with_buses
    network.add(
        "StorageUnit",
        "Hydro1",
        bus="Bus1",
        p_nom=80,
        max_hours=12.5,  # 1000 MWh / 80 MW
        efficiency_store=0.9,
        efficiency_dispatch=0.9,
    )
    return network


@pytest.fixture
def network_with_carriers(empty_network):
    """Create a network with various carriers for testing filters."""
    network = empty_network
    network.add("Bus", "Bus1")

    # Add carriers
    network.add("Carrier", "gas")
    network.add("Carrier", "coal")
    network.add("Carrier", "wind")
    network.add("Carrier", "solar")

    # Add generators with different carriers
    network.add("Generator", "CCGT", bus="Bus1", carrier="gas", p_nom=100)
    network.add("Generator", "Coal Plant", bus="Bus1", carrier="coal", p_nom=200)
    network.add(
        "Generator", "Wind Farm", bus="Bus1", carrier="", p_nom=50
    )  # Empty = VRE
    network.add(
        "Generator", "Solar Farm", bus="Bus1", carrier="", p_nom=30
    )  # Empty = VRE

    return network
