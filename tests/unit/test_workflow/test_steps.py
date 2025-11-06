"""Unit tests for workflow step functions."""

import pytest

from plexos_to_pypsa_converter.workflow.steps import (
    add_curtailment_link_step,
    scale_p_min_pu_step,
)


class TestScalePMinPuStep:
    """Test p_min_pu scaling step."""

    def test_scale_p_min_pu_basic(self, network_with_p_min_pu):
        """Test basic p_min_pu scaling."""
        network = network_with_p_min_pu

        # Store original values
        original_gas = network.generators_t.p_min_pu["Gas1"].iloc[0]
        original_coal = network.generators_t.p_min_pu["Coal1"].iloc[0]

        # Scale by 0.7
        summary = scale_p_min_pu_step(network, scaling_factor=0.7)

        # Check values were scaled
        assert network.generators_t.p_min_pu["Gas1"].iloc[0] == pytest.approx(
            original_gas * 0.7
        )
        assert network.generators_t.p_min_pu["Coal1"].iloc[0] == pytest.approx(
            original_coal * 0.7
        )

        # Check summary
        assert "scale_p_min_pu" in summary
        assert summary["scale_p_min_pu"]["scaling_factor"] == 0.7
        assert summary["scale_p_min_pu"]["generators_scaled"] == 2

    def test_scale_p_min_pu_default_factor(self, network_with_p_min_pu):
        """Test p_min_pu scaling with default factor (0.7)."""
        network = network_with_p_min_pu

        summary = scale_p_min_pu_step(network)

        assert summary["scale_p_min_pu"]["scaling_factor"] == 0.7

    def test_scale_p_min_pu_no_generators(self, network_with_snapshots):
        """Test scaling with no generators that have p_min_pu."""
        network = network_with_snapshots

        summary = scale_p_min_pu_step(network, scaling_factor=0.5)

        assert summary["scale_p_min_pu"]["generators_scaled"] == 0


class TestAddCurtailmentLinkStep:
    """Test curtailment link addition step."""

    def test_add_curtailment_link_basic(self, network_with_buses):
        """Test adding curtailment link to network."""
        network = network_with_buses

        summary = add_curtailment_link_step(
            network, bus_name="Bus1", p_nom=5000, marginal_cost=1000
        )

        # Check link was added
        assert "Curtailment_Bus1" in network.links.index

        # Check dump bus was created
        assert "Bus1_curtailment_dump" in network.buses.index

        # Check carrier was added
        assert "curtailment" in network.carriers.index

        # Check link properties
        link = network.links.loc["Curtailment_Bus1"]
        assert link["bus0"] == "Bus1"
        assert link["bus1"] == "Bus1_curtailment_dump"
        assert link["p_nom"] == 5000
        assert link["marginal_cost"] == 1000

        # Check summary
        assert "add_curtailment_link" in summary
        assert summary["add_curtailment_link"]["link_name"] == "Curtailment_Bus1"
        assert summary["add_curtailment_link"]["p_nom"] == 5000

    def test_add_curtailment_link_default_params(self, network_with_buses):
        """Test curtailment link with default parameters."""
        network = network_with_buses

        add_curtailment_link_step(network)
        # Should use "SEM" as default bus
        assert (
            "Curtailment_SEM" in network.links.index
            or "curtailment_dump" in network.buses.index
        )

    def test_add_curtailment_link_carrier_already_exists(self, network_with_buses):
        """Test adding curtailment link when carrier already exists."""
        network = network_with_buses
        network.add("Carrier", "curtailment")

        # Should not raise error
        summary = add_curtailment_link_step(network, bus_name="Bus1")

        assert "add_curtailment_link" in summary


class TestOptimizeStep:
    """Test optimization step."""

    @pytest.mark.skip(reason="Requires solver and full network setup")
    def test_optimize_step_basic(self):
        """Test basic optimization (skipped - requires solver)."""

    @pytest.mark.skip(reason="Requires solver and full network setup")
    def test_optimize_step_with_year_filter(self):
        """Test optimization with year filter (skipped - requires solver)."""

    @pytest.mark.skip(reason="Requires solver and full network setup")
    def test_optimize_step_with_snapshot_limit(self):
        """Test optimization with snapshot limit (skipped - requires solver)."""


# Note: Other step tests (create_model, load_vre_profiles, add_storage_inflows,
# apply_generator_units, parse_outages) require more complex fixtures or real
# CSV data, so they are better suited for integration tests or separate test files.
