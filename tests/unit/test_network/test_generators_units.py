"""Tests for generator Units ramp smoothing logic."""

import pandas as pd
import pypsa
import pytest

from plexos_to_pypsa_converter.network.generators_csv import (
    _apply_units_ramp_smoothing,
    _ensure_default_ramp_limits,
    validate_and_fix_generator_constraints,
)


def test_units_ramp_smoothing_handles_retirement():
    """p_min is tapered before a Units-driven shutdown, preserving ramp limits."""
    snapshots = pd.date_range("2025-06-30 21:00:00", periods=4, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "SEM")
    network.add(
        "Generator",
        "MP2",
        bus="SEM",
        p_nom=250,
        ramp_limit_up=0.3912,
        ramp_limit_down=0.3192,
        p_min_pu=0.396,
        p_max_pu=1.0,
    )

    network.generators_t.p_min_pu = pd.DataFrame(
        0.396, index=snapshots, columns=["MP2"]
    )
    network.generators_t.p_max_pu = pd.DataFrame(1.0, index=snapshots, columns=["MP2"])

    # Units-based retirement: final snapshot unavailable
    network.generators_t.p_min_pu.loc[snapshots[-1], "MP2"] = 0.0
    network.generators_t.p_max_pu.loc[snapshots[-1], "MP2"] = 0.0

    units_schedule = pd.DataFrame(
        {"MP2": [1.0, 1.0, 1.0, 0.0]},
        index=snapshots,
    )

    summary = _apply_units_ramp_smoothing(network, units_schedule)

    assert summary["ramped_generators"] == 1
    # Last online hour gets relaxed to zero so ramp down constraint is feasible
    assert network.generators_t.p_min_pu.loc[snapshots[-2], "MP2"] == pytest.approx(0.0)
    # Hour before that is partially relaxed according to ramp limit
    assert network.generators_t.p_min_pu.loc[snapshots[-3], "MP2"] == pytest.approx(
        0.198
    )


def test_default_ramp_limits_are_applied_for_missing_values():
    """Generators without ramp data should allow full hourly movement."""
    network = pypsa.Network()
    network.add("Bus", "SEM")
    network.add("Generator", "G1", bus="SEM", p_nom=50)  # NaN ramp limits by default
    network.add("Generator", "G2", bus="SEM", p_nom=20, ramp_limit_up="0")
    network.add(
        "Generator",
        "G3",
        bus="SEM",
        p_nom=15,
        ramp_limit_up=0.3,
        ramp_limit_down=0.4,
    )

    summary = _ensure_default_ramp_limits(network)

    assert summary["filled_generators"] == 2
    assert network.generators.loc["G1", "ramp_limit_up"] == pytest.approx(1.0)
    assert network.generators.loc["G1", "ramp_limit_down"] == pytest.approx(1.0)
    # Explicit zeros are lifted to the default
    assert network.generators.loc["G2", "ramp_limit_up"] == pytest.approx(1.0)
    # Pre-existing non-zero data stays untouched
    assert network.generators.loc["G3", "ramp_limit_up"] == pytest.approx(0.3)
    assert network.generators.loc["G3", "ramp_limit_down"] == pytest.approx(0.4)


def test_ramp_limits_cleared_for_fixed_dispatch():
    """Fixed-dispatch generators shouldn't carry ramp constraints."""
    snapshots = pd.date_range("2024-01-01", periods=3, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "SEM")
    network.add(
        "Generator",
        "FixedHydro",
        bus="SEM",
        p_nom=50,
        ramp_limit_up=0.5,
        ramp_limit_down=0.5,
    )
    fixed_profile = pd.Series([0.2, 0.21, 0.19], index=snapshots)
    network.generators_t.p_min_pu = pd.DataFrame({"FixedHydro": fixed_profile})
    network.generators_t.p_max_pu = pd.DataFrame({"FixedHydro": fixed_profile})

    summary = validate_and_fix_generator_constraints(network, verbose=False)
    _ensure_default_ramp_limits(network)  # Should not overwrite cleared ramps

    assert "FixedHydro" in summary["cleared_ramp_generators"]
    assert network.generators.loc["FixedHydro", "ramp_limit_up"] > 1e9
    assert network.generators.loc["FixedHydro", "ramp_limit_down"] > 1e9
