"""Tests for generator Units ramp smoothing logic."""

import pandas as pd
import pypsa
import pytest

from plexos_to_pypsa_converter.network.generators_csv import (
    _apply_units_ramp_smoothing,
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
