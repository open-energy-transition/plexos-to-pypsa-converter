"""Add slack generators (load spillage and load shedding) to the SEM model."""

import logging
from collections.abc import Iterable

import pandas as pd
import pypsa

logger = logging.getLogger(__name__)

SPILLAGE_CARRIER = "load spillage"
SHEDDING_CARRIER = "load shedding"

SPILLAGE_COST = -5000.0  # Allow dumping energy at negative cost
SHEDDING_COST = 5000.0  # Penalise unserved energy

SPILLAGE_P_NOM = 10e6  # Effectively unlimited capacity per bus
SHEDDING_P_NOM = 10e6


def _ensure_carrier(
    network: pypsa.Network, name: str, color: str, nice_name: str
) -> None:
    """Create the carrier if it does not already exist."""
    if name in network.carriers.index:
        return
    network.add("Carrier", name=name, color=color, nice_name=nice_name)


def _ensure_time_series_column(
    df: pd.DataFrame, columns: Iterable[str], value: float
) -> None:
    """Set all snapshots for the provided columns to ``value``."""
    for column in columns:
        df[column] = value


def add_slack_generators(network: pypsa.Network) -> dict:
    """Add load spillage and load shedding generators to every bus.

    PyPSA 1.0+ expects generator dispatch bounds to be carried in the
    time-series DataFrames. Earlier versions tolerated missing columns and
    would fall back to static attributes; now the bounds default to ``[-inf,
    inf]`` unless the time-series values are present. This helper ensures both
    static and time-varying attributes are populated so slack generators behave
    as true sinks/sources and keep the optimisation feasible.
    """
    buses = network.buses.index
    spillage_names = [f"{bus} load spillage" for bus in buses]
    shedding_names = [f"{bus} load shedding" for bus in buses]

    # Guarantee carriers exist.
    _ensure_carrier(
        network,
        SPILLAGE_CARRIER,
        color="#df8e23",
        nice_name="Load spillage",
    )
    _ensure_carrier(
        network,
        SHEDDING_CARRIER,
        color="#dd2e23",
        nice_name="Load shedding",
    )

    def _add_generator_if_missing(
        name: str, bus: str, carrier: str, **attrs: float
    ) -> None:
        if name in network.generators.index:
            return
        network.add(
            "Generator",
            name=name,
            bus=bus,
            carrier=carrier,
            **attrs,
        )

    for name, bus in zip(spillage_names, buses, strict=False):
        _add_generator_if_missing(
            name,
            bus,
            SPILLAGE_CARRIER,
            marginal_cost=SPILLAGE_COST,
            p_nom=SPILLAGE_P_NOM,
            p_min_pu=-1.0,
            p_max_pu=0.0,
        )

    for name, bus in zip(shedding_names, buses, strict=False):
        _add_generator_if_missing(
            name,
            bus,
            SHEDDING_CARRIER,
            marginal_cost=SHEDDING_COST,
            p_nom=SHEDDING_P_NOM,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )

    # Even if the generators already existed, make sure the static attributes
    # are set to the expected values in case PyPSA preserved older defaults.
    network.generators.loc[
        spillage_names, ["p_nom", "p_min_pu", "p_max_pu", "marginal_cost"]
    ] = (
        SPILLAGE_P_NOM,
        -1.0,
        0.0,
        SPILLAGE_COST,
    )
    network.generators.loc[
        shedding_names, ["p_nom", "p_min_pu", "p_max_pu", "marginal_cost"]
    ] = (
        SHEDDING_P_NOM,
        0.0,
        1.0,
        SHEDDING_COST,
    )

    # Populate the generators_t DataFrames with the same bounds so PyPSA 1.0+
    # creates finite dispatch limits.
    _ensure_time_series_column(network.generators_t.p_min_pu, spillage_names, -1.0)
    _ensure_time_series_column(network.generators_t.p_max_pu, spillage_names, 0.0)
    _ensure_time_series_column(network.generators_t.p_min_pu, shedding_names, 0.0)
    _ensure_time_series_column(network.generators_t.p_max_pu, shedding_names, 1.0)

    summary = {
        "slack_generators_added": len(buses),
        "spillage_names": spillage_names,
        "shedding_names": shedding_names,
    }
    logger.info("Configured slack generators for %d buses", len(buses))

    return {"add_slack": summary}
