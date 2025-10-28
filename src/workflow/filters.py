"""Generator filter presets for workflow system.

Provides named filter presets for use in registry workflow definitions.
"""

from collections.abc import Callable
from typing import Any

import pypsa


def make_filter(
    description: str, requires_network: bool, fn: Callable
) -> dict[str, Any]:
    return {
        "filter": fn,
        "requires_network": requires_network,
        "description": description,
    }


FILTER_PRESETS: dict[str, dict[str, Any]] = {
    "all": make_filter("No filtering", False, None),
    "vre_only": make_filter(
        "Only Wind/Solar", False, lambda gen: "Wind" in gen or "Solar" in gen
    ),
    "exclude_vre": make_filter(
        "Exclude VRE generators (empty carrier = VRE)",
        True,
        lambda gen, network: network.generators.at[gen, "carrier"] != "",
    ),
    "thermal_only": make_filter(
        "Only thermal/dispatchable generators",
        True,
        lambda gen, network: network.generators.at[gen, "carrier"]
        not in ["", "wind", "solar", "Wind", "Solar"],
    ),
    "has_carrier": make_filter(
        "Generators with non-empty carrier",
        True,
        lambda gen, network: network.generators.at[gen, "carrier"] != "",
    ),
    "hydro_only": make_filter(
        "Only Hydro/ROR generators",
        False,
        lambda gen: "Hydro" in gen or "hydro" in gen or "ROR" in gen,
    ),
    "exclude_vre_and_low_ramp_limits": make_filter(
        "Exclude VRE generators (empty carrier = VRE) and generators with ramp_limit_up < 0.4",
        True,
        lambda gen, network: network.generators.at[gen, "carrier"] != ""
        and network.generators.at[gen, "ramp_limit_up"] >= 0.4,
    ),
}


def resolve_filter_preset(
    filter_name: str | None, network: pypsa.Network = None
) -> Callable | None:
    """Resolve a filter preset name to a callable filter function.

    Args:
        filter_name: Name of the filter preset (e.g., "vre_only", "exclude_vre")
        network: PyPSA network (required for some filters)

    Returns:
        Callable filter function or None if filter_name is "all" or None

    Raises:
        ValueError: If filter_name is not a recognized preset
    """
    if filter_name is None or filter_name == "all":
        return None
    if filter_name not in FILTER_PRESETS:
        msg = f"Unknown filter preset: {filter_name}. Available presets: {list(FILTER_PRESETS.keys())}"
        raise ValueError(msg)
    preset = FILTER_PRESETS[filter_name]
    if preset["requires_network"]:
        if network is None:
            msg = f"Filter preset '{filter_name}' requires a network to be provided"
            raise ValueError(msg)
        return lambda gen: preset["filter"](gen, network)
    return preset["filter"]
