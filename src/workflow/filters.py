"""Generator filter presets for workflow system.

This module provides named filter presets that can be referenced in the registry
workflow definitions, eliminating the need for lambda functions in JSON.
"""

from collections.abc import Callable

import pypsa


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

    # If the filter requires network context, bind it
    if preset["requires_network"]:
        if network is None:
            msg = f"Filter preset '{filter_name}' requires a network to be provided"
            raise ValueError(msg)
        return lambda gen: preset["filter"](gen, network)
    else:
        return preset["filter"]


# Filter preset definitions
FILTER_PRESETS = {
    "all": {"filter": None, "requires_network": False, "description": "No filtering"},
    "vre_only": {
        "filter": lambda gen: "Wind" in gen or "Solar" in gen,
        "requires_network": False,
        "description": "Only generators with 'Wind' or 'Solar' in their name",
    },
    "exclude_vre": {
        "filter": lambda gen, network: network.generators.at[gen, "carrier"] != "",
        "requires_network": True,
        "description": "Exclude VRE generators (empty carrier = VRE)",
    },
    "thermal_only": {
        "filter": lambda gen, network: network.generators.at[gen, "carrier"]
        not in ["", "wind", "solar", "Wind", "Solar"],
        "requires_network": True,
        "description": "Only thermal/dispatchable generators",
    },
    "has_carrier": {
        "filter": lambda gen, network: network.generators.at[gen, "carrier"] != "",
        "requires_network": True,
        "description": "Generators with non-empty carrier",
    },
}
