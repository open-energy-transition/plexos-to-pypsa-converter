"""CSV-based cost functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based cost functions.
"""

from pathlib import Path

import pypsa

from db.csv_readers import get_property_from_static_csv, load_static_properties


def set_capital_costs_generic_csv(
    network: pypsa.Network, csv_dir: str | Path, component_type: str
):
    """Set capital costs for a component type using COAD CSV exports.

    This is the CSV-based version of set_capital_costs_generic() from costs.py.

    Parameters
    ----------
    network : Network
        The PyPSA network
    csv_dir : str | Path
        Directory containing COAD CSV exports
    component_type : str
        Component type (e.g., 'Generator', 'Storage', 'Line')

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> set_capital_costs_generic_csv(network, csv_dir, "Generator")
    """
    # Load static properties for the component type
    component_df = load_static_properties(csv_dir, component_type)

    if component_df.empty:
        print(f"Warning: No {component_type} objects found in CSV for capital costs")
        return

    # Get the appropriate component collection from network
    if component_type == "Generator":
        component_collection = network.generators
    elif component_type == "Link":
        component_collection = network.links
    elif component_type == "Line":
        component_collection = network.lines
    elif component_type == "Storage":
        component_collection = network.storage_units
    elif component_type == "Store":
        component_collection = network.stores
    else:
        print(f"Warning: Unknown component type {component_type}")
        return

    capital_costs = []

    for component_name in component_collection.index:
        # Try various property names that might contain capital cost
        cost_property_names = [
            "Build Cost",
            "Capital Cost",
            "Investment Cost",
            "CAPEX",
            "Fixed Cost",
        ]

        cost_value = None
        for prop_name in cost_property_names:
            val = get_property_from_static_csv(component_df, component_name, prop_name)
            if val is not None:
                try:
                    cost_value = float(val)
                    break
                except (ValueError, TypeError):
                    continue

        capital_costs.append(cost_value if cost_value is not None else 0.0)

    # Set capital_cost attribute
    component_collection["capital_cost"] = capital_costs

    # Count how many had costs
    non_zero_costs = sum(1 for cost in capital_costs if cost > 0)
    print(
        f"Set capital costs for {non_zero_costs}/{len(capital_costs)} {component_type}s"
    )
