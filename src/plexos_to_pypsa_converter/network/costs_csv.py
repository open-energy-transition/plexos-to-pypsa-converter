"""CSV-based cost functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based cost functions.
"""

from pathlib import Path

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.db.csv_readers import (
    get_property_from_static_csv,
    load_static_properties,
)


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


def set_battery_capital_costs_csv(network: pypsa.Network, csv_dir: str | Path) -> None:
    """Set capital costs for batteries with proper annualization.

    Follows costs.py::set_capital_costs_generic() pattern:
    - Annualizes Build Cost using WACC and lifetime
    - Adds FO&M Charge
    - Converts $/kW to $/MW

    The capital_cost is calculated as:
        annuity_factor = wacc / (1 - (1 + wacc) ** -lifetime)
        annualized_capex = build_cost * annuity_factor
        capital_cost = annualized_capex + fo_m_charge

    Parameters
    ----------
    network : Network
        The PyPSA network with storage units
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add batteries ...
    >>> set_battery_capital_costs_csv(network, csv_dir)
    """
    battery_df = load_static_properties(csv_dir, "Battery")

    if battery_df.empty or len(network.storage_units) == 0:
        print("No batteries found for capital cost calculation")
        return

    print(f"Setting capital costs for {len(network.storage_units)} batteries...")

    capital_costs = []
    processed_count = 0

    for battery_name in network.storage_units.index:
        # Extract properties - handle column name variations
        build_cost = get_property_from_static_csv(
            battery_df, battery_name, "Build Cost"
        ) or get_property_from_static_csv(battery_df, battery_name, "Capital Cost")
        wacc = get_property_from_static_csv(battery_df, battery_name, "WACC")

        # Prefer Economic Life, fallback to Technical Life
        lifetime = get_property_from_static_csv(
            battery_df, battery_name, "Economic Life"
        ) or get_property_from_static_csv(battery_df, battery_name, "Technical Life")

        fo_m = get_property_from_static_csv(battery_df, battery_name, "FO&M Charge")

        # Convert to floats
        try:
            build_cost_val = float(build_cost) if build_cost is not None else None
            wacc_val = float(wacc) if wacc is not None else None
            lifetime_val = float(lifetime) if lifetime is not None else None
            fo_m_val = float(fo_m) if fo_m is not None else None
        except (ValueError, TypeError):
            capital_costs.append(0.0)
            continue

        # Convert $/kW to $/MW (PLEXOS uses $/kW, PyPSA uses $/MW)
        build_cost_MW = build_cost_val * 1000 if build_cost_val is not None else None
        fo_m_MW = fo_m_val * 1000 if fo_m_val is not None else None

        # Calculate annualized capital cost
        if (
            build_cost_MW is not None
            and wacc_val is not None
            and lifetime_val is not None
            and lifetime_val > 0
        ):
            try:
                annuity_factor = wacc_val / (1 - (1 + wacc_val) ** (-lifetime_val))
            except ZeroDivisionError:
                annuity_factor = 1.0

            annualized_capex = build_cost_MW * annuity_factor
            capital_cost = annualized_capex + (fo_m_MW if fo_m_MW is not None else 0.0)
            capital_costs.append(capital_cost)
            processed_count += 1
        elif fo_m_MW is not None:
            # If can't annualize, at least use FO&M
            capital_costs.append(fo_m_MW)
        else:
            capital_costs.append(0.0)

    # Set capital costs
    network.storage_units["capital_cost"] = capital_costs

    print(
        f"Successfully set capital costs for {processed_count}/{len(network.storage_units)} batteries"
    )


def set_battery_marginal_costs_csv(network: pypsa.Network, csv_dir: str | Path) -> None:
    """Set marginal costs for batteries (VO&M Charge).

    Follows costs.py::set_battery_marginal_costs() pattern.

    For batteries, marginal costs are typically much lower than generators since they
    don't consume fuel. The marginal cost includes:
    - VO&M Charge: Variable operating and maintenance costs
    - Efficiency losses are already captured in efficiency parameters

    Parameters
    ----------
    network : Network
        The PyPSA network with storage units
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add batteries ...
    >>> set_battery_marginal_costs_csv(network, csv_dir)
    """
    battery_df = load_static_properties(csv_dir, "Battery")

    if battery_df.empty or len(network.storage_units) == 0:
        print("No batteries found for marginal cost calculation")
        return

    print(f"Setting marginal costs for {len(network.storage_units)} batteries...")

    snapshots = network.snapshots
    marginal_costs_dict = {}
    successful_count = 0

    for battery_name in network.storage_units.index:
        # Get VO&M Charge - handle column name variations across models
        # Try multiple property names in order of preference:
        # 1. Standard: "VO&M Charge"
        # 2. SEM style: "Charging VO&M Charge" (with .Variable suffix)
        # 3. Generic: "Variable Cost"
        vo_m = (
            get_property_from_static_csv(battery_df, battery_name, "VO&M Charge")
            or get_property_from_static_csv(
                battery_df, battery_name, "Charging VO&M Charge.Variable"
            )
            or get_property_from_static_csv(
                battery_df, battery_name, "Charging VO&M Charge"
            )
            or get_property_from_static_csv(battery_df, battery_name, "Variable Cost")
        )

        # Note: PyPSA doesn't natively support separate charge/discharge costs
        # If different charging vs discharging costs are needed, this would require
        # custom implementation (e.g., using separate Links for charge/discharge)

        try:
            vo_m_val = (
                float(vo_m) if vo_m is not None else 0.0
            )  # Default matches PLEXOS default (0)
        except (ValueError, TypeError):
            vo_m_val = 0.0  # Default matches PLEXOS default (0)

        # Create constant time series
        marginal_costs_dict[battery_name] = pd.Series(vo_m_val, index=snapshots)
        successful_count += 1

    if marginal_costs_dict:
        marginal_costs_df = pd.DataFrame(marginal_costs_dict, index=snapshots)

        # Initialize marginal_cost time series for storage units if it doesn't exist
        if not hasattr(network.storage_units_t, "marginal_cost"):
            network.storage_units_t["marginal_cost"] = pd.DataFrame(
                index=snapshots, columns=network.storage_units.index, dtype=float
            )

        # Assign time series to network
        network.storage_units_t.marginal_cost.loc[:, marginal_costs_df.columns] = (
            marginal_costs_df
        )

        print(f"Successfully set marginal costs for {successful_count} batteries")
    else:
        print("No batteries had data for marginal cost calculation")
