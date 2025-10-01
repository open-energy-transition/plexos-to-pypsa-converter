import logging

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

logger = logging.getLogger(__name__)


def set_capital_costs_generic(
    network: Network, db: PlexosDB, component_type: str, component_class: ClassEnum
) -> None:
    """Set the capital_cost for components (generators or storage units) in the PyPSA network.


    Based on 'Build Cost', 'WACC', 'Economic Life' (preferred), 'Technical Life', and
    'FO&M Charge' properties from the PlexosDB.

    The capital_cost is calculated as:
        annuity_factor = wacc / (1 - (1 + wacc) ** -lifetime)
        annualized_capex = build_cost * annuity_factor
        capital_cost = annualized_capex + fo_m_charge

    All costs are converted to $/MW (PyPSA convention).

    - Build Cost is given in $/kW, so multiply by 1000 to get $/MW.
    - FO&M Charge is given in $/kW/yr, so multiply by 1000 to get $/MW/yr.
    - capital_cost is stored in $/MW/yr.

    If build_cost, wacc, or lifetime are missing, but FO&M Charge is present,
    capital_cost is set to FO&M Charge only (converted to $/MW/yr).

    If no cost can be calculated or found, capital_cost is set to 0.

    Parameters
    ----------
    network : Network
        The PyPSA network containing the components.
    db : PlexosDB
        The Plexos database containing component data.
    component_type : str
        Type of component: "Generator" or "StorageUnit"
    component_class : ClassEnum
        PLEXOS class enum (e.g., ClassEnum.Generator, ClassEnum.Battery)

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_capital_costs_generic(network, db, "Generator", ClassEnum.Generator)
    >>> set_capital_costs_generic(network, db, "StorageUnit", ClassEnum.Battery)
    """
    if component_type == "Generator":
        components = network.generators
    elif component_type == "StorageUnit":
        components = network.storage_units
    else:
        msg = f"Unsupported component_type: {component_type}"
        raise ValueError(msg)

    print(f"Setting capital costs for {len(components)} {component_type.lower()}s...")

    capital_costs = []
    processed_count = 0

    for component_name in components.index:
        try:
            # TODO: batteries properties are stored under ClassEnum.Generator instead of ClassEnum.Battery
            props = db.get_object_properties(ClassEnum.Generator, component_name)
        except Exception as e:
            print(f"Warning: Could not get properties for {component_name}: {e}")
            capital_costs.append(0.0)
            continue

        # Extract properties
        build_costs = [
            float(p["value"]) for p in props if p["property"] == "Build Cost"
        ]
        build_cost = sum(build_costs) / len(build_costs) if build_costs else None
        wacc = next((float(p["value"]) for p in props if p["property"] == "WACC"), None)

        # Prefer Economic Life, fallback to Technical Life
        economic_life = next(
            (float(p["value"]) for p in props if p["property"] == "Economic Life"), None
        )
        technical_life = next(
            (float(p["value"]) for p in props if p["property"] == "Technical Life"),
            None,
        )
        lifetime = economic_life if economic_life is not None else technical_life
        fo_m_charge = next(
            (float(p["value"]) for p in props if p["property"] == "FO&M Charge"), None
        )

        # Convert units: $/kW -> $/MW
        build_cost_MW = build_cost * 1000 if build_cost is not None else None
        fo_m_charge_MW = fo_m_charge * 1000 if fo_m_charge is not None else None

        if build_cost_MW is None or wacc is None or lifetime is None or lifetime <= 0:
            if fo_m_charge_MW is not None:
                capital_costs.append(fo_m_charge_MW)
            else:
                capital_costs.append(0.0)
                print(
                    f"Warning: Missing or invalid capital cost data for {component_name}. Setting capital_cost to 0."
                )
            continue

        # Calculate annuity factor
        try:
            annuity_factor = wacc / (1 - (1 + wacc) ** (-lifetime))
        except ZeroDivisionError:
            annuity_factor = 1.0

        annualized_capex = build_cost_MW * annuity_factor
        capital_cost = annualized_capex + (
            fo_m_charge_MW if fo_m_charge_MW is not None else 0.0
        )
        capital_costs.append(capital_cost)
        processed_count += 1

    # Set the capital costs on the appropriate component type
    components["capital_cost"] = capital_costs

    print(
        f"Successfully set capital costs for {processed_count}/{len(components)} {component_type.lower()}s"
    )


def set_battery_marginal_costs(
    network: Network, db: PlexosDB, timeslice_csv: str | None = None
) -> None:
    """Set marginal costs for batteries in the PyPSA network.

    Based on VO&M charges and efficiency losses from the Plexos database.

    For batteries, marginal costs are typically much lower than generators since they
    don't consume fuel. The marginal cost includes:
    - VO&M Charge: Variable operating and maintenance costs
    - Efficiency losses: Costs associated with round-trip efficiency losses

    The marginal cost is calculated as:
    marginal_cost = vo_m_charge + efficiency_loss_cost

    For now, we'll use just the VO&M Charge as the marginal cost, since efficiency
    losses are already captured in the efficiency parameters.

    Parameters
    ----------
    network : Network
        The PyPSA network containing storage units.
    db : PlexosDB
        The Plexos database containing battery data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file (for future time-dependent costs).

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_battery_marginal_costs(network, db)
    """
    if len(network.storage_units) == 0:
        print("No storage units found in network. Skipping marginal cost calculation.")
        return

    print(f"Setting marginal costs for {len(network.storage_units)} batteries...")

    snapshots = network.snapshots
    marginal_costs_dict = {}
    skipped_batteries = []
    successful_count = 0

    for battery_name in network.storage_units.index:
        try:
            # Get battery properties from database
            # TODO: batteries properties are stored under ClassEnum.Generator instead of ClassEnum.Battery
            battery_props = db.get_object_properties(ClassEnum.Generator, battery_name)
        except Exception as e:
            logger.warning(
                f"Error retrieving properties for battery {battery_name}: {e}"
            )
            skipped_batteries.append(battery_name)
            continue

        # Get VO&M Charge (main component of battery marginal costs)
        vo_m_charge = None
        for prop in battery_props:
            if prop.get("property") == "VO&M Charge":
                try:
                    vo_m_charge = float(prop["value"])
                    break
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid VO&M Charge value for battery {battery_name}: {prop['value']}"
                    )

        if vo_m_charge is None:
            # For batteries without VO&M charge, set marginal cost to very low value
            vo_m_charge = 0.1  # Small positive value to avoid zero marginal costs
            logger.info(
                f"No VO&M Charge found for battery {battery_name}, using default: {vo_m_charge}"
            )

        # For batteries, marginal cost is primarily the VO&M charge
        # Efficiency losses are already handled by the efficiency parameters
        marginal_cost_value = vo_m_charge

        # Create time series with constant marginal cost
        marginal_cost_ts = pd.Series(marginal_cost_value, index=snapshots)
        marginal_costs_dict[battery_name] = marginal_cost_ts
        successful_count += 1

        logger.debug(f"Battery {battery_name}: marginal_cost={marginal_cost_value}")

    # Create DataFrame from marginal costs dictionary
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
        print("No batteries had complete data for marginal cost calculation")

    # Report skipped batteries
    if skipped_batteries:
        print(
            f"Skipped marginal costs for {len(skipped_batteries)} batteries due to missing properties:"
        )
        for battery in skipped_batteries:
            print(f"  - {battery}")
