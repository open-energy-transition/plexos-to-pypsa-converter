"""Core network setup functions using CSV-based approach."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from plexosdb import PlexosDB
from pypsa import Network

from db.csv_readers import load_static_properties
from network.carriers_csv import parse_fuel_prices_csv
from network.core import (
    _add_loads_with_participation_factors,
    add_loads_flexible,
    add_snapshots,
    parse_demand_data,
)
from network.costs_csv import (
    set_battery_capital_costs_csv,
    set_battery_marginal_costs_csv,
)
from network.generators_csv import port_generators_csv
from network.investment import (
    configure_investment_periods,
    load_group_profiles,
    load_manifest,
)
from network.links_csv import port_links_csv
from network.storage_csv import port_storage_csv

logger = logging.getLogger(__name__)


def add_buses_csv(network: Network, csv_dir: str | Path) -> int:
    """Add buses from Node.csv to network.

    Parameters
    ----------
    network : Network
        PyPSA network to add buses to
    csv_dir : str | Path
        Directory containing CSV exports

    Returns
    -------
    int
        Number of buses added
    """
    node_df = load_static_properties(csv_dir, "Node")

    if node_df.empty:
        logger.warning("No Node.csv found, cannot add buses")
        return 0

    count = 0
    for node_name in node_df.index:
        # Get voltage if available
        v_nom = None
        if "Voltage" in node_df.columns:
            v_nom_val = node_df.at[node_name, "Voltage"]
            if pd.notna(v_nom_val):
                try:
                    v_nom = float(v_nom_val)
                except (ValueError, TypeError):
                    logger.debug(
                        f"Could not parse voltage for {node_name}: {v_nom_val}"
                    )

        # Add bus
        if v_nom:
            network.add("Bus", node_name, v_nom=v_nom)
        else:
            network.add("Bus", node_name)

        count += 1

    logger.info(f"Added {count} buses from CSV")
    return count


def add_carriers_csv(network: Network, csv_dir: str | Path) -> int:
    """Add carriers from Fuel.csv to network.

    Parameters
    ----------
    network : Network
        PyPSA network to add carriers to
    csv_dir : str | Path
        Directory containing CSV exports

    Returns
    -------
    int
        Number of carriers added
    """
    fuel_df = load_static_properties(csv_dir, "Fuel")

    if fuel_df.empty:
        logger.warning("No Fuel.csv found, cannot add carriers")
        return 0

    count = 0
    for fuel_name in fuel_df.index:
        if fuel_name not in network.carriers.index:
            network.add("Carrier", fuel_name)
            count += 1

    logger.info(f"Added {count} carriers from CSV")
    return count


def setup_network_csv(
    network: Network,
    csv_dir: str | Path,
    db: PlexosDB | None = None,
    snapshots_source: str | None = None,
    demand_source: str | None = None,
    timeslice_csv: str | None = None,
    vre_profiles_path: str | None = None,
    model_name: str | None = None,
    inflow_path: str | None = None,
    target_node: str | None = None,
    aggregate_node_name: str | None = None,
    demand_assignment_strategy: str = "per_node",
    demand_target_node: str | None = None,
    transmission_as_lines: bool = False,
    bidirectional_links: bool = True,
    load_scenario: str | None = None,
    use_investment_periods: bool = False,
    investment_manifest: str | None = None,
    investment_group: str = "demand",
) -> dict:
    """Set up PyPSA network from CSV exports.

    This is the CSV equivalent of setup_network() in core.py.

    Parameters
    ----------
    network : Network
        PyPSA network to populate
    csv_dir : str | Path
        Directory containing COAD CSV exports
    db : PlexosDB, optional
        PlexosDB object (still needed for snapshots/demand in some cases)
    snapshots_source : str, optional
        Path to snapshots data source
    demand_source : str, optional
        Path to demand data source
    timeslice_csv : str, optional
        Path to timeslice CSV file
    vre_profiles_path : str, optional
        Path to VRE profile files
    model_name : str, optional
        PLEXOS model name to use
    inflow_path : str, optional
        Path to hydro inflow data
    target_node : str, optional
        Target node for generator reassignment (target_node strategy)
    aggregate_node_name : str, optional
        Aggregate node name (aggregate_node strategy)
    demand_assignment_strategy : str, default "per_node"
        Strategy for generator/component assignment: "per_node", "target_node", or "aggregate_node"
    demand_target_node : str, optional
        Specific node to assign ALL demand to, regardless of demand_assignment_strategy.
        This allows keeping generators on their original nodes (per_node) while
        consolidating all demand to one node. Example: "SEM"
    transmission_as_lines : bool, default False
        Use Line components instead of Links for transmission
    bidirectional_links : bool, default True
        If True, Links with zero/NaN min flows will be set to allow bidirectional
        flow (p_min_pu = -1). If False, min flows of zero/NaN will be treated
        as unidirectional (p_min_pu = 0).
    load_scenario : str, optional
        For demand data with multiple scenarios (iterations), specify which scenario
        to use. If None, defaults to first scenario. Example: "iteration_1"
    use_investment_periods : bool, default False
        If True, build the network using representative investment-period profiles
        rather than the full chronological traces.
    investment_manifest : str, optional
        Path to the investment_periods.json manifest (absolute or relative to
        the model directory) produced by prepare_investment_periods_step.
    investment_group : str, default "demand"
        Manifest group key that contains the load profiles to import.

    Returns
    -------
    dict
        Setup summary with component counts and metadata
    """
    csv_dir = Path(csv_dir)

    logger.info("Setting up network from CSVs...")
    logger.info(f"  CSV directory: {csv_dir}")
    logger.info(f"  Demand strategy: {demand_assignment_strategy}")

    manifest_data: dict | None = None
    manifest_path_obj: Path | None = None
    manifest_base: Path | None = None
    investment_info: dict[str, Any] | None = None

    # 1. Add buses
    num_buses = add_buses_csv(network, csv_dir)

    # 2. Add carriers
    num_carriers = add_carriers_csv(network, csv_dir)

    load_summary: dict[str, Any] | None = None

    if use_investment_periods:
        logger.info("Investment-period mode enabled; loading manifest and snapshots...")
        if investment_manifest is not None:
            manifest_path_obj = Path(investment_manifest)
            if not manifest_path_obj.is_absolute():
                manifest_path_obj = (Path.cwd() / manifest_path_obj).resolve()
        else:
            manifest_path_obj = (
                csv_dir.parent.parent / "investment_periods" / "investment_periods.json"
            ).resolve()

        manifest_data = load_manifest(manifest_path_obj)
        manifest_base = manifest_path_obj.parent

        (
            _,
            period_weights,
            _,
            profile_offsets,
            period_base_dates,
        ) = configure_investment_periods(
            network=network,
            manifest=manifest_data,
            base_dir=manifest_base,
        )

        logger.info("Loading demand profiles from investment manifest...")
        demand_profiles = load_group_profiles(
            manifest_data,
            manifest_base,
            investment_group,
            profile_offsets=profile_offsets,
            period_base_dates=period_base_dates,
        )
        demand_profiles = demand_profiles.reindex(network.snapshots)
        column_mapping = {
            column: column.split("_", 1)[0] for column in demand_profiles.columns
        }
        collapsed_demand = demand_profiles.T.groupby(column_mapping).sum().T.fillna(0.0)
        load_summary = _apply_investment_period_loads(
            network=network,
            demand_df=collapsed_demand,
            demand_target_node=demand_target_node,
        )

        investment_info = {
            "manifest": str(manifest_path_obj),
            "periods": period_weights.to_dict(),
        }
    else:
        logger.info("Setting up snapshots and demand from CSV files...")
        if snapshots_source:
            add_snapshots(network, snapshots_source)
        else:
            logger.warning(
                "No snapshots_source provided, network will need snapshots set manually"
            )

        if demand_source:
            if demand_assignment_strategy == "participation_factors":
                logger.info("  Using participation factors strategy")
                demand_df = parse_demand_data(demand_source)
                _add_loads_with_participation_factors(
                    network=network,
                    demand_df=demand_df,
                    csv_dir=csv_dir,
                    load_scenario=load_scenario,
                )
            else:
                demand_node = None
                demand_aggregate = None

                if demand_target_node:
                    demand_node = demand_target_node
                    logger.info(f"  Assigning all demand to node: {demand_target_node}")
                elif demand_assignment_strategy == "target_node":
                    demand_node = target_node
                elif demand_assignment_strategy == "aggregate_node":
                    demand_aggregate = aggregate_node_name

                add_loads_flexible(
                    network=network,
                    demand_source=demand_source,
                    target_node=demand_node,
                    aggregate_node_name=demand_aggregate,
                    load_scenario=load_scenario,
                )
    # 4. Add generators
    logger.info("Adding generators from CSV...")
    port_generators_csv(
        network=network,
        csv_dir=csv_dir,
        timeslice_csv=timeslice_csv,
        vre_profiles_path=vre_profiles_path,
        target_node=target_node,
    )

    # NOTE: Generator Units time series (retirements, new builds) are NOT applied here.
    # Units MUST be applied AFTER VRE profiles are loaded, otherwise they get overwritten.
    # Call apply_generator_units_timeseries_csv() manually after loading VRE profiles.

    # 5. Add storage
    logger.info("Adding storage from CSV...")
    port_storage_csv(
        network=network,
        csv_dir=csv_dir,
        timeslice_csv=timeslice_csv,
    )

    # 5b. Set battery costs
    if len(network.storage_units) > 0:
        logger.info("Setting battery costs...")

        set_battery_capital_costs_csv(network, csv_dir)
        set_battery_marginal_costs_csv(network, csv_dir)

    # 6. Add transmission links
    logger.info("Adding transmission links from CSV...")
    if transmission_as_lines:
        logger.warning(
            "transmission_as_lines=True not yet supported in CSV mode. "
            "Using Links instead."
        )
    port_links_csv(
        network=network,
        csv_dir=csv_dir,
        timeslice_csv=timeslice_csv,
        target_node=target_node,
        bidirectional_links=bidirectional_links,
    )

    # 7. Set fuel prices
    logger.info("Setting fuel prices from CSV...")
    fuel_prices = parse_fuel_prices_csv(
        csv_dir=csv_dir,
        network=network,
        timeslice_csv=timeslice_csv,
    )

    # Apply fuel prices to carriers (if carriers_t exists)
    if not fuel_prices.empty and len(network.snapshots) > 0:
        # Initialize carriers_t if it doesn't exist
        if not hasattr(network, "carriers_t") or network.carriers_t is None:
            network.carriers_t = {}

        # Initialize marginal_cost DataFrame if it doesn't exist
        if "marginal_cost" not in network.carriers_t:
            network.carriers_t["marginal_cost"] = pd.DataFrame(
                index=network.snapshots,
                columns=network.carriers.index,
            )

        for carrier in fuel_prices.columns:
            if carrier in network.carriers.index:
                network.carriers_t["marginal_cost"][carrier] = fuel_prices[carrier]

    # 8. Demand aggregation is handled by add_loads_flexible above
    # No additional processing needed here

    summary = {
        "method": "csv",
        "csv_dir": str(csv_dir),
        "buses": num_buses,
        "carriers": num_carriers,
        "generators": len(network.generators),
        "storage": len(network.storage_units),
        "links": len(network.links),
        "snapshots": len(network.snapshots),
        "demand_strategy": demand_assignment_strategy,
    }

    if target_node:
        summary["target_node"] = target_node
    if aggregate_node_name:
        summary["aggregate_node"] = aggregate_node_name
    if load_summary:
        summary["loads"] = load_summary
    if investment_info:
        summary["investment"] = investment_info

    summary["snapshots_mode"] = (
        "investment_periods" if use_investment_periods else "chronological"
    )

    logger.info("CSV-based network setup complete")
    logger.info(f"  Buses: {num_buses}")
    logger.info(f"  Carriers: {num_carriers}")
    logger.info(f"  Generators: {len(network.generators)}")
    logger.info(f"  Storage units: {len(network.storage_units)}")
    logger.info(f"  Links: {len(network.links)}")
    logger.info(f"  Snapshots: {len(network.snapshots)}")

    return summary


def _apply_investment_period_loads(
    network: Network,
    demand_df: pd.DataFrame,
    demand_target_node: str | None,
) -> dict[str, Any]:
    """Assign investment-period demand profiles to network loads."""
    load_names: list[str] = []

    if demand_target_node:
        total_series = demand_df.sum(axis=1)
        load_name = f"Load_{demand_target_node}"
        if load_name not in network.loads.index:
            if demand_target_node not in network.buses.index:
                logger.warning(
                    "Demand target node '%s' not found in buses. Skipping load creation.",
                    demand_target_node,
                )
                return {"mode": "investment_periods", "loads": load_names}
            network.add("Load", load_name, bus=demand_target_node)
        network.loads_t.p_set.loc[:, load_name] = total_series.to_numpy()
        load_names.append(load_name)
        logger.info("  Assigned aggregated demand to load %s", load_name)
        return {"mode": "investment_periods", "loads": load_names}

    for zone in demand_df.columns:
        load_name = f"Load_{zone}"
        if zone not in network.buses.index:
            logger.warning(
                "Bus '%s' missing for investment-period load; skipping column.",
                zone,
            )
            continue
        if load_name not in network.loads.index:
            network.add("Load", load_name, bus=zone)
        network.loads_t.p_set.loc[:, load_name] = demand_df[zone].to_numpy()
        load_names.append(load_name)

    logger.info("  Added %d loads from investment-period profiles", len(load_names))
    return {"mode": "investment_periods", "loads": load_names}
