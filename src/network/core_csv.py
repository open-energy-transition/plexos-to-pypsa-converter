"""Core network setup functions using CSV-based approach."""

import logging
from pathlib import Path

import pandas as pd
from plexosdb import PlexosDB
from pypsa import Network

from db.csv_readers import load_static_properties
from network.carriers_csv import parse_fuel_prices_csv
from network.core import add_loads_flexible, add_snapshots
from network.generators_csv import port_generators_csv
from network.links_csv import port_links_csv
from network.storage_csv import add_storage_csv

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
    transmission_as_lines: bool = False,
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
        Target node for demand aggregation (target_node strategy)
    aggregate_node_name : str, optional
        Aggregate node name (aggregate_node strategy)
    demand_assignment_strategy : str, default "per_node"
        Strategy for demand assignment: "per_node", "target_node", or "aggregate_node"
    transmission_as_lines : bool, default False
        Use Line components instead of Links for transmission

    Returns
    -------
    dict
        Setup summary with component counts and metadata
    """
    csv_dir = Path(csv_dir)

    logger.info("Setting up network from CSVs...")
    logger.info(f"  CSV directory: {csv_dir}")
    logger.info(f"  Demand strategy: {demand_assignment_strategy}")

    # 1. Add buses
    num_buses = add_buses_csv(network, csv_dir)

    # 2. Add carriers
    num_carriers = add_carriers_csv(network, csv_dir)

    # 3. Set up snapshots and demand (still uses PlexosDB for now if available)
    # TODO: Could be migrated to CSV in future
    if db is not None:
        logger.info("Setting snapshots from database...")
        # Use snapshots_source to set up time index
        if snapshots_source:
            add_snapshots(network, snapshots_source)
        else:
            logger.warning(
                "No snapshots_source provided, network will need snapshots set manually"
            )

        logger.info("Adding demand from database...")
        # Use add_loads_flexible which handles different demand formats
        if demand_source:
            add_loads_flexible(
                network=network,
                demand_source=demand_source,
                target_node=target_node
                if demand_assignment_strategy == "target_node"
                else None,
                aggregate_node_name=aggregate_node_name
                if demand_assignment_strategy == "aggregate_node"
                else None,
            )
    else:
        logger.warning(
            "No database provided, skipping snapshots/demand setup. "
            "Network will need snapshots set manually."
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

    # 5. Add storage
    logger.info("Adding storage from CSV...")
    add_storage_csv(
        network=network,
        csv_dir=csv_dir,
        timeslice_csv=timeslice_csv,
    )

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

    logger.info("CSV-based network setup complete")
    logger.info(f"  Buses: {num_buses}")
    logger.info(f"  Carriers: {num_carriers}")
    logger.info(f"  Generators: {len(network.generators)}")
    logger.info(f"  Storage units: {len(network.storage_units)}")
    logger.info(f"  Links: {len(network.links)}")
    logger.info(f"  Snapshots: {len(network.snapshots)}")

    return summary
