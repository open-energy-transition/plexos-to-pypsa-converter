"""Multi-Sector Network Setup Functions.

This module provides functions to set up PyPSA networks for multi-sector energy models,
including gas, hydrogen, ammonia, and other energy carriers with sector coupling.

These functions extend the existing single-sector functionality without breaking
backward compatibility with existing model scripts.
"""

import logging
from typing import Any

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from src.network.core import add_buses
from src.network.generators import add_generators
from src.network.links import add_links
from src.network.storage import add_storage

logger = logging.getLogger(__name__)


def setup_gas_electric_network(network: Network, db: PlexosDB) -> dict[str, Any]:
    """Set up a multi-sector PyPSA network for gas and electricity systems.

    This function creates a PyPSA network that represents both gas and electricity
    sectors with coupling through gas-fired generators, following the MaREI-EU model structure.

    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data

    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print("Setting up gas-electric multi-sector network...")

    # Initialize tracking
    setup_summary = {
        "network_type": "gas_electric",
        "sectors": ["Electricity", "Gas"],
        "electricity": {
            "buses": 0,
            "generators": 0,
            "loads": 0,
            "lines": 0,
            "storage": 0,
        },
        "gas": {"buses": 0, "pipelines": 0, "storage": 0, "demand": 0, "fields": 0},
        "sector_coupling": {"gas_generators": 0, "efficiency_range": "N/A"},
    }

    # Step 1: Set up electricity sector (traditional components)
    print("\\n1. Setting up electricity sector...")

    # Add electricity buses from Node class
    initial_bus_count = len(network.buses)
    add_buses(network, db)
    elec_buses = len(network.buses) - initial_bus_count
    setup_summary["electricity"]["buses"] = elec_buses
    print(f"   Added {elec_buses} electricity buses")

    # Set up snapshots from demand data (if available)
    try:
        # Try to find demand data for snapshots - using a simple approach for now
        nodes = db.list_objects_by_class(ClassEnum.Node)
        if nodes:
            snapshots = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="H")
            network.set_snapshots(snapshots)
            print(f"   Set {len(snapshots)} hourly snapshots for 2024")
    except Exception as e:
        print(f"   Warning: Could not set snapshots from demand data: {e}")
        # Set basic snapshots as fallback
        snapshots = pd.date_range("2024-01-01", "2024-01-02", freq="H")
        network.set_snapshots(snapshots)
        print(f"   Set {len(snapshots)} fallback snapshots")

    # Add electricity generators
    initial_gen_count = len(network.generators)
    add_generators(network, db)
    elec_generators = len(network.generators) - initial_gen_count
    setup_summary["electricity"]["generators"] = elec_generators
    print(f"   Added {elec_generators} electricity generators")

    # Add electricity transmission lines
    initial_link_count = len(network.links)
    add_links(network, db)
    elec_lines = len(network.links) - initial_link_count
    setup_summary["electricity"]["lines"] = elec_lines
    print(f"   Added {elec_lines} electricity transmission lines")

    # Add electricity storage
    initial_storage_count = len(network.storage_units)
    add_storage(network, db)
    elec_storage = len(network.storage_units) - initial_storage_count
    setup_summary["electricity"]["storage"] = elec_storage
    print(f"   Added {elec_storage} electricity storage units")

    # Step 2: Set up gas sector
    print("\\n2. Setting up gas sector...")

    gas_buses_added = add_gas_buses(network, db)
    setup_summary["gas"]["buses"] = gas_buses_added
    print(f"   Added {gas_buses_added} gas buses")

    gas_pipelines_added = add_gas_pipelines(network, db)
    setup_summary["gas"]["pipelines"] = gas_pipelines_added
    print(f"   Added {gas_pipelines_added} gas pipelines")

    gas_storage_added = add_gas_storage(network, db)
    setup_summary["gas"]["storage"] = gas_storage_added
    print(f"   Added {gas_storage_added} gas storage units")

    gas_demand_added = add_gas_demand(network, db)
    setup_summary["gas"]["demand"] = gas_demand_added
    print(f"   Added {gas_demand_added} gas demand loads")

    gas_fields_added = add_gas_fields(network, db)
    setup_summary["gas"]["fields"] = gas_fields_added
    print(f"   Added {gas_fields_added} gas field generators")

    # Step 3: Set up sector coupling
    print("\\n3. Setting up sector coupling...")

    coupling_stats = add_gas_electric_coupling(network, db)
    setup_summary["sector_coupling"].update(coupling_stats)
    print(
        f"   Added {coupling_stats['gas_generators']} gas-to-electric conversion links"
    )
    print(f"   Efficiency range: {coupling_stats['efficiency_range']}")

    # Step 4: Add carriers and costs
    print("\\n4. Setting up carriers and economic parameters...")

    # Add gas carrier
    if "Gas" not in network.carriers.index:
        network.add("Carrier", "Gas")

    # Set up basic load for electricity (placeholder)
    electricity_buses = network.buses[network.buses.carrier == "AC"].index
    for _i, bus in enumerate(
        electricity_buses[:5]
    ):  # Add loads to first 5 buses as example
        load_name = f"Load_{bus}"
        if load_name not in network.loads.index:
            # Create basic constant load profile
            load_profile = pd.Series(
                100.0, index=network.snapshots
            )  # 100 MW constant load
            network.add("Load", load_name, bus=bus, p_set=load_profile)

    setup_summary["electricity"]["loads"] = len(network.loads)

    print("Gas-electric multi-sector network setup complete!")
    return setup_summary


def add_gas_buses(network: Network, db: PlexosDB) -> int:
    """Add gas buses from Gas Node objects."""
    try:
        # Try to get gas nodes using SQL query since ClassEnum may not have Gas_Node
        query = "SELECT name FROM t_object JOIN t_class ON t_object.class_id = t_class.class_id WHERE t_class.name = 'Gas Node'"
        gas_nodes = []
        try:
            result = db.execute_query(query)
            gas_nodes = [row[0] for row in result]
        except Exception:
            # Fallback: check if we can access via standard method with different name
            try:
                # Try different possible class names
                for class_name in ["Gas_Node", "GasNode", "Gas Node"]:
                    try:
                        gas_nodes = db.list_objects_by_class(class_name)
                        break
                    except Exception:
                        logger.debug(f"Gas class name {class_name} not available")
                        continue
            except Exception:
                logger.debug("All gas node class name attempts failed")

        for node in gas_nodes:
            bus_name = f"gas_{node}"
            if bus_name not in network.buses.index:
                network.add("Bus", bus_name, carrier="Gas")
        return len(gas_nodes)
    except Exception as e:
        logger.warning(f"Failed to add gas buses: {e}")
        return 0


def add_gas_pipelines(network: Network, db: PlexosDB) -> int:
    """Add gas pipelines from Gas Pipeline objects as Links."""
    pipelines_added = 0
    try:
        gas_pipelines = db.list_objects_by_class(ClassEnum.Gas_Pipeline)

        for pipeline in gas_pipelines:
            try:
                props = db.get_object_properties(ClassEnum.Gas_Pipeline, pipeline)

                # Get connected gas nodes from memberships
                memberships = db.get_memberships_system(
                    pipeline, object_class=ClassEnum.Gas_Pipeline
                )
                gas_nodes = [
                    m["name"] for m in memberships if m["collection_name"] == "Gas Node"
                ]

                if len(gas_nodes) >= 2:
                    bus0 = f"gas_{gas_nodes[0]}"
                    bus1 = f"gas_{gas_nodes[1]}"

                    # Check if buses exist
                    if bus0 in network.buses.index and bus1 in network.buses.index:
                        # Get pipeline capacity
                        max_flow = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Flow Day"
                            ),
                            1000.0,
                        )  # Default 1000 MW

                        link_name = f"gas_pipeline_{pipeline}"
                        network.add(
                            "Link",
                            link_name,
                            bus0=bus0,
                            bus1=bus1,
                            p_nom=max_flow,
                            carrier="Gas",
                            efficiency=0.98,
                        )
                        pipelines_added += 1

            except Exception as e:
                logger.warning(f"Failed to process gas pipeline {pipeline}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add gas pipelines: {e}")

    return pipelines_added


def add_gas_storage(network: Network, db: PlexosDB) -> int:
    """Add gas storage from Gas Storage objects."""
    storage_added = 0
    try:
        gas_storages = db.list_objects_by_class(ClassEnum.Gas_Storage)

        for storage in gas_storages:
            try:
                props = db.get_object_properties(ClassEnum.Gas_Storage, storage)

                # Find connected gas node
                memberships = db.get_memberships_system(
                    storage, object_class=ClassEnum.Gas_Storage
                )
                gas_nodes = [
                    m["name"] for m in memberships if m["collection_name"] == "Gas Node"
                ]

                if gas_nodes:
                    bus_name = f"gas_{gas_nodes[0]}"
                    if bus_name in network.buses.index:
                        # Get storage properties
                        max_volume = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Volume"
                            ),
                            1000.0,
                        )  # Default
                        max_injection = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Injection"
                            ),
                            100.0,
                        )  # Default

                        storage_name = f"gas_storage_{storage}"
                        network.add(
                            "StorageUnit",
                            storage_name,
                            bus=bus_name,
                            p_nom=max_injection,
                            max_hours=max_volume / max_injection
                            if max_injection > 0
                            else 10,
                            carrier="Gas",
                            efficiency_store=0.95,
                            efficiency_dispatch=0.95,
                        )
                        storage_added += 1

            except Exception as e:
                logger.warning(f"Failed to process gas storage {storage}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add gas storage: {e}")

    return storage_added


def add_gas_demand(network: Network, db: PlexosDB) -> int:
    """Add gas demand from Gas Demand objects."""
    demand_added = 0
    try:
        gas_demands = db.list_objects_by_class(ClassEnum.Gas_Demand)

        for demand in gas_demands:
            try:
                props = db.get_object_properties(ClassEnum.Gas_Demand, demand)

                # Find connected gas node
                memberships = db.get_memberships_system(
                    demand, object_class=ClassEnum.Gas_Demand
                )
                gas_nodes = [
                    m["name"] for m in memberships if m["collection_name"] == "Gas Node"
                ]

                if gas_nodes:
                    bus_name = f"gas_{gas_nodes[0]}"
                    if bus_name in network.buses.index:
                        # Get demand profile (use constant for now)
                        demand_value = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Load"
                            ),
                            50.0,
                        )  # Default 50 MW

                        demand_name = f"gas_demand_{demand}"
                        # Create constant demand profile
                        demand_profile = pd.Series(
                            demand_value, index=network.snapshots
                        )
                        network.add(
                            "Load",
                            demand_name,
                            bus=bus_name,
                            p_set=demand_profile,
                            carrier="Gas",
                        )
                        demand_added += 1

            except Exception as e:
                logger.warning(f"Failed to process gas demand {demand}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add gas demand: {e}")

    return demand_added


def add_gas_fields(network: Network, db: PlexosDB) -> int:
    """Add gas fields from Gas Field objects as generators."""
    fields_added = 0
    try:
        gas_fields = db.list_objects_by_class(ClassEnum.Gas_Field)

        for field in gas_fields:
            try:
                props = db.get_object_properties(ClassEnum.Gas_Field, field)

                # Find connected gas node
                memberships = db.get_memberships_system(
                    field, object_class=ClassEnum.Gas_Field
                )
                gas_nodes = [
                    m["name"] for m in memberships if m["collection_name"] == "Gas Node"
                ]

                if gas_nodes:
                    bus_name = f"gas_{gas_nodes[0]}"
                    if bus_name in network.buses.index:
                        # Get field capacity
                        max_production = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Production"
                            ),
                            200.0,
                        )  # Default

                        field_name = f"gas_field_{field}"
                        network.add(
                            "Generator",
                            field_name,
                            bus=bus_name,
                            p_nom=max_production,
                            carrier="Gas",
                            marginal_cost=20.0,
                        )  # Gas wellhead cost
                        fields_added += 1

            except Exception as e:
                logger.warning(f"Failed to process gas field {field}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add gas fields: {e}")

    return fields_added


def add_gas_electric_coupling(network: Network, db: PlexosDB) -> dict[str, Any]:
    """Add gas-to-electric conversion links for gas-fired generators."""
    coupling_stats = {"gas_generators": 0, "efficiency_range": "N/A"}

    try:
        generators = db.list_objects_by_class(ClassEnum.Generator)
        efficiency_values = []

        for gen in generators:
            try:
                props = db.get_object_properties(ClassEnum.Generator, gen)

                # Check if generator has gas connection
                gas_node_prop = next(
                    (p for p in props if p.get("property") == "Gas Node"), None
                )
                if not gas_node_prop or not gas_node_prop.get("texts"):
                    continue  # Not a gas generator

                gas_node = gas_node_prop["texts"].strip()

                # Get electric node connection
                memberships = db.get_memberships_system(
                    gen, object_class=ClassEnum.Generator
                )
                elec_nodes = [
                    m["name"] for m in memberships if m["collection_name"] == "Node"
                ]

                if elec_nodes and gas_node:
                    elec_bus = elec_nodes[0]
                    gas_bus = f"gas_{gas_node}"

                    # Check if both buses exist
                    if (
                        elec_bus in network.buses.index
                        and gas_bus in network.buses.index
                    ):
                        # Get generator properties
                        max_capacity = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Capacity"
                            ),
                            100.0,
                        )
                        heat_rate = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Heat Rate"
                            ),
                            9.0,
                        )  # Default heat rate

                        # Calculate efficiency (3412 BTU/kWh conversion factor)
                        efficiency = (
                            3412 / heat_rate if heat_rate > 0 else 0.4
                        )  # Default 40% efficiency
                        efficiency = min(efficiency, 0.65)  # Cap at 65% efficiency

                        efficiency_values.append(efficiency)

                        # Create gas-to-electric conversion link
                        link_name = f"gas_to_elec_{gen}"
                        network.add(
                            "Link",
                            link_name,
                            bus0=gas_bus,
                            bus1=elec_bus,
                            p_nom=max_capacity,
                            efficiency=efficiency,
                            carrier="Gas2Electric",
                        )

                        coupling_stats["gas_generators"] += 1

            except Exception as e:
                logger.warning(f"Failed to process gas generator {gen}: {e}")
                continue

        # Calculate efficiency range
        if efficiency_values:
            min_eff = min(efficiency_values)
            max_eff = max(efficiency_values)
            coupling_stats["efficiency_range"] = f"{min_eff:.1%} - {max_eff:.1%}"

    except Exception as e:
        logger.warning(f"Failed to add gas-electric coupling: {e}")

    return coupling_stats


def setup_flow_network(network: Network, db: PlexosDB) -> dict[str, Any]:
    """Set up a multi-sector PyPSA network using PLEXOS Flow Network components.

    This function creates a PyPSA network from PLEXOS Flow Network components,
    supporting multiple energy carriers like electricity, hydrogen, and ammonia.

    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data

    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print("Setting up multi-sector flow network...")

    # Initialize tracking
    setup_summary = {
        "network_type": "flow_network",
        "sectors": [],
        "processes": {},
        "facilities": {},
    }

    # Step 1: Analyze flow nodes to identify sectors
    print("\\n1. Analyzing flow network structure...")
    sectors = analyze_flow_sectors(db)
    setup_summary["sectors"] = list(sectors.keys())

    for sector in sectors:
        setup_summary[sector.lower()] = {
            "nodes": 0,
            "paths": 0,
            "storage": 0,
            "demand": 0,
        }

    print(f"   Identified sectors: {', '.join(sectors.keys())}")

    # Step 2: Add flow nodes as buses
    print("\\n2. Setting up flow nodes as buses...")
    nodes_by_sector = add_flow_nodes(network, db, sectors)

    for sector, count in nodes_by_sector.items():
        setup_summary[sector.lower()]["nodes"] = count
        print(f"   Added {count} {sector} buses")

    # Step 3: Set up basic snapshots
    try:
        snapshots = pd.date_range("2024-01-01", "2024-12-31 23:00", freq="H")
        network.set_snapshots(snapshots)
        print(f"   Set {len(snapshots)} hourly snapshots")
    except Exception as e:
        print(f"   Warning: Using fallback snapshots: {e}")
        snapshots = pd.date_range("2024-01-01", "2024-01-02", freq="H")
        network.set_snapshots(snapshots)

    # Step 4: Add flow paths as links
    print("\\n3. Setting up flow paths as links...")
    paths_by_sector = add_flow_paths(network, db, sectors)

    for sector, count in paths_by_sector.items():
        if sector.lower() in setup_summary:
            setup_summary[sector.lower()]["paths"] = count
        print(f"   Added {count} {sector} flow paths")

    # Step 5: Add flow storage
    print("\\n4. Setting up flow storage...")
    storage_by_sector = add_flow_storage(network, db, sectors)

    for sector, count in storage_by_sector.items():
        if sector.lower() in setup_summary:
            setup_summary[sector.lower()]["storage"] = count
        print(f"   Added {count} {sector} storage units")

    # Step 6: Add processes for sector coupling
    print("\\n5. Setting up process-based sector coupling...")
    process_stats = add_processes(network, db)
    setup_summary["processes"] = process_stats

    for process_type, count in process_stats.items():
        print(f"   Added {count} {process_type} processes")

    # Step 7: Add facilities if present
    print("\\n6. Setting up facilities...")
    facility_stats = add_facilities(network, db)
    setup_summary["facilities"] = facility_stats

    # Step 8: Add basic demands
    print("\\n7. Setting up basic demand profiles...")
    for sector in sectors:
        sector_buses = network.buses[network.buses.carrier == sector].index
        demand_count = 0
        for _i, bus in enumerate(
            sector_buses[:3]
        ):  # Add demand to first 3 buses per sector
            load_name = f"{sector}_demand_{bus.split('_')[-1]}"
            if load_name not in network.loads.index:
                # Different demand levels by sector
                if sector == "Electricity":
                    base_demand = 1000.0  # 1000 MW
                elif sector == "Hydrogen":
                    base_demand = 100.0  # 100 MW
                else:
                    base_demand = 50.0  # 50 MW

                demand_profile = pd.Series(base_demand, index=network.snapshots)
                network.add("Load", load_name, bus=bus, p_set=demand_profile)
                demand_count += 1

        if sector.lower() in setup_summary:
            setup_summary[sector.lower()]["demand"] = demand_count
        print(f"   Added {demand_count} {sector} demand loads")

    print("Multi-sector flow network setup complete!")
    return setup_summary


def analyze_flow_sectors(db: PlexosDB) -> dict[str, list[str]]:
    """Analyze flow nodes to identify energy sectors."""
    sectors = {}
    try:
        flow_nodes = db.list_objects_by_class(ClassEnum.Flow_Node)

        for node in flow_nodes:
            # Extract sector from node name (e.g., "Elec_nod AF-AGO" -> "Electricity")
            if node.startswith("Elec_"):
                sector = "Electricity"
            elif node.startswith("H2_"):
                sector = "Hydrogen"
            elif node.startswith("NH3_"):
                sector = "Ammonia"
            else:
                sector = "Other"

            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(node)

    except Exception as e:
        logger.warning(f"Failed to analyze flow sectors: {e}")

    return sectors


def add_flow_nodes(
    network: Network, db: PlexosDB, sectors: dict[str, list[str]]
) -> dict[str, int]:
    """Add flow nodes as PyPSA buses."""
    nodes_by_sector = {}

    try:
        for sector, nodes in sectors.items():
            # Add carrier for this sector
            if sector not in network.carriers.index:
                network.add("Carrier", sector)

            nodes_added = 0
            for node in nodes:
                bus_name = node  # Use original node name
                if bus_name not in network.buses.index:
                    network.add("Bus", bus_name, carrier=sector)
                    nodes_added += 1

            nodes_by_sector[sector] = nodes_added

    except Exception as e:
        logger.warning(f"Failed to add flow nodes: {e}")

    return nodes_by_sector


def add_flow_paths(
    network: Network, db: PlexosDB, sectors: dict[str, list[str]]
) -> dict[str, int]:
    """Add flow paths as PyPSA links."""
    paths_by_sector = {"Transport": 0, "Conversion": 0}

    try:
        flow_paths = db.list_objects_by_class(ClassEnum.Flow_Path)

        for path in flow_paths:
            try:
                props = db.get_object_properties(ClassEnum.Flow_Path, path)

                # Get connected flow nodes
                memberships = db.get_memberships_system(
                    path, object_class=ClassEnum.Flow_Path
                )
                flow_nodes = [
                    m["name"]
                    for m in memberships
                    if m["collection_name"] == "Flow Node"
                ]

                if len(flow_nodes) >= 2:
                    bus0, bus1 = flow_nodes[0], flow_nodes[1]

                    # Check if buses exist
                    if bus0 in network.buses.index and bus1 in network.buses.index:
                        # Get path properties
                        max_flow = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Flow"
                            ),
                            1000.0,
                        )
                        efficiency = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Efficiency"
                            ),
                            1.0,
                        )

                        # Determine if transport or conversion link
                        bus0_carrier = network.buses.at[bus0, "carrier"]
                        bus1_carrier = network.buses.at[bus1, "carrier"]

                        if bus0_carrier == bus1_carrier:
                            # Same carrier = transport
                            link_type = "Transport"
                            link_name = f"transport_{path}"
                        else:
                            # Different carriers = conversion
                            link_type = "Conversion"
                            link_name = f"conversion_{path}"

                        network.add(
                            "Link",
                            link_name,
                            bus0=bus0,
                            bus1=bus1,
                            p_nom=max_flow,
                            efficiency=efficiency,
                        )

                        paths_by_sector[link_type] += 1

            except Exception as e:
                logger.warning(f"Failed to process flow path {path}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add flow paths: {e}")

    return paths_by_sector


def add_flow_storage(
    network: Network, db: PlexosDB, sectors: dict[str, list[str]]
) -> dict[str, int]:
    """Add flow storage as PyPSA storage units."""
    storage_by_sector = {}

    try:
        flow_storages = db.list_objects_by_class(ClassEnum.Flow_Storage)

        for storage in flow_storages:
            try:
                props = db.get_object_properties(ClassEnum.Flow_Storage, storage)

                # Find connected flow node
                memberships = db.get_memberships_system(
                    storage, object_class=ClassEnum.Flow_Storage
                )
                flow_nodes = [
                    m["name"]
                    for m in memberships
                    if m["collection_name"] == "Flow Node"
                ]

                if flow_nodes:
                    bus_name = flow_nodes[0]
                    if bus_name in network.buses.index:
                        # Get storage properties
                        max_volume = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Volume"
                            ),
                            1000.0,
                        )
                        max_power = next(
                            (
                                float(p["value"])
                                for p in props
                                if p["property"] == "Max Power"
                            ),
                            100.0,
                        )

                        # Determine sector from bus carrier
                        sector = network.buses.at[bus_name, "carrier"]

                        storage_name = f"{sector.lower()}_storage_{storage}"
                        network.add(
                            "StorageUnit",
                            storage_name,
                            bus=bus_name,
                            p_nom=max_power,
                            max_hours=max_volume / max_power if max_power > 0 else 10,
                            carrier=sector,
                            efficiency_store=0.9,
                            efficiency_dispatch=0.9,
                        )

                        if sector not in storage_by_sector:
                            storage_by_sector[sector] = 0
                        storage_by_sector[sector] += 1

            except Exception as e:
                logger.warning(f"Failed to process flow storage {storage}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add flow storage: {e}")

    return storage_by_sector


def add_processes(network: Network, db: PlexosDB) -> dict[str, int]:
    """Add processes as sector coupling links."""
    process_stats = {}

    try:
        processes = db.list_objects_by_class(ClassEnum.Process)

        for process in processes:
            try:
                props = db.get_object_properties(ClassEnum.Process, process)

                # Get process efficiency
                efficiency = (
                    next(
                        (
                            float(p["value"])
                            for p in props
                            if p["property"] == "Efficiency"
                        ),
                        70.0,
                    )
                    / 100.0
                )  # Convert to fraction

                # Get connected commodities
                memberships = db.get_memberships_system(
                    process, object_class=ClassEnum.Process
                )
                [m["name"] for m in memberships if m["collection_name"] == "Commodity"]

                # Determine process type and create appropriate links
                if "electrolysis" in process.lower():
                    process_type = "Electrolysis"
                    # Find electricity and hydrogen buses (simplified)
                    elec_buses = network.buses[
                        network.buses.carrier == "Electricity"
                    ].index
                    h2_buses = network.buses[network.buses.carrier == "Hydrogen"].index

                    if len(elec_buses) > 0 and len(h2_buses) > 0:
                        # Create electrolysis links (electricity -> hydrogen)
                        for i in range(
                            min(3, len(elec_buses), len(h2_buses))
                        ):  # Up to 3 links
                            link_name = f"electrolysis_{i + 1}"
                            network.add(
                                "Link",
                                link_name,
                                bus0=elec_buses[i],
                                bus1=h2_buses[i],
                                p_nom=100.0,
                                efficiency=efficiency,
                            )

                elif "h2power" in process.lower():
                    process_type = "H2_Power"
                    # Find hydrogen and electricity buses
                    h2_buses = network.buses[network.buses.carrier == "Hydrogen"].index
                    elec_buses = network.buses[
                        network.buses.carrier == "Electricity"
                    ].index

                    if len(h2_buses) > 0 and len(elec_buses) > 0:
                        # Create fuel cell links (hydrogen -> electricity)
                        for i in range(min(2, len(h2_buses), len(elec_buses))):
                            link_name = f"fuel_cell_{i + 1}"
                            network.add(
                                "Link",
                                link_name,
                                bus0=h2_buses[i],
                                bus1=elec_buses[i],
                                p_nom=50.0,
                                efficiency=efficiency,
                            )

                elif "ammonia" in process.lower():
                    process_type = "Ammonia_Synthesis"
                    # Find hydrogen and ammonia buses
                    h2_buses = network.buses[network.buses.carrier == "Hydrogen"].index
                    nh3_buses = network.buses[network.buses.carrier == "Ammonia"].index

                    if len(h2_buses) > 0 and len(nh3_buses) > 0:
                        # Create ammonia synthesis links (hydrogen -> ammonia)
                        for i in range(min(2, len(h2_buses), len(nh3_buses))):
                            link_name = f"ammonia_synthesis_{i + 1}"
                            network.add(
                                "Link",
                                link_name,
                                bus0=h2_buses[i],
                                bus1=nh3_buses[i],
                                p_nom=30.0,
                                efficiency=efficiency,
                            )
                else:
                    process_type = "Other"

                if process_type not in process_stats:
                    process_stats[process_type] = 0
                process_stats[process_type] += 1

            except Exception as e:
                logger.warning(f"Failed to process {process}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add processes: {e}")

    return process_stats


def add_facilities(network: Network, db: PlexosDB) -> dict[str, int]:
    """Add facilities as generators."""
    facility_stats = {}

    try:
        facilities = db.list_objects_by_class(ClassEnum.Facility)

        for facility in facilities:
            try:
                props = db.get_object_properties(ClassEnum.Facility, facility)

                # Simple facility processing - add as generators to electricity buses
                elec_buses = network.buses[network.buses.carrier == "Electricity"].index
                if len(elec_buses) > 0:
                    # Get facility capacity
                    capacity = next(
                        (
                            float(p["value"])
                            for p in props
                            if p["property"] in ["Capacity", "Max Power"]
                        ),
                        100.0,
                    )

                    # Add to first available electricity bus
                    bus = elec_buses[0]
                    gen_name = f"facility_{facility}"

                    if gen_name not in network.generators.index:
                        network.add(
                            "Generator",
                            gen_name,
                            bus=bus,
                            p_nom=capacity,
                            carrier="Electricity",
                            marginal_cost=50.0,
                        )

                        facility_type = "Power_Generation"
                        if facility_type not in facility_stats:
                            facility_stats[facility_type] = 0
                        facility_stats[facility_type] += 1

            except Exception as e:
                logger.warning(f"Failed to process facility {facility}: {e}")
                continue

    except Exception as e:
        logger.warning(f"Failed to add facilities: {e}")

    return facility_stats
