"""Multi-Sector Network Setup Functions (CSV-based).

This module provides functions to set up PyPSA networks for multi-sector energy models
using the exported CSV data from PLEXOS-COAD. This approach is more robust than trying
to use ClassEnum for multi-sector classes that may not be available.
"""

import ast
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypsa import Network

logger = logging.getLogger(__name__)


def setup_gas_electric_network_csv(network: Network, csv_path: str) -> dict[str, Any]:
    """Set up a multi-sector PyPSA network for gas and electricity using CSV data.

    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    csv_path : str
        Path to directory containing CSV files exported from PLEXOS-COAD

    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print(f"Setting up gas-electric network from CSV data: {csv_path}")

    # Initialize tracking
    setup_summary = {
        "network_type": "gas_electric_csv",
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

    try:
        # Step 1: Set up electricity sector
        print("\\n1. Setting up electricity sector...")

        # Add electricity buses from Node.csv
        elec_buses = add_electricity_buses_csv(network, csv_path)
        setup_summary["electricity"]["buses"] = elec_buses
        print(f"   Added {elec_buses} electricity buses")

        # Set basic snapshots
        snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="H")
        network.set_snapshots(snapshots)
        print(f"   Set {len(snapshots)} hourly snapshots")

        # Add electricity generators from Generator.csv
        elec_generators = add_electricity_generators_csv(network, csv_path)
        setup_summary["electricity"]["generators"] = elec_generators
        print(f"   Added {elec_generators} electricity generators")

        # Add electricity transmission lines from Line.csv
        elec_lines = add_electricity_lines_csv(network, csv_path)
        setup_summary["electricity"]["lines"] = elec_lines
        print(f"   Added {elec_lines} electricity transmission lines")

        # Add electricity storage from Storage.csv
        elec_storage = add_electricity_storage_csv(network, csv_path)
        setup_summary["electricity"]["storage"] = elec_storage
        print(f"   Added {elec_storage} electricity storage units")

        # Step 2: Set up gas sector
        print("\\n2. Setting up gas sector...")

        gas_buses = add_gas_buses_csv(network, csv_path)
        setup_summary["gas"]["buses"] = gas_buses
        print(f"   Added {gas_buses} gas buses")

        gas_pipelines = add_gas_pipelines_csv(network, csv_path)
        setup_summary["gas"]["pipelines"] = gas_pipelines
        print(f"   Added {gas_pipelines} gas pipelines")

        gas_storage = add_gas_storage_csv(network, csv_path)
        setup_summary["gas"]["storage"] = gas_storage
        print(f"   Added {gas_storage} gas storage units")

        gas_demand = add_gas_demand_csv(network, csv_path)
        setup_summary["gas"]["demand"] = gas_demand
        print(f"   Added {gas_demand} gas demand loads")

        gas_fields = add_gas_fields_csv(network, csv_path)
        setup_summary["gas"]["fields"] = gas_fields
        print(f"   Added {gas_fields} gas field generators")

        # Step 3: Set up sector coupling
        print("\\n3. Setting up sector coupling...")

        coupling_stats = add_gas_electric_coupling_csv(network, csv_path)
        setup_summary["sector_coupling"].update(coupling_stats)
        print(
            f"   Added {coupling_stats['gas_generators']} gas-to-electric conversion links"
        )
        print(f"   Efficiency range: {coupling_stats['efficiency_range']}")

        # Step 4: Add basic loads
        print("\\n4. Adding basic demand profiles...")

        # Add simple loads to electricity buses
        elec_bus_list = [
            bus
            for bus in network.buses.index
            if network.buses.at[bus, "carrier"] == "AC"
        ]
        for i, bus in enumerate(elec_bus_list[:5]):  # First 5 buses
            load_name = f"elec_load_{bus}"
            if load_name not in network.loads.index:
                # Create load profile with some variation
                base_load = 1000 * (1 + 0.1 * i)  # 1000-1400 MW base
                load_profile = pd.Series(
                    base_load
                    * (
                        1
                        + 0.3
                        * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)
                    ),
                    index=network.snapshots,
                )
                network.add("Load", load_name, bus=bus, p_set=load_profile)

        setup_summary["electricity"]["loads"] = len(
            [l for l in network.loads.index if "elec_load" in l]
        )

    except Exception:
        logger.exception("Error setting up gas-electric network")
        raise

    print("Gas-electric multi-sector network setup complete!")
    return setup_summary


def add_electricity_buses_csv(network: Network, csv_path: str) -> int:
    """Add electricity buses from Node.csv."""
    buses_added = 0
    try:
        node_file = str(Path(csv_path) / "Node.csv")
        if Path(node_file).exists():
            nodes_df = pd.read_csv(node_file)

            for _, node_row in nodes_df.iterrows():
                node_name = node_row["object"]
                if node_name not in network.buses.index:
                    network.add("Bus", node_name, carrier="AC", v_nom=110)
                    buses_added += 1

    except Exception as e:
        logger.warning(f"Failed to add electricity buses: {e}")

    return buses_added


def add_gas_buses_csv(network: Network, csv_path: str) -> int:
    """Add gas buses from Gas Node.csv."""
    buses_added = 0
    try:
        gas_node_file = str(Path(csv_path) / "Gas Node.csv")
        if Path(gas_node_file).exists():
            gas_nodes_df = pd.read_csv(gas_node_file)

            # Add gas carrier if not exists
            if "Gas" not in network.carriers.index:
                network.add("Carrier", "Gas")

            for _, node_row in gas_nodes_df.iterrows():
                node_name = node_row["object"]
                bus_name = f"gas_{node_name}"
                if bus_name not in network.buses.index:
                    network.add("Bus", bus_name, carrier="Gas")
                    buses_added += 1

    except Exception as e:
        logger.warning(f"Failed to add gas buses: {e}")

    return buses_added


def add_electricity_generators_csv(network: Network, csv_path: str) -> int:
    """Add electricity generators from Generator.csv."""
    generators_added = 0
    try:
        gen_file = str(Path(csv_path) / "Generator.csv")
        if Path(gen_file).exists():
            generators_df = pd.read_csv(gen_file)

            for _, gen_row in generators_df.iterrows():
                gen_name = gen_row["object"]

                # Get properties
                node = gen_row.get("Node", "")
                max_capacity = gen_row.get("Max Capacity", 100)
                fuel = gen_row.get("Fuel", "Unknown")

                # Skip if no valid node
                if not node or pd.isna(node) or node not in network.buses.index:
                    continue

                # Convert capacity
                try:
                    p_nom = float(max_capacity) if pd.notna(max_capacity) else 100.0
                except Exception:
                    p_nom = 100.0

                if gen_name not in network.generators.index:
                    network.add(
                        "Generator",
                        gen_name,
                        bus=node,
                        p_nom=p_nom,
                        carrier=fuel,
                        marginal_cost=50.0,
                    )
                    generators_added += 1

    except Exception as e:
        logger.warning(f"Failed to add electricity generators: {e}")

    return generators_added


def add_electricity_lines_csv(network: Network, csv_path: str) -> int:
    """Add electricity transmission lines from Line.csv."""
    lines_added = 0
    try:
        line_file = str(Path(csv_path) / "Line.csv")
        if Path(line_file).exists():
            lines_df = pd.read_csv(line_file)

            for _, line_row in lines_df.iterrows():
                line_row["object"]

                # For now, create basic links between first few buses
                # In practice, you'd extract the actual node connections from PLEXOS data
                elec_buses = [
                    b
                    for b in network.buses.index
                    if network.buses.at[b, "carrier"] == "AC"
                ]
                if len(elec_buses) >= 2:
                    # Create links between adjacent buses
                    for i in range(min(len(elec_buses) - 1, 10)):  # Limit to 10 links
                        link_name = f"line_{i}_{elec_buses[i]}_{elec_buses[i + 1]}"
                        if link_name not in network.links.index:
                            network.add(
                                "Link",
                                link_name,
                                bus0=elec_buses[i],
                                bus1=elec_buses[i + 1],
                                p_nom=1000,
                                efficiency=0.97,
                            )
                            lines_added += 1
                            break  # Only add one per line object

    except Exception as e:
        logger.warning(f"Failed to add electricity lines: {e}")

    return lines_added


def add_electricity_storage_csv(network: Network, csv_path: str) -> int:
    """Add electricity storage from Storage.csv."""
    storage_added = 0
    try:
        storage_file = str(Path(csv_path) / "Storage.csv")
        if Path(storage_file).exists():
            storage_df = pd.read_csv(storage_file)

            elec_buses = [
                b for b in network.buses.index if network.buses.at[b, "carrier"] == "AC"
            ]

            for i, (_, storage_row) in enumerate(storage_df.iterrows()):
                if i >= len(elec_buses):
                    break

                storage_name = storage_row["object"]
                bus = elec_buses[i]

                # Get storage properties
                max_volume = storage_row.get("Max Volume", 1000)
                max_power = storage_row.get("Max Generation", 100)

                try:
                    p_nom = float(max_power) if pd.notna(max_power) else 100.0
                    volume = float(max_volume) if pd.notna(max_volume) else 1000.0
                    max_hours = volume / p_nom if p_nom > 0 else 10.0
                except Exception:
                    p_nom = 100.0
                    max_hours = 10.0

                if storage_name not in network.storage_units.index:
                    network.add(
                        "StorageUnit",
                        storage_name,
                        bus=bus,
                        p_nom=p_nom,
                        max_hours=max_hours,
                        efficiency_store=0.9,
                        efficiency_dispatch=0.9,
                    )
                    storage_added += 1

    except Exception as e:
        logger.warning(f"Failed to add electricity storage: {e}")

    return storage_added


def add_gas_pipelines_csv(network: Network, csv_path: str) -> int:
    """Add gas pipelines from Gas Pipeline.csv."""
    pipelines_added = 0
    try:
        pipeline_file = str(Path(csv_path) / "Gas Pipeline.csv")
        if Path(pipeline_file).exists():
            pipelines_df = pd.read_csv(pipeline_file)

            [b for b in network.buses.index if network.buses.at[b, "carrier"] == "Gas"]

            for _, pipeline_row in pipelines_df.iterrows():
                pipeline_name = pipeline_row["object"]

                # Extract connected nodes from Gas Node column
                gas_nodes_str = pipeline_row.get("Gas Node", "")
                if pd.notna(gas_nodes_str) and gas_nodes_str:
                    # Parse the gas nodes (they might be in format like "['AT', 'DE']")
                    try:
                        if gas_nodes_str.startswith("["):
                            gas_nodes = ast.literal_eval(gas_nodes_str)
                        else:
                            gas_nodes = gas_nodes_str.split(",")

                        if len(gas_nodes) >= 2:
                            # bus0 = f"gas_{gas_nodes[0].strip().strip('\\'\"')}"
                            # bus1 = f"gas_{gas_nodes[1].strip().strip('\\'\"')}"
                            node0_clean = gas_nodes[0].strip().strip("'\"")
                            node1_clean = gas_nodes[1].strip().strip("'\"")
                            bus0 = f"gas_{node0_clean}"
                            bus1 = f"gas_{node1_clean}"

                            if (
                                bus0 in network.buses.index
                                and bus1 in network.buses.index
                            ):
                                max_flow = pipeline_row.get("Max Flow Day", 1000)
                                try:
                                    p_nom = (
                                        float(max_flow)
                                        if pd.notna(max_flow)
                                        else 1000.0
                                    )
                                except Exception:
                                    p_nom = 1000.0

                                link_name = f"gas_pipeline_{pipeline_name}"
                                if link_name not in network.links.index:
                                    network.add(
                                        "Link",
                                        link_name,
                                        bus0=bus0,
                                        bus1=bus1,
                                        p_nom=p_nom,
                                        efficiency=0.98,
                                        carrier="Gas",
                                    )
                                    pipelines_added += 1
                    except Exception as exc:
                        logger.debug(
                            "Skipping gas pipeline %s due to error: %s",
                            pipeline_name,
                            exc,
                        )

    except Exception as e:
        logger.warning(f"Failed to add gas pipelines: {e}")

    return pipelines_added


def add_gas_storage_csv(network: Network, csv_path: str) -> int:
    """Add gas storage from Gas Storage.csv."""
    storage_added = 0
    try:
        gas_storage_file = str(Path(csv_path) / "Gas Storage.csv")
        if Path(gas_storage_file).exists():
            gas_storage_df = pd.read_csv(gas_storage_file)

            gas_buses = [
                b
                for b in network.buses.index
                if network.buses.at[b, "carrier"] == "Gas"
            ]

            for i, (_, storage_row) in enumerate(gas_storage_df.iterrows()):
                if i >= len(gas_buses):
                    break

                storage_name = storage_row["object"]
                bus = gas_buses[i % len(gas_buses)]  # Cycle through gas buses

                # Get storage properties
                max_volume = storage_row.get("Max Volume", 10000)
                max_injection = storage_row.get("Max Injection", 200)

                try:
                    p_nom = float(max_injection) if pd.notna(max_injection) else 200.0
                    volume = float(max_volume) if pd.notna(max_volume) else 10000.0
                    max_hours = volume / p_nom if p_nom > 0 else 50.0
                except Exception:
                    p_nom = 200.0
                    max_hours = 50.0

                storage_name_gas = f"gas_storage_{storage_name}"
                if storage_name_gas not in network.storage_units.index:
                    network.add(
                        "StorageUnit",
                        storage_name_gas,
                        bus=bus,
                        p_nom=p_nom,
                        max_hours=max_hours,
                        carrier="Gas",
                        efficiency_store=0.95,
                        efficiency_dispatch=0.95,
                    )
                    storage_added += 1

    except Exception as e:
        logger.warning(f"Failed to add gas storage: {e}")

    return storage_added


def add_gas_demand_csv(network: Network, csv_path: str) -> int:
    """Add gas demand from Gas Demand.csv."""
    demand_added = 0
    try:
        gas_demand_file = str(Path(csv_path) / "Gas Demand.csv")
        if Path(gas_demand_file).exists():
            gas_demand_df = pd.read_csv(gas_demand_file)

            gas_buses = [
                b
                for b in network.buses.index
                if network.buses.at[b, "carrier"] == "Gas"
            ]

            for i, (_, demand_row) in enumerate(gas_demand_df.iterrows()):
                if i >= len(gas_buses):
                    break

                demand_name = demand_row["object"]
                bus = gas_buses[i % len(gas_buses)]

                # Create gas demand profile
                base_demand = 200 * (1 + 0.1 * i)  # Varying demand
                demand_profile = pd.Series(base_demand, index=network.snapshots)

                load_name = f"gas_demand_{demand_name}"
                if load_name not in network.loads.index:
                    network.add(
                        "Load", load_name, bus=bus, p_set=demand_profile, carrier="Gas"
                    )
                    demand_added += 1

    except Exception as e:
        logger.warning(f"Failed to add gas demand: {e}")

    return demand_added


def add_gas_fields_csv(network: Network, csv_path: str) -> int:
    """Add gas fields from Gas Field.csv."""
    fields_added = 0
    try:
        gas_field_file = str(Path(csv_path) / "Gas Field.csv")
        if Path(gas_field_file).exists():
            gas_field_df = pd.read_csv(gas_field_file)

            gas_buses = [
                b
                for b in network.buses.index
                if network.buses.at[b, "carrier"] == "Gas"
            ]

            for i, (_, field_row) in enumerate(gas_field_df.iterrows()):
                if i >= len(gas_buses):
                    break

                field_name = field_row["object"]
                bus = gas_buses[i % len(gas_buses)]

                # Get field production capacity
                max_production = field_row.get("Max Production", 500)

                try:
                    p_nom = float(max_production) if pd.notna(max_production) else 500.0
                except Exception:
                    p_nom = 500.0

                gen_name = f"gas_field_{field_name}"
                if gen_name not in network.generators.index:
                    network.add(
                        "Generator",
                        gen_name,
                        bus=bus,
                        p_nom=p_nom,
                        carrier="Gas",
                        marginal_cost=25.0,
                    )  # Gas wellhead cost
                    fields_added += 1

    except Exception as e:
        logger.warning(f"Failed to add gas fields: {e}")

    return fields_added


def add_gas_electric_coupling_csv(network: Network, csv_path: str) -> dict[str, Any]:
    """Add gas-to-electric conversion from Generator.csv with Gas Node connections."""
    coupling_stats = {"gas_generators": 0, "efficiency_range": "N/A"}

    try:
        gen_file = str(Path(csv_path) / "Generator.csv")
        if Path(gen_file).exists():
            generators_df = pd.read_csv(gen_file)

            efficiency_values = []

            for _, gen_row in generators_df.iterrows():
                gen_name = gen_row["object"]
                node = gen_row.get("Node", "")
                gas_node = gen_row.get("Gas Node", "")

                # Skip if not a gas generator
                if pd.isna(gas_node) or not gas_node or gas_node == "":
                    continue

                # Check if electric and gas buses exist
                if node in network.buses.index:
                    gas_bus = f"gas_{gas_node.strip()}"
                    if gas_bus in network.buses.index:
                        # Get generator properties
                        max_capacity = gen_row.get("Max Capacity", 100)
                        heat_rate = gen_row.get("Heat Rate", 9.0)

                        try:
                            p_nom = (
                                float(max_capacity) if pd.notna(max_capacity) else 100.0
                            )
                            hr = float(heat_rate) if pd.notna(heat_rate) else 9.0

                            # Calculate efficiency (3412 BTU/kWh conversion factor)
                            efficiency = 3412 / hr if hr > 0 else 0.4
                            efficiency = min(efficiency, 0.65)  # Cap at 65%

                            efficiency_values.append(efficiency)

                            # Create gas-to-electric conversion link
                            link_name = f"gas_to_elec_{gen_name}"
                            if link_name not in network.links.index:
                                network.add(
                                    "Link",
                                    link_name,
                                    bus0=gas_bus,
                                    bus1=node,
                                    p_nom=p_nom,
                                    efficiency=efficiency,
                                    carrier="Gas2Electric",
                                )
                                coupling_stats["gas_generators"] += 1
                        except Exception as exc:
                            logger.debug(
                                "Skipping gas generator %s due to error: %s",
                                gen_name,
                                exc,
                            )

            # Calculate efficiency range
            if efficiency_values:
                min_eff = min(efficiency_values)
                max_eff = max(efficiency_values)
                coupling_stats["efficiency_range"] = f"{min_eff:.1%} - {max_eff:.1%}"

    except Exception as e:
        logger.warning(f"Failed to add gas-electric coupling: {e}")

    return coupling_stats


# Additional function for PLEXOS-MESSAGE flow network (hydrogen, ammonia)
def setup_flow_network_csv(network: Network, csv_path: str) -> dict[str, Any]:
    """Set up multi-sector flow network from CSV data."""
    print(f"Setting up flow network from CSV data: {csv_path}")

    setup_summary = {"network_type": "flow_network_csv", "sectors": [], "processes": {}}

    try:
        # Step 1: Add flow nodes as buses
        sectors = add_flow_nodes_csv(network, csv_path)
        setup_summary["sectors"] = list(sectors.keys())

        for sector in sectors:
            setup_summary[sector.lower()] = {
                "nodes": sectors[sector],
                "paths": 0,
                "storage": 0,
                "demand": 0,
            }

        # Step 2: Set basic snapshots
        snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="H")
        network.set_snapshots(snapshots)

        # Step 3: Add flow paths as links
        paths = add_flow_paths_csv(network, csv_path)
        for sector, count in paths.items():
            if sector.lower() in setup_summary:
                setup_summary[sector.lower()]["paths"] = count

        # Step 4: Add processes for sector coupling
        processes = add_processes_csv(network, csv_path)
        setup_summary["processes"] = processes

        # Step 5: Add basic demands
        for sector in sectors:
            if sectors[sector] > 0:  # Only if we have buses for this sector
                add_flow_demand_csv(network, sector)

    except Exception:
        logger.exception("Error setting up flow network")
        raise

    return setup_summary


def add_flow_nodes_csv(network: Network, csv_path: str) -> dict[str, int]:
    """Add flow nodes from Flow Node.csv."""
    sectors = {}

    try:
        flow_node_file = str(Path(csv_path) / "Flow Node.csv")
        if Path(flow_node_file).exists():
            flow_nodes_df = pd.read_csv(flow_node_file)

            for _, node_row in flow_nodes_df.iterrows():
                node_name = node_row["object"]

                # Determine sector from node name
                if node_name.startswith("Elec_"):
                    sector = "Electricity"
                elif node_name.startswith("H2_"):
                    sector = "Hydrogen"
                elif node_name.startswith("NH3_"):
                    sector = "Ammonia"
                else:
                    sector = "Other"

                # Add carrier if needed
                if sector not in network.carriers.index:
                    network.add("Carrier", sector)

                # Add bus
                if node_name not in network.buses.index:
                    network.add("Bus", node_name, carrier=sector)

                    if sector not in sectors:
                        sectors[sector] = 0
                    sectors[sector] += 1

    except Exception as e:
        logger.warning(f"Failed to add flow nodes: {e}")

    return sectors


def add_flow_paths_csv(network: Network, csv_path: str) -> dict[str, int]:
    """Add flow paths from Flow Path.csv."""
    paths = {"Transport": 0, "Conversion": 0}

    try:
        flow_path_file = str(Path(csv_path) / "Flow Path.csv")
        if Path(flow_path_file).exists():
            flow_paths_df = pd.read_csv(flow_path_file)

            for _, path_row in flow_paths_df.iterrows():
                path_name = path_row["object"]

                # Get connected flow nodes
                flow_nodes_str = path_row.get("Flow Node", "")
                if pd.notna(flow_nodes_str) and flow_nodes_str:
                    try:
                        if flow_nodes_str.startswith("["):
                            flow_nodes = ast.literal_eval(flow_nodes_str)
                        else:
                            flow_nodes = flow_nodes_str.split(",")

                        if len(flow_nodes) >= 2:
                            # bus0 = flow_nodes[0].strip().strip('\\'\"')
                            # bus1 = flow_nodes[1].strip().strip('\\'\"')
                            bus0 = flow_nodes[0].strip().strip("'\"")
                            bus1 = flow_nodes[1].strip().strip("'\"")

                            if (
                                bus0 in network.buses.index
                                and bus1 in network.buses.index
                            ):
                                # Get path properties
                                max_flow = path_row.get("Max Flow", 1000)
                                efficiency = path_row.get("Efficiency", 1.0)

                                try:
                                    p_nom = (
                                        float(max_flow)
                                        if pd.notna(max_flow)
                                        else 1000.0
                                    )
                                    eff = (
                                        float(efficiency)
                                        if pd.notna(efficiency) and efficiency != ""
                                        else 1.0
                                    )
                                except Exception:
                                    p_nom = 1000.0
                                    eff = 1.0

                                # Determine link type
                                bus0_carrier = network.buses.at[bus0, "carrier"]
                                bus1_carrier = network.buses.at[bus1, "carrier"]

                                if bus0_carrier == bus1_carrier:
                                    link_type = "Transport"
                                    link_name = f"transport_{path_name}"
                                else:
                                    link_type = "Conversion"
                                    link_name = f"conversion_{path_name}"

                                if link_name not in network.links.index:
                                    network.add(
                                        "Link",
                                        link_name,
                                        bus0=bus0,
                                        bus1=bus1,
                                        p_nom=p_nom,
                                        efficiency=eff,
                                    )
                                    paths[link_type] += 1
                    except Exception as exc:
                        logger.debug(
                            "Skipping flow path %s due to error: %s",
                            path_name,
                            exc,
                        )

    except Exception as e:
        logger.warning(f"Failed to add flow paths: {e}")

    return paths


def add_processes_csv(network: Network, csv_path: str) -> dict[str, int]:
    """Add processes from Process.csv."""
    processes = {}

    try:
        process_file = str(Path(csv_path) / "Process.csv")
        if Path(process_file).exists():
            process_df = pd.read_csv(process_file)

            for _, process_row in process_df.iterrows():
                process_name = process_row["object"]
                efficiency_val = process_row.get("Efficiency", 70)

                try:
                    efficiency = (
                        float(efficiency_val) / 100.0
                        if pd.notna(efficiency_val)
                        else 0.7
                    )
                except Exception:
                    efficiency = 0.7

                # Determine process type and create links
                if "electrolysis" in process_name.lower():
                    process_type = "Electrolysis"
                    # Create electricity -> hydrogen links
                    elec_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Electricity"
                    ]
                    h2_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Hydrogen"
                    ]

                    for i in range(min(3, len(elec_buses), len(h2_buses))):
                        link_name = f"electrolysis_{i + 1}"
                        if link_name not in network.links.index:
                            network.add(
                                "Link",
                                link_name,
                                bus0=elec_buses[i],
                                bus1=h2_buses[i],
                                p_nom=100,
                                efficiency=efficiency,
                            )

                elif "h2power" in process_name.lower():
                    process_type = "H2_Power"
                    # Create hydrogen -> electricity links
                    h2_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Hydrogen"
                    ]
                    elec_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Electricity"
                    ]

                    for i in range(min(2, len(h2_buses), len(elec_buses))):
                        link_name = f"fuel_cell_{i + 1}"
                        if link_name not in network.links.index:
                            network.add(
                                "Link",
                                link_name,
                                bus0=h2_buses[i],
                                bus1=elec_buses[i],
                                p_nom=50,
                                efficiency=efficiency,
                            )

                elif "ammonia" in process_name.lower():
                    process_type = "Ammonia"
                    # Create hydrogen -> ammonia links
                    h2_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Hydrogen"
                    ]
                    nh3_buses = [
                        b
                        for b in network.buses.index
                        if network.buses.at[b, "carrier"] == "Ammonia"
                    ]

                    for i in range(min(2, len(h2_buses), len(nh3_buses))):
                        link_name = f"ammonia_synthesis_{i + 1}"
                        if link_name not in network.links.index:
                            network.add(
                                "Link",
                                link_name,
                                bus0=h2_buses[i],
                                bus1=nh3_buses[i],
                                p_nom=30,
                                efficiency=efficiency,
                            )
                else:
                    process_type = "Other"

                if process_type not in processes:
                    processes[process_type] = 0
                processes[process_type] += 1

    except Exception as e:
        logger.warning(f"Failed to add processes: {e}")

    return processes


def add_flow_demand_csv(network: Network, sector: str) -> None:
    """Add basic demand profiles to flow network buses."""
    try:
        sector_buses = [
            b for b in network.buses.index if network.buses.at[b, "carrier"] == sector
        ]

        for i, bus in enumerate(sector_buses[:3]):  # First 3 buses per sector
            load_name = f"{sector.lower()}_demand_{i + 1}"
            if load_name not in network.loads.index:
                # Different base demands by sector
                if sector == "Electricity":
                    base_demand = 1500 * (1 + 0.1 * i)
                elif sector == "Hydrogen":
                    base_demand = 150 * (1 + 0.1 * i)
                elif sector == "Ammonia":
                    base_demand = 75 * (1 + 0.1 * i)
                else:
                    base_demand = 100 * (1 + 0.1 * i)

                # Create demand profile with daily variation
                demand_profile = pd.Series(
                    base_demand
                    * (
                        1
                        + 0.3
                        * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)
                    ),
                    index=network.snapshots,
                )

                network.add(
                    "Load", load_name, bus=bus, p_set=demand_profile, carrier=sector
                )

    except Exception as e:
        logger.warning(f"Failed to add {sector} demand: {e}")
