import logging
from typing import Any

import numpy as np
import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from network.progress import create_progress_tracker

logger = logging.getLogger(__name__)


def discover_flow_classes(db: PlexosDB) -> dict[str, list[str]]:
    """Discover flow-related classes using the PlexosDB list_classes method.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping class categories to class names
    """
    flow_classes: dict[str, list[str]] = {"flow": [], "process": [], "facility": []}

    try:
        # Get all available classes using PlexosDB method
        all_classes = db.list_classes()

        for class_name in all_classes:
            class_name_lower = class_name.lower()
            if class_name_lower.startswith("flow"):
                flow_classes["flow"].append(class_name)
            elif class_name_lower.startswith("process"):
                flow_classes["process"].append(class_name)
            elif class_name_lower.startswith("facility"):
                flow_classes["facility"].append(class_name)

        logger.info(f"Discovered flow classes: {flow_classes}")

    except Exception as e:
        logger.warning(f"Failed to discover flow classes: {e}")

    return flow_classes


def get_flow_objects_by_class_name(db: PlexosDB, class_name: str) -> list[str]:
    """Get all objects of a specific flow class using PlexosDB methods.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Name of the flow class to query

    Returns
    -------
    List[str]
        List of flow object names
    """
    objects = []

    try:
        # Use the built-in PlexosDB method
        objects = db.list_objects_by_class(class_name)
        logger.debug(f"Found {len(objects)} objects for flow class {class_name}")

    except Exception as e:
        logger.warning(f"Failed to get objects for flow class {class_name}: {e}")

    return objects


def get_flow_object_properties(
    db: PlexosDB, class_name: str, object_name: str
) -> list[dict[str, Any]]:
    """Get properties for a flow object using PlexosDB methods.

    Note: Flow classes may not have direct ClassEnum mappings, so this function
    handles cases where ClassEnum lookup fails gracefully.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the flow object
    object_name : str
        Name of the flow object to query properties for

    Returns
    -------
    List[Dict[str, Any]]
        List of property dictionaries
    """
    properties: list[dict[str, Any]] = []

    try:
        # Flow classes might not have direct ClassEnum mappings
        # Use generic property lookup if available
        logger.debug(
            f"Flow class '{class_name}' may not have ClassEnum mapping, using alternative approach"
        )

    except Exception as e:
        logger.debug(
            f"Failed to get properties for flow object {object_name} (class: {class_name}): {e}"
        )

    return properties


def get_flow_object_memberships(
    db: PlexosDB, class_name: str, object_name: str
) -> list[dict[str, Any]]:
    """Get membership relationships for a flow object using PlexosDB methods.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the flow object
    object_name : str
        Name of the flow object to query memberships for

    Returns
    -------
    List[Dict[str, Any]]
        List of membership relationships
    """
    memberships = []

    try:
        # Use the built-in PlexosDB method with class enum if available
        if class_name in {"Flow Path", "Facility"}:
            # Use string-based method which works in debug script
            memberships = db.get_memberships_system(
                object_name, object_class=class_name
            )

        else:
            # Default approach for other classes
            memberships = db.get_memberships_system(
                object_name, object_class=class_name
            )

        logger.debug(
            f"Found {len(memberships)} memberships for flow object {object_name}"
        )

    except Exception as e:
        logger.warning(f"Failed to get memberships for flow object {object_name}: {e}")

    return memberships


def extract_plexos_properties(
    properties: list[dict[str, Any]], property_mappings: dict[str, str]
) -> dict[str, Any]:
    """Extract and map PLEXOS properties to PyPSA attributes.

    Parameters
    ----------
    properties : List[Dict[str, Any]]
        List of PLEXOS property dictionaries
    property_mappings : Dict[str, str]
        Mapping from PLEXOS property names to PyPSA attribute names

    Returns
    -------
    Dict[str, Any]
        Dictionary of PyPSA attributes with extracted values
    """
    extracted = {}

    for prop in properties:
        prop_name = prop.get("property", "")
        if prop_name in property_mappings:
            pypsa_attr = property_mappings[prop_name]
            try:
                # Try to convert to float, fallback to original value
                value = (
                    float(prop.get("value", 0))
                    if prop.get("value") is not None
                    else None
                )
                if value is not None:
                    extracted[pypsa_attr] = value
            except (ValueError, TypeError):
                # Keep original value if conversion fails
                extracted[pypsa_attr] = prop.get("value")

    return extracted


def map_efficiency_from_heat_rate(heat_rate: float) -> float:
    """Convert thermal heat rate to efficiency.

    Parameters
    ----------
    heat_rate : float
        Heat rate in BTU/kWh

    Returns
    -------
    float
        Efficiency (0-1 range)
    """
    if heat_rate <= 0:
        return 0.4  # Default efficiency

    # Standard conversion: 3412 BTU/kWh is 100% efficient
    efficiency = 3412 / heat_rate
    return min(efficiency, 0.65)  # Cap at 65% efficiency


def extract_capacity_properties(properties: list[dict[str, Any]]) -> dict[str, float]:
    """Extract capacity-related properties from PLEXOS data.

    Parameters
    ----------
    properties : List[Dict[str, Any]]
        List of PLEXOS property dictionaries

    Returns
    -------
    Dict[str, float]
        Dictionary with capacity properties (p_nom, e_nom, etc.)
    """
    capacity_mappings = {
        "Max Capacity": "p_nom",
        "Max Flow": "p_nom",
        "Max Flow Day": "p_nom",
        "Max Injection": "p_nom",
        "Max Volume": "e_nom",
        "Max Storage": "e_nom",
        "Units": "units",
    }

    return extract_plexos_properties(properties, capacity_mappings)


def process_facility_memberships(
    db: PlexosDB, facility_class: str, facility_name: str
) -> dict[str, list[str]]:
    """Process Facility memberships to extract connected Flow Nodes and other components.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    facility_class : str
        Class name of the Facility
    facility_name : str
        Name of the Facility object

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping component types to lists of connected objects
    """
    connections: dict[str, list[str]] = {
        "flow_nodes": [],
        "processes": [],
        "gas_nodes": [],
        "electric_nodes": [],
    }

    try:
        memberships = get_flow_object_memberships(db, facility_class, facility_name)

        for membership in memberships:
            child_class = membership.get("child_class_name", "").lower()
            child_name = membership.get("child_object_name", "")
            membership.get("parent_class_name", "").lower()
            membership.get("parent_object_name", "")

            # Based on debug results: Facilities connect to Flow Nodes as CHILDREN
            # So we look for Flow Nodes in the child position
            if "flow" in child_class and "node" in child_class:
                if child_name not in connections["flow_nodes"]:
                    connections["flow_nodes"].append(child_name)
            elif "process" in child_class:
                if child_name not in connections["processes"]:
                    connections["processes"].append(child_name)
            elif "gas" in child_class and "node" in child_class:
                if child_name not in connections["gas_nodes"]:
                    connections["gas_nodes"].append(child_name)
            elif (
                child_class == "node"
                and child_name not in connections["electric_nodes"]
            ):
                connections["electric_nodes"].append(child_name)

    except Exception as e:
        logger.warning(
            f"Failed to process memberships for facility {facility_name}: {e}"
        )

    return connections


def identify_conversion_processes(
    facility_name: str, processes: list[str]
) -> dict[str, Any]:
    """Identify the type of conversion process and determine appropriate parameters.

    Parameters
    ----------
    facility_name : str
        Name of the Facility
    processes : List[str]
        List of connected Process names

    Returns
    -------
    Dict[str, Any]
        Dictionary with process type and parameters
    """
    facility_lower = facility_name.lower()
    process_str = " ".join(processes).lower()

    # Process identification based on name patterns
    if "nh32power" in facility_lower or "ammonia_to_power" in facility_lower:
        return {
            "type": "NH3_to_Electric",
            "efficiency": 0.40,  # NH3 fuel cell efficiency
            "carriers": ("NH3", "AC"),
            "description": "Ammonia to electricity conversion",
        }

    elif "haberbosch" in facility_lower or "ammonia_synthesis" in facility_lower:
        return {
            "type": "H2_to_NH3",
            "efficiency": 0.60,  # Haber-Bosch synthesis efficiency
            "carriers": ("H2", "NH3"),
            "description": "Hydrogen to ammonia synthesis (Haber-Bosch)",
        }

    elif "ammoniacrack" in facility_lower or "ammonia_crack" in facility_lower:
        return {
            "type": "NH3_to_H2",
            "efficiency": 0.70,  # Ammonia cracking efficiency
            "carriers": ("NH3", "H2"),
            "description": "Ammonia cracking to hydrogen",
        }

    elif "electrolysis" in facility_lower or "electrolysis" in process_str:
        return {
            "type": "Electric_to_H2",
            "efficiency": 0.75,  # Electrolyzer efficiency
            "carriers": ("AC", "H2"),
            "description": "Water electrolysis",
        }

    elif "fuelcell" in facility_lower or "fuel_cell" in process_str:
        return {
            "type": "H2_to_Electric",
            "efficiency": 0.50,  # Fuel cell efficiency
            "carriers": ("H2", "AC"),
            "description": "Hydrogen fuel cell",
        }

    else:
        return {
            "type": "Generic_Conversion",
            "efficiency": 0.75,  # Default efficiency
            "carriers": ("Unknown", "Unknown"),
            "description": "Generic conversion process",
        }


def determine_carrier_from_node_name(node_name: str) -> str:
    """Determine carrier type from Flow Node name using naming conventions.

    Parameters
    ----------
    node_name : str
        Name of the Flow Node

    Returns
    -------
    str
        Carrier name
    """
    node_lower = node_name.lower()

    if node_name.startswith("NH3_") or "nh3" in node_lower or "ammonia" in node_lower:
        return "NH3"
    elif node_name.startswith("H2_") or "h2" in node_lower or "hydrogen" in node_lower:
        return "H2"
    elif (
        node_name.startswith("Elec_")
        or "elec" in node_lower
        or "electric" in node_lower
    ):
        return "AC"
    elif node_name.startswith("Gas_") or "gas" in node_lower:
        return "Gas"
    else:
        return "Other"


def add_flow_buses(network: Network, db: PlexosDB) -> dict[str, int]:
    """Add flow nodes as buses, organized by sector with enhanced carrier detection.

    Parameters
    ----------
    network : Network
        PyPSA network to add buses to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    Dict[str, int]
        Dictionary mapping sector names to number of buses added
    """
    sector_counts = {}

    try:
        flow_classes = discover_flow_classes(db)

        for flow_class in flow_classes["flow"]:
            if "node" in flow_class.lower():
                flow_objects = get_flow_objects_by_class_name(db, flow_class)

                for node_name in flow_objects:
                    # Use enhanced carrier detection
                    carrier = determine_carrier_from_node_name(node_name)

                    # Map carriers to PyPSA standard names
                    if carrier == "NH3":
                        sector = "Ammonia"
                    elif carrier == "H2":
                        sector = "Hydrogen"
                    elif carrier == "AC":
                        sector = "Electricity"
                    elif carrier == "Gas":
                        sector = "Gas"
                    else:
                        sector = "Other"

                    # Add carrier if needed
                    if carrier not in network.carriers.index:
                        network.add("Carrier", carrier)

                    # Get node properties for enhanced bus parameters
                    try:
                        # Try to get properties using ClassEnum.FlowNode
                        props = db.get_object_properties(ClassEnum.FlowNode, node_name)
                        extract_capacity_properties(props)

                        # Determine v_nom based on carrier type
                        if carrier == "AC":
                            v_nom = 110.0  # kV for electricity
                        elif carrier == "Gas":
                            v_nom = 50.0  # bar for gas pressure
                        elif carrier == "H2":
                            v_nom = 30.0  # bar for hydrogen pressure
                        elif carrier == "NH3":
                            v_nom = 10.0  # bar for ammonia pressure
                        else:
                            v_nom = 1.0  # Default

                        # Add bus with enhanced parameters
                        if node_name not in network.buses.index:
                            network.add("Bus", node_name, carrier=carrier, v_nom=v_nom)

                            if sector not in sector_counts:
                                sector_counts[sector] = 0
                            sector_counts[sector] += 1

                    except Exception as e:
                        # Fallback to basic bus creation
                        logger.debug(f"Could not get properties for {node_name}: {e}")
                        if node_name not in network.buses.index:
                            network.add("Bus", node_name, carrier=carrier)

                            if sector not in sector_counts:
                                sector_counts[sector] = 0
                            sector_counts[sector] += 1

    except Exception:
        logger.exception("Failed to add flow buses")

    total_buses = sum(sector_counts.values())
    print(f"Added {total_buses} flow buses across sectors: {sector_counts}")
    return sector_counts


def add_flow_paths(
    network: Network,
    db: PlexosDB,
    timeslice_csv: str | None = None,
    testing_mode: bool = False,
) -> dict[str, int]:
    """Add flow paths as links with enhanced carrier detection and property extraction.

    This function processes PLEXOS Flow Path objects and creates appropriate PyPSA Links:
    - Transport links for same-carrier connections (e.g., NH3 → NH3 shipping)
    - Conversion links for cross-carrier connections (e.g., H2 → NH3)

    Parameters
    ----------
    network : Network
        PyPSA network to add paths to
    db : PlexosDB
        PLEXOS database connection
    timeslice_csv : str, optional
        Path to timeslice CSV file for time-dependent properties
    testing_mode : bool, optional
        If True, process limited subset for testing

    Returns
    -------
    Dict[str, int]
        Dictionary with transport and conversion link counts
    """
    paths = {"Transport": 0, "Conversion": 0, "Shipping": 0}

    try:
        flow_classes = discover_flow_classes(db)

        # Count total flow paths first for progress tracking
        total_paths = 0
        for flow_class in flow_classes["flow"]:
            if "path" in flow_class.lower():
                path_objects = get_flow_objects_by_class_name(db, flow_class)
                total_paths += len(path_objects)

        print(f"  Found {total_paths} flow paths to process")

        # Initialize progress tracker
        if total_paths > 0:
            path_progress = create_progress_tracker(
                total_paths,
                "Processing Flow Paths",
                testing_mode=testing_mode,
                test_limit=500,
            )
            path_index = 0

            for flow_class in flow_classes["flow"]:
                if "path" in flow_class.lower():
                    path_objects = get_flow_objects_by_class_name(db, flow_class)

                    for path_name in path_objects:
                        # Check if we should process this path (testing mode)
                        if not path_progress.should_process(path_index):
                            break

                        # Get path properties for capacity and efficiency
                        try:
                            # Try to get properties using ClassEnum.FlowPath if available
                            props = []
                            try:
                                if hasattr(ClassEnum, "FlowPath"):
                                    props = db.get_object_properties(
                                        ClassEnum.FlowPath, path_name
                                    )
                            except Exception:
                                pass

                            # Extract capacity properties
                            capacity_props = extract_capacity_properties(props)
                            p_nom = capacity_props.get(
                                "p_nom", 1000.0
                            )  # Default capacity

                        except Exception as e:
                            logger.debug(
                                f"Could not get properties for flow path {path_name}: {e}"
                            )
                            p_nom = 1000.0  # Default capacity

                        # Get connected flow nodes using direct method (same as debug script)
                        memberships = db.get_memberships_system(
                            path_name, object_class=flow_class
                        )

                        # Extract flow nodes - try multiple approaches for connection detection
                        connected_nodes = []
                        for membership in memberships:
                            # Direct plexosdb uses different field names: 'class' and 'name'
                            obj_class = membership.get("class", "").lower()
                            obj_name = membership.get("name", "")

                            # Look for Flow Node connections
                            if (
                                "flow" in obj_class
                                and "node" in obj_class
                                and obj_name not in connected_nodes
                            ):
                                connected_nodes.append(obj_name)

                        # Debug logging for first few paths
                        if path_index < 5:
                            print(f"    Debug - Path {path_index}: {path_name}")
                            print(f"    Raw memberships found: {len(memberships)}")
                            if memberships:
                                print(f"    First membership sample: {memberships[0]}")
                                for j, mem in enumerate(memberships[:4]):
                                    child_class = mem.get("child_class_name", "")
                                    child_name = mem.get("child_object_name", "")
                                    parent_class = mem.get("parent_class_name", "")
                                    parent_name = mem.get("parent_object_name", "")
                                    print(
                                        f"      Mem {j + 1}: {parent_class} '{parent_name}' -> {child_class} '{child_name}'"
                                    )
                            print(f"    Connected nodes: {connected_nodes}")
                            print(f"    P_nom: {p_nom}")

                        if len(connected_nodes) >= 2:
                            bus0, bus1 = connected_nodes[0], connected_nodes[1]

                            if (
                                bus0 in network.buses.index
                                and bus1 in network.buses.index
                            ):
                                # Get carrier information for both buses
                                bus0_carrier = network.buses.at[bus0, "carrier"]
                                bus1_carrier = network.buses.at[bus1, "carrier"]

                                # Determine link type and efficiency based on carriers
                                if bus0_carrier == bus1_carrier:
                                    # Same carrier - transport link
                                    if (
                                        "shipping" in path_name.lower()
                                        or "route" in path_name.lower()
                                    ):
                                        link_type = "Shipping"
                                        efficiency = (
                                            0.95  # Shipping efficiency with losses
                                        )
                                        link_name = f"shipping_{path_name}"
                                    else:
                                        link_type = "Transport"
                                        efficiency = (
                                            0.98  # Pipeline/transmission efficiency
                                        )
                                        link_name = f"transport_{path_name}"

                                    link_carrier = (
                                        bus0_carrier  # Use the common carrier
                                    )

                                else:
                                    # Different carriers - conversion link
                                    link_type = "Conversion"
                                    link_name = f"conversion_{path_name}"
                                    link_carrier = f"{bus0_carrier}2{bus1_carrier}"

                                    # Set efficiency based on conversion type
                                    if bus0_carrier == "H2" and bus1_carrier == "NH3":
                                        efficiency = 0.60  # H2 to NH3 synthesis
                                    elif bus0_carrier == "NH3" and bus1_carrier == "H2":
                                        efficiency = 0.70  # NH3 cracking
                                    elif bus0_carrier == "AC" and bus1_carrier == "H2":
                                        efficiency = 0.75  # Electrolysis
                                    elif bus0_carrier == "H2" and bus1_carrier == "AC":
                                        efficiency = 0.50  # Fuel cell
                                    elif bus0_carrier == "NH3" and bus1_carrier == "AC":
                                        efficiency = 0.40  # NH3 fuel cell
                                    else:
                                        efficiency = (
                                            0.75  # Default conversion efficiency
                                        )

                                # Add carrier if needed
                                if link_carrier not in network.carriers.index:
                                    network.add("Carrier", link_carrier)

                                # Create the link
                                if link_name not in network.links.index:
                                    network.add(
                                        "Link",
                                        link_name,
                                        bus0=bus0,
                                        bus1=bus1,
                                        p_nom=p_nom,
                                        efficiency=efficiency,
                                        carrier=link_carrier,
                                    )

                                    paths[link_type] += 1

                                    # Debug logging for first few links
                                    if paths[link_type] <= 3:
                                        print(
                                            f"     Added {link_type.lower()} link: {link_name}"
                                        )
                                        print(
                                            f"      {bus0} ({bus0_carrier}) -> {bus1} ({bus1_carrier})"
                                        )
                                        print(
                                            f"      Efficiency: {efficiency:.1%}, Capacity: {p_nom}"
                                        )
                            elif path_index < 5:
                                print(f"     Bus(es) not found: {bus0}, {bus1}")
                        elif path_index < 5:
                            print(
                                f"     Insufficient connected nodes: {connected_nodes}"
                            )

                        # Update progress
                        path_progress.update(path_index + 1, path_name)
                        path_index += 1

                        # Break out of both loops if we've hit the limit
                        if not path_progress.should_process(path_index):
                            break

                    # Break outer loop if we've hit the limit
                    if path_index >= path_progress.items_to_process:
                        break

            # Finish progress tracking
            total_added = sum(paths.values())
            path_progress.finish(total_added)

    except Exception:
        logger.exception("Failed to add flow paths")

    print(
        f"Added {paths['Transport']} transport links, {paths['Conversion']} conversion links, and {paths['Shipping']} shipping links"
    )
    return paths


def add_processes(network: Network, db: PlexosDB) -> dict[str, int]:
    """Add processes as sector coupling links.

    Parameters
    ----------
    network : Network
        PyPSA network to add processes to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    Dict[str, int]
        Dictionary mapping process types to counts
    """
    process_stats = {}

    try:
        flow_classes = discover_flow_classes(db)

        # Process discovered process classes
        for process_class in flow_classes["process"]:
            process_objects = get_flow_objects_by_class_name(db, process_class)

            for process_name in process_objects:
                # Get process properties (would extract efficiency if properties were available)
                get_flow_object_properties(db, process_class, process_name)
                efficiency = 70.0  # Default

                try:
                    eff = float(efficiency) / 100.0 if efficiency else 0.7
                except Exception:
                    eff = 0.7

                # Determine process type and create links based on name patterns
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
                                efficiency=eff,
                            )

                elif (
                    "h2power" in process_name.lower()
                    or "fuel_cell" in process_name.lower()
                ):
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
                                efficiency=eff,
                            )

                elif "ammonia" in process_name.lower():
                    process_type = "Ammonia_Synthesis"
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
                                efficiency=eff,
                            )
                else:
                    process_type = "Other"

                if process_type not in process_stats:
                    process_stats[process_type] = 0
                process_stats[process_type] += 1

    except Exception:
        logger.exception("Failed to add processes")

    total_processes = sum(process_stats.values())
    print(f"Added {total_processes} processes: {process_stats}")
    return process_stats


def add_facilities(
    network: Network, db: PlexosDB, testing_mode: bool = False
) -> dict[str, int]:
    """Add facilities as conversion links, generators, or loads with enhanced property extraction.

    This function processes PLEXOS Facility objects and converts them to appropriate PyPSA components:
    - Conversion facilities (NH32Power, HaberBosch, etc.) → Links with proper efficiency
    - Generation facilities → Generators with capacity and profiles
    - Demand facilities → Loads with demand profiles

    Parameters
    ----------
    network : Network
        PyPSA network to add facilities to
    db : PlexosDB
        PLEXOS database connection
    testing_mode : bool, optional
        If True, process limited subset for testing

    Returns
    -------
    Dict[str, int]
        Dictionary with facility statistics including conversion links
    """
    facility_stats = {
        "conversion_links": 0,
        "generators": 0,
        "loads": 0,
        "storage": 0,
        "skipped": 0,
    }

    try:
        flow_classes = discover_flow_classes(db)

        # Count total facilities first
        total_facilities = 0
        for facility_class in flow_classes.get("facility", []):
            facility_objects = get_flow_objects_by_class_name(db, facility_class)
            total_facilities += len(facility_objects)

        print(f"  Found {total_facilities} facilities to process")

        # Initialize progress tracker for facilities
        if total_facilities > 0:
            facility_progress = create_progress_tracker(
                total_facilities,
                "Processing Facilities",
                testing_mode=testing_mode,
                test_limit=1000,
            )
            facility_index = 0

            # Process facility classes
            for facility_class in flow_classes.get("facility", []):
                facility_objects = get_flow_objects_by_class_name(db, facility_class)

                for facility_name in facility_objects:
                    # Check if we should process this facility (testing mode)
                    if not facility_progress.should_process(facility_index):
                        break

                    # Get facility properties for capacity and efficiency
                    try:
                        # Use ClassEnum.Facility now that it exists
                        props = db.get_object_properties(
                            ClassEnum.Facility, facility_name
                        )

                        # Extract capacity properties
                        capacity_props = extract_capacity_properties(props)
                        p_nom = capacity_props.get("p_nom", 100.0)  # Default capacity

                    except Exception as e:
                        logger.debug(
                            f"Could not get properties for facility {facility_name}: {e}"
                        )
                        props = []
                        p_nom = 100.0  # Default capacity

                    # Process facility memberships to find connections using direct method
                    memberships = db.get_memberships_system(
                        facility_name, object_class=facility_class
                    )

                    # Extract connected flow nodes directly using correct field names for direct plexosdb
                    flow_nodes = []
                    processes = []
                    for membership in memberships:
                        obj_class = membership.get("class", "").lower()
                        obj_name = membership.get("name", "")

                        if "flow" in obj_class and "node" in obj_class:
                            if obj_name not in flow_nodes:
                                flow_nodes.append(obj_name)
                        elif "process" in obj_class and obj_name not in processes:
                            processes.append(obj_name)

                    # Identify the conversion process type
                    process_info = identify_conversion_processes(
                        facility_name, processes
                    )

                    # Debug logging for first few facilities
                    if facility_index < 5:
                        print(f"    Debug - Facility {facility_index}: {facility_name}")
                        print(f"    Flow nodes: {flow_nodes}")
                        print(f"    Process type: {process_info['type']}")
                        print(f"    Efficiency: {process_info['efficiency']}")
                        print(f"    P_nom: {p_nom}")

                    # Process based on facility type and connections
                    if (
                        len(flow_nodes) >= 2
                        and process_info["type"] != "Generic_Conversion"
                    ):
                        # Multi-node conversion facility - create as conversion link
                        bus0, bus1 = flow_nodes[0], flow_nodes[1]

                        # Determine correct bus order based on conversion type
                        if process_info["type"] == "NH3_to_Electric":
                            # NH3 bus → Electric bus
                            nh3_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "NH3"
                                ),
                                None,
                            )
                            elec_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "AC"
                                ),
                                None,
                            )
                            if nh3_bus and elec_bus:
                                bus0, bus1 = nh3_bus, elec_bus

                        elif process_info["type"] == "H2_to_NH3":
                            # H2 bus → NH3 bus
                            h2_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "H2"
                                ),
                                None,
                            )
                            nh3_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "NH3"
                                ),
                                None,
                            )
                            if h2_bus and nh3_bus:
                                bus0, bus1 = h2_bus, nh3_bus

                        elif process_info["type"] == "NH3_to_H2":
                            # NH3 bus → H2 bus
                            nh3_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "NH3"
                                ),
                                None,
                            )
                            h2_bus = next(
                                (
                                    bus
                                    for bus in flow_nodes
                                    if determine_carrier_from_node_name(bus) == "H2"
                                ),
                                None,
                            )
                            if nh3_bus and h2_bus:
                                bus0, bus1 = nh3_bus, h2_bus

                        if bus0 in network.buses.index and bus1 in network.buses.index:
                            link_name = f"facility_{facility_name}"
                            if link_name not in network.links.index:
                                # Create conversion link
                                carrier_from, carrier_to = process_info["carriers"]
                                link_carrier = (
                                    f"{carrier_from}2{carrier_to}"
                                    if carrier_from != "Unknown"
                                    else "conversion"
                                )

                                # Add conversion carrier if needed
                                if link_carrier not in network.carriers.index:
                                    network.add("Carrier", link_carrier)

                                network.add(
                                    "Link",
                                    link_name,
                                    bus0=bus0,
                                    bus1=bus1,
                                    p_nom=p_nom,
                                    efficiency=process_info["efficiency"],
                                    carrier=link_carrier,
                                )

                                facility_stats["conversion_links"] += 1

                                # Debug logging for first few conversion links
                                if facility_stats["conversion_links"] <= 5:
                                    print(f"     Added conversion link: {link_name}")
                                    print(
                                        f"      {bus0} -> {bus1} (efficiency: {process_info['efficiency']:.1%})"
                                    )
                        else:
                            facility_stats["skipped"] += 1
                            if facility_index < 5:
                                print(f"     Bus(es) not found: {bus0}, {bus1}")

                    elif len(flow_nodes) >= 1:
                        primary_bus = flow_nodes[0]

                        if primary_bus in network.buses.index:
                            # Check if this is a generation facility
                            generator_keywords = [
                                "solar",
                                "wind",
                                "hydro",
                                "nuclear",
                                "gas",
                                "biomass",
                                "geothermal",
                                "coal",
                                "oil",
                                "pv",
                                "thermal",
                                "power",
                                "gen",
                            ]
                            is_generator = any(
                                keyword in facility_name.lower()
                                or keyword in " ".join(processes).lower()
                                for keyword in generator_keywords
                            )

                            if is_generator:
                                # Add as generator
                                gen_name = f"gen_{facility_name}"
                                if gen_name not in network.generators.index:
                                    # Create generation profile based on type
                                    if "solar" in facility_name.lower():
                                        profile = pd.Series(
                                            [
                                                max(
                                                    0,
                                                    0.8
                                                    * np.sin(
                                                        (h % 24) * np.pi / 12
                                                        - np.pi / 2
                                                    ),
                                                )
                                                for h in range(len(network.snapshots))
                                            ],
                                            index=network.snapshots,
                                        )
                                    elif "wind" in facility_name.lower():
                                        rng = np.random.default_rng()
                                        profile = pd.Series(
                                            [
                                                0.3 + 0.4 * rng.random()
                                                for _ in range(len(network.snapshots))
                                            ],
                                            index=network.snapshots,
                                        )
                                    else:
                                        # Baseload profile
                                        profile = pd.Series(
                                            [0.8] * len(network.snapshots),
                                            index=network.snapshots,
                                        )

                                    network.add(
                                        "Generator",
                                        gen_name,
                                        bus=primary_bus,
                                        p_nom=p_nom,
                                        p_max_pu=profile,
                                        marginal_cost=50,
                                    )
                                    facility_stats["generators"] += 1
                            else:
                                # Add as load (demand facility)
                                load_name = f"load_{facility_name}"
                                if load_name not in network.loads.index:
                                    base_demand = (
                                        p_nom * 0.5
                                    )  # 50% of facility capacity as base demand
                                    demand_profile = pd.Series(
                                        [base_demand] * len(network.snapshots),
                                        index=network.snapshots,
                                    )
                                    network.add(
                                        "Load",
                                        load_name,
                                        bus=primary_bus,
                                        p_set=demand_profile,
                                    )
                                    facility_stats["loads"] += 1
                        else:
                            facility_stats["skipped"] += 1
                            if facility_index < 5:
                                print(f"     Primary bus not found: {primary_bus}")
                    else:
                        facility_stats["skipped"] += 1
                        if facility_index < 5:
                            print(
                                f"     No flow nodes found for facility: {facility_name}"
                            )

                    # Update progress
                    facility_progress.update(facility_index + 1, facility_name)
                    facility_index += 1

                    # Break out of both loops if we've hit the limit
                    if not facility_progress.should_process(facility_index):
                        break

                # Break outer loop if we've hit the limit
                if facility_index >= facility_progress.items_to_process:
                    break

            # Finish facility progress tracking
            if total_facilities > 0:
                total_added = (
                    facility_stats["conversion_links"]
                    + facility_stats["generators"]
                    + facility_stats["loads"]
                    + facility_stats["storage"]
                )
                facility_progress.finish(total_added)

    except Exception:
        logger.exception("Failed to add facilities")

    print(
        f"Added {facility_stats['conversion_links']} conversion links, {facility_stats['generators']} generators, and {facility_stats['loads']} loads from facilities"
    )
    return facility_stats


def add_flow_demand(network: Network) -> None:
    """Add basic demand profiles to flow network buses by sector.

    Parameters
    ----------
    network : Network
        PyPSA network to add demand to
    """
    try:
        # Get buses by carrier/sector
        sectors = ["Electricity", "Hydrogen", "Ammonia", "Other"]

        for sector in sectors:
            sector_buses = [
                b
                for b in network.buses.index
                if network.buses.at[b, "carrier"] == sector
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

    except Exception:
        logger.exception("Failed to add flow demand")


def port_flow_network(
    network: Network,
    db: PlexosDB,
    timeslice_csv: str | None = None,
    testing_mode: bool = False,
) -> dict[str, Any]:
    """Comprehensive function to set up multi-sector flow networks using PLEXOS Flow Network components.

    This function follows established patterns from electricity sector modules
    and combines all flow network operations:
    - Adds flow nodes as buses (electricity, hydrogen, ammonia sectors)
    - Adds flow paths as links with transport/conversion logic
    - Adds processes as sector coupling links
    - Adds facilities as generators/loads
    - Adds basic demand profiles

    Parameters
    ----------
    network : Network
        The PyPSA network to set up as flow network.
    db : PlexosDB
        The Plexos database containing flow network data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.

    Returns
    -------
    Dict[str, Any]
        Summary statistics of flow network setup

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> flow_summary = port_flow_network(network, db,
    ...                                  timeslice_csv="path/to/timeslice.csv")
    """
    print("Starting flow network porting process...")

    summary = {
        "network_type": "flow_network",
        "sectors": [],
        "buses_by_sector": {},
        "paths": {},
        "processes": {},
        "facilities": {},
    }

    try:
        # Step 1: Add flow nodes as buses
        print("1. Adding flow nodes as buses...")
        summary["buses_by_sector"] = add_flow_buses(network, db)
        summary["sectors"] = list(summary["buses_by_sector"].keys())

        # Set basic snapshots if not already set
        if len(network.snapshots) == 0:
            snapshots = pd.date_range("2030-01-01", "2030-12-31 23:00", freq="h")
            network.set_snapshots(snapshots)
            print(f"Set {len(snapshots)} hourly snapshots")

        # Step 2: Add flow paths as links
        print("2. Adding flow paths as links...")
        summary["paths"] = add_flow_paths(
            network, db, timeslice_csv=timeslice_csv, testing_mode=testing_mode
        )

        # Step 3: Add processes for sector coupling
        print("3. Adding processes for sector coupling...")
        summary["processes"] = add_processes(network, db)

        # Step 4: Add facilities as generators/loads
        print("4. Adding facilities as generators/loads...")
        summary["facilities"] = add_facilities(network, db, testing_mode=testing_mode)

        # Step 5: Add basic demand profiles
        print("5. Adding basic demand profiles...")
        add_flow_demand(network)

    except Exception:
        logger.exception("Error in flow network porting")
        raise

    total_buses = sum(summary["buses_by_sector"].values())
    total_links = summary["paths"].get("Transport", 0) + summary["paths"].get(
        "Conversion", 0
    )
    total_processes = sum(summary["processes"].values())

    print("Flow network porting complete!")
    print(f"  Added {total_buses} buses across {len(summary['sectors'])} sectors")
    print(f"  Added {total_links} links ({summary['paths']})")
    print(f"  Added {total_processes} processes ({summary['processes']})")

    # Enhanced facility reporting
    conversion_links = summary["facilities"].get("conversion_links", 0)
    generators = summary["facilities"].get("generators", 0)
    loads = summary["facilities"].get("loads", 0)
    storage = summary["facilities"].get("storage", 0)

    print(
        f"  Added {conversion_links} conversion links from facilities (SOLVES ORPHANED BUSES)"
    )
    print(
        f"  Added {generators} generators, {loads} loads, and {storage} storage from facilities"
    )

    return summary
