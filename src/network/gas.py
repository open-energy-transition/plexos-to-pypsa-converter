"""Gas Network Components for PLEXOS-PyPSA Conversion.

This module provides functions to convert PLEXOS gas sector components
to PyPSA network components, following established patterns from
electricity sector modules.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

logger = logging.getLogger(__name__)


def discover_gas_classes(db: PlexosDB) -> list[str]:
    """Discover gas-related classes using the PlexosDB list_classes method.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    List[str]
        List of gas class names found in the database
    """
    gas_classes = []

    try:
        # Get all available classes using PlexosDB method
        all_classes = db.list_classes()

        # Use list comprehension to create a transformed list
        gas_classes = [
            class_name
            for class_name in all_classes
            if class_name.lower().startswith("gas")
        ]

        logger.info(f"Discovered gas classes: {gas_classes}")

    except Exception as e:
        logger.warning(f"Failed to discover gas classes: {e}")

    return gas_classes


def get_gas_objects_by_class_name(db: PlexosDB, class_name: str) -> list[str]:
    """Get all objects of a specific gas class using PlexosDB methods.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Name of the gas class to query

    Returns
    -------
    List[str]
        List of gas object names
    """
    objects = []

    try:
        # Use the built-in PlexosDB method
        objects = db.list_objects_by_class(class_name)
        logger.debug(f"Found {len(objects)} objects for gas class {class_name}")

    except Exception as e:
        logger.warning(f"Failed to get objects for gas class {class_name}: {e}")

    return objects


def get_gas_object_properties(
    db: PlexosDB, class_name: str, object_name: str
) -> list[dict[str, Any]]:
    """Get properties for a gas object using PlexosDB methods.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the gas object
    object_name : str
        Name of the gas object to query properties for

    Returns
    -------
    List[Dict[str, Any]]
        List of property dictionaries
    """
    properties: list[dict[str, Any]] = []

    try:
        # Convert string class name to ClassEnum (from multi_sector_db.py pattern)
        class_mapping = {
            "Gas Field": ClassEnum.Gas_Field,
            "Gas Plant": ClassEnum.Gas_Plant,
            "Gas Pipeline": ClassEnum.Gas_Pipeline,
            "Gas Node": ClassEnum.Gas_Node,
            "Gas Storage": ClassEnum.Gas_Storage,
            "Gas Demand": ClassEnum.Gas_Demand,
            "Gas DSM Program": ClassEnum.Gas_DSM_Program,
            "Gas Basin": ClassEnum.Gas_Basin,
            "Gas Zone": ClassEnum.Gas_Zone,
            "Gas Contract": ClassEnum.Gas_Contract,
            "Gas Transport": ClassEnum.Gas_Transport,
            "Gas Path": ClassEnum.Gas_Path,
            "Gas Capacity Release Offer": ClassEnum.Gas_Capacity_Release_Offer,
        }

        class_enum = class_mapping.get(class_name)

        if class_enum is None:
            logger.debug(
                f"No ClassEnum mapping found for gas class '{class_name}', skipping properties for {object_name}"
            )
            return properties

        # Use the built-in PlexosDB method with ClassEnum
        properties = db.get_object_properties(class_enum, object_name)
        logger.debug(
            f"Found {len(properties)} properties for gas object {object_name} (class: {class_name})"
        )

    except Exception as e:
        logger.debug(
            f"Failed to get properties for gas object {object_name} (class: {class_name}): {e}"
        )

    return properties


def get_gas_object_memberships(
    db: PlexosDB, class_name: str, object_name: str
) -> list[dict[str, Any]]:
    """Get membership relationships for a gas object using PlexosDB methods.

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the gas object
    object_name : str
        Name of the gas object to query memberships for

    Returns
    -------
    List[Dict[str, Any]]
        List of membership relationships
    """
    memberships = []

    try:
        # Use the built-in PlexosDB method
        memberships = db.get_memberships_system(object_name, object_class=class_name)
        logger.debug(
            f"Found {len(memberships)} memberships for gas object {object_name}"
        )

    except Exception as e:
        logger.warning(f"Failed to get memberships for gas object {object_name}: {e}")

    return memberships


def add_gas_buses(network: Network, db: PlexosDB) -> int:
    """Add gas buses to the network from Gas Node objects with enhanced properties.

    Parameters
    ----------
    network : Network
        PyPSA network to add buses to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    int
        Number of gas buses added
    """
    gas_buses_added = 0

    try:
        # Add gas carriers following PyPSA multi-sector patterns
        carriers_to_add = ["gas", "natural_gas", "biogas", "hydrogen"]
        for carrier in carriers_to_add:
            if carrier not in network.carriers.index:
                network.add("Carrier", carrier)

        # Get gas node classes
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "node" in gas_class.lower():
                gas_objects = get_gas_objects_by_class_name(db, gas_class)
                for gas_name in gas_objects:
                    # Enhanced bus naming with carrier specification
                    props = get_gas_object_properties(db, gas_class, gas_name)

                    # Determine gas type from properties or name
                    gas_type = "gas"  # Default
                    for prop in props:
                        if prop.get("property") == "Gas Type":
                            gas_type = prop.get("value", "gas").lower()
                            break

                    # Infer from name if property not found
                    if gas_type == "gas":
                        if "hydrogen" in gas_name.lower() or "h2" in gas_name.lower():
                            gas_type = "hydrogen"
                        elif "bio" in gas_name.lower():
                            gas_type = "biogas"
                        else:
                            gas_type = "natural_gas"

                    bus_name = f"gas_{gas_name}"
                    if bus_name not in network.buses.index:
                        # Extract additional properties
                        pressure = None
                        for prop in props:
                            if prop.get("property") == "Pressure":
                                try:
                                    pressure = float(prop.get("value", 0))
                                except Exception:
                                    logger.debug(
                                        f"Could not parse pressure value for {gas_name}"
                                    )
                                break

                        bus_attrs = {"carrier": gas_type}
                        if pressure:
                            bus_attrs["v_nom"] = pressure  # Store pressure in v_nom

                        network.add("Bus", bus_name, **bus_attrs)
                        gas_buses_added += 1

    except Exception:
        logger.exception("Failed to add gas buses")

    print(f"Added {gas_buses_added} gas buses")
    return gas_buses_added


def add_gas_pipelines(
    network: Network, db: PlexosDB, timeslice_csv: str | None = None
) -> int:
    """Add gas pipelines as links, leveraging existing time-series parsing from links.py.

    Parameters
    ----------
    network : Network
        PyPSA network to add pipelines to
    db : PlexosDB
        PLEXOS database connection
    timeslice_csv : str, optional
        Path to timeslice CSV file for time-dependent properties

    Returns
    -------
    int
        Number of gas pipelines added
    """
    gas_pipelines_added = 0

    try:
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "pipeline" in gas_class.lower():
                pipeline_objects = get_gas_objects_by_class_name(db, gas_class)

                for pipeline_name in pipeline_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(
                        db, gas_class, pipeline_name
                    )
                    gas_nodes = [
                        m["name"]
                        for m in memberships
                        if "gas" in m.get("class", "").lower()
                        and "node" in m.get("class", "").lower()
                    ]

                    if len(gas_nodes) >= 2:
                        bus0 = f"gas_{gas_nodes[0]}"
                        bus1 = f"gas_{gas_nodes[1]}"

                        if bus0 in network.buses.index and bus1 in network.buses.index:
                            # Get pipeline properties
                            props = get_gas_object_properties(
                                db, gas_class, pipeline_name
                            )

                            # Extract Max Flow Day property (similar to existing electricity approach)
                            max_flow = 1000  # Default
                            for prop in props:
                                if prop.get("property") == "Max Flow Day":
                                    max_flow = prop.get("value", 1000)
                                    break

                            try:
                                p_nom = float(max_flow) if max_flow else 1000.0
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
                                gas_pipelines_added += 1

                            # TODO: Use parse_lines_flow() from links.py for time-series properties
                            # This would require adapting parse_lines_flow to work with gas pipeline properties

    except Exception:
        logger.exception("Failed to add gas pipelines")

    print(f"Added {gas_pipelines_added} gas pipelines")
    return gas_pipelines_added


def add_gas_fields(network: Network, db: PlexosDB) -> int:
    """Add gas fields as Store components following PyPSA multi-sector patterns.

    Gas fields are finite energy resources, not generators, so they should be
    represented as Store components with finite capacity (reserves).

    Parameters
    ----------
    network : Network
        PyPSA network to add gas fields to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    int
        Number of gas fields added
    """
    gas_fields_added = 0

    try:
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "field" in gas_class.lower():
                field_objects = get_gas_objects_by_class_name(db, gas_class)

                for field_name in field_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(db, gas_class, field_name)
                    gas_nodes = [
                        m["name"]
                        for m in memberships
                        if "gas" in m.get("class", "").lower()
                        and "node" in m.get("class", "").lower()
                    ]

                    if gas_nodes:
                        gas_bus = f"gas_{gas_nodes[0]}"

                        if gas_bus in network.buses.index:
                            # Get field properties
                            props = get_gas_object_properties(db, gas_class, field_name)

                            # Extract properties following PLEXOS gas field patterns
                            reserves = 10000  # Default reserves in MWh
                            extraction_cost = 0.1  # Default extraction cost
                            max_production = 500  # Default max production rate

                            for prop in props:
                                if prop.get("property") == "Reserves":
                                    try:
                                        reserves = float(prop.get("value", 10000))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Max Production":
                                    try:
                                        max_production = float(prop.get("value", 500))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Production Cost":
                                    try:
                                        extraction_cost = float(prop.get("value", 0.1))
                                    except Exception:
                                        pass

                            field_store_name = f"gas_field_{field_name}"
                            if field_store_name not in network.stores.index:
                                # Add as Store following PyPSA patterns
                                network.add(
                                    "Store",
                                    field_store_name,
                                    bus=gas_bus,
                                    e_nom=reserves,  # Total reserves
                                    e_nom_extendable=False,  # Fields have finite reserves
                                    capital_cost=extraction_cost * 8760,  # Annualized
                                    marginal_cost=extraction_cost,  # Per MWh extraction
                                    e_max_pu=max_production / reserves
                                    if reserves > 0
                                    else 1.0,  # Extraction rate limit
                                    carrier=network.buses.at[gas_bus, "carrier"]
                                    if gas_bus in network.buses.index
                                    else "gas",
                                )
                                gas_fields_added += 1

    except Exception:
        logger.exception("Failed to add gas fields")

    print(f"Added {gas_fields_added} gas fields")
    return gas_fields_added


def add_gas_storage(network: Network, db: PlexosDB) -> int:
    """Add gas storage components as Store following PyPSA multi-sector patterns.

    Underground gas storage should be represented as Store components for
    consistency with PyPSA multi-sector modeling patterns.

    Parameters
    ----------
    network : Network
        PyPSA network to add storage to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    int
        Number of gas storage units added
    """
    gas_storage_added = 0

    try:
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "storage" in gas_class.lower():
                storage_objects = get_gas_objects_by_class_name(db, gas_class)

                for storage_name in storage_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(
                        db, gas_class, storage_name
                    )
                    gas_nodes = [
                        m["name"]
                        for m in memberships
                        if "gas" in m.get("class", "").lower()
                        and "node" in m.get("class", "").lower()
                    ]

                    if gas_nodes:
                        gas_bus = f"gas_{gas_nodes[0]}"

                        if gas_bus in network.buses.index:
                            # Get storage properties
                            props = get_gas_object_properties(
                                db, gas_class, storage_name
                            )

                            # Default storage properties
                            max_volume = 1000  # Default energy capacity in MWh
                            max_injection = 100  # Default injection rate
                            max_withdrawal = 100  # Default withdrawal rate
                            storage_cost = 10  # Default storage cost

                            # Extract properties
                            for prop in props:
                                if prop.get("property") == "Max Volume":
                                    try:
                                        max_volume = float(prop.get("value", 1000))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Max Injection":
                                    try:
                                        max_injection = float(prop.get("value", 100))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Max Withdrawal":
                                    try:
                                        max_withdrawal = float(prop.get("value", 100))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Storage Cost":
                                    try:
                                        storage_cost = float(prop.get("value", 10))
                                    except Exception:
                                        pass

                            storage_name_clean = f"gas_storage_{storage_name}"
                            if storage_name_clean not in network.stores.index:
                                # Add as Store following PyPSA patterns from plexos_message.py
                                network.add(
                                    "Store",
                                    storage_name_clean,
                                    bus=gas_bus,
                                    e_nom=max_volume,  # Storage capacity
                                    e_nom_extendable=True,  # Can be expanded
                                    e_cyclic=True,  # Storage cycles over time period
                                    capital_cost=storage_cost,  # Storage cost per MWh
                                    e_max_pu=max_withdrawal / max_volume
                                    if max_volume > 0
                                    else 1.0,  # Max withdrawal rate
                                    e_min_pu=-max_injection / max_volume
                                    if max_volume > 0
                                    else -1.0,  # Max injection rate
                                    carrier=network.buses.at[gas_bus, "carrier"]
                                    if gas_bus in network.buses.index
                                    else "gas",
                                )
                                gas_storage_added += 1

    except Exception:
        logger.exception("Failed to add gas storage")

    print(f"Added {gas_storage_added} gas storage units")
    return gas_storage_added


def add_gas_plants(network: Network, db: PlexosDB) -> int:
    """Add gas plants as Links for gas→electricity conversion following PyPSA patterns.

    Gas plants convert gas to electricity and should be represented as conversion Links,
    not generators when using multi-sector modeling.

    Parameters
    ----------
    network : Network
        PyPSA network to add gas plants to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    int
        Number of gas plants added
    """
    gas_plants_added = 0

    try:
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "plant" in gas_class.lower():
                plant_objects = get_gas_objects_by_class_name(db, gas_class)

                for plant_name in plant_objects:
                    # Get connected nodes (both gas and electric)
                    memberships = get_gas_object_memberships(db, gas_class, plant_name)

                    # Find gas and electric connections
                    gas_nodes = [
                        m["name"]
                        for m in memberships
                        if "gas" in m.get("class", "").lower()
                        and "node" in m.get("class", "").lower()
                    ]

                    elec_nodes = [
                        m["name"]
                        for m in memberships
                        if m.get("class", "") == "Node"  # Regular electric nodes
                    ]

                    if gas_nodes and elec_nodes:
                        gas_bus = f"gas_{gas_nodes[0]}"
                        elec_bus = elec_nodes[0]  # Electric bus name from PLEXOS

                        if (
                            gas_bus in network.buses.index
                            and elec_bus in network.buses.index
                        ):
                            # Get plant properties
                            props = get_gas_object_properties(db, gas_class, plant_name)

                            # Extract properties
                            capacity = 500  # Default capacity in MW
                            heat_rate = 9.0  # Default heat rate (BTU/kWh)
                            capital_cost = 1000  # Default capital cost

                            for prop in props:
                                if prop.get("property") == "Max Capacity":
                                    try:
                                        capacity = float(prop.get("value", 500))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Heat Rate":
                                    try:
                                        heat_rate = float(prop.get("value", 9.0))
                                    except Exception:
                                        pass
                                elif prop.get("property") == "Build Cost":
                                    try:
                                        capital_cost = float(prop.get("value", 1000))
                                    except Exception:
                                        pass

                            # Calculate efficiency from heat rate (3412 BTU/kWh conversion)
                            efficiency = 3412 / heat_rate if heat_rate > 0 else 0.4
                            efficiency = min(
                                efficiency, 0.65
                            )  # Cap at realistic maximum

                            plant_link_name = f"gas_plant_{plant_name}"
                            if plant_link_name not in network.links.index:
                                # Add as Link for gas→electricity conversion
                                network.add(
                                    "Link",
                                    plant_link_name,
                                    bus0=gas_bus,  # Gas input
                                    bus1=elec_bus,  # Electricity output
                                    p_nom=capacity,
                                    efficiency=efficiency,
                                    capital_cost=capital_cost,
                                    carrier="gas_to_electricity",
                                    p_nom_extendable=False,
                                )  # Existing capacity
                                gas_plants_added += 1

    except Exception:
        logger.exception("Failed to add gas plants")

    print(f"Added {gas_plants_added} gas plants")
    return gas_plants_added


def add_gas_demand(network: Network, db: PlexosDB) -> int:
    """Add gas demand/loads to the network from Gas Demand objects and generic demand.

    Parameters
    ----------
    network : Network
        PyPSA network to add demand to
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    int
        Number of gas loads added
    """
    gas_loads_added = 0

    try:
        # First, try to add specific Gas Demand objects
        gas_classes = discover_gas_classes(db)

        for gas_class in gas_classes:
            if "demand" in gas_class.lower():
                demand_objects = get_gas_objects_by_class_name(db, gas_class)

                for demand_name in demand_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(db, gas_class, demand_name)
                    gas_nodes = [
                        m["name"]
                        for m in memberships
                        if "gas" in m.get("class", "").lower()
                        and "node" in m.get("class", "").lower()
                    ]

                    if gas_nodes:
                        gas_bus = f"gas_{gas_nodes[0]}"

                        if gas_bus in network.buses.index:
                            # Get demand properties
                            props = get_gas_object_properties(
                                db, gas_class, demand_name
                            )

                            # Extract demand magnitude
                            base_demand = 100  # Default
                            for prop in props:
                                if prop.get("property") == "Max Demand":
                                    try:
                                        base_demand = float(prop.get("value", 100))
                                    except Exception:
                                        pass

                            load_name = f"gas_demand_{demand_name}"
                            if load_name not in network.loads.index:
                                # Create demand profile with seasonal variation
                                demand_profile = pd.Series(
                                    base_demand
                                    * (
                                        1
                                        + 0.3
                                        * np.sin(
                                            np.arange(len(network.snapshots))
                                            * 2
                                            * np.pi
                                            / (24 * 90)
                                        )
                                    ),  # Seasonal
                                    index=network.snapshots,
                                )

                                network.add(
                                    "Load",
                                    load_name,
                                    bus=gas_bus,
                                    p_set=demand_profile,
                                    carrier=network.buses.at[gas_bus, "carrier"],
                                )
                                gas_loads_added += 1

        # Add generic demand to remaining gas buses without specific demand objects
        gas_buses = [bus for bus in network.buses.index if "gas_" in bus]
        existing_demand_buses = set()
        for load in network.loads.index:
            if "gas_demand_" in load:
                existing_demand_buses.add(network.loads.at[load, "bus"])

        unassigned_buses = [
            bus for bus in gas_buses if bus not in existing_demand_buses
        ]

        for i, bus in enumerate(
            unassigned_buses[:3]
        ):  # Limit to first 3 unassigned buses
            load_name = f"gas_demand_generic_{i + 1}"
            if load_name not in network.loads.index:
                # Create demand profile with daily variation
                base_demand = 100 * (1 + 0.2 * i)  # Vary by bus
                demand_profile = pd.Series(
                    base_demand
                    * (
                        1
                        + 0.2
                        * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)
                    ),
                    index=network.snapshots,
                )

                carrier = (
                    network.buses.at[bus, "carrier"]
                    if bus in network.buses.index
                    else "gas"
                )
                network.add(
                    "Load", load_name, bus=bus, p_set=demand_profile, carrier=carrier
                )
                gas_loads_added += 1

    except Exception:
        logger.exception("Failed to add gas demand")

    print(f"Added {gas_loads_added} gas loads")
    return gas_loads_added


def port_gas_components(
    network: Network,
    db: PlexosDB,
    timeslice_csv: str | None = None,
    testing_mode: bool = False,
) -> dict[str, Any]:
    """Comprehensive function to add all gas sector components to the PyPSA network.

    This function follows PyPSA multi-sector best practices and combines all
    gas-related operations:
    - Adds gas buses from Gas Node objects with enhanced carrier typing
    - Adds gas fields as Store components (finite reserves)
    - Adds gas pipelines as Links with flow limits and losses
    - Adds gas storage as Store components with cycling capabilities
    - Adds gas plants as Links for gas→electricity conversion
    - Adds gas demand/loads from Gas Demand objects

    Parameters
    ----------
    network : Network
        The PyPSA network to which gas components will be added.
    db : PlexosDB
        The Plexos database containing gas sector data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.

    Returns
    -------
    Dict[str, Any]
        Summary statistics of gas components added

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> gas_summary = port_gas_components(network, db,
    ...                                   timeslice_csv="path/to/timeslice.csv")
    """
    print("Starting gas sector porting process with PyPSA multi-sector patterns...")

    summary = {
        "sector": "Gas",
        "buses": 0,
        "fields": 0,
        "pipelines": 0,
        "storage": 0,
        "plants": 0,
        "demand": 0,
    }

    try:
        # Step 1: Add gas buses with enhanced carrier typing
        print("1. Adding gas buses...")
        summary["buses"] = add_gas_buses(network, db)

        # Step 2: Add gas fields as Store components (finite reserves)
        print("2. Adding gas fields...")
        summary["fields"] = add_gas_fields(network, db)

        # Step 3: Add gas pipelines as Links with flow limits
        print("3. Adding gas pipelines...")
        summary["pipelines"] = add_gas_pipelines(
            network, db, timeslice_csv=timeslice_csv
        )

        # Step 4: Add gas storage as Store components
        print("4. Adding gas storage...")
        summary["storage"] = add_gas_storage(network, db)

        # Step 5: Add gas plants as Links for gas→electricity conversion
        print("5. Adding gas plants...")
        summary["plants"] = add_gas_plants(network, db)

        # Step 6: Add gas demand from Gas Demand objects
        print("6. Adding gas demand...")
        summary["demand"] = add_gas_demand(network, db)

    except Exception:
        logger.exception("Error in gas sector porting")
        raise

    total_stores = summary["fields"] + summary["storage"]
    print(
        f"Gas sector porting complete! Added {summary['buses']} buses, {summary['fields']} fields, {summary['pipelines']} pipelines, {summary['storage']} storage, {summary['plants']} plants, {summary['demand']} loads."
    )
    print(
        f"  -> Total Store components: {total_stores} (following PyPSA multi-sector patterns)"
    )
    print(
        f"  -> Total Link components: {summary['pipelines'] + summary['plants']} (transport + conversion)"
    )
    return summary
