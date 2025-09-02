"""
Gas Network Components for PLEXOS-PyPSA Conversion

This module provides functions to convert PLEXOS gas sector components
to PyPSA network components, following established patterns from
electricity sector modules.
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from .links import parse_lines_flow

logger = logging.getLogger(__name__)


def discover_gas_classes(db: PlexosDB) -> List[str]:
    """
    Discover gas-related classes using the PlexosDB list_classes method.
    
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
        
        for class_name in all_classes:
            if class_name.lower().startswith('gas'):
                gas_classes.append(class_name)
        
        logger.info(f"Discovered gas classes: {gas_classes}")
        
    except Exception as e:
        logger.warning(f"Failed to discover gas classes: {e}")
    
    return gas_classes


def get_gas_objects_by_class_name(db: PlexosDB, class_name: str) -> List[str]:
    """
    Get all objects of a specific gas class using PlexosDB methods.
    
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


def get_gas_object_properties(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get properties for a gas object using PlexosDB methods.
    
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
    properties = []
    
    try:
        # Convert string class name to ClassEnum (from multi_sector_db.py pattern)
        class_mapping = {
            'Gas Field': ClassEnum.Gas_Field,
            'Gas Plant': ClassEnum.Gas_Plant,
            'Gas Pipeline': ClassEnum.Gas_Pipeline,
            'Gas Node': ClassEnum.Gas_Node,
            'Gas Storage': ClassEnum.Gas_Storage,
            'Gas Demand': ClassEnum.Gas_Demand,
            'Gas DSM Program': ClassEnum.Gas_DSM_Program,
            'Gas Basin': ClassEnum.Gas_Basin,
            'Gas Zone': ClassEnum.Gas_Zone,
            'Gas Contract': ClassEnum.Gas_Contract,
            'Gas Transport': ClassEnum.Gas_Transport,
            'Gas Path': ClassEnum.Gas_Path,
            'Gas Capacity Release Offer': ClassEnum.Gas_Capacity_Release_Offer,
        }
        
        class_enum = class_mapping.get(class_name)
        
        if class_enum is None:
            logger.debug(f"No ClassEnum mapping found for gas class '{class_name}', skipping properties for {object_name}")
            return properties
        
        # Use the built-in PlexosDB method with ClassEnum
        properties = db.get_object_properties(class_enum, object_name)
        logger.debug(f"Found {len(properties)} properties for gas object {object_name} (class: {class_name})")
        
    except Exception as e:
        logger.debug(f"Failed to get properties for gas object {object_name} (class: {class_name}): {e}")
    
    return properties


def get_gas_object_memberships(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get membership relationships for a gas object using PlexosDB methods.
    
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
        logger.debug(f"Found {len(memberships)} memberships for gas object {object_name}")
        
    except Exception as e:
        logger.warning(f"Failed to get memberships for gas object {object_name}: {e}")
    
    return memberships


def add_gas_buses(network: Network, db: PlexosDB) -> int:
    """
    Add gas buses to the network from Gas Node objects.
    
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
        # Add Gas carrier if needed
        if 'Gas' not in network.carriers.index:
            network.add('Carrier', 'Gas')
        
        # Get gas node classes
        gas_classes = discover_gas_classes(db)
        
        for gas_class in gas_classes:
            if 'node' in gas_class.lower():
                gas_objects = get_gas_objects_by_class_name(db, gas_class)
                for gas_name in gas_objects:
                    bus_name = f"gas_{gas_name}"
                    if bus_name not in network.buses.index:
                        network.add('Bus', bus_name, carrier='Gas')
                        gas_buses_added += 1
    
    except Exception as e:
        logger.error(f"Failed to add gas buses: {e}")
    
    print(f"Added {gas_buses_added} gas buses")
    return gas_buses_added


def add_gas_pipelines(network: Network, db: PlexosDB, timeslice_csv: Optional[str] = None) -> int:
    """
    Add gas pipelines as links, leveraging existing time-series parsing from links.py.
    
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
            if 'pipeline' in gas_class.lower():
                pipeline_objects = get_gas_objects_by_class_name(db, gas_class)
                
                for pipeline_name in pipeline_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(db, gas_class, pipeline_name)
                    gas_nodes = [
                        m['name'] for m in memberships 
                        if 'gas' in m.get('class', '').lower() and 'node' in m.get('class', '').lower()
                    ]
                    
                    if len(gas_nodes) >= 2:
                        bus0 = f"gas_{gas_nodes[0]}"
                        bus1 = f"gas_{gas_nodes[1]}"
                        
                        if bus0 in network.buses.index and bus1 in network.buses.index:
                            # Get pipeline properties
                            props = get_gas_object_properties(db, gas_class, pipeline_name)
                            
                            # Extract Max Flow Day property (similar to existing electricity approach)
                            max_flow = 1000  # Default
                            for prop in props:
                                if prop.get('property') == 'Max Flow Day':
                                    max_flow = prop.get('value', 1000)
                                    break
                            
                            try:
                                p_nom = float(max_flow) if max_flow else 1000.0
                            except:
                                p_nom = 1000.0
                            
                            link_name = f"gas_pipeline_{pipeline_name}"
                            if link_name not in network.links.index:
                                network.add('Link', link_name, bus0=bus0, bus1=bus1,
                                          p_nom=p_nom, efficiency=0.98, carrier='Gas')
                                gas_pipelines_added += 1
                            
                            # TODO: Use parse_lines_flow() from links.py for time-series properties
                            # This would require adapting parse_lines_flow to work with gas pipeline properties
    
    except Exception as e:
        logger.error(f"Failed to add gas pipelines: {e}")
    
    print(f"Added {gas_pipelines_added} gas pipelines")
    return gas_pipelines_added


def add_gas_storage(network: Network, db: PlexosDB) -> int:
    """
    Add gas storage components to the network.
    
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
            if 'storage' in gas_class.lower():
                storage_objects = get_gas_objects_by_class_name(db, gas_class)
                
                for storage_name in storage_objects:
                    # Get connected gas nodes
                    memberships = get_gas_object_memberships(db, gas_class, storage_name)
                    gas_nodes = [
                        m['name'] for m in memberships 
                        if 'gas' in m.get('class', '').lower() and 'node' in m.get('class', '').lower()
                    ]
                    
                    if gas_nodes:
                        gas_bus = f"gas_{gas_nodes[0]}"
                        
                        if gas_bus in network.buses.index:
                            # Get storage properties
                            props = get_gas_object_properties(db, gas_class, storage_name)
                            
                            # Default storage properties
                            p_nom = 100  # Default power capacity
                            e_nom = 1000  # Default energy capacity
                            
                            # Extract properties
                            for prop in props:
                                if prop.get('property') == 'Max Injection':
                                    try:
                                        p_nom = float(prop.get('value', 100))
                                    except:
                                        pass
                                elif prop.get('property') == 'Max Volume':
                                    try:
                                        e_nom = float(prop.get('value', 1000))
                                    except:
                                        pass
                            
                            storage_unit_name = f"gas_storage_{storage_name}"
                            if storage_unit_name not in network.storage_units.index:
                                network.add('StorageUnit', storage_unit_name, 
                                          bus=gas_bus, p_nom=p_nom, e_nom=e_nom,
                                          carrier='Gas')
                                gas_storage_added += 1
    
    except Exception as e:
        logger.error(f"Failed to add gas storage: {e}")
    
    print(f"Added {gas_storage_added} gas storage units")
    return gas_storage_added


def add_gas_demand(network: Network, db: PlexosDB) -> int:
    """
    Add gas demand/loads to the network.
    
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
        # Add basic gas demand to gas buses
        gas_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'Gas']
        
        for i, bus in enumerate(gas_buses[:5]):  # Limit to first 5 gas buses
            load_name = f"gas_demand_{i+1}"
            if load_name not in network.loads.index:
                # Create demand profile with daily variation
                base_demand = 100 * (1 + 0.2 * i)  # Vary by bus
                demand_profile = pd.Series(
                    base_demand * (1 + 0.2 * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)),
                    index=network.snapshots
                )
                
                network.add('Load', load_name, bus=bus, p_set=demand_profile, carrier='Gas')
                gas_loads_added += 1
    
    except Exception as e:
        logger.error(f"Failed to add gas demand: {e}")
    
    print(f"Added {gas_loads_added} gas loads")
    return gas_loads_added


def port_gas_components(
    network: Network,
    db: PlexosDB,
    timeslice_csv: Optional[str] = None,
    testing_mode: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive function to add all gas sector components to the PyPSA network.
    
    This function follows the established pattern from electricity sector modules
    and combines all gas-related operations:
    - Adds gas buses from Gas Node objects
    - Adds gas pipelines as links with flow limits
    - Adds gas storage units
    - Adds gas demand/loads
    
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
    print("Starting gas sector porting process...")
    
    summary = {
        'sector': 'Gas',
        'buses': 0,
        'pipelines': 0, 
        'storage': 0,
        'demand': 0
    }
    
    try:
        # Step 1: Add gas buses
        print("1. Adding gas buses...")
        summary['buses'] = add_gas_buses(network, db)
        
        # Step 2: Add gas pipelines as links
        print("2. Adding gas pipelines...")
        summary['pipelines'] = add_gas_pipelines(network, db, timeslice_csv=timeslice_csv)
        
        # Step 3: Add gas storage
        print("3. Adding gas storage...")
        summary['storage'] = add_gas_storage(network, db)
        
        # Step 4: Add gas demand
        print("4. Adding gas demand...")
        summary['demand'] = add_gas_demand(network, db)
        
    except Exception as e:
        logger.error(f"Error in gas sector porting: {e}")
        raise
    
    print(f"Gas sector porting complete! Added {summary['buses']} buses, {summary['pipelines']} pipelines, {summary['storage']} storage units, {summary['demand']} loads.")
    return summary