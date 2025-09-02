"""
Multi-Sector Network Setup Functions (Database-based)

This module provides functions to set up PyPSA networks for multi-sector energy models
using direct database queries to discover Gas and Flow network classes dynamically.
This approach is more robust than relying on ClassEnum for multi-sector classes.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from .progress import BatchProgressTracker, create_progress_tracker
from .generators import port_generators
from .links import port_links
from .storage import add_storage
from .gas import port_gas_components
from .flows import port_flow_network
from .core import setup_network

logger = logging.getLogger(__name__)


def get_class_enum_from_string(class_name: str) -> Optional[ClassEnum]:
    """
    Convert a string class name to the corresponding ClassEnum value.
    
    Parameters
    ----------
    class_name : str
        Name of the PLEXOS class as a string
        
    Returns
    -------
    Optional[ClassEnum]
        Corresponding ClassEnum value, or None if not found
    """
    # Create mapping of string names to ClassEnum values
    class_mapping = {
        # Standard classes
        'Generator': ClassEnum.Generator,
        'Node': ClassEnum.Node,
        'Line': ClassEnum.Line,
        'Storage': ClassEnum.Storage,
        'Battery': ClassEnum.Battery,
        'Constraint': ClassEnum.Constraint,
        'Fuel': ClassEnum.Fuel,
        'Emission': ClassEnum.Emission,
        'DataFile': ClassEnum.DataFile,
        
        # Gas classes
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
        
        # Flow classes (if they exist in ClassEnum)
        'Flow Control': getattr(ClassEnum, 'Flow_Control', None),
        'Flow Network': getattr(ClassEnum, 'Flow_Network', None),
        'Flow Node': getattr(ClassEnum, 'Flow_Node', None),
        'Flow Path': getattr(ClassEnum, 'Flow_Path', None),
        'Flow Storage': getattr(ClassEnum, 'Flow_Storage', None),
        
        # Process classes (if they exist)
        'Process': getattr(ClassEnum, 'Process', None),
        
        # Facility classes (if they exist)
        'Facility': getattr(ClassEnum, 'Facility', None),
    }
    
    # Remove None values (classes that don't exist in ClassEnum)
    class_mapping = {k: v for k, v in class_mapping.items() if v is not None}
    
    return class_mapping.get(class_name)


def discover_multi_sector_classes(db: PlexosDB) -> Dict[str, List[str]]:
    """
    Discover multi-sector classes using the PlexosDB list_classes method.
    
    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping class categories to class names
    """
    multi_sector_classes = {
        'gas': [],
        'flow': [],
        'process': [],
        'facility': []
    }
    
    try:
        # Get all available classes using PlexosDB method
        all_classes = db.list_classes()
        
        for class_name in all_classes:
            class_name_lower = class_name.lower()
            if class_name_lower.startswith('gas'):
                multi_sector_classes['gas'].append(class_name)
            elif class_name_lower.startswith('flow'):
                multi_sector_classes['flow'].append(class_name)
            elif class_name_lower.startswith('process'):
                multi_sector_classes['process'].append(class_name)
            elif class_name_lower.startswith('facility'):
                multi_sector_classes['facility'].append(class_name)
        
        logger.info(f"Discovered multi-sector classes: {multi_sector_classes}")
        
    except Exception as e:
        logger.warning(f"Failed to discover multi-sector classes: {e}")
    
    return multi_sector_classes


def get_objects_by_class_name(db: PlexosDB, class_name: str) -> List[str]:
    """
    Get all objects of a specific class using PlexosDB methods.
    
    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Name of the class to query
        
    Returns
    -------
    List[str]
        List of object names
    """
    objects = []
    
    try:
        # Use the built-in PlexosDB method
        objects = db.list_objects_by_class(class_name)
        logger.debug(f"Found {len(objects)} objects for class {class_name}")
        
    except Exception as e:
        logger.warning(f"Failed to get objects for class {class_name}: {e}")
    
    return objects


def get_object_properties_by_name(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get properties for an object using PlexosDB methods.
    
    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the object
    object_name : str
        Name of the object to query properties for
        
    Returns
    -------
    List[Dict[str, Any]]
        List of property dictionaries
    """
    properties = []
    
    try:
        # Convert string class name to ClassEnum
        class_enum = get_class_enum_from_string(class_name)
        
        if class_enum is None:
            logger.debug(f"No ClassEnum mapping found for class '{class_name}', skipping properties for {object_name}")
            return properties
        
        # Use the built-in PlexosDB method with ClassEnum
        properties = db.get_object_properties(class_enum, object_name)
        logger.debug(f"Found {len(properties)} properties for object {object_name} (class: {class_name})")
        
    except Exception as e:
        logger.debug(f"Failed to get properties for object {object_name} (class: {class_name}): {e}")
    
    return properties


def get_object_memberships(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get membership relationships for an object using PlexosDB methods.
    
    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
    class_name : str
        Class name of the object
    object_name : str
        Name of the object to query memberships for
        
    Returns
    -------
    List[Dict[str, Any]]
        List of membership relationships
    """
    memberships = []
    
    try:
        # Use the built-in PlexosDB method
        memberships_system = db.get_memberships_system(object_name, object_class=class_name)
        
        # Return the memberships as they are - they already have the right format
        memberships = memberships_system
        
        logger.debug(f"Found {len(memberships)} memberships for object {object_name}")
        
    except Exception as e:
        logger.warning(f"Failed to get memberships for object {object_name}: {e}")
    
    return memberships


def setup_gas_electric_network_db(network: Network, db: PlexosDB, generators_as_links: bool = False, testing_mode: bool = False, 
                                  timeslice_csv: Optional[str] = None, vre_profiles_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Set up a multi-sector PyPSA network for gas and electricity using existing modules.
    
    This function now orchestrates existing electricity and gas sector modules
    to eliminate code redundancy while maintaining the same functionality.
    
    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data
    generators_as_links : bool, optional
        If True, represent conventional generators (coal, gas, nuclear, etc.) as Links
        connecting fuel buses to electric buses. If False, use standard Generators.
        Default False.
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.
        Default False creates complete model.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    vre_profiles_path : str, optional
        Path to the folder containing VRE generation profile files.
        
    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print("Setting up gas-electric network using existing modules...")
    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")
    if generators_as_links:
        print("🔗 GENERATORS-AS-LINKS: Converting conventional generators to fuel→electric Links")
    
    # Initialize tracking
    setup_summary = {
        'network_type': 'gas_electric_db',
        'sectors': ['Electricity', 'Gas'],
        'electricity': {'buses': 0, 'generators': 0, 'loads': 0, 'lines': 0, 'storage': 0},
        'gas': {'buses': 0, 'pipelines': 0, 'storage': 0, 'demand': 0},
        'sector_coupling': {
            'generators_as_links': generators_as_links,
            'fuel_types': []
        }
    }
    
    try:
        # Step 1: Set up basic network structure using core module
        print("\n1. Setting up basic network structure...")
        
        # Add carriers automatically discovered from database
        print("   Discovering carriers from PLEXOS database...")
        from .core import discover_carriers_from_db
        carriers_to_add = discover_carriers_from_db(db)
        
        added_carriers = []
        for carrier in carriers_to_add:
            if carrier not in network.carriers.index:
                network.add('Carrier', carrier)
                added_carriers.append(carrier)
        print(f"   Added {len(added_carriers)} carriers automatically")
        
        # Add electricity buses from Node class
        node_objects = get_objects_by_class_name(db, 'Node')
        for node_name in node_objects:
            if node_name not in network.buses.index:
                network.add('Bus', node_name, carrier='AC', v_nom=110)
        
        setup_summary['electricity']['buses'] = len(node_objects)
        
        # Set basic snapshots
        snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
        network.set_snapshots(snapshots)
        print(f"   Added {len(node_objects)} electricity buses and {len(snapshots)} snapshots")
        
        # Step 2: Port electricity sector components using existing modules
        print("\n2. Setting up electricity sector using port_generators()...")
        
        # Use the refactored generators module with generators-as-links support
        gen_summary = port_generators(
            network=network,
            db=db,
            timeslice_csv=timeslice_csv,
            vre_profiles_path=vre_profiles_path,
            generators_as_links=generators_as_links,
            fuel_bus_prefix="fuel_"
        )
        
        setup_summary['electricity']['generators'] = len(network.generators)
        if generators_as_links:
            # Count generator-links (fuel→electric conversion links)
            fuel_buses = [bus for bus in network.buses.index if 'fuel_' in bus]
            elec_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'AC']
            generator_links = [link for link in network.links.index 
                             if network.links.at[link, 'bus0'] in fuel_buses and 
                                network.links.at[link, 'bus1'] in elec_buses]
            setup_summary['sector_coupling']['generator_links'] = len(generator_links)
            print(f"   Created {len(generator_links)} generator-links and {len(network.generators)} generators")
        else:
            print(f"   Added {len(network.generators)} electricity generators")
        
        # Step 3: Port transmission links using existing modules
        print("\n3. Setting up transmission links using port_links()...")
        
        port_links(network=network, db=db, timeslice_csv=timeslice_csv)
        setup_summary['electricity']['lines'] = len(network.links)
        print(f"   Added {len(network.links)} transmission links")
        
        # Step 4: Port storage using existing modules
        print("\n4. Setting up storage using port_storage()...")
        
        try:
            add_storage(network=network, db=db, timeslice_csv=timeslice_csv)
            setup_summary['electricity']['storage'] = len(network.storage_units)
            print(f"   Added {len(network.storage_units)} storage units")
        except Exception as e:
            logger.warning(f"Storage porting failed: {e}")
            setup_summary['electricity']['storage'] = 0
        
        # Step 5: Set up gas sector using new gas module
        print("\n5. Setting up gas sector using port_gas_components()...")
        
        gas_summary = port_gas_components(network=network, db=db, timeslice_csv=timeslice_csv, testing_mode=testing_mode)
        setup_summary['gas'].update(gas_summary)
        
        # Step 6: Add sector coupling (gas-electric coupling)
        print("\n6. Setting up gas-electric sector coupling...")
        
        coupling_stats = add_gas_electric_coupling_db(network, db)
        setup_summary['sector_coupling'].update(coupling_stats)
        if coupling_stats.get('gas_generators', 0) > 0:
            print(f"   Added {coupling_stats['gas_generators']} gas-to-electric conversion links")
            print(f"   Efficiency range: {coupling_stats['efficiency_range']}")
        
        # Step 7: Add basic loads for load balancing
        print("\n7. Adding load balancing...")
        
        # Add loads to electricity buses
        elec_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'AC']
        total_gen_capacity = sum(network.generators.p_nom) if len(network.generators) > 0 else len(elec_buses) * 1000
        
        if generators_as_links:
            # Include generator-link capacity in total calculation
            fuel_buses = [bus for bus in network.buses.index if 'fuel_' in bus]
            gen_links = network.links[(network.links.bus0.isin(fuel_buses)) & (network.links.bus1.isin(elec_buses))]
            gen_link_capacity = sum(gen_links.p_nom * gen_links.efficiency) if len(gen_links) > 0 else 0
            total_gen_capacity += gen_link_capacity
        
        load_per_bus = total_gen_capacity / len(elec_buses) * 0.3 if len(elec_buses) > 0 else 100
        
        for i, bus in enumerate(elec_buses):
            load_name = f"elec_load_{bus}"
            if load_name not in network.loads.index:
                base_load = load_per_bus * (0.8 + 0.4 * np.random.random())
                load_profile = pd.Series(
                    base_load * (1 + 0.3 * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)),
                    index=network.snapshots
                )
                network.add('Load', load_name, bus=bus, p_set=load_profile)
        
        setup_summary['electricity']['loads'] = len([l for l in network.loads.index if 'elec_load' in l])
        print(f"   Added {setup_summary['electricity']['loads']} electricity loads")
        
    except Exception as e:
        logger.error(f"Error setting up gas-electric network: {e}")
        raise
    
    print("\nGas-electric multi-sector network setup complete!")
    print(f"  Electricity: {setup_summary['electricity']}")
    print(f"  Gas: {setup_summary['gas']}")
    return setup_summary


def add_gas_electric_coupling_db(network: Network, db: PlexosDB) -> Dict[str, Any]:
    """Add gas-to-electric conversion from generators with gas connections."""
    coupling_stats = {'gas_generators': 0, 'efficiency_range': 'N/A'}
    
    try:
        generator_objects = get_objects_by_class_name(db, 'Generator')
        efficiency_values = []
        
        for gen_name in generator_objects:
            # Get generator properties and memberships
            props = get_object_properties_by_name(db, 'Generator', gen_name)
            memberships = get_object_memberships(db, 'Generator', gen_name)
            
            # Check if generator has gas connection
            gas_node = None
            elec_node = None
            
            for membership in memberships:
                if membership.get('class') == 'Node':
                    elec_node = membership.get('name')
                elif 'gas' in membership.get('class', '').lower() and 'node' in membership.get('class', '').lower():
                    gas_node = membership.get('name')
            
            # Also check properties for Gas Node references
            if not gas_node:
                for prop in props:
                    if prop.get('property') == 'Gas Node':
                        gas_node = prop.get('value')
                        break
            
            if elec_node and gas_node:
                elec_bus = elec_node
                gas_bus = f"gas_{gas_node}"
                
                # Check if both buses exist
                if elec_bus in network.buses.index and gas_bus in network.buses.index:
                    # Get generator properties
                    max_capacity = 100  # Default
                    heat_rate = 9.0  # Default
                    
                    for prop in props:
                        if prop.get('property') == 'Max Capacity':
                            max_capacity = prop.get('value', 100)
                        elif prop.get('property') == 'Heat Rate':
                            heat_rate = prop.get('value', 9.0)
                    
                    try:
                        p_nom = float(max_capacity) if max_capacity else 100.0
                        hr = float(heat_rate) if heat_rate else 9.0
                        
                        # Calculate efficiency (3412 BTU/kWh conversion factor)
                        efficiency = 3412 / hr if hr > 0 else 0.4
                        efficiency = min(efficiency, 0.65)  # Cap at 65%
                        
                        efficiency_values.append(efficiency)
                        
                        # Create gas-to-electric conversion link
                        link_name = f"gas_to_elec_{gen_name}"
                        if link_name not in network.links.index:
                            network.add('Link', link_name,
                                      bus0=gas_bus, bus1=elec_bus, p_nom=p_nom,
                                      efficiency=efficiency, carrier='Gas2Electric')
                            coupling_stats['gas_generators'] += 1
                    except:
                        pass
        
        # Calculate efficiency range
        if efficiency_values:
            min_eff = min(efficiency_values)
            max_eff = max(efficiency_values)
            coupling_stats['efficiency_range'] = f"{min_eff:.1%} - {max_eff:.1%}"
            
    except Exception as e:
        logger.warning(f"Failed to add gas-electric coupling: {e}")
    
    return coupling_stats


def setup_flow_network_db(network: Network, db: PlexosDB, testing_mode: bool = False, timeslice_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Set up a multi-sector PyPSA network using PLEXOS Flow Network components with the flows.py module.
    
    This function now uses the dedicated flows.py module to eliminate code redundancy
    while maintaining the same functionality.
    
    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties
        
    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print("Setting up flow network using flows.py module...")
    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")
    
    try:
        # Use the dedicated flows.py module for complete flow network setup
        setup_summary = port_flow_network(
            network=network,
            db=db,
            timeslice_csv=timeslice_csv,
            testing_mode=testing_mode
        )
        
        # Update summary format to match expected interface
        setup_summary['network_type'] = 'flow_network_db'
        
    except Exception as e:
        logger.error(f"Error setting up flow network: {e}")
        raise
    
    print("Flow network multi-sector setup complete using flows.py module!")
    return setup_summary


def add_processes_db(network: Network, db: PlexosDB, multi_sector_classes: Dict[str, List[str]]) -> Dict[str, int]:
    """Add processes as sector coupling links using database queries."""
    process_stats = {}
    
    try:
        # Process discovered process classes
        for process_class in multi_sector_classes['process']:
            process_objects = get_objects_by_class_name(db, process_class)
            
            for process_name in process_objects:
                # Get process properties
                props = get_object_properties_by_name(db, process_class, process_name)
                efficiency = 70.0  # Default
                for prop in props:
                    if prop.get('property') == 'Efficiency':
                        efficiency = prop.get('value', 70.0)
                        break
                
                try:
                    eff = float(efficiency) / 100.0 if efficiency else 0.7
                except:
                    eff = 0.7
                
                # Determine process type and create links
                if 'electrolysis' in process_name.lower():
                    process_type = 'Electrolysis'
                    # Create electricity -> hydrogen links
                    elec_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Electricity']
                    h2_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Hydrogen']
                    
                    for i in range(min(3, len(elec_buses), len(h2_buses))):
                        link_name = f"electrolysis_{i+1}"
                        if link_name not in network.links.index:
                            network.add('Link', link_name,
                                      bus0=elec_buses[i], bus1=h2_buses[i],
                                      p_nom=100, efficiency=eff)
                            
                elif 'h2power' in process_name.lower() or 'fuel_cell' in process_name.lower():
                    process_type = 'H2_Power'
                    # Create hydrogen -> electricity links
                    h2_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Hydrogen']
                    elec_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Electricity']
                    
                    for i in range(min(2, len(h2_buses), len(elec_buses))):
                        link_name = f"fuel_cell_{i+1}"
                        if link_name not in network.links.index:
                            network.add('Link', link_name,
                                      bus0=h2_buses[i], bus1=elec_buses[i],
                                      p_nom=50, efficiency=eff)
                            
                elif 'ammonia' in process_name.lower():
                    process_type = 'Ammonia_Synthesis'
                    # Create hydrogen -> ammonia links
                    h2_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Hydrogen']
                    nh3_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == 'Ammonia']
                    
                    for i in range(min(2, len(h2_buses), len(nh3_buses))):
                        link_name = f"ammonia_synthesis_{i+1}"
                        if link_name not in network.links.index:
                            network.add('Link', link_name,
                                      bus0=h2_buses[i], bus1=nh3_buses[i],
                                      p_nom=30, efficiency=eff)
                else:
                    process_type = 'Other'
                
                if process_type not in process_stats:
                    process_stats[process_type] = 0
                process_stats[process_type] += 1
                
    except Exception as e:
        logger.warning(f"Failed to add processes: {e}")
    
    return process_stats


def add_facilities_db(network: Network, db: PlexosDB, multi_sector_classes: Dict[str, List[str]], testing_mode: bool = False) -> Dict[str, int]:
    """Add facilities as generators or loads using database queries."""
    facility_stats = {'generators': 0, 'loads': 0, 'skipped': 0}
    
    try:
        # Count total facilities first
        total_facilities = 0
        for facility_class in multi_sector_classes.get('facility', []):
            facility_objects = get_objects_by_class_name(db, facility_class)
            total_facilities += len(facility_objects)
        
        print(f"  Found {total_facilities} facilities to process")
        
        # Initialize progress tracker for facilities
        if total_facilities > 0:
            facility_progress = create_progress_tracker(total_facilities, "Processing Facilities", 
                                                      testing_mode=testing_mode, test_limit=1000)
            facility_index = 0
            
            # Process facility classes
            for facility_class in multi_sector_classes.get('facility', []):
                facility_objects = get_objects_by_class_name(db, facility_class)
                
                for facility_name in facility_objects:
                    # Check if we should process this facility (testing mode)
                    if not facility_progress.should_process(facility_index):
                        break
                    
                    # Get facility memberships to find connected flow nodes and processes
                    memberships = get_object_memberships(db, facility_class, facility_name)
                
                    # Extract connected flow nodes using specific collection names (like Flow Paths)
                    flow_nodes = []
                    for m in memberships:
                        # Look for specific collection names that contain flow nodes
                        if 'flow' in m.get('class', '').lower() and 'node' in m.get('class', '').lower():
                            flow_nodes.append(m["name"])
                    
                    # Extract process type
                    processes = [m["name"] for m in memberships if m.get('class', '').lower() == 'process']
                
                    # Debug logging - log more variety
                    if facility_index < 5 or facility_index % 200 == 0:  # Log first 5 + every 200th
                        print(f"    Debug - Facility {facility_index}: {facility_name}")
                        print(f"    Flow nodes found: {flow_nodes}")
                        print(f"    Processes found: {processes}")
                        if facility_index < 5:
                            print(f"    Available buses: {list(network.buses.index)[:5]}...")  # First 5 buses
                
                    if len(flow_nodes) >= 1:
                        primary_bus = flow_nodes[0]  # Use first flow node as primary bus
                        process_name = processes[0] if processes else ''   # Use first process if available
                        
                        # Broaden generator classification logic
                        generator_keywords = ['solar', 'wind', 'hydro', 'nuclear', 'gas', 'biomass', 'geothermal', 
                                            'coal', 'oil', 'pv', 'onshore', 'offshore', 'thermal', 'power', 'gen']
                        is_generator = any(gen_type in facility_name.lower() or gen_type in process_name.lower() 
                                         for gen_type in generator_keywords)
                        
                        # Debug logging
                        if facility_index < 5 or facility_index % 200 == 0:
                            print(f"    Primary bus: {primary_bus}, Bus exists: {primary_bus in network.buses.index}")
                            print(f"    Is generator: {is_generator}")
                            if is_generator:
                                print(f"    ⭐ GENERATOR FOUND at index {facility_index}!")
                        
                        if primary_bus in network.buses.index:
                            if is_generator:
                                # Add as generator
                                gen_name = f"gen_{facility_name}"
                                if gen_name not in network.generators.index:
                                    # Determine capacity based on facility type
                                    if 'solar' in facility_name.lower():
                                        p_nom = 100
                                    elif 'wind' in facility_name.lower():
                                        p_nom = 150
                                    elif 'nuclear' in facility_name.lower():
                                        p_nom = 1000
                                    elif 'hydro' in facility_name.lower():
                                        p_nom = 500
                                    else:
                                        p_nom = 200  # Default
                                    
                                    # Create generation profile
                                    if 'solar' in facility_name.lower():
                                        # Solar profile with daily cycle
                                        profile = pd.Series([
                                            max(0, 0.8 * np.sin((h % 24) * np.pi / 12 - np.pi/2)) 
                                            for h in range(len(network.snapshots))
                                        ], index=network.snapshots)
                                    elif 'wind' in facility_name.lower():
                                        # Variable wind profile
                                        profile = pd.Series([
                                            0.3 + 0.4 * np.random.random() 
                                            for _ in range(len(network.snapshots))
                                        ], index=network.snapshots)
                                    else:
                                        # Baseload profile
                                        profile = pd.Series([0.8] * len(network.snapshots), index=network.snapshots)
                                    
                                    network.add('Generator', gen_name, bus=primary_bus, 
                                              p_nom=p_nom, p_max_pu=profile, marginal_cost=50)
                                    facility_stats['generators'] += 1
                                    
                                    # Debug logging for first few generators
                                    if facility_stats['generators'] <= 5:
                                        print(f"    ✓ Added generator: {gen_name} to bus {primary_bus} (p_nom={p_nom})")
                                        print(f"    Network now has {len(network.generators)} generators total")
                            
                            elif len(flow_nodes) >= 2:
                                # Multi-node facility - create as conversion link between flow nodes
                                link_name = f"facility_{facility_name}"
                                if link_name not in network.links.index:
                                    bus0, bus1 = flow_nodes[0], flow_nodes[1]
                                    if bus0 in network.buses.index and bus1 in network.buses.index:
                                        # Conversion efficiency based on process type
                                        if 'electrolysis' in process_name.lower():
                                            efficiency = 0.7
                                        elif 'ammonia' in process_name.lower():
                                            efficiency = 0.6
                                        else:
                                            efficiency = 0.75
                                        
                                        network.add('Link', link_name, bus0=bus0, bus1=bus1, 
                                                  p_nom=100, efficiency=efficiency)
                                        facility_stats['generators'] += 1  # Count as generator equivalent
                            
                            else:
                                # Single node facility that's not a generator - add as load
                                load_name = f"load_{facility_name}"
                                if load_name not in network.loads.index:
                                    base_demand = 50
                                    demand_profile = pd.Series([base_demand] * len(network.snapshots), index=network.snapshots)
                                    network.add('Load', load_name, bus=primary_bus, p_set=demand_profile)
                                    facility_stats['loads'] += 1
                        else:
                            facility_stats['skipped'] += 1
                            if facility_index < 5:  # Debug first few failures
                                print(f"    ❌ Bus not found: {primary_bus}")
                    else:
                        facility_stats['skipped'] += 1
                        if facility_index < 5:  # Debug first few failures
                            print(f"    ❌ No flow nodes found for facility: {facility_name}")
                    
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
                total_added = facility_stats['generators'] + facility_stats['loads']
                facility_progress.finish(total_added)
                        
    except Exception as e:
        logger.warning(f"Failed to add facilities: {e}")
    
    return facility_stats


def add_flow_demand_db(network: Network, sector: str) -> None:
    """Add basic demand profiles to flow network buses."""
    try:
        sector_buses = [b for b in network.buses.index if network.buses.at[b, 'carrier'] == sector]
        
        for i, bus in enumerate(sector_buses[:3]):  # First 3 buses per sector
            load_name = f"{sector.lower()}_demand_{i+1}"
            if load_name not in network.loads.index:
                # Different base demands by sector
                if sector == 'Electricity':
                    base_demand = 1500 * (1 + 0.1 * i)
                elif sector == 'Hydrogen':
                    base_demand = 150 * (1 + 0.1 * i)
                elif sector == 'Ammonia':
                    base_demand = 75 * (1 + 0.1 * i)
                else:
                    base_demand = 100 * (1 + 0.1 * i)
                
                # Create demand profile with daily variation
                demand_profile = pd.Series(
                    base_demand * (1 + 0.3 * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)),
                    index=network.snapshots
                )
                
                network.add('Load', load_name, bus=bus, p_set=demand_profile, carrier=sector)
                
    except Exception as e:
        logger.warning(f"Failed to add {sector} demand: {e}")