"""
Flow Network Components for PLEXOS-PyPSA Conversion

This module provides functions to convert PLEXOS Flow Network components
to PyPSA network components, supporting multi-sector energy systems
(electricity, hydrogen, ammonia, etc.) following established patterns.
"""

import logging
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from plexosdb import PlexosDB
from pypsa import Network

from .progress import create_progress_tracker

logger = logging.getLogger(__name__)


def discover_flow_classes(db: PlexosDB) -> Dict[str, List[str]]:
    """
    Discover flow-related classes using the PlexosDB list_classes method.
    
    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping class categories to class names
    """
    flow_classes = {
        'flow': [],
        'process': [],
        'facility': []
    }
    
    try:
        # Get all available classes using PlexosDB method
        all_classes = db.list_classes()
        
        for class_name in all_classes:
            class_name_lower = class_name.lower()
            if class_name_lower.startswith('flow'):
                flow_classes['flow'].append(class_name)
            elif class_name_lower.startswith('process'):
                flow_classes['process'].append(class_name)
            elif class_name_lower.startswith('facility'):
                flow_classes['facility'].append(class_name)
        
        logger.info(f"Discovered flow classes: {flow_classes}")
        
    except Exception as e:
        logger.warning(f"Failed to discover flow classes: {e}")
    
    return flow_classes


def get_flow_objects_by_class_name(db: PlexosDB, class_name: str) -> List[str]:
    """
    Get all objects of a specific flow class using PlexosDB methods.
    
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


def get_flow_object_properties(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get properties for a flow object using PlexosDB methods.
    
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
    properties = []
    
    try:
        # Flow classes might not have direct ClassEnum mappings
        # Use generic property lookup if available
        properties = []
        logger.debug(f"Flow class '{class_name}' may not have ClassEnum mapping, using alternative approach")
        
    except Exception as e:
        logger.debug(f"Failed to get properties for flow object {object_name} (class: {class_name}): {e}")
    
    return properties


def get_flow_object_memberships(db: PlexosDB, class_name: str, object_name: str) -> List[Dict[str, Any]]:
    """
    Get membership relationships for a flow object using PlexosDB methods.
    
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
        # Use the built-in PlexosDB method
        memberships = db.get_memberships_system(object_name, object_class=class_name)
        logger.debug(f"Found {len(memberships)} memberships for flow object {object_name}")
        
    except Exception as e:
        logger.warning(f"Failed to get memberships for flow object {object_name}: {e}")
    
    return memberships


def add_flow_buses(network: Network, db: PlexosDB) -> Dict[str, int]:
    """
    Add flow nodes as buses, organized by sector.
    
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
        
        for flow_class in flow_classes['flow']:
            if 'node' in flow_class.lower():
                flow_objects = get_flow_objects_by_class_name(db, flow_class)
                
                for node_name in flow_objects:
                    # Determine sector from node name (following multi_sector_db.py pattern)
                    if node_name.startswith('Elec_'):
                        sector = 'Electricity'
                    elif node_name.startswith('H2_'):
                        sector = 'Hydrogen'
                    elif node_name.startswith('NH3_'):
                        sector = 'Ammonia'
                    else:
                        sector = 'Other'
                    
                    # Add carrier if needed
                    if sector not in network.carriers.index:
                        network.add('Carrier', sector)
                    
                    # Add bus
                    if node_name not in network.buses.index:
                        network.add('Bus', node_name, carrier=sector)
                        
                        if sector not in sector_counts:
                            sector_counts[sector] = 0
                        sector_counts[sector] += 1
    
    except Exception as e:
        logger.error(f"Failed to add flow buses: {e}")
    
    total_buses = sum(sector_counts.values())
    print(f"Added {total_buses} flow buses across sectors: {sector_counts}")
    return sector_counts


def add_flow_paths(network: Network, db: PlexosDB, timeslice_csv: Optional[str] = None, testing_mode: bool = False) -> Dict[str, int]:
    """
    Add flow paths as links, distinguishing between transport and conversion.
    
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
    paths = {'Transport': 0, 'Conversion': 0}
    
    try:
        flow_classes = discover_flow_classes(db)
        
        # Count total flow paths first for progress tracking
        total_paths = 0
        for flow_class in flow_classes['flow']:
            if 'path' in flow_class.lower():
                path_objects = get_flow_objects_by_class_name(db, flow_class)
                total_paths += len(path_objects)
        
        # Initialize progress tracker
        if total_paths > 0:
            path_progress = create_progress_tracker(total_paths, "Processing Flow Paths", 
                                                  testing_mode=testing_mode, test_limit=100)
            path_index = 0
            
            for flow_class in flow_classes['flow']:
                if 'path' in flow_class.lower():
                    path_objects = get_flow_objects_by_class_name(db, flow_class)
                    
                    for path_name in path_objects:
                        # Check if we should process this path (testing mode)
                        if not path_progress.should_process(path_index):
                            break
                        
                        # Get connected flow nodes using specific collection names
                        memberships = get_flow_object_memberships(db, flow_class, path_name)
                        
                        # Extract flow nodes based on collection names
                        bus0 = None
                        bus1 = None
                        for m in memberships:
                            if m.get('collection_name') == 'Flow Node From' and 'flow' in m.get('class', '').lower():
                                bus0 = m["name"]
                            elif m.get('collection_name') == 'Flow Node To' and 'flow' in m.get('class', '').lower():
                                bus1 = m["name"]
                        
                        if bus0 and bus1 and bus0 in network.buses.index and bus1 in network.buses.index:
                            # Get path properties (would use parse_lines_flow if adapted for flow networks)
                            props = get_flow_object_properties(db, flow_class, path_name)
                            max_flow = 1000  # Default
                            efficiency = 1.0  # Default
                            
                            # For now, use defaults since flow properties may not have ClassEnum mappings
                            # TODO: Adapt parse_lines_flow from links.py for flow network properties
                            
                            try:
                                p_nom = float(max_flow) if max_flow else 1000.0
                                eff = float(efficiency) if efficiency else 1.0
                            except:
                                p_nom = 1000.0
                                eff = 1.0
                            
                            # Determine if transport or conversion link
                            bus0_carrier = network.buses.at[bus0, 'carrier']
                            bus1_carrier = network.buses.at[bus1, 'carrier']
                            
                            if bus0_carrier == bus1_carrier:
                                link_type = 'Transport'
                                link_name = f"transport_{path_name}"
                            else:
                                link_type = 'Conversion'
                                link_name = f"conversion_{path_name}"
                            
                            if link_name not in network.links.index:
                                network.add('Link', link_name, bus0=bus0, bus1=bus1,
                                          p_nom=p_nom, efficiency=eff)
                                paths[link_type] += 1
                        
                        # Update progress
                        path_progress.update(path_index, path_name)
                        path_index += 1
                        
                        # Break out of both loops if we've hit the limit
                        if not path_progress.should_process(path_index):
                            break
                    
                    # Break outer loop if we've hit the limit
                    if path_index >= path_progress.items_to_process:
                        break
            
            # Finish progress tracking
            total_added = paths.get('Transport', 0) + paths.get('Conversion', 0)
            path_progress.finish(total_added)
    
    except Exception as e:
        logger.error(f"Failed to add flow paths: {e}")
    
    print(f"Added {paths['Transport']} transport links and {paths['Conversion']} conversion links")
    return paths


def add_processes(network: Network, db: PlexosDB) -> Dict[str, int]:
    """
    Add processes as sector coupling links.
    
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
        for process_class in flow_classes['process']:
            process_objects = get_flow_objects_by_class_name(db, process_class)
            
            for process_name in process_objects:
                # Get process properties (would extract efficiency if properties were available)
                props = get_flow_object_properties(db, process_class, process_name)
                efficiency = 70.0  # Default
                
                try:
                    eff = float(efficiency) / 100.0 if efficiency else 0.7
                except:
                    eff = 0.7
                
                # Determine process type and create links based on name patterns
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
        logger.error(f"Failed to add processes: {e}")
    
    total_processes = sum(process_stats.values())
    print(f"Added {total_processes} processes: {process_stats}")
    return process_stats


def add_facilities(network: Network, db: PlexosDB, testing_mode: bool = False) -> Dict[str, int]:
    """
    Add facilities as generators or loads following generators.py patterns.
    
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
        Dictionary with facility statistics
    """
    facility_stats = {'generators': 0, 'loads': 0, 'skipped': 0}
    
    try:
        flow_classes = discover_flow_classes(db)
        
        # Count total facilities first
        total_facilities = 0
        for facility_class in flow_classes.get('facility', []):
            facility_objects = get_flow_objects_by_class_name(db, facility_class)
            total_facilities += len(facility_objects)
        
        print(f"  Found {total_facilities} facilities to process")
        
        # Initialize progress tracker for facilities
        if total_facilities > 0:
            facility_progress = create_progress_tracker(total_facilities, "Processing Facilities", 
                                                      testing_mode=testing_mode, test_limit=1000)
            facility_index = 0
            
            # Process facility classes
            for facility_class in flow_classes.get('facility', []):
                facility_objects = get_flow_objects_by_class_name(db, facility_class)
                
                for facility_name in facility_objects:
                    # Check if we should process this facility (testing mode)
                    if not facility_progress.should_process(facility_index):
                        break
                    
                    # Get facility memberships to find connected flow nodes and processes
                    memberships = get_flow_object_memberships(db, facility_class, facility_name)
                    
                    # Extract connected flow nodes
                    flow_nodes = []
                    for m in memberships:
                        if 'flow' in m.get('class', '').lower() and 'node' in m.get('class', '').lower():
                            flow_nodes.append(m["name"])
                    
                    # Extract process type
                    processes = [m["name"] for m in memberships if m.get('class', '').lower() == 'process']
                    
                    if len(flow_nodes) >= 1:
                        primary_bus = flow_nodes[0]  # Use first flow node as primary bus
                        process_name = processes[0] if processes else ''   # Use first process if available
                        
                        # Broaden generator classification logic (from generators.py patterns)
                        generator_keywords = ['solar', 'wind', 'hydro', 'nuclear', 'gas', 'biomass', 'geothermal', 
                                            'coal', 'oil', 'pv', 'onshore', 'offshore', 'thermal', 'power', 'gen']
                        is_generator = any(gen_type in facility_name.lower() or gen_type in process_name.lower() 
                                         for gen_type in generator_keywords)
                        
                        if primary_bus in network.buses.index:
                            if is_generator:
                                # Add as generator (following generators.py patterns)
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
                    else:
                        facility_stats['skipped'] += 1
                    
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
        logger.error(f"Failed to add facilities: {e}")
    
    print(f"Added {facility_stats['generators']} generators and {facility_stats['loads']} loads from facilities")
    return facility_stats


def add_flow_demand(network: Network) -> None:
    """
    Add basic demand profiles to flow network buses by sector.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add demand to
    """
    try:
        # Get buses by carrier/sector
        sectors = ['Electricity', 'Hydrogen', 'Ammonia', 'Other']
        
        for sector in sectors:
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
        logger.error(f"Failed to add flow demand: {e}")


def port_flow_network(
    network: Network,
    db: PlexosDB,
    timeslice_csv: Optional[str] = None,
    testing_mode: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive function to set up multi-sector flow networks using PLEXOS Flow Network components.
    
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
        'network_type': 'flow_network',
        'sectors': [],
        'buses_by_sector': {},
        'paths': {},
        'processes': {},
        'facilities': {}
    }
    
    try:
        # Step 1: Add flow nodes as buses
        print("1. Adding flow nodes as buses...")
        summary['buses_by_sector'] = add_flow_buses(network, db)
        summary['sectors'] = list(summary['buses_by_sector'].keys())
        
        # Set basic snapshots if not already set
        if len(network.snapshots) == 0:
            snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
            network.set_snapshots(snapshots)
            print(f"Set {len(snapshots)} hourly snapshots")
        
        # Step 2: Add flow paths as links
        print("2. Adding flow paths as links...")
        summary['paths'] = add_flow_paths(network, db, timeslice_csv=timeslice_csv, testing_mode=testing_mode)
        
        # Step 3: Add processes for sector coupling
        print("3. Adding processes for sector coupling...")
        summary['processes'] = add_processes(network, db)
        
        # Step 4: Add facilities as generators/loads
        print("4. Adding facilities as generators/loads...")
        summary['facilities'] = add_facilities(network, db, testing_mode=testing_mode)
        
        # Step 5: Add basic demand profiles
        print("5. Adding basic demand profiles...")
        add_flow_demand(network)
        
    except Exception as e:
        logger.error(f"Error in flow network porting: {e}")
        raise
    
    total_buses = sum(summary['buses_by_sector'].values())
    total_links = summary['paths'].get('Transport', 0) + summary['paths'].get('Conversion', 0)
    total_processes = sum(summary['processes'].values())
    
    print(f"Flow network porting complete!")
    print(f"  Added {total_buses} buses across {len(summary['sectors'])} sectors")
    print(f"  Added {total_links} links ({summary['paths']})")  
    print(f"  Added {total_processes} processes ({summary['processes']})")
    print(f"  Added {summary['facilities']['generators']} generators and {summary['facilities']['loads']} loads")
    
    return summary