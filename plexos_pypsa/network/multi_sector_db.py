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


def setup_gas_electric_network_db(network: Network, db: PlexosDB, generators_as_links: bool = False, testing_mode: bool = False) -> Dict[str, Any]:
    """
    Set up a multi-sector PyPSA network for gas and electricity using database queries.
    
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
        
    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    print("Setting up gas-electric network using database queries...")
    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")
    if generators_as_links:
        print("🔗 GENERATORS-AS-LINKS: Converting conventional generators to fuel→electric Links")
    
    # Initialize tracking
    setup_summary = {
        'network_type': 'gas_electric_db',
        'sectors': ['Electricity', 'Gas'],
        'electricity': {'buses': 0, 'generators': 0, 'loads': 0, 'lines': 0, 'storage': 0},
        'gas': {'buses': 0, 'pipelines': 0, 'storage': 0, 'demand': 0, 'fields': 0},
        'sector_coupling': {
            'gas_generators': 0, 
            'efficiency_range': 'N/A',
            'generator_links': 0,
            'renewable_generators': 0, 
            'fuel_types': []
        }
    }
    
    try:
        # Step 1: Discover multi-sector classes
        print("\n1. Discovering multi-sector classes...")
        multi_sector_classes = discover_multi_sector_classes(db)
        
        # Step 2: Set up electricity sector (use existing functionality)
        print("\n2. Setting up electricity sector...")
        
        # Add carriers first
        if 'AC' not in network.carriers.index:
            network.add('Carrier', 'AC')
        if 'Gas' not in network.carriers.index:
            network.add('Carrier', 'Gas')
        
        # Add common generator carriers
        generator_carriers = ['Coal', 'Natural Gas', 'Nuclear', 'Hydro', 'Wind', 'Solar', 
                            'Biomass', 'Oil', 'Unknown', 'Wind Onshore', 'Wind Offshore', 
                            'Solar PV', 'Biomass Waste', 'Natural Gas CCGT', 'Natural Gas OCGT',
                            'Solids Fired', 'Lignite', 'Hard Coal', 'Gas Turbine', 'Steam Turbine']
        for carrier in generator_carriers:
            if carrier not in network.carriers.index:
                network.add('Carrier', carrier)
        
        # Add link/conversion carriers
        link_carriers = ['Gas2Electric', 'Electricity', 'Hydrogen', 'Ammonia']
        for carrier in link_carriers:
            if carrier not in network.carriers.index:
                network.add('Carrier', carrier)
        
        # Add electricity buses from Node class
        node_objects = get_objects_by_class_name(db, 'Node')
        for node_name in node_objects:
            if node_name not in network.buses.index:
                network.add('Bus', node_name, carrier='AC', v_nom=110)
        
        setup_summary['electricity']['buses'] = len(node_objects)
        print(f"   Added {len(node_objects)} electricity buses")
        
        # Set basic snapshots
        snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
        network.set_snapshots(snapshots)
        print(f"   Set {len(snapshots)} hourly snapshots")
        
        # Add electricity generators (with optional generators-as-links conversion)
        generator_objects = get_objects_by_class_name(db, 'Generator')
        generators_added = 0
        generator_links_added = 0
        fuel_types_used = set()
        
        # Apply testing mode limit
        if testing_mode and len(generator_objects) > 100:
            print(f"   ⚠️  TESTING MODE: Processing only 100 of {len(generator_objects)} generators")
            generator_objects = generator_objects[:100]
        
        for gen_name in generator_objects:
            # Get generator properties and memberships
            props = get_object_properties_by_name(db, 'Generator', gen_name)
            memberships = get_object_memberships(db, 'Generator', gen_name)
            
            # Find connected node
            node_name = None
            for membership in memberships:
                if membership.get('class') == 'Node':
                    node_name = membership.get('name')
                    break
            
            if node_name and node_name in network.buses.index:
                max_capacity = 100  # Default
                for prop in props:
                    if prop.get('property') == 'Max Capacity':
                        max_capacity = prop.get('value', 100)
                        break
                
                try:
                    p_nom = float(max_capacity) if max_capacity else 100.0
                except:
                    p_nom = 100.0
                
                # Determine carrier from generator name
                carrier = 'Unknown'
                gen_name_lower = gen_name.lower()
                if 'wind onshore' in gen_name_lower:
                    carrier = 'Wind Onshore'
                elif 'wind offshore' in gen_name_lower:
                    carrier = 'Wind Offshore'
                elif 'wind' in gen_name_lower:
                    carrier = 'Wind'
                elif 'solar' in gen_name_lower or 'pv' in gen_name_lower:
                    carrier = 'Solar PV'
                elif 'biomass waste' in gen_name_lower:
                    carrier = 'Biomass Waste'
                elif 'biomass' in gen_name_lower:
                    carrier = 'Biomass'
                elif 'natural gas ccgt' in gen_name_lower or 'ccgt' in gen_name_lower:
                    carrier = 'Natural Gas CCGT'
                elif 'natural gas ocgt' in gen_name_lower or 'ocgt' in gen_name_lower:
                    carrier = 'Natural Gas OCGT'
                elif 'natural gas' in gen_name_lower or 'gas' in gen_name_lower:
                    carrier = 'Natural Gas'
                elif 'nuclear' in gen_name_lower:
                    carrier = 'Nuclear'
                elif 'solids fired' in gen_name_lower:
                    carrier = 'Solids Fired'
                elif 'lignite' in gen_name_lower:
                    carrier = 'Lignite'
                elif 'hard coal' in gen_name_lower:
                    carrier = 'Hard Coal'
                elif 'coal' in gen_name_lower:
                    carrier = 'Coal'
                elif 'hydro' in gen_name_lower:
                    carrier = 'Hydro'
                elif 'oil' in gen_name_lower:
                    carrier = 'Oil'
                
                # Determine if this is a conventional (fuel-based) generator
                conventional_fuels = ['Natural Gas', 'Natural Gas CCGT', 'Natural Gas OCGT', 
                                    'Coal', 'Hard Coal', 'Lignite', 'Nuclear', 'Oil', 'Biomass', 
                                    'Biomass Waste', 'Solids Fired']
                is_conventional = carrier in conventional_fuels
                
                if generators_as_links and is_conventional:
                    # Create fuel bus if it doesn't exist
                    fuel_bus_name = f"{carrier.replace(' ', '_')}_bus"
                    if fuel_bus_name not in network.buses.index:
                        # Add fuel carrier if needed
                        fuel_carrier = carrier.replace(' CCGT', '').replace(' OCGT', '').replace(' Waste', '')
                        if fuel_carrier not in network.carriers.index:
                            network.add('Carrier', fuel_carrier)
                        network.add('Bus', fuel_bus_name, carrier=fuel_carrier)
                        fuel_types_used.add(fuel_carrier)
                    
                    # Create generator-link (fuel bus → electric bus)
                    link_name = f"gen_link_{gen_name}"
                    if link_name not in network.links.index:
                        # Set efficiency based on technology
                        if 'CCGT' in carrier:
                            efficiency = 0.55  # Combined cycle
                        elif 'OCGT' in carrier:
                            efficiency = 0.35  # Open cycle  
                        elif 'Nuclear' in carrier:
                            efficiency = 0.33  # Nuclear thermal
                        elif 'Coal' in carrier:
                            efficiency = 0.40  # Coal thermal
                        elif 'Biomass' in carrier:
                            efficiency = 0.30  # Biomass thermal
                        else:
                            efficiency = 0.40  # Default thermal
                        
                        # Fuel cost (marginal cost of fuel, not electricity)
                        fuel_cost = 25.0 if 'Gas' in carrier else 20.0
                        
                        network.add('Link', link_name, 
                                  bus0=fuel_bus_name, bus1=node_name,
                                  p_nom=p_nom, efficiency=efficiency, 
                                  marginal_cost=fuel_cost)
                        generator_links_added += 1
                        
                        # Add infinite fuel supply (simplified)
                        fuel_supply_name = f"fuel_supply_{fuel_carrier.replace(' ', '_')}"
                        if fuel_supply_name not in network.generators.index:
                            # Fuel supply should be cheaper than generator-link to ensure proper dispatch
                            network.add('Generator', fuel_supply_name, 
                                      bus=fuel_bus_name, p_nom=99999, 
                                      carrier=fuel_carrier, marginal_cost=0.1)  # Very cheap fuel
                            print(f"   🔍 DEBUG - Added fuel supply: {fuel_supply_name} on {fuel_bus_name} (p_nom=99999, mc=0.1)")
                            print(f"   🔍 DEBUG - Generator-link: {link_name} {fuel_bus_name}→{node_name} (p_nom={p_nom}, eff={efficiency:.2f}, mc={fuel_cost})")
                    
                else:
                    # Add as standard generator (renewables or when generators_as_links=False)
                    if gen_name not in network.generators.index:
                        network.add('Generator', gen_name, bus=node_name, p_nom=p_nom,
                                  carrier=carrier, marginal_cost=50.0)
                        generators_added += 1
        
        setup_summary['electricity']['generators'] = generators_added
        setup_summary['sector_coupling']['generator_links'] = generator_links_added  
        setup_summary['sector_coupling']['renewable_generators'] = generators_added
        setup_summary['sector_coupling']['fuel_types'] = list(fuel_types_used)
        
        if generators_as_links:
            print(f"   Added {generator_links_added} generator-links (conventional)")
            print(f"   Added {generators_added} generators (renewable)")  
            print(f"   Fuel types: {list(fuel_types_used)}")
        else:
            print(f"   Added {generators_added} electricity generators")
        
        # Step 3: Set up gas sector
        print("\n3. Setting up gas sector...")
        
        # Add gas buses from discovered gas classes
        gas_buses_added = 0
        for gas_class in multi_sector_classes['gas']:
            if 'node' in gas_class.lower():
                gas_objects = get_objects_by_class_name(db, gas_class)
                for gas_name in gas_objects:
                    bus_name = f"gas_{gas_name}"
                    if bus_name not in network.buses.index:
                        if 'Gas' not in network.carriers.index:
                            network.add('Carrier', 'Gas')
                        network.add('Bus', bus_name, carrier='Gas')
                        gas_buses_added += 1
        
        setup_summary['gas']['buses'] = gas_buses_added
        print(f"   Added {gas_buses_added} gas buses")
        
        # Add gas pipelines
        gas_pipelines_added = 0
        for gas_class in multi_sector_classes['gas']:
            if 'pipeline' in gas_class.lower():
                pipeline_objects = get_objects_by_class_name(db, gas_class)
                for pipeline_name in pipeline_objects:
                    # Get connected gas nodes
                    memberships = get_object_memberships(db, gas_class, pipeline_name)
                    gas_nodes = [m['name'] for m in memberships if 'gas' in m.get('class', '').lower() and 'node' in m.get('class', '').lower()]
                    
                    if len(gas_nodes) >= 2:
                        bus0 = f"gas_{gas_nodes[0]}"
                        bus1 = f"gas_{gas_nodes[1]}"
                        
                        if bus0 in network.buses.index and bus1 in network.buses.index:
                            props = get_object_properties_by_name(db, gas_class, pipeline_name)
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
        
        setup_summary['gas']['pipelines'] = gas_pipelines_added
        print(f"   Added {gas_pipelines_added} gas pipelines")
        
        # Step 4: Set up sector coupling
        print("\n4. Setting up sector coupling...")
        
        coupling_stats = add_gas_electric_coupling_db(network, db)
        setup_summary['sector_coupling'].update(coupling_stats)
        print(f"   Added {coupling_stats['gas_generators']} gas-to-electric conversion links")
        print(f"   Efficiency range: {coupling_stats['efficiency_range']}")
        
        # Step 5: Add basic loads
        print("\n5. Adding basic demand profiles...")
        
        # Add loads to ALL electricity buses to balance generation
        elec_bus_list = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'AC']
        
        # Calculate total generation capacity differently for generators-as-links mode
        if generators_as_links:
            # For generators-as-links: count renewable generators + generator-link capacity (with efficiency)
            renewable_gen_capacity = sum(network.generators[
                network.generators.bus.isin(elec_bus_list)
            ].p_nom)
            
            # Generator-links capacity (fuel→electric links)
            fuel_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] in ['Natural Gas', 'Biomass', 'Nuclear', 'Oil', 'Solids Fired']]
            generator_link_capacity = sum(network.links[
                (network.links.bus0.isin(fuel_buses)) & 
                (network.links.bus1.isin(elec_bus_list))
            ].p_nom * network.links[
                (network.links.bus0.isin(fuel_buses)) & 
                (network.links.bus1.isin(elec_bus_list))
            ].efficiency)
            
            total_generation_capacity = renewable_gen_capacity + generator_link_capacity
            print(f"   🔍 DEBUG - Renewable generators: {renewable_gen_capacity:.0f} MW")
            print(f"   🔍 DEBUG - Generator-links (with efficiency): {generator_link_capacity:.0f} MW") 
            print(f"   🔍 DEBUG - Total generation capacity: {total_generation_capacity:.0f} MW")
        else:
            # Standard mode: all generators
            total_generation_capacity = sum(network.generators.p_nom)
            print(f"   🔍 DEBUG - Standard generators: {total_generation_capacity:.0f} MW")
        
        total_generation_capacity = max(total_generation_capacity, len(elec_bus_list) * 1000)  # Ensure minimum capacity
        load_per_bus = total_generation_capacity / len(elec_bus_list) * 0.3  # 30% capacity factor (more conservative)
        
        for i, bus in enumerate(elec_bus_list):
            load_name = f"elec_load_{bus}"
            if load_name not in network.loads.index:
                # Vary load by bus with realistic country-based scaling
                base_load = load_per_bus * (0.8 + 0.4 * np.random.random())  # 80-120% of average
                load_profile = pd.Series(
                    base_load * (1 + 0.3 * np.sin(np.arange(len(network.snapshots)) * 2 * np.pi / 24)),
                    index=network.snapshots
                )
                network.add('Load', load_name, bus=bus, p_set=load_profile)
        
        setup_summary['electricity']['loads'] = len([l for l in network.loads.index if 'elec_load' in l])
        
        # Debug load-generation balance
        total_load = sum([network.loads.at[load, 'p_set'].sum() for load in network.loads.index if 'elec_load' in load])
        print(f"   🔍 DEBUG - Total load created: {total_load/1e6:.1f} TWh/year")
        print(f"   🔍 DEBUG - Load-generation ratio: {total_load/(total_generation_capacity*8760):.2f}")
        
        # Add real transmission lines from PLEXOS Line objects
        print("   Adding real transmission lines from PLEXOS model...")
        lines_added = 0
        
        # Get PLEXOS Line objects
        line_objects = get_objects_by_class_name(db, 'Line')
        
        for line_name in line_objects:
            # Get line properties
            props = get_object_properties_by_name(db, 'Line', line_name)
            
            # Get connected nodes
            memberships = get_object_memberships(db, 'Line', line_name)
            
            node_from = None
            node_to = None
            for membership in memberships:
                if membership.get('collection_name') == 'Node From':
                    node_from = membership.get('name')
                elif membership.get('collection_name') == 'Node To':
                    node_to = membership.get('name')
            
            # Only add line if both nodes exist in the network
            if node_from and node_to and node_from in network.buses.index and node_to in network.buses.index:
                # Extract capacity from Max Flow property
                s_nom = 1000  # Default capacity (MW)
                for prop in props:
                    if prop.get('property') == 'Max Flow':
                        try:
                            s_nom = float(prop.get('value', 1000))
                        except:
                            s_nom = 1000
                        break
                
                # Add realistic impedance values for transmission lines
                # Typical transmission line R/X ratio ≈ 0.1-0.3
                # Using conservative values for stability
                line_length_factor = 1.0  # Could be refined based on geography
                r = 0.01 * line_length_factor  # Resistance
                x = 0.03 * line_length_factor  # Reactance (higher X/R for transmission)
                
                if line_name not in network.lines.index:
                    network.add('Line', line_name, 
                              bus0=node_from, bus1=node_to, 
                              s_nom=abs(s_nom), r=r, x=x)
                    lines_added += 1
        
        setup_summary['electricity']['lines'] = lines_added
        print(f"   Added {lines_added} real transmission lines from PLEXOS model")
        
    except Exception as e:
        logger.error(f"Error setting up gas-electric network: {e}")
        raise
    
    print("Gas-electric multi-sector network setup complete!")
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


def setup_flow_network_db(network: Network, db: PlexosDB, testing_mode: bool = False) -> Dict[str, Any]:
    """
    Set up a multi-sector PyPSA network using PLEXOS Flow Network components with database queries.
    
    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing
        
    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector
    """
    # Initialize batch progress tracker
    batch_tracker = BatchProgressTracker("Flow Network Setup", testing_mode=testing_mode)
    
    setup_summary = {
        'network_type': 'flow_network_db',
        'sectors': [],
        'processes': {}
    }
    
    try:
        # Step 1: Discover multi-sector classes
        print("\n1. Discovering multi-sector classes...")
        multi_sector_classes = discover_multi_sector_classes(db)
        
        # Step 2: Add flow nodes as buses
        print("\n2. Setting up flow nodes as buses...")
        sectors = {}
        
        for flow_class in multi_sector_classes['flow']:
            if 'node' in flow_class.lower():
                flow_objects = get_objects_by_class_name(db, flow_class)
                
                for node_name in flow_objects:
                    
                    # Determine sector from node name
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
                        
                        if sector not in sectors:
                            sectors[sector] = 0
                        sectors[sector] += 1
        
        setup_summary['sectors'] = list(sectors.keys())
        for sector in sectors:
            setup_summary[sector.lower()] = {
                'nodes': sectors[sector], 'paths': 0, 'storage': 0, 'demand': 0
            }
        
        # Step 3: Set basic snapshots
        snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
        network.set_snapshots(snapshots)
        
        # Step 4: Add flow paths as links
        print("\n3. Setting up flow paths as links...")
        paths = {'Transport': 0, 'Conversion': 0}
        
        # Count total flow paths first
        total_paths = 0
        for flow_class in multi_sector_classes['flow']:
            if 'path' in flow_class.lower():
                path_objects = get_objects_by_class_name(db, flow_class)
                total_paths += len(path_objects)
        
        # Initialize progress tracker for flow paths
        if total_paths > 0:
            path_progress = batch_tracker.add_batch(total_paths, "Processing Flow Paths", test_limit=100)
            path_index = 0
            
            for flow_class in multi_sector_classes['flow']:
                if 'path' in flow_class.lower():
                    path_objects = get_objects_by_class_name(db, flow_class)
                    
                    for path_name in path_objects:
                        # Check if we should process this path (testing mode)
                        if not path_progress.should_process(path_index):
                            break
                        
                        # Get connected flow nodes using specific collection names
                        memberships = get_object_memberships(db, flow_class, path_name)
                    
                    # Extract flow nodes based on collection names
                    bus0 = None
                    bus1 = None
                    for m in memberships:
                        if m.get('collection_name') == 'Flow Node From' and 'flow' in m.get('class', '').lower():
                            bus0 = m["name"]
                        elif m.get('collection_name') == 'Flow Node To' and 'flow' in m.get('class', '').lower():
                            bus1 = m["name"]
                    
                    if bus0 and bus1:
                        
                        if bus0 in network.buses.index and bus1 in network.buses.index:
                            # Get path properties
                            props = get_object_properties_by_name(db, flow_class, path_name)
                            max_flow = 1000  # Default
                            efficiency = 1.0  # Default
                            
                            for prop in props:
                                if prop.get('property') == 'Max Flow':
                                    max_flow = prop.get('value', 1000)
                                elif prop.get('property') == 'Efficiency':
                                    efficiency = prop.get('value', 1.0)
                            
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
                        
                        # Break out of both loops if we've hit the limit
                        if not path_progress.should_process(path_index):
                            break
                    
                    # Break outer loop if we've hit the limit
                    if path_index >= path_progress.items_to_process:
                        break
            
            # Finish flow path progress tracking
            path_progress.finish(paths.get('Transport', 0) + paths.get('Conversion', 0))
        
        # Update sector path counts
        for sector in sectors:
            setup_summary[sector.lower()]['paths'] = paths.get('Transport', 0) + paths.get('Conversion', 0)
        
        # Step 5: Add processes for sector coupling
        print("\n4. Setting up process-based sector coupling...")
        process_stats = add_processes_db(network, db, multi_sector_classes)
        setup_summary['processes'] = process_stats
        
        # Step 6: Add facilities as generators/loads
        print("\n5. Setting up facilities as generators/loads...")
        facility_stats = add_facilities_db(network, db, multi_sector_classes, testing_mode)
        setup_summary['facilities'] = facility_stats
        
        # Step 7: Add basic demands (only to buses without facilities)  
        print("\n6. Setting up basic demand profiles...")
        for sector in sectors:
            if sectors[sector] > 0:  # Only if we have buses for this sector
                add_flow_demand_db(network, sector)
        
    except Exception as e:
        logger.error(f"Error setting up flow network: {e}")
        raise
    finally:
        # Finish batch progress tracking
        batch_tracker.finish()
    
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