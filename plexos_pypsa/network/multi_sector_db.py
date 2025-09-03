"""
Multi-Sector Network Setup Functions (Database-based)

This module provides functions to set up PyPSA networks for multi-sector energy models
using direct database queries to discover Gas and Flow network classes dynamically.
Enhanced with CSV data integration following PyPSA multi-sector best practices.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import os

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


def sync_network_carriers(network: Network) -> None:
    """
    Discover and add missing carriers from existing network components.
    
    This function inspects all network components (generators, links, loads, buses) 
    and adds any carriers they reference that don't already exist in network.carriers.
    This ensures PyPSA consistency by preventing "carrier not defined" warnings.
    
    Parameters
    ----------
    network : Network
        PyPSA network to sync carriers for
    """
    missing_carriers = set()
    
    # Extract carriers from all network components
    if len(network.links) > 0 and 'carrier' in network.links.columns:
        missing_carriers.update(network.links.carrier.dropna().unique())
    if len(network.generators) > 0 and 'carrier' in network.generators.columns:
        missing_carriers.update(network.generators.carrier.dropna().unique())
    if len(network.loads) > 0 and 'carrier' in network.loads.columns:
        missing_carriers.update(network.loads.carrier.dropna().unique())
    if len(network.buses) > 0 and 'carrier' in network.buses.columns:
        missing_carriers.update(network.buses.carrier.dropna().unique())
    
    # Add any missing carriers
    added_count = 0
    for carrier in missing_carriers:
        if carrier not in network.carriers.index:
            network.add('Carrier', carrier)
            added_count += 1
            
    if added_count > 0:
        print(f"   Automatically added {added_count} missing carriers from network components")


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


# ============================================================================
# CSV Data Integration Functions (PyPSA Multi-Sector Best Practices)
# ============================================================================

def load_csv_costs(inputs_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load cost data from CSV files in the PLEXOS-MESSAGE inputs folder.
    
    Parameters
    ----------
    inputs_folder : str
        Path to the inputs folder containing CSV files
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing cost dataframes: 'build_costs', 'foms', 'voms', 'fuel_prices'
    """
    cost_data = {}
    
    try:
        # Load build costs (capital costs)
        build_costs_path = os.path.join(inputs_folder, "BuildCosts.csv")
        if os.path.exists(build_costs_path):
            cost_data['build_costs'] = pd.read_csv(build_costs_path, index_col='YEAR')
            logger.info(f"Loaded BuildCosts.csv with {len(cost_data['build_costs'].columns)} technology-region combinations")
        
        # Load fixed O&M costs
        foms_path = os.path.join(inputs_folder, "FOMs.csv")  
        if os.path.exists(foms_path):
            cost_data['foms'] = pd.read_csv(foms_path, index_col='YEAR')
            logger.info(f"Loaded FOMs.csv with {len(cost_data['foms'].columns)} technology-region combinations")
            
        # Load variable O&M costs
        voms_path = os.path.join(inputs_folder, "VOMs.csv")
        if os.path.exists(voms_path):
            cost_data['voms'] = pd.read_csv(voms_path, index_col='YEAR')
            logger.info(f"Loaded VOMs.csv with {len(cost_data['voms'].columns)} technology-region combinations")
            
        # Load fuel prices with emissions
        fuel_prices_path = os.path.join(inputs_folder, "prices_w_emissions_permwh.csv")
        if os.path.exists(fuel_prices_path):
            cost_data['fuel_prices'] = pd.read_csv(fuel_prices_path)
            logger.info(f"Loaded fuel prices with {len(cost_data['fuel_prices'].columns)} fuel types")
            
    except Exception as e:
        logger.warning(f"Error loading CSV cost data: {e}")
    
    return cost_data


def load_time_series_data(inputs_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load time series data from CSV files following PyPSA patterns.
    
    Parameters
    ----------
    inputs_folder : str
        Path to the inputs folder containing CSV files
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing time series dataframes
    """
    time_series_data = {}
    
    try:
        # Load electricity demand (primary snapshots source)
        load_path = os.path.join(inputs_folder, "Load.csv")
        if os.path.exists(load_path):
            load_df = pd.read_csv(load_path, parse_dates=['Datetime'], index_col='Datetime')
            time_series_data['electricity_demand'] = load_df
            logger.info(f"Loaded Load.csv with {len(load_df.columns)} regions and {len(load_df)} time steps")
            
        # Load hydrogen demand
        h2_demand_path = os.path.join(inputs_folder, "H2_Demand_With_Blending.csv")
        if os.path.exists(h2_demand_path):
            h2_df = pd.read_csv(h2_demand_path, parse_dates=['Datetime'], index_col='Datetime')
            time_series_data['hydrogen_demand'] = h2_df
            logger.info(f"Loaded H2 demand with {len(h2_df.columns)} regions")
            
        # Load VRE profiles
        solar_path = os.path.join(inputs_folder, "Solar_Inverted.csv")
        if os.path.exists(solar_path):
            solar_df = pd.read_csv(solar_path, parse_dates=['Datetime'], index_col='Datetime')
            time_series_data['solar_profiles'] = solar_df
            logger.info(f"Loaded Solar profiles with {len(solar_df.columns)} regions")
            
        wind_path = os.path.join(inputs_folder, "Wind_Inverted.csv") 
        if os.path.exists(wind_path):
            wind_df = pd.read_csv(wind_path, parse_dates=['Datetime'], index_col='Datetime')
            time_series_data['wind_profiles'] = wind_df
            logger.info(f"Loaded Wind profiles with {len(wind_df.columns)} regions")
            
        # Load hydro inflows (monthly data)
        hydro_path = os.path.join(inputs_folder, "Hydro.csv")
        if os.path.exists(hydro_path):
            hydro_df = pd.read_csv(hydro_path, index_col='NAME')
            time_series_data['hydro_inflows'] = hydro_df
            logger.info(f"Loaded Hydro inflows for {len(hydro_df)} facilities")
            
    except Exception as e:
        logger.warning(f"Error loading time series data: {e}")
        
    return time_series_data


def load_infrastructure_data(inputs_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Load infrastructure data for multi-sector modeling.
    
    Parameters
    ----------
    inputs_folder : str
        Path to the inputs folder containing CSV files
        
    Returns  
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing infrastructure dataframes
    """
    infrastructure_data = {}
    
    try:
        # Load H2 pipeline data
        pipelines_path = os.path.join(inputs_folder, "H2Pipelines.csv")
        if os.path.exists(pipelines_path):
            pipelines_df = pd.read_csv(pipelines_path)
            infrastructure_data['h2_pipelines'] = pipelines_df
            logger.info(f"Loaded {len(pipelines_df)} H2 pipelines")
            
        # Load losses data 
        losses_path = os.path.join(inputs_folder, "Losses.csv")
        if os.path.exists(losses_path):
            losses_df = pd.read_csv(losses_path)
            infrastructure_data['losses'] = losses_df
            logger.info(f"Loaded transmission losses data")
            
    except Exception as e:
        logger.warning(f"Error loading infrastructure data: {e}")
        
    return infrastructure_data


def extract_region_from_column(column_name: str) -> str:
    """
    Extract region identifier from CSV column names following MESSAGE patterns.
    
    Examples: 'Solar_fac EU-DEU' -> 'EU-DEU', 'H2_mkt EU-FRA' -> 'EU-FRA'
    """
    parts = column_name.split()
    if len(parts) >= 2:
        return parts[-1]  # Last part is usually the region
    return column_name


def get_technology_costs(cost_data: Dict[str, pd.DataFrame], technology: str, region: str, year: int = 2050) -> Dict[str, float]:
    """
    Extract costs for a specific technology and region from loaded CSV data.
    
    Parameters
    ----------
    cost_data : Dict[str, pd.DataFrame]
        Loaded cost data from load_csv_costs()
    technology : str
        Technology type (e.g., 'Solar_fac', 'Electrolysis_fac')
    region : str
        Region identifier (e.g., 'EU-DEU')
    year : int, optional
        Year for cost lookup, default 2050
        
    Returns
    -------
    Dict[str, float]
        Dictionary with 'capital_cost', 'fixed_cost', 'marginal_cost'
    """
    costs = {'capital_cost': 0.0, 'fixed_cost': 0.0, 'marginal_cost': 0.0}
    
    try:
        tech_region_col = f"{technology} {region}"
        
        # Extract capital cost from BuildCosts.csv
        if 'build_costs' in cost_data and tech_region_col in cost_data['build_costs'].columns:
            if year in cost_data['build_costs'].index:
                costs['capital_cost'] = cost_data['build_costs'].at[year, tech_region_col]
                
        # Extract fixed O&M cost from FOMs.csv
        if 'foms' in cost_data and tech_region_col in cost_data['foms'].columns:
            if year in cost_data['foms'].index:
                costs['fixed_cost'] = cost_data['foms'].at[year, tech_region_col]
                
        # Extract variable O&M cost from VOMs.csv  
        if 'voms' in cost_data and tech_region_col in cost_data['voms'].columns:
            if year in cost_data['voms'].index:
                costs['marginal_cost'] = cost_data['voms'].at[year, tech_region_col]
                
    except Exception as e:
        logger.debug(f"Error extracting costs for {technology} in {region}: {e}")
        
    return costs


def create_multi_sector_buses(network: Network, regions: List[str]) -> Dict[str, List[str]]:
    """
    Create multi-sector buses following PyPSA best practices for sector coupling.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add buses to
    regions : List[str]
        List of region identifiers
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping sectors to list of bus names created
    """
    sector_buses = {
        'electricity': [],
        'hydrogen': [], 
        'ammonia': [],
        'gas': []
    }
    
    # Add carriers first following PyPSA patterns
    carriers = ['AC', 'hydrogen', 'ammonia', 'gas']
    for carrier in carriers:
        if carrier not in network.carriers.index:
            network.add('Carrier', carrier)
    
    # Create sector-specific buses for each region
    for region in regions:
        # Electricity bus (following existing AC pattern)
        elec_bus = f"{region}_electricity"
        if elec_bus not in network.buses.index:
            network.add('Bus', elec_bus, carrier='AC', v_nom=110.0)
            sector_buses['electricity'].append(elec_bus)
            
        # Hydrogen bus following PyPSA sector coupling patterns
        h2_bus = f"{region}_hydrogen"
        if h2_bus not in network.buses.index:
            network.add('Bus', h2_bus, carrier='hydrogen')
            sector_buses['hydrogen'].append(h2_bus)
            
        # Ammonia bus for synthetic fuel sector
        nh3_bus = f"{region}_ammonia"
        if nh3_bus not in network.buses.index:
            network.add('Bus', nh3_bus, carrier='ammonia')
            sector_buses['ammonia'].append(nh3_bus)
            
        # Gas bus (if needed for existing gas infrastructure)
        gas_bus = f"{region}_gas"
        if gas_bus not in network.buses.index:
            network.add('Bus', gas_bus, carrier='gas')
            sector_buses['gas'].append(gas_bus)
    
    logger.info(f"Created multi-sector buses: {len(sector_buses['electricity'])} electricity, "
                f"{len(sector_buses['hydrogen'])} hydrogen, {len(sector_buses['ammonia'])} ammonia")
    
    return sector_buses


def create_sector_coupling_links(network: Network, sector_buses: Dict[str, List[str]], cost_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Create sector coupling links following PyPSA best practices.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add links to
    sector_buses : Dict[str, List[str]]
        Dictionary mapping sectors to bus names from create_multi_sector_buses()
    cost_data : Dict[str, pd.DataFrame]
        Cost data from load_csv_costs()
        
    Returns
    -------
    Dict[str, int]
        Dictionary with counts of links created by type
    """
    link_counts = {
        'electrolysis': 0,
        'h2_power': 0,
        'haber_bosch': 0,
        'ammonia_crack': 0
    }
    
    # Create sector coupling links for each region
    electricity_buses = sector_buses.get('electricity', [])
    hydrogen_buses = sector_buses.get('hydrogen', [])
    ammonia_buses = sector_buses.get('ammonia', [])
    
    for i, elec_bus in enumerate(electricity_buses):
        # Extract region from bus name (assuming pattern: "{region}_electricity")
        region = elec_bus.replace('_electricity', '')
        
        if i < len(hydrogen_buses):
            h2_bus = hydrogen_buses[i]
            
            # 1. Electrolysis: Electricity -> Hydrogen
            electrolysis_costs = get_technology_costs(cost_data, 'Electrolysis_fac', region)
            elec_link = f"{region}_electrolysis"
            if elec_link not in network.links.index:
                network.add('Link', elec_link,
                          bus0=elec_bus,
                          bus1=h2_bus,
                          carrier='Electrolysis',
                          efficiency=0.7,  # PyPSA typical electrolysis efficiency
                          capital_cost=electrolysis_costs.get('capital_cost', 1000),
                          marginal_cost=electrolysis_costs.get('marginal_cost', 5),
                          p_nom_extendable=True)
                link_counts['electrolysis'] += 1
                
            # 2. H2Power (Fuel Cell): Hydrogen -> Electricity  
            h2power_costs = get_technology_costs(cost_data, 'H2Power_fac', region)
            h2power_link = f"{region}_h2power"
            if h2power_link not in network.links.index:
                network.add('Link', h2power_link,
                          bus0=h2_bus,
                          bus1=elec_bus, 
                          carrier='H2Power',
                          efficiency=0.5,  # PyPSA typical fuel cell efficiency
                          capital_cost=h2power_costs.get('capital_cost', 1500),
                          marginal_cost=h2power_costs.get('marginal_cost', 10),
                          p_nom_extendable=True)
                link_counts['h2_power'] += 1
        
        if i < len(ammonia_buses):
            h2_bus = hydrogen_buses[i] if i < len(hydrogen_buses) else None
            nh3_bus = ammonia_buses[i]
            
            if h2_bus:
                # 3. Haber-Bosch: Hydrogen -> Ammonia
                haber_costs = get_technology_costs(cost_data, 'HaberBosch_fac', region)
                haber_link = f"{region}_haber_bosch" 
                if haber_link not in network.links.index:
                    network.add('Link', haber_link,
                              bus0=h2_bus,
                              bus1=nh3_bus,
                              carrier='HaberBosch',
                              efficiency=0.8,  # Haber-Bosch process efficiency
                              capital_cost=haber_costs.get('capital_cost', 2000),
                              marginal_cost=haber_costs.get('marginal_cost', 20),
                              p_nom_extendable=True)
                    link_counts['haber_bosch'] += 1
                    
                # 4. Ammonia Cracking: Ammonia -> Hydrogen  
                crack_costs = get_technology_costs(cost_data, 'AmmoniaCrack_fac', region)
                crack_link = f"{region}_ammonia_crack"
                if crack_link not in network.links.index:
                    network.add('Link', crack_link,
                              bus0=nh3_bus,
                              bus1=h2_bus,
                              carrier='AmmoniaCrack', 
                              efficiency=0.9,  # Ammonia cracking efficiency
                              capital_cost=crack_costs.get('capital_cost', 800),
                              marginal_cost=crack_costs.get('marginal_cost', 15),
                              p_nom_extendable=True)
                    link_counts['ammonia_crack'] += 1
    
    logger.info(f"Created sector coupling links: {link_counts}")
    return link_counts


def add_multi_sector_storage(network: Network, sector_buses: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Add storage for different sectors following PyPSA Store patterns.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add storage to
    sector_buses : Dict[str, List[str]]  
        Dictionary mapping sectors to bus names
        
    Returns
    -------
    Dict[str, int]
        Dictionary with storage units added by sector
    """
    storage_counts = {'hydrogen': 0, 'ammonia': 0}
    
    # Add hydrogen storage
    for h2_bus in sector_buses.get('hydrogen', []):
        region = h2_bus.replace('_hydrogen', '')
        store_name = f"{region}_h2_storage"
        if store_name not in network.stores.index:
            network.add('Store', store_name,
                      bus=h2_bus,
                      carrier='hydrogen',
                      e_cyclic=True,  # PyPSA pattern for cyclic storage
                      e_nom_extendable=True,
                      capital_cost=50)  # €/MWh storage cost
            storage_counts['hydrogen'] += 1
            
    # Add ammonia storage  
    for nh3_bus in sector_buses.get('ammonia', []):
        region = nh3_bus.replace('_ammonia', '')
        store_name = f"{region}_nh3_storage"
        if store_name not in network.stores.index:
            network.add('Store', store_name,
                      bus=nh3_bus,
                      carrier='ammonia', 
                      e_cyclic=True,
                      e_nom_extendable=True,
                      capital_cost=25)  # €/MWh storage cost
            storage_counts['ammonia'] += 1
    
    logger.info(f"Added multi-sector storage: {storage_counts}")
    return storage_counts


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
        
        # Step 6: Add enhanced sector coupling (gas-electric coupling)
        print("\n6. Setting up enhanced gas-electric sector coupling...")
        
        coupling_stats = add_gas_electric_coupling_db(network, db, generators_as_links=generators_as_links)
        setup_summary['sector_coupling'].update(coupling_stats)
        
        # Print coupling summary
        if coupling_stats.get('gas_plants_added', 0) > 0:
            print(f"   Gas plants (from gas.py): {coupling_stats['gas_plants_added']} links")
        if coupling_stats.get('gas_generators', 0) > 0:
            print(f"   Gas-fired generators: {coupling_stats['gas_generators']} links")
            print(f"   Efficiency range: {coupling_stats['efficiency_range']}")
        if coupling_stats.get('sector_coupling_links', 0) > 0:
            print(f"   Multi-sector links (electrolysis/fuel_cell): {coupling_stats['sector_coupling_links']} links")
        if coupling_stats.get('fuel_types'):
            print(f"   Fuel types: {', '.join(coupling_stats['fuel_types'])}")
        
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


def add_gas_electric_coupling_db(network: Network, db: PlexosDB, generators_as_links: bool = False) -> Dict[str, Any]:
    """
    Add enhanced gas-to-electric conversion following PyPSA multi-sector patterns.
    
    This function creates comprehensive sector coupling between gas and electricity sectors,
    integrating with the enhanced gas network components and following established patterns
    from plexos_message.py.
    """
    coupling_stats = {
        'gas_generators': 0, 
        'efficiency_range': 'N/A',
        'gas_plants_added': 0,
        'fuel_types': [],
        'sector_coupling_links': 0
    }
    
    try:
        # 1. Process existing gas plants (already added by gas.py)
        gas_plant_links = [link for link in network.links.index if 'gas_plant_' in link]
        coupling_stats['gas_plants_added'] = len(gas_plant_links)
        
        # 2. Process electricity generators with gas connections (for generators_as_links mode)
        if generators_as_links:
            generator_objects = get_objects_by_class_name(db, 'Generator')
            efficiency_values = []
            fuel_types_found = set()
            
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
                        fuel_type = 'Natural Gas'  # Default
                        
                        for prop in props:
                            if prop.get('property') == 'Max Capacity':
                                max_capacity = prop.get('value', 100)
                            elif prop.get('property') == 'Heat Rate':
                                heat_rate = prop.get('value', 9.0)
                            elif prop.get('property') == 'Fuel':
                                fuel_type = prop.get('value', 'Natural Gas')
                        
                        try:
                            p_nom = float(max_capacity) if max_capacity else 100.0
                            hr = float(heat_rate) if heat_rate else 9.0
                            
                            # Calculate efficiency (3412 BTU/kWh conversion factor)
                            efficiency = 3412 / hr if hr > 0 else 0.4
                            efficiency = min(efficiency, 0.65)  # Cap at 65%
                            
                            efficiency_values.append(efficiency)
                            fuel_types_found.add(fuel_type)
                            
                            # Create gas-to-electric conversion link (generators-as-links mode)
                            link_name = f"gas_to_elec_{gen_name}"
                            if link_name not in network.links.index:
                                network.add('Link', link_name,
                                          bus0=gas_bus, bus1=elec_bus, p_nom=p_nom,
                                          efficiency=efficiency, carrier='gas_to_electricity',
                                          marginal_cost=0.01)  # Small marginal cost for gas conversion
                                coupling_stats['gas_generators'] += 1
                        except:
                            pass
            
            coupling_stats['fuel_types'] = list(fuel_types_found)
            
            # Calculate efficiency range
            if efficiency_values:
                min_eff = min(efficiency_values)
                max_eff = max(efficiency_values)
                coupling_stats['efficiency_range'] = f"{min_eff:.1%} - {max_eff:.1%}"
        
        # 3. Add additional sector coupling following plexos_message.py patterns
        # Check for any hydrogen or other gas types for multi-sector coupling
        hydrogen_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'hydrogen']
        elec_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'AC']
        
        if hydrogen_buses and elec_buses:
            # Add power-to-gas (electrolysis) links
            for i, (elec_bus, h2_bus) in enumerate(zip(elec_buses[:3], hydrogen_buses[:3])):
                electrolysis_link = f"power_to_gas_{i+1}"
                if electrolysis_link not in network.links.index:
                    network.add('Link', electrolysis_link,
                              bus0=elec_bus, bus1=h2_bus,
                              p_nom_extendable=True,
                              efficiency=0.7,  # Electrolysis efficiency
                              capital_cost=1000,  # €/MW
                              carrier='electrolysis')
                    coupling_stats['sector_coupling_links'] += 1
            
            # Add gas-to-power (fuel cell) links
            for i, (h2_bus, elec_bus) in enumerate(zip(hydrogen_buses[:2], elec_buses[:2])):
                fuel_cell_link = f"gas_to_power_{i+1}"
                if fuel_cell_link not in network.links.index:
                    network.add('Link', fuel_cell_link,
                              bus0=h2_bus, bus1=elec_bus,
                              p_nom_extendable=True,
                              efficiency=0.5,  # Fuel cell efficiency
                              capital_cost=1500,  # €/MW
                              carrier='fuel_cell')
                    coupling_stats['sector_coupling_links'] += 1
        
        # 4. Total sector coupling links
        total_coupling = (coupling_stats['gas_generators'] + 
                         coupling_stats['gas_plants_added'] + 
                         coupling_stats['sector_coupling_links'])
        
        logger.info(f"Enhanced gas-electric coupling complete: {total_coupling} total coupling links")
            
    except Exception as e:
        logger.warning(f"Failed to add enhanced gas-electric coupling: {e}")
    
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


def setup_enhanced_flow_network_with_csv(network: Network, db: PlexosDB, inputs_folder: str, 
                                        testing_mode: bool = False, timeslice_csv: Optional[str] = None) -> Dict[str, Any]:
    """
    Set up enhanced multi-sector PyPSA network with CSV data integration.
    
    This function combines PLEXOS Flow Network components with CSV data following
    PyPSA multi-sector best practices for electricity, hydrogen, and ammonia sectors.
    
    Parameters
    ----------
    network : pypsa.Network
        Empty PyPSA network to populate
    db : PlexosDB
        PLEXOS database containing model data
    inputs_folder : str
        Path to folder containing CSV data files (BuildCosts.csv, Load.csv, etc.)
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties
        
    Returns
    -------
    Dict[str, Any]
        Setup summary with statistics for each sector and CSV data integration
    """
    print("Setting up enhanced multi-sector flow network with CSV data integration...")
    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")
    
    setup_summary = {
        'network_type': 'enhanced_flow_network_csv',
        'sectors': ['Electricity', 'Hydrogen', 'Ammonia'],
        'csv_data_loaded': False,
        'multi_sector_buses': {},
        'sector_coupling_links': {},
        'csv_integration': {}
    }
    
    try:
        # Step 1: Load CSV data
        print("\n1. Loading CSV data...")
        cost_data = load_csv_costs(inputs_folder)
        time_series_data = load_time_series_data(inputs_folder)
        infrastructure_data = load_infrastructure_data(inputs_folder)
        
        if cost_data or time_series_data:
            setup_summary['csv_data_loaded'] = True
            setup_summary['csv_integration'] = {
                'cost_files': len(cost_data),
                'time_series_files': len(time_series_data),
                'infrastructure_files': len(infrastructure_data)
            }
            print(f"   Loaded {len(cost_data)} cost files, {len(time_series_data)} time series files")
        
        # Step 2: Set up basic flow network using existing flows.py
        print("\n2. Setting up base flow network...")
        flow_summary = port_flow_network(
            network=network,
            db=db,
            timeslice_csv=timeslice_csv,
            testing_mode=testing_mode
        )
        setup_summary.update(flow_summary)
        
        # Step 3: Extract regions from electricity demand or flow nodes
        print("\n3. Discovering regions for multi-sector architecture...")
        regions = []
        
        # Try to get regions from Load.csv first
        if 'electricity_demand' in time_series_data:
            elec_demand = time_series_data['electricity_demand'] 
            for col in elec_demand.columns:
                region = extract_region_from_column(col)
                if region not in regions:
                    regions.append(region)
                    
        # Fallback: use existing buses as regions
        if not regions:
            existing_buses = list(network.buses.index)[:10]  # Limit for testing
            regions = [bus.replace('_electricity', '').replace('_Electric', '') for bus in existing_buses]
            regions = list(set(regions))  # Remove duplicates
            
        print(f"   Found {len(regions)} regions: {regions[:5]}..." if len(regions) > 5 else f"   Found {len(regions)} regions: {regions}")
        
        # Step 4: Create multi-sector bus architecture
        print("\n4. Creating multi-sector bus architecture...")
        sector_buses = create_multi_sector_buses(network, regions)
        setup_summary['multi_sector_buses'] = {
            sector: len(buses) for sector, buses in sector_buses.items()
        }
        
        # Step 5: Create sector coupling links with CSV cost data
        print("\n5. Creating sector coupling links...")
        coupling_links = create_sector_coupling_links(network, sector_buses, cost_data)
        setup_summary['sector_coupling_links'] = coupling_links
        
        # Step 6: Add multi-sector storage
        print("\n6. Adding multi-sector storage...")
        storage_counts = add_multi_sector_storage(network, sector_buses)
        setup_summary['multi_sector_storage'] = storage_counts
        
        # Step 7: Set up snapshots from Load.csv
        if 'electricity_demand' in time_series_data:
            print("\n7. Setting up snapshots from Load.csv...")
            load_df = time_series_data['electricity_demand']
            network.set_snapshots(load_df.index)
            setup_summary['snapshots_source'] = 'Load.csv'
            setup_summary['snapshots_count'] = len(load_df.index)
            print(f"   Set {len(load_df.index)} snapshots from Load.csv")
        
        # Step 8: Add multi-sector loads
        print("\n8. Adding multi-sector loads...")
        loads_added = add_multi_sector_loads_from_csv(network, sector_buses, time_series_data)
        setup_summary['multi_sector_loads'] = loads_added
        
        # Step 9: Add H2 pipeline infrastructure
        if 'h2_pipelines' in infrastructure_data:
            print("\n9. Adding H2 pipeline infrastructure...")
            h2_links = add_h2_pipeline_network(network, infrastructure_data['h2_pipelines'], sector_buses)
            setup_summary['h2_pipelines'] = h2_links
            
        # Step 10: Update summary
        setup_summary['total_buses'] = len(network.buses)
        setup_summary['total_links'] = len(network.links)
        setup_summary['total_loads'] = len(network.loads)
        setup_summary['total_stores'] = len(network.stores)
        
        # Step 11: Sync any missing carriers from network components
        print("\n11. Syncing network carriers...")
        sync_network_carriers(network)
        
    except Exception as e:
        logger.error(f"Error setting up enhanced flow network: {e}")
        raise
    
    print("\nEnhanced multi-sector flow network with CSV integration complete!")
    return setup_summary


def add_multi_sector_loads_from_csv(network: Network, sector_buses: Dict[str, List[str]], 
                                   time_series_data: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    """
    Add multi-sector loads using CSV time series data following PyPSA Load patterns.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add loads to
    sector_buses : Dict[str, List[str]]
        Dictionary mapping sectors to bus names
    time_series_data : Dict[str, pd.DataFrame]
        Time series data from load_time_series_data()
        
    Returns
    -------
    Dict[str, int]
        Dictionary with load counts by sector
    """
    load_counts = {'electricity': 0, 'hydrogen': 0, 'ammonia': 0}
    
    try:
        # Add electricity loads from Load.csv
        if 'electricity_demand' in time_series_data:
            elec_demand = time_series_data['electricity_demand']
            
            for col in elec_demand.columns:
                region = extract_region_from_column(col)
                elec_bus = f"{region}_electricity"
                
                if elec_bus in network.buses.index:
                    load_name = f"{region}_elec_load"
                    if load_name not in network.loads.index:
                        network.add('Load', load_name,
                                  bus=elec_bus,
                                  p_set=elec_demand[col],
                                  carrier='electricity')
                        load_counts['electricity'] += 1
        
        # Add hydrogen loads from H2_Demand_With_Blending.csv
        if 'hydrogen_demand' in time_series_data:
            h2_demand = time_series_data['hydrogen_demand']
            
            for col in h2_demand.columns:
                region = extract_region_from_column(col)
                h2_bus = f"{region}_hydrogen"
                
                if h2_bus in network.buses.index:
                    load_name = f"{region}_h2_load"
                    if load_name not in network.loads.index:
                        network.add('Load', load_name,
                                  bus=h2_bus,
                                  p_set=h2_demand[col],
                                  carrier='hydrogen')
                        load_counts['hydrogen'] += 1
        
        # Add basic ammonia demand (if no specific CSV available)
        for nh3_bus in sector_buses.get('ammonia', []):
            region = nh3_bus.replace('_ammonia', '')
            load_name = f"{region}_nh3_load"
            if load_name not in network.loads.index and len(network.snapshots) > 0:
                # Create basic demand profile
                base_demand = 50  # MW base demand
                demand_profile = pd.Series([base_demand] * len(network.snapshots), 
                                         index=network.snapshots)
                network.add('Load', load_name,
                          bus=nh3_bus,
                          p_set=demand_profile,
                          carrier='ammonia')
                load_counts['ammonia'] += 1
                
    except Exception as e:
        logger.warning(f"Error adding multi-sector loads: {e}")
    
    logger.info(f"Added multi-sector loads: {load_counts}")
    return load_counts


def add_h2_pipeline_network(network: Network, pipelines_df: pd.DataFrame, 
                           sector_buses: Dict[str, List[str]]) -> int:
    """
    Add hydrogen pipeline network from H2Pipelines.csv following PyPSA Link patterns.
    
    Parameters
    ----------
    network : Network
        PyPSA network to add pipelines to
    pipelines_df : pd.DataFrame
        DataFrame from H2Pipelines.csv with columns: Name, Build Cost, FO&M Charge, Efficiency
    sector_buses : Dict[str, List[str]]
        Dictionary mapping sectors to bus names
        
    Returns
    -------
    int
        Number of H2 pipelines added
    """
    pipelines_added = 0
    hydrogen_buses = sector_buses.get('hydrogen', [])
    
    # Create lookup for hydrogen buses by region
    h2_bus_lookup = {}
    for bus in hydrogen_buses:
        region = bus.replace('_hydrogen', '')
        h2_bus_lookup[region] = bus
    
    try:
        for _, pipeline in pipelines_df.iterrows():
            pipeline_name = pipeline['Name']
            
            # Parse pipeline name to extract source and destination regions
            # Expected format: H2PipelineAF-AGO-AF-COD
            if 'H2Pipeline' in pipeline_name:
                # Remove prefix and split by remaining parts
                regions_part = pipeline_name.replace('H2Pipeline', '')
                # Split by potential separators
                if '-' in regions_part:
                    parts = regions_part.split('-')
                    if len(parts) >= 4:  # AF-AGO-AF-COD format
                        source_region = f"{parts[0]}-{parts[1]}" # AF-AGO
                        dest_region = f"{parts[2]}-{parts[3]}" # AF-COD
                        
                        source_bus = h2_bus_lookup.get(source_region)
                        dest_bus = h2_bus_lookup.get(dest_region)
                        
                        if source_bus and dest_bus and source_bus in network.buses.index and dest_bus in network.buses.index:
                            link_name = f"h2_pipeline_{source_region}_{dest_region}"
                            
                            if link_name not in network.links.index:
                                # Extract pipeline properties
                                capital_cost = float(pipeline.get('Build Cost', 1000000)) / 1000  # Convert to per MW
                                fixed_cost = float(pipeline.get('FO&M Charge', 10000)) / 1000    # Convert to per MW/year
                                efficiency = float(pipeline.get('Efficiency', 98)) / 100         # Convert to decimal
                                
                                network.add('Link', link_name,
                                          bus0=source_bus,
                                          bus1=dest_bus,
                                          carrier='hydrogen_transport',
                                          capital_cost=capital_cost,
                                          marginal_cost=fixed_cost / 8760,  # Convert annual to hourly
                                          efficiency=efficiency,
                                          p_nom_extendable=True)
                                pipelines_added += 1
                                
    except Exception as e:
        logger.warning(f"Error adding H2 pipelines: {e}")
    
    logger.info(f"Added {pipelines_added} H2 pipelines")
    return pipelines_added


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


# ============================================================================
# MaREI CSV Data Integration Function
# ============================================================================

def setup_marei_csv_network(
    network: Network, 
    db: PlexosDB, 
    csv_data_path: str,
    infrastructure_scenario: str = "PCI",
    pricing_scheme: str = "Production",
    generators_as_links: bool = False
) -> Dict[str, Any]:
    """
    Enhanced MaREI network setup with CSV data integration.
    
    This function creates a comprehensive EU multi-sector model by combining:
    - PLEXOS database topology and generator data
    - MaREI CSV data for detailed demand profiles and infrastructure
    - PyPSA multi-sector patterns for gas/electricity coupling
    
    Parameters
    ----------
    network : Network
        PyPSA network to populate
    db : PlexosDB
        PLEXOS database connection
    csv_data_path : str
        Path to MaREI CSV Files directory
    infrastructure_scenario : str, default "PCI"
        Infrastructure scenario ('PCI', 'High', 'Low')
    pricing_scheme : str, default "Production" 
        Gas pricing scheme ('Production', 'Postage', 'Trickle', 'Uniform')
    generators_as_links : bool, default False
        If True, represent conventional generators as fuel→electric Links
        
    Returns
    -------
    Dict[str, Any]
        Comprehensive setup summary with CSV integration statistics
    """
    from .csv_loaders import load_marei_full_dataset
    
    print("Setting up MaREI EU multi-sector network with CSV data integration...")
    print(f"  Infrastructure scenario: {infrastructure_scenario}")
    print(f"  Gas pricing scheme: {pricing_scheme}")
    print(f"  Generators as links: {generators_as_links}")
    
    setup_summary = {
        'network_type': 'marei_csv_enhanced',
        'sectors': ['Electricity', 'Gas'],
        'csv_data_loaded': False,
        'csv_integration': {},
        'infrastructure_scenario': infrastructure_scenario,
        'pricing_scheme': pricing_scheme,
        'electricity': {'buses': 0, 'generators': 0, 'loads': 0, 'lines': 0, 'storage': 0},
        'gas': {'buses': 0, 'pipelines': 0, 'storage': 0, 'demand': 0, 'fields': 0, 'plants': 0},
        'sector_coupling': {'generators_as_links': generators_as_links, 'fuel_types': []},
        'eu_countries': []
    }
    
    try:
        # Step 1: Load complete MaREI CSV dataset
        print("\n1. Loading MaREI CSV dataset...")
        marei_data = load_marei_full_dataset(
            csv_base_dir=csv_data_path,
            scenario=infrastructure_scenario,
            pricing_scheme=pricing_scheme
        )
        
        if marei_data:
            setup_summary['csv_data_loaded'] = True
            setup_summary['csv_integration'] = {
                'data_categories': len(marei_data),
                'available_datasets': list(marei_data.keys())
            }
            print(f"  ✓ Loaded {len(marei_data)} data categories")
        
        # Step 2: Set up basic network structure from PlexosDB
        print("\n2. Setting up base network structure from PlexosDB...")
        
        # Add carriers automatically discovered from database
        print("   Discovering carriers from PLEXOS database...")
        from .core import discover_carriers_from_db
        carriers_to_add = discover_carriers_from_db(db)
        
        added_carriers = []
        for carrier in carriers_to_add:
            if carrier not in network.carriers.index:
                network.add('Carrier', carrier)
                added_carriers.append(carrier)
        print(f"   Added {len(added_carriers)} carriers automatically from database")
        
        # Get electricity buses from PlexosDB
        node_objects = get_objects_by_class_name(db, 'Node')
        eu_countries = set()
        
        for node_name in node_objects:
            # Extract country code from node name
            country_code = node_name[:2] if len(node_name) >= 2 else node_name
            eu_countries.add(country_code)
            
            if node_name not in network.buses.index:
                network.add('Bus', node_name, carrier='AC', v_nom=110)
        
        setup_summary['electricity']['buses'] = len(node_objects)
        setup_summary['eu_countries'] = sorted(list(eu_countries))
        print(f"  ✓ Added {len(node_objects)} electricity buses for {len(eu_countries)} countries")
        
        # Step 3: Create gas network buses for EU countries
        print("\n3. Creating gas network infrastructure...")
        
        # Create gas buses for each EU country
        gas_buses_created = 0
        for country in eu_countries:
            gas_bus_name = f"gas_{country}"
            if gas_bus_name not in network.buses.index:
                network.add('Bus', gas_bus_name, carrier='gas')
                gas_buses_created += 1
        
        setup_summary['gas']['buses'] = gas_buses_created
        print(f"  ✓ Created {gas_buses_created} gas buses")
        
        # Step 4: Set up snapshots from CSV data
        print("\n4. Configuring time snapshots...")
        
        # Use electricity demand CSV as snapshots source if available
        if 'electricity_demand' in marei_data:
            # Get any country's demand data to extract time structure
            elec_demand = next(iter(marei_data['electricity_demand'].values()))
            if hasattr(elec_demand, 'index') and len(elec_demand.index) > 0:
                network.set_snapshots(elec_demand.index)
                print(f"  ✓ Set {len(elec_demand.index)} snapshots from electricity demand CSV")
            else:
                # Fall back to full year
                snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
                network.set_snapshots(snapshots)
                print(f"  ✓ Set {len(snapshots)} snapshots (full year default)")
        else:
            # Fall back to full year
            snapshots = pd.date_range('2030-01-01', '2030-12-31 23:00', freq='h')
            network.set_snapshots(snapshots)
            print(f"  ✓ Set {len(snapshots)} snapshots (full year default)")
        
        # Step 5: Port electricity sector using existing modules
        print("\n5. Setting up electricity sector from PlexosDB...")
        
        gen_summary = port_generators(
            network=network,
            db=db,
            generators_as_links=generators_as_links,
            fuel_bus_prefix="fuel_"
        )
        setup_summary['electricity']['generators'] = len(network.generators)
        
        # Port transmission links
        port_links(network=network, db=db)
        setup_summary['electricity']['lines'] = len(network.links)
        
        # Port electricity storage
        try:
            add_storage(network=network, db=db)
            setup_summary['electricity']['storage'] = len(network.storage_units)
        except Exception as e:
            logger.warning(f"Storage porting failed: {e}")
            setup_summary['electricity']['storage'] = 0
        
        print(f"  ✓ Electricity sector: {setup_summary['electricity']}")
        
        # Step 6: Integrate gas and electricity demand from CSV
        print("\n6. Integrating CSV demand data...")
        
        demand_stats = add_marei_csv_loads(network, marei_data, eu_countries)
        setup_summary['electricity']['loads'] += demand_stats.get('electricity_loads', 0)
        setup_summary['gas']['demand'] = demand_stats.get('gas_loads', 0)
        
        # Step 7: Add gas infrastructure from CSV
        print("\n7. Adding gas infrastructure from CSV...")
        
        gas_infra_stats = add_marei_gas_infrastructure(network, marei_data, eu_countries)
        setup_summary['gas'].update(gas_infra_stats)
        
        # Step 8: Set up gas sector from PlexosDB (enhanced with CSV)
        print("\n8. Setting up gas sector from PlexosDB...")
        
        gas_summary = port_gas_components(network=network, db=db, testing_mode=False)
        # Update with CSV-enhanced values
        for key, value in gas_summary.items():
            if key in setup_summary['gas'] and isinstance(value, int):
                setup_summary['gas'][key] += value
            else:
                setup_summary['gas'][key] = value
        
        # Step 9: Add sector coupling
        print("\n9. Setting up gas-electricity sector coupling...")
        
        coupling_stats = add_marei_sector_coupling(network, db, generators_as_links)
        setup_summary['sector_coupling'].update(coupling_stats)
        
        # Step 10: Apply gas pricing from CSV
        if 'gas_pricing' in marei_data and marei_data['gas_pricing']:
            print(f"\n10. Applying {pricing_scheme} gas pricing...")
            apply_marei_gas_pricing(network, marei_data['gas_pricing'], pricing_scheme)
        
        # Final summary
        total_buses = len(network.buses)
        total_generators = len(network.generators) 
        total_links = len(network.links)
        total_loads = len(network.loads)
        total_storage = len(network.storage_units) + len(network.stores)
        
        print(f"\n✓ MaREI CSV-enhanced network complete!")
        print(f"  Total components: {total_buses} buses, {total_generators} generators, {total_links} links, {total_loads} loads, {total_storage} storage")
        
        # Update final summary
        setup_summary.update({
            'total_buses': total_buses,
            'total_generators': total_generators,
            'total_links': total_links,
            'total_loads': total_loads,
            'total_storage': total_storage
        })
        
        # Sync any missing carriers from network components
        print("\nSyncing network carriers...")
        sync_network_carriers(network)
        
    except Exception as e:
        logger.error(f"Error setting up MaREI CSV network: {e}")
        raise
    
    return setup_summary


def add_marei_csv_loads(network: Network, marei_data: Dict, eu_countries: List[str]) -> Dict[str, int]:
    """Add electricity and gas loads from MaREI CSV data."""
    load_stats = {'electricity_loads': 0, 'gas_loads': 0, 'missing_countries': []}
    
    try:
        # Add electricity loads from Load Files CSV
        if 'electricity_demand' in marei_data:
            elec_demands = marei_data['electricity_demand']
            
            for country, demand_profile in elec_demands.items():
                # Find matching electricity bus
                elec_bus = None
                for bus_name in network.buses.index:
                    if (network.buses.at[bus_name, 'carrier'] == 'AC' and 
                        (bus_name.startswith(country) or bus_name.endswith(country))):
                        elec_bus = bus_name
                        break
                
                if elec_bus and len(demand_profile) > 0:
                    load_name = f"elec_load_{country}"
                    if load_name not in network.loads.index:
                        # Ensure demand profile matches network snapshots
                        if len(network.snapshots) == len(demand_profile):
                            demand_series = pd.Series(demand_profile.values, index=network.snapshots)
                        else:
                            # Resample or extend to match snapshots
                            demand_series = pd.Series(
                                [demand_profile.iloc[i % len(demand_profile)] for i in range(len(network.snapshots))],
                                index=network.snapshots
                            )
                        
                        network.add('Load', load_name, bus=elec_bus, p_set=demand_series, carrier='electricity')
                        load_stats['electricity_loads'] += 1
                        
                        if load_stats['electricity_loads'] <= 3:  # Debug first few
                            print(f"    ✓ Added electricity load for {country}: {demand_series.mean():.1f} MW avg")
                else:
                    load_stats['missing_countries'].append(f"{country} (elec)")
        
        # Add gas loads from Gas Demand CSV
        if 'gas_demand' in marei_data:
            gas_demands = marei_data['gas_demand']
            
            for country, demand_profile in gas_demands.items():
                gas_bus = f"gas_{country}"
                
                if gas_bus in network.buses.index and len(demand_profile) > 0:
                    load_name = f"gas_load_{country}"
                    if load_name not in network.loads.index:
                        # Ensure demand profile matches network snapshots
                        if len(network.snapshots) == len(demand_profile):
                            demand_series = pd.Series(demand_profile.values, index=network.snapshots)
                        else:
                            # Resample or extend to match snapshots
                            demand_series = pd.Series(
                                [demand_profile.iloc[i % len(demand_profile)] for i in range(len(network.snapshots))],
                                index=network.snapshots
                            )
                        
                        network.add('Load', load_name, bus=gas_bus, p_set=demand_series, carrier='gas')
                        load_stats['gas_loads'] += 1
                        
                        if load_stats['gas_loads'] <= 3:  # Debug first few
                            print(f"    ✓ Added gas load for {country}: {demand_series.mean():.1f} MW avg")
                else:
                    load_stats['missing_countries'].append(f"{country} (gas)")
        
        print(f"  ✓ Added {load_stats['electricity_loads']} electricity loads, {load_stats['gas_loads']} gas loads")
        if load_stats['missing_countries'][:5]:  # Show first 5 missing
            print(f"    Missing buses for: {load_stats['missing_countries'][:5]}...")
            
    except Exception as e:
        logger.warning(f"Error adding MaREI CSV loads: {e}")
    
    return load_stats


def add_marei_gas_infrastructure(network: Network, marei_data: Dict, eu_countries: List[str]) -> Dict[str, int]:
    """Add gas infrastructure from MaREI CSV data."""
    infra_stats = {'pipelines': 0, 'storage': 0, 'lng': 0, 'fields': 0}
    
    try:
        # Add gas pipelines from infrastructure CSV
        if 'infrastructure' in marei_data and 'flow' in marei_data['infrastructure']:
            flow_data = marei_data['infrastructure']['flow']
            
            if not flow_data.empty and len(flow_data.columns) > 1:
                # Parse pipeline connections from column names (country pairs)
                pipeline_connections = [col for col in flow_data.columns if '-' in col and col != 'Year']
                
                for connection in pipeline_connections:
                    if '-' in connection:
                        parts = connection.split('-')
                        if len(parts) == 2:
                            country1, country2 = parts[0], parts[1]
                            gas_bus1 = f"gas_{country1}"
                            gas_bus2 = f"gas_{country2}"
                            
                            if (gas_bus1 in network.buses.index and gas_bus2 in network.buses.index):
                                pipeline_name = f"gas_pipeline_{country1}_{country2}"
                                
                                if pipeline_name not in network.links.index:
                                    # Get capacity from CSV data (use 2030 value if available)
                                    capacity = 100  # Default MW
                                    if len(flow_data) > 0:
                                        capacity_value = flow_data[connection].iloc[0] if not flow_data[connection].empty else 100
                                        try:
                                            capacity = float(capacity_value) if capacity_value else 100
                                        except:
                                            capacity = 100
                                    
                                    network.add('Link', pipeline_name,
                                              bus0=gas_bus1, bus1=gas_bus2,
                                              p_nom=capacity, efficiency=0.95,
                                              carrier='gas_transport', marginal_cost=0.001)
                                    infra_stats['pipelines'] += 1
                
                print(f"    ✓ Added {infra_stats['pipelines']} gas pipelines")
        
        # Add gas storage from storage CSV files
        storage_files = ['storage_cap', 'storage_inj', 'storage_with']
        if 'infrastructure' in marei_data:
            for storage_type in storage_files:
                if storage_type in marei_data['infrastructure']:
                    storage_data = marei_data['infrastructure'][storage_type]
                    
                    if not storage_data.empty:
                        # Parse storage locations from column names
                        storage_locations = [col for col in storage_data.columns if col not in ['Year'] and len(col) <= 3]
                        
                        for country in storage_locations:
                            gas_bus = f"gas_{country}"
                            if gas_bus in network.buses.index:
                                storage_name = f"gas_storage_{country}"
                                
                                if storage_name not in network.stores.index:
                                    # Get storage capacity
                                    capacity = 1000  # Default MWh
                                    if len(storage_data) > 0 and country in storage_data.columns:
                                        cap_value = storage_data[country].iloc[0] if not storage_data[country].empty else 1000
                                        try:
                                            capacity = float(cap_value) if cap_value else 1000
                                        except:
                                            capacity = 1000
                                    
                                    network.add('Store', storage_name, bus=gas_bus,
                                              e_nom=capacity, e_cyclic=True,
                                              carrier='gas', capital_cost=50)
                                    infra_stats['storage'] += 1
                                    break  # Only add one storage per country
        
        # Add LNG terminals
        if 'infrastructure' in marei_data and 'lng' in marei_data['infrastructure']:
            lng_data = marei_data['infrastructure']['lng']
            
            if not lng_data.empty:
                lng_countries = [col for col in lng_data.columns if col not in ['Year'] and len(col) <= 3]
                
                for country in lng_countries:
                    gas_bus = f"gas_{country}"
                    if gas_bus in network.buses.index:
                        lng_name = f"lng_terminal_{country}"
                        
                        if lng_name not in network.stores.index:
                            # Get LNG capacity
                            capacity = 500  # Default MWh
                            if country in lng_data.columns and len(lng_data) > 0:
                                cap_value = lng_data[country].iloc[0] if not lng_data[country].empty else 500
                                try:
                                    capacity = float(cap_value) if cap_value else 500
                                except:
                                    capacity = 500
                            
                            network.add('Store', lng_name, bus=gas_bus,
                                      e_nom=capacity, e_cyclic=False,
                                      carrier='gas', capital_cost=100)
                            infra_stats['lng'] += 1
        
        print(f"    ✓ Gas infrastructure: {infra_stats['pipelines']} pipelines, {infra_stats['storage']} storage, {infra_stats['lng']} LNG terminals")
        
    except Exception as e:
        logger.warning(f"Error adding MaREI gas infrastructure: {e}")
    
    return infra_stats


def add_marei_sector_coupling(network: Network, db: PlexosDB, generators_as_links: bool) -> Dict[str, Any]:
    """Add enhanced sector coupling for MaREI model."""
    coupling_stats = {
        'gas_to_elec_links': 0,
        'gas_generators': 0,
        'efficiency_range': 'N/A',
        'fuel_types': []
    }
    
    try:
        # Get all gas and electricity buses
        gas_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'gas']
        elec_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'AC']
        
        # Create gas-to-electricity conversion links for each country
        country_pairs = []
        for gas_bus in gas_buses:
            if gas_bus.startswith('gas_'):
                country = gas_bus.replace('gas_', '')
                # Find matching electricity bus
                matching_elec_bus = None
                for elec_bus in elec_buses:
                    if elec_bus.startswith(country) or elec_bus.endswith(country):
                        matching_elec_bus = elec_bus
                        break
                
                if matching_elec_bus:
                    country_pairs.append((gas_bus, matching_elec_bus, country))
        
        # Add gas-to-electricity conversion links
        efficiency_values = []
        for gas_bus, elec_bus, country in country_pairs:
            link_name = f"gas_to_elec_{country}"
            
            if link_name not in network.links.index:
                # Default gas plant efficiency
                efficiency = 0.45  # 45% efficiency for gas-fired generation
                efficiency_values.append(efficiency)
                
                network.add('Link', link_name,
                          bus0=gas_bus, bus1=elec_bus,
                          p_nom_extendable=True, efficiency=efficiency,
                          carrier='gas_to_electricity', marginal_cost=0.01)
                coupling_stats['gas_to_elec_links'] += 1
        
        if efficiency_values:
            min_eff = min(efficiency_values)
            max_eff = max(efficiency_values)
            coupling_stats['efficiency_range'] = f"{min_eff:.1%} - {max_eff:.1%}"
        
        coupling_stats['fuel_types'] = ['Natural Gas']
        
        # Additional generators-as-links processing if enabled
        if generators_as_links:
            generator_objects = get_objects_by_class_name(db, 'Generator')
            
            for gen_name in generator_objects[:10]:  # Process first 10 for performance
                props = get_object_properties_by_name(db, 'Generator', gen_name)
                memberships = get_object_memberships(db, 'Generator', gen_name)
                
                # Check for gas connection
                gas_connected = False
                elec_node = None
                
                for membership in memberships:
                    if membership.get('class') == 'Node':
                        elec_node = membership.get('name')
                
                for prop in props:
                    if prop.get('property') == 'Fuel' and 'gas' in str(prop.get('value', '')).lower():
                        gas_connected = True
                        break
                
                if gas_connected and elec_node and elec_node in network.buses.index:
                    # Find matching gas bus for this generator's country
                    country = elec_node[:2] if len(elec_node) >= 2 else elec_node
                    gas_bus = f"gas_{country}"
                    
                    if gas_bus in network.buses.index:
                        gen_link_name = f"gen_link_{gen_name}"
                        if gen_link_name not in network.links.index:
                            network.add('Link', gen_link_name,
                                      bus0=gas_bus, bus1=elec_node,
                                      p_nom=100, efficiency=0.45,
                                      carrier='gas_generation')
                            coupling_stats['gas_generators'] += 1
        
        print(f"    ✓ Sector coupling: {coupling_stats['gas_to_elec_links']} gas-to-elec links")
        if generators_as_links and coupling_stats['gas_generators'] > 0:
            print(f"    ✓ Generator links: {coupling_stats['gas_generators']} gas generator links")
            
    except Exception as e:
        logger.warning(f"Error adding MaREI sector coupling: {e}")
    
    return coupling_stats


def apply_marei_gas_pricing(network: Network, pricing_data: Dict, pricing_scheme: str) -> None:
    """Apply gas pricing from MaREI CSV to gas buses and links."""
    try:
        if pricing_scheme.lower() not in pricing_data:
            print(f"    Warning: Pricing scheme '{pricing_scheme}' not found in data")
            return
        
        pricing_df = pricing_data[pricing_scheme.lower()]
        
        # Apply base gas pricing to gas buses (set marginal costs)
        gas_buses = [bus for bus in network.buses.index if network.buses.at[bus, 'carrier'] == 'gas']
        base_gas_price = 30.0  # Default €/MWh
        
        if len(pricing_df) > 0 and 'price' in pricing_df.columns:
            base_gas_price = float(pricing_df['price'].iloc[0]) if not pricing_df['price'].empty else 30.0
        
        # Update marginal costs for gas-related components
        gas_links = [link for link in network.links.index if 'gas' in link.lower()]
        for link in gas_links:
            if network.links.at[link, 'carrier'] in ['gas_transport', 'gas_to_electricity']:
                current_cost = network.links.at[link, 'marginal_cost']
                network.links.at[link, 'marginal_cost'] = max(current_cost, base_gas_price * 0.01)
        
        print(f"    ✓ Applied {pricing_scheme} pricing: {base_gas_price:.1f} €/MWh base price")
        
    except Exception as e:
        logger.warning(f"Error applying gas pricing: {e}")