"""
Data-driven model creation interface for PLEXOS-PyPSA conversion.

This module provides a unified interface for creating PyPSA models from PLEXOS
data using automatic file path discovery from the database.
"""

from pathlib import Path
from typing import Dict, Optional

import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.db.discovery import discover_model_paths
from plexos_pypsa.network.core import setup_network


def create_model_from_xml(
    xml_file_path: str,
    main_directory: Optional[str] = None,
    demand_assignment_strategy: Optional[str] = None,
    target_node: Optional[str] = None,
    aggregate_node_name: Optional[str] = None,
    **override_paths,
) -> pypsa.Network:
    """
    Create a PyPSA model from PLEXOS XML using automatic data discovery.

    This function automatically discovers data file dependencies from the PLEXOS
    database and creates a complete PyPSA network model.

    Parameters
    ----------
    xml_file_path : str
        Path to the PLEXOS XML file
    main_directory : str, optional
        Main directory containing the model data. If not provided, uses the
        directory containing the XML file
    demand_assignment_strategy : str, optional
        Strategy for demand assignment: 'per_node', 'target_node', 'aggregate_node'
        If not provided, will be auto-inferred from discovered files
    target_node : str, optional
        Target node name for 'target_node' strategy
    aggregate_node_name : str, optional
        Name for aggregate node in 'aggregate_node' strategy
    **override_paths
        Optional path overrides for specific data types:
        - demand_source: Override demand files path
        - snapshots_source: Override snapshots source path
        - vre_profiles_path: Override VRE profiles path
        - inflow_path: Override hydro inflows path
        - timeslice_csv: Override timeslice CSV file path

    Returns
    -------
    pypsa.Network
        Configured PyPSA network with all components

    Examples
    --------
    Basic usage with automatic discovery:
    >>> network = create_model_from_xml("model.xml")

    With explicit main directory:
    >>> network = create_model_from_xml("path/to/model.xml",
    ...                                 main_directory="/data/model_root")

    With demand strategy override:
    >>> network = create_model_from_xml("model.xml",
    ...                                 demand_assignment_strategy="target_node",
    ...                                 target_node="SEM")

    With path overrides:
    >>> network = create_model_from_xml("model.xml",
    ...                                 demand_source="/custom/demand/path",
    ...                                 vre_profiles_path="/custom/vre/path")
    """
    # Determine main directory
    if main_directory is None:
        main_directory = str(Path(xml_file_path).parent)

    print(f"Creating model from: {xml_file_path}")
    print(f"Main directory: {main_directory}")

    # Load PlexosDB from XML file
    print("Loading PLEXOS database...")
    db = PlexosDB.from_xml(xml_file_path)

    # Discover data paths from database
    discovery_result = discover_model_paths(db, main_directory)

    # Print discovery summary
    print("\nDiscovered model structure:")
    for key, value in discovery_result["structure_info"].items():
        print(f"  {key}: {value}")

    # Prepare setup_network arguments
    setup_args = discovery_result["setup_paths"].copy()

    # Apply any overrides
    for key, value in override_paths.items():
        if value is not None:
            setup_args[key] = value
            print(f"Path override: {key} = {value}")

    # Determine demand assignment strategy if not provided
    if demand_assignment_strategy is None:
        structure_info = discovery_result["structure_info"]

        if structure_info["demand_pattern"] == "per_node":
            demand_assignment_strategy = "per_node"
            print("Auto-detected demand strategy: per_node")
        elif target_node:
            demand_assignment_strategy = "target_node"
            print(f"Using target_node strategy with node: {target_node}")
        elif aggregate_node_name:
            demand_assignment_strategy = "aggregate_node"
            print(f"Using aggregate_node strategy with name: {aggregate_node_name}")
        else:
            # Default fallback
            demand_assignment_strategy = "per_node"
            print("Using default demand strategy: per_node")

    # Add strategy-specific arguments
    if demand_assignment_strategy == "target_node" and target_node:
        setup_args["target_node"] = target_node
    elif demand_assignment_strategy == "aggregate_node" and aggregate_node_name:
        setup_args["aggregate_node_name"] = aggregate_node_name

    # Initialize PyPSA network
    network = pypsa.Network()

    # Set up complete network
    print(f"\nSetting up network with strategy: {demand_assignment_strategy}")
    print("Setup arguments:")
    for key, value in setup_args.items():
        print(f"  {key}: {value}")

    setup_summary = setup_network(network, db, **setup_args)

    # Print setup summary
    print("\nNetwork setup complete:")
    print(f"  Demand assignment: {setup_summary.get('mode', 'unknown')}")
    print(f"  Format type: {setup_summary.get('format_type', 'unknown')}")

    if setup_summary.get("format_type") == "iteration":
        print(
            f"  Iterations processed: {setup_summary.get('iterations_processed', 'N/A')}"
        )

    print(f"  Loads created: {setup_summary.get('loads_added', 0)}")
    if setup_summary.get("loads_skipped", 0) > 0:
        print(f"  Loads skipped: {setup_summary['loads_skipped']}")

    print(f"  Total buses: {len(network.buses)}")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total links: {len(network.links)}")
    print(f"  Total storage units: {len(network.storage_units)}")

    if (
        hasattr(network.storage_units_t, "inflow")
        and len(network.storage_units_t.inflow.columns) > 0
    ):
        print(f"  Storage with inflows: {len(network.storage_units_t.inflow.columns)}")

    print(f"  Total snapshots: {len(network.snapshots)}")

    return network


def create_aemo_model_data_driven(
    xml_file_path: str, main_directory: Optional[str] = None
) -> pypsa.Network:
    """
    Create AEMO model using data-driven approach.

    Parameters
    ----------
    xml_file_path : str
        Path to AEMO PLEXOS XML file
    main_directory : str, optional
        Main directory containing model data

    Returns
    -------
    pypsa.Network
        Configured AEMO PyPSA network
    """
    return create_model_from_xml(
        xml_file_path=xml_file_path,
        main_directory=main_directory,
        demand_assignment_strategy="per_node",  # AEMO uses traditional per-node assignment
    )


def create_caiso_model_data_driven(
    xml_file_path: str, main_directory: Optional[str] = None
) -> pypsa.Network:
    """
    Create CAISO model using data-driven approach.

    Parameters
    ----------
    xml_file_path : str
        Path to CAISO PLEXOS XML file
    main_directory : str, optional
        Main directory containing model data

    Returns
    -------
    pypsa.Network
        Configured CAISO PyPSA network
    """
    return create_model_from_xml(
        xml_file_path=xml_file_path,
        main_directory=main_directory,
        demand_assignment_strategy="aggregate_node",
        aggregate_node_name="CAISO_Load_Aggregate",
    )


def create_sem_model_data_driven(
    xml_file_path: str, main_directory: Optional[str] = None
) -> pypsa.Network:
    """
    Create SEM model using data-driven approach.

    Parameters
    ----------
    xml_file_path : str
        Path to SEM PLEXOS XML file
    main_directory : str, optional
        Main directory containing model data

    Returns
    -------
    pypsa.Network
        Configured SEM PyPSA network
    """
    return create_model_from_xml(
        xml_file_path=xml_file_path,
        main_directory=main_directory,
        demand_assignment_strategy="target_node",
        target_node="SEM",
    )


def validate_discovered_paths(
    discovery_result: Dict, main_directory: str
) -> Dict[str, bool]:
    """
    Validate that discovered file paths actually exist.

    Parameters
    ----------
    discovery_result : Dict
        Result from discover_model_paths()
    main_directory : str
        Main model directory

    Returns
    -------
    Dict[str, bool]
        Dictionary mapping path types to existence status
    """
    validation_results = {}

    resolved_paths = discovery_result["resolved_paths"]

    for path_type, paths in resolved_paths.items():
        if not paths:
            validation_results[path_type] = False
            continue

        # Check if at least one file of this type exists
        exists = any(Path(path).exists() for path in paths)
        validation_results[path_type] = exists

        if not exists:
            print(f"Warning: No existing files found for {path_type}")
            print(f"  Searched paths: {paths[:3]}...")  # Show first 3 paths

    return validation_results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_driven.py <xml_file_path> [main_directory]")
        sys.exit(1)

    xml_path = sys.argv[1]
    main_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Creating model from: {xml_path}")
    network = create_model_from_xml(xml_path, main_dir)

    print("\nModel created successfully!")
    print(
        f"Network summary: {len(network.buses)} buses, {len(network.generators)} generators"
    )
