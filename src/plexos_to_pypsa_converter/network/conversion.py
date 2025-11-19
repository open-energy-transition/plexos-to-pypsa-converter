"""Unified factory for creating PyPSA models from PLEXOS data.

This module provides a single entry point for creating all supported model types
from the MODEL_REGISTRY, handling electricity-only, multi-sector gas-electric,
and multi-sector flow models with a consistent interface.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.network.core import setup_network
from plexos_to_pypsa_converter.network.core_csv import setup_network_csv
from plexos_to_pypsa_converter.utils.csv_export import (
    check_csv_export_exists,
    export_csvs_from_xml,
)
from plexos_to_pypsa_converter.utils.model_paths import (
    get_model_directory,
    get_model_xml_path,
)


def _merge_configs(default_config: dict, overrides: dict) -> dict:
    """Merge default configuration with user overrides.

    Parameters
    ----------
    default_config : dict
        Default configuration from MODEL_REGISTRY
    overrides : dict
        User-provided configuration overrides

    Returns
    -------
    dict
        Merged configuration with overrides taking precedence
    """
    merged = default_config.copy()
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def _resolve_cross_model_dependency(model_id: str, dependency_key: str) -> str | None:
    """Resolve cross-model dependencies like VRE profiles from other models.

    Parameters
    ----------
    model_id : str
        ID of the current model being created
    dependency_key : str
        Key for the dependency (e.g., "vre_profiles_model_id")

    Returns
    -------
    str or None
        Resolved path to the dependency data, or None if not found
    """
    registry_entry = MODEL_REGISTRY.get(model_id, {})
    default_config = registry_entry.get("default_config", {})
    cross_deps = default_config.get("cross_model_dependencies", {})

    if dependency_key in cross_deps:
        dep_model_id = cross_deps[dependency_key]
        dep_model_dir = get_model_directory(dep_model_id)
        if dep_model_dir:
            return str(dep_model_dir)

    return None


def _create_electricity_model(
    model_id: str,
    xml_file: Path,
    model_dir: Path,
    config: dict,
    db: PlexosDB,
    use_csv: bool = False,
) -> tuple[pypsa.Network, dict]:
    """Create electricity-only PyPSA model.

    Parameters
    ----------
    model_id : str
        Model identifier from MODEL_REGISTRY
    xml_file : Path
        Path to PLEXOS XML file
    model_dir : Path
        Path to model directory
    config : dict
        Merged configuration (defaults + overrides)
    db : PlexosDB
        Loaded PLEXOS database
    use_csv : bool, default False
        If True, use CSV-based conversion approach

    Returns
    -------
    tuple[pypsa.Network, dict]
        Network and setup summary
    """
    network = pypsa.Network()

    if use_csv:
        # CSV-based path
        csv_dir = config.get("csv_dir")
        if csv_dir is None:
            msg = "csv_dir must be provided when use_csv=True"
            raise ValueError(msg)

        # Auto-detection logic for snapshots_source (same as PlexosDB path)
        snapshots_source = config.get("snapshots_source")
        demand_source = config.get("demand_source")

        # Check for explicit demand file configuration
        if config.get("demand_file") and snapshots_source is None:
            demand_file_path = model_dir / config["demand_file"]
            if demand_file_path.exists():
                snapshots_source = str(demand_file_path)
                demand_source = str(demand_file_path)
            else:
                msg = f"Configured demand_file not found: {demand_file_path}"
                raise FileNotFoundError(msg)

        # Auto-determine paths if not explicitly provided
        if snapshots_source is None:
            # Check for common snapshot directories
            candidates = [
                "Traces/demand",  # AEMO pattern
                "demand",  # Simple pattern
                "LoadProfile",  # CAISO pattern
                "load",  # Alternative pattern
            ]
            for candidate in candidates:
                candidate_path = model_dir / candidate
                if candidate_path.exists():
                    snapshots_source = str(candidate_path)
                    break

        if demand_source is None:
            demand_source = snapshots_source

        print("Setting up electricity model from CSVs...")
        setup_summary = setup_network_csv(
            network=network,
            csv_dir=csv_dir,
            db=db,
            snapshots_source=snapshots_source,
            demand_source=demand_source,
            timeslice_csv=config.get("timeslice_csv"),
            vre_profiles_path=config.get("vre_profiles_path"),
            model_name=config.get("model_name"),
            inflow_path=config.get("inflow_path"),
            target_node=config.get("target_node"),
            aggregate_node_name=config.get("aggregate_node_name"),
            demand_assignment_strategy=config.get(
                "demand_assignment_strategy", "per_node"
            ),
            demand_target_node=config.get("demand_target_node"),
            transmission_as_lines=config.get("transmission_as_lines", False),
            bidirectional_links=config.get("bidirectional_links", True),
            load_scenario=config.get("load_scenario"),
        )

        return network, setup_summary

    # Original PlexosDB-based path
    network = pypsa.Network()

    # Get demand assignment strategy
    strategy = config.get("demand_assignment_strategy", "per_node")

    # Build setup_network arguments
    setup_args = {
        "network": network,
        "db": db,
        "snapshots_source": config.get("snapshots_source"),
        "demand_source": config.get("demand_source"),
        "timeslice_csv": config.get("timeslice_csv"),
        "vre_profiles_path": config.get("vre_profiles_path"),
        "model_name": config.get("model_name"),
        "inflow_path": config.get("inflow_path"),
        "transmission_as_lines": config.get("transmission_as_lines", False),
    }

    # Check for explicit demand file configuration (takes precedence over auto-detection)
    if config.get("demand_file") and setup_args["snapshots_source"] is None:
        demand_file_path = model_dir / config["demand_file"]
        if demand_file_path.exists():
            setup_args["snapshots_source"] = str(demand_file_path)
            setup_args["demand_source"] = str(demand_file_path)
        else:
            msg = f"Configured demand_file not found: {demand_file_path}"
            raise FileNotFoundError(msg)

    # Auto-determine paths if not explicitly provided
    if setup_args["snapshots_source"] is None:
        # Check for common snapshot directories (including nested paths for AEMO)
        candidates = [
            "Traces/demand",  # AEMO pattern
            "demand",  # Simple pattern
            "LoadProfile",  # CAISO pattern
            "load",  # Alternative pattern
        ]
        for candidate in candidates:
            candidate_path = model_dir / candidate
            if candidate_path.exists():
                setup_args["snapshots_source"] = str(candidate_path)
                break

    if setup_args["demand_source"] is None:
        setup_args["demand_source"] = setup_args["snapshots_source"]

    # Resolve cross-model dependencies (e.g., VRE profiles from AEMO)
    if setup_args["vre_profiles_path"] is None:
        dep_path = _resolve_cross_model_dependency(model_id, "vre_profiles_model_id")
        if dep_path:
            # For cross-model VRE profiles, check for Traces subdirectory
            traces_path = Path(dep_path) / "Traces"
            if traces_path.exists():
                setup_args["vre_profiles_path"] = str(traces_path)
            else:
                setup_args["vre_profiles_path"] = dep_path
        else:
            # Fall back to model directory, checking for Traces subdirectory
            traces_path = model_dir / "Traces"
            if traces_path.exists():
                setup_args["vre_profiles_path"] = str(traces_path)
            else:
                setup_args["vre_profiles_path"] = str(model_dir)

    if setup_args["inflow_path"] is None:
        # Check for common hydro inflow directories
        for hydro_candidate in ["Traces/hydro", "hydro"]:
            hydro_path = model_dir / hydro_candidate
            if hydro_path.exists():
                setup_args["inflow_path"] = str(hydro_path)
                break

    # Add strategy-specific arguments
    if strategy == "target_node":
        setup_args["target_node"] = config.get("target_node")
    elif strategy == "aggregate_node":
        setup_args["aggregate_node_name"] = config.get("aggregate_node_name")

    # Add demand-specific target node (independent of strategy)
    if "demand_target_node" in config:
        setup_args["demand_target_node"] = config.get("demand_target_node")

    # Add load scenario parameter
    setup_args["load_scenario"] = config.get("load_scenario")

    # Set up network
    print(f"Setting up {strategy} electricity model...")
    setup_summary = setup_network(**setup_args)

    return network, setup_summary


def create_model(model_id: str, **config_overrides: dict) -> tuple[pypsa.Network, dict]:
    """Create a PyPSA model from PLEXOS data using MODEL_REGISTRY configuration.

    This is the unified factory function for creating all supported model types.
    It handles:
    - Electricity-only models (AEMO, CAISO, SEM, NREL, PLEXOS-World)
    - Multi-sector gas+electric models (MaREI-EU)
    - Multi-sector flow models (PLEXOS-MESSAGE)

    The function automatically:
    - Loads the model XML and database
    - Applies default configuration from MODEL_REGISTRY
    - Merges user overrides
    - Routes to appropriate setup function based on model_type
    - Handles cross-model dependencies (e.g., VRE profiles)

    Parameters
    ----------
    model_id : str
        Model identifier from MODEL_REGISTRY (e.g., "aemo-2024-isp-progressive-change")
    **config_overrides
        Optional configuration overrides. Available options depend on model type:

        Electricity models:
        - use_csv : bool
            Enable CSV-based conversion (auto-generates CSVs from XML if needed)
        - demand_assignment_strategy : str
            "per_node", "target_node", or "aggregate_node"
        - target_node : str
            Target node name for target_node strategy
        - aggregate_node_name : str
            Aggregate node name for aggregate_node strategy
        - model_name : str
            PLEXOS model name to use
        - snapshots_source : str
            Path to snapshots data
        - demand_source : str
            Path to demand data
        - vre_profiles_path : str
            Path to VRE profiles
        - inflow_path : str
            Path to hydro inflow data
        - timeslice_csv : str
            Path to timeslice CSV file
        - transmission_as_lines : bool
            Use Line components instead of Links
        - load_scenario : str, optional
            For demand data with multiple scenarios (iterations), specify which
            scenario to use. If None, defaults to first scenario.
            Example: "iteration_1" to select first scenario explicitly

        Gas+Electric models (MaREI-EU):
        - use_csv : bool
            Enable CSV data integration (default False)
        - csv_data_path : str
            Path to CSV Files directory
        - infrastructure_scenario : str
            "PCI", "High", or "Low"
        - pricing_scheme : str
            "Production", "Postage", "Trickle", or "Uniform"
        - generators_as_links : bool
            Represent generators as Links for sector coupling
        - testing_mode : bool
            Process limited subsets for testing

        Flow models (PLEXOS-MESSAGE):
        - use_csv : bool
            Enable CSV data integration (default from registry)
        - inputs_folder : str
            Path to Inputs folder with CSV data
        - testing_mode : bool
            Process limited subsets for testing

    Returns
    -------
    tuple[pypsa.Network, dict]
        Created PyPSA network and setup summary dictionary

    Raises
    ------
    ValueError
        If model_id is not in MODEL_REGISTRY
    FileNotFoundError
        If model data is not found (suggests running auto-download if recipe exists)

    Examples
    --------
    Create AEMO model with defaults:
    >>> network, summary = create_model("aemo-2024-isp-progressive-change")

    Create CAISO model (uses default aggregate_node strategy):
    >>> network, summary = create_model("caiso-irp23")

    Create SEM model with custom target node:
    >>> network, summary = create_model("sem-2024-2032", target_node="CustomNode")

    Create SEM model with CSV-based conversion (auto-generates CSVs if needed):
    >>> network, summary = create_model("sem-2024-2032", use_csv=True)

    Create MaREI model with CSV integration:
    >>> network, summary = create_model("marei-eu",
    ...                                  use_csv=True,
    ...                                  infrastructure_scenario="High")

    Create PLEXOS-MESSAGE model in testing mode:
    >>> network, summary = create_model("plexos-message", testing_mode=True)
    """
    # Validate model_id
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        msg = f"Unknown model_id: '{model_id}'. Available models: {available}"
        raise ValueError(msg)

    # Get model metadata
    model_metadata = MODEL_REGISTRY[model_id]
    model_type = model_metadata.get("model_type")
    default_config = model_metadata.get("default_config", {})

    # Check if model type is specified
    # Check if model type is specified
    if not model_type:
        msg = (
            f"Model '{model_id}' has no model_type in MODEL_REGISTRY. "
            "This model may not be fully configured for the factory pattern."
        )
        raise ValueError(msg)
    # Get XML file path
    xml_file = get_model_xml_path(model_id)
    if xml_file is None:
        recipe_available = "recipe" in model_metadata
        msg = (
            f"Model '{model_id}' not found in src/examples/data/. "
            "Please download and extract the model data."
        )
        if recipe_available:
            msg += (
                "\n\nTip: This model has an auto-download recipe. "
                "You can use the recipe system to automatically download it."
            )
        raise FileNotFoundError(msg)

    # Get model directory
    model_dir = get_model_directory(model_id)
    if model_dir is None:
        msg = f"Could not determine directory for model '{model_id}'"
        raise FileNotFoundError(msg)

    # Merge configurations
    config = _merge_configs(default_config, config_overrides)

    # Check for CSV usage and auto-generate if needed
    use_csv = config.get("use_csv", False)
    csv_dir = None

    if use_csv and model_type in ["electricity", "multi_sector_gas_electric"]:
        print(f"\nChecking for CSV exports in {model_dir}...")

        # Check if CSV export already exists
        csv_export_path = check_csv_export_exists(model_dir)

        if csv_export_path:
            print(f"Found existing CSV export at: {csv_export_path}")
            csv_dir = csv_export_path
        else:
            print("CSV export not found. Generating CSVs from XML...")
            print("This may take a few minutes...")

            # Generate CSVs from XML
            csv_output_dir = model_dir / "csvs_from_xml"
            try:
                export_csvs_from_xml(xml_file, csv_output_dir)
                print(f"Successfully exported CSVs to: {csv_output_dir}")
            except Exception as e:
                msg = f"Failed to export CSVs from XML: {e}"
                print(f"Error: {msg}")
                print("Falling back to PlexosDB-based approach")
                use_csv = False
            else:
                # Verify the export
                csv_export_path = check_csv_export_exists(model_dir)
                if not csv_export_path:
                    msg = "CSV export completed but verification failed"
                    print(f"Error: {msg}")
                    print("Falling back to PlexosDB-based approach")
                    use_csv = False
                else:
                    csv_dir = csv_export_path

        # Store csv_dir in config for use by _create_electricity_model
        if csv_dir:
            config["csv_dir"] = csv_dir

    print(f"Creating model: {model_metadata['name']}")
    print(f"Model type: {model_type}")
    print(f"XML file: {xml_file}")
    print(f"Model directory: {model_dir}")

    # Load PLEXOS database
    db = None
    if not use_csv:
        print("\nLoading PLEXOS database...")
        db = PlexosDB.from_xml(str(xml_file))

    # Route to appropriate creation function based on model type
    if model_type == "electricity":
        network, setup_summary = _create_electricity_model(
            model_id, xml_file, model_dir, config, db, use_csv=use_csv
        )
    else:
        msg = (
            f"Unknown model_type: '{model_type}' for model '{model_id}'. "
            "Supported types: electricity"
        )
        raise ValueError(msg)

    # Print summary
    print("\n" + "=" * 60)
    print(f"{model_metadata['name'].upper()} - SETUP COMPLETE")
    print("=" * 60)
    print(f"Total buses: {len(network.buses)}")
    print(f"Total generators: {len(network.generators)}")
    print(f"Total links: {len(network.links)}")
    print(f"Total storage units: {len(network.storage_units)}")
    print(f"Total stores: {len(network.stores)}")
    print(f"Total loads: {len(network.loads)}")
    print(f"Total snapshots: {len(network.snapshots)}")

    return network, setup_summary


def create_model_with_optimization(
    model_id: str,
    solver_name: str = "highs",
    snapshots_per_year: int | None = None,
    solver_options: dict | None = None,
    **config_overrides: dict,
) -> tuple[pypsa.Network, dict]:
    """Create and optimize a PyPSA model in one step.

    This convenience function creates a model using create_model() and then
    runs optimization with the specified solver configuration.

    Parameters
    ----------
    model_id : str
        Model identifier from MODEL_REGISTRY
    solver_name : str, default "highs"
        Solver to use for optimization
    snapshots_per_year : int, optional
        If provided, optimize on subset with this many snapshots per year.
        If None, optimizes on all snapshots.
    solver_options : dict, optional
        Solver-specific options to pass to network.optimize()
    **config_overrides
        Configuration overrides passed to create_model()

    Returns
    -------
    tuple[pypsa.Network, dict]
        Optimized network and setup summary

    Examples
    --------
    Create and optimize AEMO model with subset:
    >>> network, summary = create_model_with_optimization(
    ...     "aemo-2024-isp-progressive-change",
    ...     snapshots_per_year=50
    ... )

    Create and optimize with custom solver:
    >>> network, summary = create_model_with_optimization(
    ...     "sem-2024-2032",
    ...     solver_name="gurobi",
    ...     solver_options={"Threads": 4},
    ...     snapshots_per_year=60
    ... )
    """
    # Create model
    network, setup_summary = create_model(model_id, **config_overrides)

    # Select snapshots
    if snapshots_per_year is not None:
        print(f"\nSelecting {snapshots_per_year} snapshots per year...")
        snapshots_by_year: defaultdict[int, list] = defaultdict(list)
        for snap in network.snapshots:
            year = pd.Timestamp(snap).year
            if len(snapshots_by_year[year]) < snapshots_per_year:
                snapshots_by_year[year].append(snap)

        snapshots = [snap for snaps in snapshots_by_year.values() for snap in snaps]
        print(f"  Selected {len(snapshots)} total snapshots")
    else:
        snapshots = network.snapshots
        print(f"\nOptimizing with all {len(snapshots)} snapshots")

    # Run consistency check
    print("\nRunning consistency check...")
    network.consistency_check()
    print(" Consistency check passed")

    # Optimize
    print(f"\nOptimizing with {solver_name}...")
    optimize_args = {"solver_name": solver_name, "snapshots": snapshots}
    if solver_options:
        optimize_args["solver_options"] = solver_options

    network.optimize(**optimize_args)  # type: ignore
    print(" Optimization complete")
    print(f"  Objective value: {network.objective:.2f}")

    return network, setup_summary
