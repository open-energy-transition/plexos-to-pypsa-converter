"""Pre-defined workflow step implementations.

This module provides a library of standard workflow steps that can be composed
into model processing pipelines via registry workflow definitions.
"""

from pathlib import Path

import pypsa

from network.conversion import create_model
from network.generators_csv import (
    apply_generator_units_timeseries_csv,
    load_data_file_profiles_csv,
)
from network.outages import (
    apply_outage_schedule,
    build_outage_schedule,
    generate_stochastic_outages_csv,
    parse_explicit_outages_from_properties,
)
from network.storage_csv import add_storage_inflows_csv
from workflow.filters import resolve_filter_preset


def create_model_step(
    model_id: str,
    use_csv: bool = True,
    **create_model_kwargs,
) -> tuple[pypsa.Network, dict]:
    """Step: Initialize PyPSA network from PLEXOS model.

    Args:
        model_id: Model identifier from registry
        use_csv: Whether to use CSV-based data loading
        **create_model_kwargs: Additional arguments for create_model()

    Returns:
        (network, summary) tuple
    """
    network, summary = create_model(model_id, use_csv=use_csv, **create_model_kwargs)
    return network, {"create_model": summary}


def scale_p_min_pu_step(
    network: pypsa.Network,
    scaling_factor: float = 0.7,
) -> dict:
    """Step: Scale minimum generation constraints (p_min_pu) for all generators.

    Useful for models with tight must-run constraints that cause infeasibility.

    Args:
        network: PyPSA network (modified in-place)
        scaling_factor: Multiplier for p_min_pu values (default: 0.7)

    Returns:
        Summary dict with count of generators scaled
    """
    scaled_count = 0
    for gen in network.generators.index:
        if gen in network.generators_t.p_min_pu.columns:
            network.generators_t.p_min_pu[gen] *= scaling_factor
            scaled_count += 1

    return {
        "scale_p_min_pu": {
            "scaling_factor": scaling_factor,
            "generators_scaled": scaled_count,
        }
    }


def add_curtailment_link_step(
    network: pypsa.Network,
    bus_name: str = "SEM",
    p_nom: float = 5000,
    marginal_cost: float = 1000,
) -> dict:
    """Step: Add curtailment/slack link to absorb excess must-run generation.

    Creates a dummy bus and link that consumes excess power when minimum
    generation exceeds demand.

    Args:
        network: PyPSA network (modified in-place)
        bus_name: Source bus to connect curtailment link to
        p_nom: Maximum curtailment capacity (MW)
        marginal_cost: Penalty for using curtailment ($/MWh)

    Returns:
        Summary dict with curtailment link properties
    """
    # Add curtailment carrier if needed
    if "curtailment" not in network.carriers.index:
        network.add("Carrier", "curtailment")

    # Add dummy dump bus
    dump_bus = f"{bus_name}_curtailment_dump"
    network.add("Bus", dump_bus)

    # Add Link from source bus to dump
    link_name = f"Curtailment_{bus_name}"
    network.add(
        "Link",
        link_name,
        bus0=bus_name,
        bus1=dump_bus,
        p_nom=p_nom,
        marginal_cost=marginal_cost,
    )

    return {
        "add_curtailment_link": {
            "link_name": link_name,
            "bus0": bus_name,
            "bus1": dump_bus,
            "p_nom": p_nom,
            "marginal_cost": marginal_cost,
        }
    }


def load_vre_profiles_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    profiles_path: str | Path,
    property_name: str = "Rating",
    target_property: str = "p_max_pu",
    target_type: str = "generators_t",
    apply_mode: str = "replace",
    scenario: int = 1,
    generator_filter: str | None = None,
    carrier_mapping: dict | None = None,
    value_scaling: float = 1.0,
    manual_mappings: dict | None = None,
) -> dict:
    """Step: Load VRE generation profiles from CSV Data Files.

    Args:
        network: PyPSA network (modified in-place)
        csv_dir: Directory containing csvs_from_xml
        profiles_path: Base path to model directory
        property_name: PLEXOS property name (default: "Rating")
        target_property: PyPSA property to update (default: "p_max_pu")
        target_type: Target component type (default: "generators_t")
        apply_mode: How to apply profiles ("replace" or "dispatch")
        scenario: Scenario number to filter
        generator_filter: Filter preset name (e.g., "vre_only")
        carrier_mapping: Dict mapping categories to carrier names
        value_scaling: Scaling factor for profile values
        manual_mappings: Manual generator-to-profile mappings

    Returns:
        Summary dict from load_data_file_profiles_csv()
    """
    # Resolve filter preset
    filter_fn = resolve_filter_preset(generator_filter, network)

    summary = load_data_file_profiles_csv(
        network=network,
        csv_dir=csv_dir,
        profiles_path=profiles_path,
        property_name=property_name,
        target_property=target_property,
        target_type=target_type,
        apply_mode=apply_mode,
        scenario=scenario,
        generator_filter=filter_fn,
        carrier_mapping=carrier_mapping or {},
        value_scaling=value_scaling,
        manual_mappings=manual_mappings or {},
    )

    return {"load_vre_profiles": summary}


def add_storage_inflows_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    inflow_path: str | Path,
) -> dict:
    """Step: Add natural inflow time series to storage units (hydro).

    Args:
        network: PyPSA network (modified in-place)
        csv_dir: Directory containing csvs_from_xml
        inflow_path: Base path to model directory

    Returns:
        Summary dict from add_storage_inflows_csv()
    """
    summary = add_storage_inflows_csv(
        network=network,
        csv_dir=csv_dir,
        inflow_path=inflow_path,
    )

    return {"add_storage_inflows": summary}


def apply_generator_units_step(
    network: pypsa.Network,
    csv_dir: str | Path,
) -> dict:
    """Step: Apply generator Units time series (retirements, builds, capacity scaling).

    IMPORTANT: Must be called AFTER VRE profiles are loaded.

    Args:
        network: PyPSA network (modified in-place)
        csv_dir: Directory containing csvs_from_xml

    Returns:
        Summary dict from apply_generator_units_timeseries_csv()
    """
    summary = apply_generator_units_timeseries_csv(network, csv_dir)

    return {"apply_generator_units": summary}


def parse_outages_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    include_explicit: bool = True,
    explicit_property: str = "Units Out",
    include_forced: bool = True,
    include_maintenance: bool = True,
    generator_filter: str = "exclude_vre",
    random_seed: int = 42,
) -> dict:
    """Step: Parse explicit outages and generate stochastic outages, then apply to network.

    Args:
        network: PyPSA network (modified in-place)
        csv_dir: Directory containing csvs_from_xml
        include_explicit: Parse explicit outages from CSV property
        explicit_property: Property name for explicit outages
        include_forced: Generate Monte Carlo forced outages
        include_maintenance: Generate PASA-like maintenance outages
        generator_filter: Filter preset name (default: "exclude_vre")
        random_seed: Random seed for reproducibility

    Returns:
        Combined summary from explicit and stochastic outages
    """
    summary = {}

    # Get demand profile for intelligent maintenance scheduling
    try:
        demand = network.loads_t.p_set.sum(axis=1)
        has_demand = True
    except Exception:
        demand = None
        has_demand = False

    # Parse explicit outages
    explicit_events = []
    if include_explicit:
        explicit_events = parse_explicit_outages_from_properties(
            csv_dir=csv_dir,
            network=network,
            property_name=explicit_property,
        )
        summary["explicit_outages"] = len(explicit_events)

    # Generate stochastic outages
    filter_fn = resolve_filter_preset(generator_filter, network)

    stochastic_events = generate_stochastic_outages_csv(
        csv_dir=csv_dir,
        network=network,
        include_forced=include_forced,
        include_maintenance=include_maintenance,
        demand_profile=demand if has_demand else None,
        random_seed=random_seed,
        existing_outage_events=explicit_events if include_explicit else None,
        generator_filter=filter_fn,
    )
    summary["stochastic_outages"] = len(stochastic_events)

    # Build and apply outage schedule
    all_events = explicit_events + stochastic_events
    schedule = build_outage_schedule(all_events, network.snapshots)
    outage_summary = apply_outage_schedule(network, schedule)

    summary.update(outage_summary)

    return {"parse_outages": summary}


def optimize_step(
    network: pypsa.Network,
    year: int | None = None,
    solver_config: dict | None = None,
) -> dict:
    """Step: Run PyPSA network optimization.

    Args:
        network: PyPSA network to optimize
        year: Optional year to subset snapshots (None = all snapshots)
        solver_config: Solver configuration dict with:
            - solver_name: Solver to use (e.g., "gurobi")
            - solver_options: Dict of solver-specific options

    Returns:
        Summary dict with optimization status
    """
    # Run consistency check before optimization
    network.consistency_check()

    # Subset snapshots by year if specified
    if year is not None:
        snapshots = network.snapshots[network.snapshots.year == year]
    else:
        snapshots = network.snapshots

    # Apply default solver config if not provided
    if solver_config is None:
        solver_config = {
            "solver_name": "gurobi",
            "solver_options": {
                "Threads": 6,
                "Method": 2,  # barrier
                "Crossover": 0,
                "BarConvTol": 1.0e-5,
                "Seed": 123,
                "AggFill": 0,
                "PreDual": 0,
                "GURO_PAR_BARDENSETHRESH": 200,
            },
        }

    # Run optimization
    network.optimize(snapshots=snapshots, **solver_config)

    return {
        "optimize": {
            "status": network.results.status,
            "objective": network.objective,
            "snapshots_count": len(snapshots),
            "year_filter": year,
        }
    }


# Registry of all available step functions
STEP_REGISTRY = {
    "create_model": create_model_step,
    "scale_p_min_pu": scale_p_min_pu_step,
    "add_curtailment_link": add_curtailment_link_step,
    "load_vre_profiles": load_vre_profiles_step,
    "add_storage_inflows": add_storage_inflows_step,
    "apply_generator_units": apply_generator_units_step,
    "parse_outages": parse_outages_step,
    "optimize": optimize_step,
}
