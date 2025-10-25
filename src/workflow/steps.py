"""Pre-defined workflow step implementations for workflow system."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

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
from network.ramp import fix_outage_ramp_conflicts
from network.slack import add_slack_generators
from network.storage_csv import add_storage_inflows_csv
from workflow.filters import resolve_filter_preset


def create_model_step(
    model_id: str,
    use_csv: bool = True,
    **create_model_kwargs,
) -> tuple[pypsa.Network, dict]:
    """Step: Initialize PyPSA network from PLEXOS model."""
    network, summary = create_model(model_id, use_csv=use_csv, **create_model_kwargs)
    return network, {"create_model": summary}


def scale_p_min_pu_step(
    network: pypsa.Network,
    scaling_factor: float = 0.7,
) -> dict:
    """Step: Scale minimum generation constraints (p_min_pu) for all generators."""
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
    """Step: Add curtailment/slack link to absorb excess must-run generation."""
    if "curtailment" not in network.carriers.index:
        network.add("Carrier", "curtailment")
    dump_bus = f"{bus_name}_curtailment_dump"
    network.add("Bus", dump_bus)
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
    """Step: Load VRE generation profiles from CSV Data Files."""
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


def load_hydro_dispatch_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    profiles_path: str | Path,
    scenario: str | int = "Value",
    generator_filter: str = "hydro_only",
    load_rating: bool = True,
    load_min_stable: bool = True,
) -> dict:
    """Step: Load hydro dispatch profiles (Rating and Min Stable Level).

    This step loads time-varying dispatch schedules for run-of-river and dispatchable
    hydro generators. Unlike VRE profiles (which are capacity factors for intermittent
    generation), hydro dispatch profiles represent operational constraints and schedules.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    profiles_path : str | Path
        Base directory containing hydro dispatch profile CSV files
    scenario : str | int, default "Value"
        Which scenario column to use. Hydro dispatch is typically deterministic,
        so default is "Value". For stochastic hydro, pass scenario number (1, 2, etc.)
    generator_filter : str, default "hydro_only"
        Filter preset name (e.g., "hydro_only", "all")
    load_rating : bool, default True
        Load Rating profiles as p_max_pu
    load_min_stable : bool, default True
        Load Min Stable Level profiles as p_min_pu

    Returns
    -------
    dict
        Summary with processed/skipped/failed generator counts for each property
    """
    summary = {}
    filter_fn = resolve_filter_preset(generator_filter, network)

    # Load Rating profiles (p_max_pu) for hydro dispatch schedules
    if load_rating:
        rating_summary = load_data_file_profiles_csv(
            network=network,
            csv_dir=csv_dir,
            profiles_path=profiles_path,
            property_name="Rating",
            target_property="p_max_pu",
            target_type="generators_t",
            apply_mode="replace",
            scenario=scenario,
            generator_filter=filter_fn,
            carrier_mapping={"Hydro": "hydro", "ROR": "hydro"},
        )
        summary["rating"] = rating_summary

    # Load Min Stable Level profiles (p_min_pu) for must-run constraints
    if load_min_stable:
        min_summary = load_data_file_profiles_csv(
            network=network,
            csv_dir=csv_dir,
            profiles_path=profiles_path,
            property_name="Min Stable Level",
            target_property="p_min_pu",
            target_type="generators_t",
            apply_mode="replace",
            scenario=scenario,
            generator_filter=filter_fn,
        )
        summary["min_stable"] = min_summary

    return {"load_hydro_dispatch": summary}


def add_storage_inflows_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    inflow_path: str | Path,
) -> dict:
    """Step: Add natural inflow time series to storage units (hydro)."""
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
    """Step: Apply generator Units time series (retirements, builds, capacity scaling)."""
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
    """Step: Parse explicit outages and generate stochastic outages, then apply to network."""
    summary = {}
    try:
        demand = network.loads_t.p_set.sum(axis=1)
        has_demand = True
    except Exception:
        demand = None
        has_demand = False
    explicit_events = []
    filter_fn = resolve_filter_preset(generator_filter, network)
    if include_explicit:
        explicit_events = parse_explicit_outages_from_properties(
            csv_dir=csv_dir,
            network=network,
            property_name=explicit_property,
            generator_filter=filter_fn,
        )
        summary["explicit_outages"] = len(explicit_events)
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
    """Step: Run PyPSA network optimization."""
    network.consistency_check()
    if year is not None:
        snapshots = network.snapshots[network.snapshots.year == year]
    else:
        snapshots = network.snapshots
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
    res = network.optimize(snapshots=snapshots, **solver_config)
    return {
        "optimize": {
            "solve": res[0],
            "status": res[1],
            "snapshots_count": len(snapshots),
            "year_filter": year,
        }
    }


# Registry of all available step functions
STEP_REGISTRY: dict[str, Callable[..., Any]] = {
    "create_model": create_model_step,
    "scale_p_min_pu": scale_p_min_pu_step,
    "add_curtailment_link": add_curtailment_link_step,
    "load_vre_profiles": load_vre_profiles_step,
    "load_hydro_dispatch": load_hydro_dispatch_step,
    "add_storage_inflows": add_storage_inflows_step,
    "apply_generator_units": apply_generator_units_step,
    "parse_outages": parse_outages_step,
    "fix_outage_ramps": fix_outage_ramp_conflicts,
    "add_slack": add_slack_generators,
    "optimize": optimize_step,
}
