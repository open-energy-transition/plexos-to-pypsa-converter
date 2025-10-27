"""Pre-defined workflow step implementations for workflow system."""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pypsa

from network.conversion import create_model
from network.generators_csv import (
    apply_generator_units_timeseries_csv,
    load_data_file_profiles_csv,
)
from network.investment import (
    InvestmentPeriod,
    build_day_type_classifier,
    compute_period_statistics,
    load_directory_timeseries,
    write_profile_csv,
)
from network.outages import (
    apply_outage_schedule,
    build_outage_schedule,
    generate_stochastic_outages_csv,
    load_outages_from_monthly_files,
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


def prepare_investment_periods_step(
    model_dir: str,
    timeslice_csv: str,
    trace_groups: Mapping[str, str],
    periods: Sequence[Mapping[str, int]],
    network: pypsa.Network | None = None,
    enabled: bool = True,
    group_patterns: Mapping[str, Sequence[str]] | None = None,
    default_group: str = "base",
    output_subdir: str = "investment_periods",
    csv_pattern: str = "*.csv",
    file_limit: int | None = None,
    metadata_filename: str = "investment_periods.json",
) -> dict:
    """Step: Aggregate traces into representative-day investment period datasets."""
    if not enabled:
        return {
            "prepare_investment_periods": {
                "enabled": False,
            }
        }

    base_dir = Path(model_dir)
    timeslice_path = Path(timeslice_csv)
    if not timeslice_path.is_absolute():
        timeslice_path = base_dir / timeslice_csv

    if group_patterns is None or not group_patterns:
        group_patterns = {"base": []}

    classifier = build_day_type_classifier(
        timeslice_path,
        group_patterns,
        default_group=default_group,
    )

    investment_periods = []
    for period in periods:
        label = period.get("label")
        if label is None:
            msg = "Each investment period requires a 'label' key."
            raise ValueError(msg)
        start_year = period.get("start_year", period.get("start"))
        end_year = period.get("end_year", period.get("end"))
        if start_year is None or end_year is None:
            msg = f"Investment period '{label}' missing start_year/end_year definition."
            raise ValueError(msg)
        investment_periods.append(
            InvestmentPeriod(
                label=str(label),
                start_year=int(start_year),
                end_year=int(end_year),
            )
        )

    output_dir = base_dir / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, list[dict[str, float | str]]]] = {}
    summary_totals: dict[str, dict[str, float]] = {}

    for group_label, trace_path in trace_groups.items():
        target_path = Path(trace_path)
        if not target_path.is_absolute():
            target_path = base_dir / trace_path
        data = load_directory_timeseries(
            target_path,
            pattern=csv_pattern,
            limit=file_limit,
        )
        stats = compute_period_statistics(data, investment_periods, classifier)
        metadata[group_label] = {}
        summary_totals[group_label] = {}

        for period_label, profiles in stats.items():
            metadata[group_label][period_label] = []
            total_weight = 0.0
            for profile in profiles:
                filename = f"{group_label}__{period_label}__{profile.name.replace(' ', '_')}.csv"
                write_profile_csv(profile, output_dir / filename)
                metadata[group_label][period_label].append(
                    {
                        "name": profile.name,
                        "weight_years": profile.weight_years,
                        "csv": filename,
                    }
                )
                total_weight += profile.weight_years
            summary_totals[group_label][period_label] = total_weight

    metadata_path = output_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "prepare_investment_periods": {
            "output_dir": str(output_dir),
            "metadata_file": str(metadata_path),
            "group_totals": summary_totals,
        }
    }


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


def load_monthly_outages_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    units_out_dir: str | Path,
    scenario: str | int | None = None,
    generator_filter: str | None = None,
    ramp_aware: bool = True,
) -> dict:
    """Step: Load pre-computed monthly outage schedules and apply to network.

    This step is designed for models like CAISO IRP23 that provide pre-computed
    monthly outage files (e.g., UnitsOut data) instead of requiring stochastic
    outage generation.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports (for generator metadata)
    units_out_dir : str | Path
        Directory containing monthly outage files organized in M01-M12 subdirectories
    scenario : str | int | None, default None
        Which scenario column to use from monthly files (e.g., 1, 2, "Value")
    generator_filter : str | None, default None
        Filter preset name (e.g., "exclude_vre", "all")
    ramp_aware : bool, default True
        Enable ramp-aware outage application with gradual startup/shutdown zones

    Returns
    -------
    dict
        Summary with outage loading statistics and application results
    """
    filter_fn = resolve_filter_preset(generator_filter, network)

    # Load outage schedules from monthly files
    outage_schedule = load_outages_from_monthly_files(
        units_out_dir=units_out_dir,
        network=network,
        scenario=scenario,
        generator_filter=filter_fn,
    )

    # Apply outage schedule to network with ramp-aware startup/shutdown
    outage_summary = apply_outage_schedule(
        network,
        outage_schedule,
        ramp_aware=ramp_aware,
    )

    return {
        "load_monthly_outages": {
            "generators_with_outages": len(outage_schedule.columns),
            "snapshots": len(outage_schedule),
            "scenario": scenario,
            "ramp_aware": ramp_aware,
            **outage_summary,
        }
    }


def optimize_step(
    network: pypsa.Network,
    year: int | None = None,
    solver_config: dict | None = None,
) -> dict:
    """Step: Run PyPSA network optimization."""
    network.consistency_check()

    # Debug: Print network state
    print("\n=== Network State Before Optimization ===")
    print(f"Buses: {len(network.buses)}")
    print(f"Generators: {len(network.generators)}")
    print(f"Loads: {len(network.loads)}")
    print(f"Links: {len(network.links)}")
    print(f"Storage Units: {len(network.storage_units)}")
    print(f"Snapshots: {len(network.snapshots)}")
    print("=========================================\n")

    if year is not None:
        snapshots = network.snapshots[network.snapshots.year == year]
    else:
        snapshots = network.snapshots

    print(f"Snapshots for optimization: {len(snapshots)}")
    print(f"Year filter: {year}")

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

    print(f"Solver config: {solver_config}")
    print(f"Calling network.optimize with {len(snapshots)} snapshots...\n")

    res = network.optimize(snapshots=snapshots, **solver_config)
    return {
        "optimize": {
            "solve": res[0],
            "status": res[1],
            "snapshots_count": len(snapshots),
            "year_filter": year,
        }
    }


def save_network_step(
    network: pypsa.Network,
    model_id: str,
    output_dir: str | Path = "src/examples/results",
) -> dict:
    """Step: Save solved network to NetCDF file.

    Saves the network to {output_dir}/{model_id}/network.nc,
    creating directories if they don't exist.
    """
    output_path = Path(output_dir) / model_id
    output_path.mkdir(parents=True, exist_ok=True)

    netcdf_file = output_path / "solved_network.nc"
    network.export_to_netcdf(str(netcdf_file))

    return {
        "save_network": {
            "path": str(netcdf_file),
            "size_mb": netcdf_file.stat().st_size / (1024 * 1024),
        }
    }


# Registry of all available step functions
STEP_REGISTRY: dict[str, Callable[..., Any]] = {
    "prepare_investment_periods": prepare_investment_periods_step,
    "create_model": create_model_step,
    "scale_p_min_pu": scale_p_min_pu_step,
    "add_curtailment_link": add_curtailment_link_step,
    "load_vre_profiles": load_vre_profiles_step,
    "load_hydro_dispatch": load_hydro_dispatch_step,
    "add_storage_inflows": add_storage_inflows_step,
    "apply_generator_units": apply_generator_units_step,
    "parse_outages": parse_outages_step,
    "load_monthly_outages": load_monthly_outages_step,
    "fix_outage_ramps": fix_outage_ramp_conflicts,
    "add_slack": add_slack_generators,
    "optimize": optimize_step,
    "save_network": save_network_step,
}
