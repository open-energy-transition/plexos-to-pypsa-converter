"""Pre-defined workflow step implementations for workflow system."""

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pypsa

from network.conversion import create_model
from network.generators_csv import (
    apply_generator_units_timeseries_csv,
    load_data_file_profiles_csv,
)
from network.investment import (
    InvestmentPeriod,
    build_day_type_classifier,
    get_snapshot_timestamps,
    load_directory_timeseries,
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

logger = logging.getLogger(__name__)


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
    periods: Sequence[Mapping[str, int]],
    trace_groups: Mapping[str, Any] | None = None,
    timeslice_csv: str | None = None,
    use_representative_days: bool = True,
    unique_day_labels: bool = False,
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
    if trace_groups is None:
        trace_groups = _discover_default_trace_groups(base_dir)
    else:
        trace_groups = dict(trace_groups)

    if use_representative_days:
        if timeslice_csv is None:
            msg = "timeslice_csv must be provided when use_representative_days=True"
            raise ValueError(msg)
        timeslice_path = Path(timeslice_csv)
        if not timeslice_path.is_absolute():
            timeslice_path = base_dir / timeslice_csv

        if group_patterns is None or not group_patterns:
            group_patterns = {"base": []}

        if unique_day_labels:

            def classifier(ts: pd.Timestamp) -> str:
                return ts.strftime("%Y-%m-%d")

        else:
            classifier = build_day_type_classifier(
                timeslice_path,
                group_patterns,
                default_group=default_group,
            )
    else:
        classifier = None

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

    if use_representative_days:
        (
            metadata,
            period_bases,
            summary_totals,
            period_weights,
        ) = _prepare_representative_groups(
            base_dir=base_dir,
            trace_groups=trace_groups,
            investment_periods=investment_periods,
            classifier=classifier,
            output_dir=output_dir,
            csv_pattern=csv_pattern,
            file_limit=file_limit,
        )
        format_tag = "representative"
    else:
        (
            metadata,
            period_bases,
            summary_totals,
            period_weights,
        ) = _prepare_chronological_groups(
            base_dir=base_dir,
            trace_groups=trace_groups,
            investment_periods=investment_periods,
            output_dir=output_dir,
            csv_pattern=csv_pattern,
            file_limit=file_limit,
        )
        format_tag = "chronological"

    meta_section = metadata.setdefault("_meta", {})
    if period_bases:
        meta_section["period_bases"] = period_bases
    if period_weights:
        meta_section["period_weights"] = period_weights
    meta_section["format"] = format_tag

    metadata_path = output_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return {
        "prepare_investment_periods": {
            "output_dir": str(output_dir),
            "metadata_file": str(metadata_path),
            "group_totals": summary_totals,
        }
    }


def _prepare_representative_groups(
    **_: Any,
) -> tuple[dict, dict[str, str], dict[str, dict[str, float]], dict[str, float]]:
    """Print placeholder until representative-day aggregation is reintroduced.

    The simplified investment-period workflow currently focuses on full chronology
    support. Representative-day aggregation will be restored in a future revision
    once the streamlined path is stable.
    """
    msg = (
        "Representative-day aggregation is temporarily unavailable in the simplified "
        "investment-period workflow. Set use_representative_days=False to generate a "
        "chronological manifest."
    )
    raise NotImplementedError(msg)


def _prepare_chronological_groups(
    base_dir: Path,
    trace_groups: Mapping[str, Any],
    investment_periods: Sequence[InvestmentPeriod],
    output_dir: Path,
    csv_pattern: str = "*.csv",
    file_limit: int | None = None,
) -> tuple[dict, dict[str, str], dict[str, dict[str, float]], dict[str, float]]:
    """Generate manifest metadata for full-chronology investment-period runs."""
    normalized_groups: dict[str, dict[str, str]] = {}
    resolved_groups: dict[str, tuple[Path, str]] = {}

    for name, spec in trace_groups.items():
        path, pattern, meta = _resolve_trace_group(
            base_dir,
            spec,
            csv_pattern,
        )
        normalized_groups[name] = meta
        resolved_groups[name] = (path, pattern or csv_pattern)

    if not resolved_groups:
        msg = "trace_groups must contain at least one entry (e.g. 'demand')."
        raise ValueError(msg)

    if "demand" in resolved_groups:
        demand_path, demand_pattern = resolved_groups["demand"]
    else:
        demand_path, demand_pattern = next(iter(resolved_groups.values()))
        logger.warning(
            "No 'demand' trace group supplied; using '%s' to infer chronology.",
            next(iter(normalized_groups)),
        )

    demand_df = load_directory_timeseries(
        demand_path,
        pattern=demand_pattern,
        limit=file_limit,
    )
    demand_df = demand_df.sort_index()
    if demand_df.empty:
        msg = f"No demand profiles found in {demand_path}"
        raise ValueError(msg)

    timestamps = demand_df.index.unique().sort_values()
    if timestamps.empty:
        msg = f"No timestamps discovered in demand traces at {demand_path}"
        raise ValueError(msg)

    period_lookup = []
    for ts in timestamps:
        period = None
        for investment_period in investment_periods:
            if investment_period.contains(ts):
                period = investment_period.label
                break
        period_lookup.append(period)

    chronology_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "period": period_lookup,
        }
    )
    chronology_df = chronology_df.dropna(subset=["period"])
    chronology_df["period"] = chronology_df["period"].astype(str)

    if chronology_df.empty:
        msg = (
            "No timestamps fell within the specified investment periods. "
            "Check the date ranges in the demand traces and period definitions."
        )
        raise ValueError(msg)

    ts_series = chronology_df["timestamp"]
    step_series = ts_series.diff().shift(-1)
    if step_series.dropna().empty:
        fallback_step = pd.Timedelta(minutes=60)
    else:
        fallback_step = step_series.dropna().mode().iloc[0]
    step_series = step_series.fillna(fallback_step)
    chronology_df["weight_years"] = (step_series / pd.Timedelta(hours=8760)).astype(
        float
    )

    snapshot_csv = output_dir / "chronological_snapshots.csv"
    chronology_df.to_csv(snapshot_csv, index=False)

    period_weights_series = chronology_df.groupby("period")["weight_years"].sum()
    period_weights = {
        period: float(weight) for period, weight in period_weights_series.items()
    }

    period_bases = {
        period: ts_series.loc[chronology_df["period"] == period].iloc[0].isoformat()
        for period in period_weights
    }

    summary_totals = {"snapshots": period_weights.copy()}

    relative_root = Path(os.path.relpath(base_dir, output_dir)).as_posix()

    metadata = {
        "_meta": {
            "format": "chronological",
            "trace_groups": normalized_groups,
            "timeline": {
                "start": ts_series.iloc[0].isoformat(),
                "end": ts_series.iloc[-1].isoformat(),
            },
            "time_step_minutes": int(round(fallback_step / pd.Timedelta(minutes=1))),
            "periods": [
                {
                    "label": period.label,
                    "start_year": period.start_year,
                    "end_year": period.end_year,
                }
                for period in investment_periods
            ],
            "model_root": relative_root,
        },
        "snapshots": {
            "csv": snapshot_csv.name,
            "columns": ["timestamp", "period", "weight_years"],
            "total_years": float(sum(period_weights.values())),
            "count": int(len(chronology_df)),
            "period_weights_years": period_weights.copy(),
        },
    }

    return metadata, period_bases, summary_totals, period_weights


def _resolve_trace_group(
    base_dir: Path,
    spec: Any,
    default_pattern: str,
) -> tuple[Path, str, dict[str, str]]:
    """Normalize trace group specification into filesystem paths."""
    if isinstance(spec, str):
        raw_path = Path(spec)
        pattern = default_pattern
    elif isinstance(spec, Mapping):
        raw_path_value = spec.get("path")
        if raw_path_value is None:
            msg = "trace_groups entries must provide a 'path'."
            raise ValueError(msg)
        raw_path = Path(raw_path_value)
        pattern = spec.get("pattern", default_pattern)
    else:
        msg = "trace_groups values must be a string path or mapping."
        raise TypeError(msg)

    path = raw_path if raw_path.is_absolute() else base_dir / raw_path
    meta_path = raw_path.as_posix()

    return path, pattern, {"path": meta_path, "pattern": pattern}


def _discover_default_trace_groups(base_dir: Path) -> dict[str, Any]:
    """Auto-detect common trace directories within a model export."""
    search_map: dict[str, list[tuple[str, str]]] = {
        "demand": [
            ("Traces/demand", "*.csv"),
            ("Traces/load", "*.csv"),
            ("demand", "*.csv"),
            ("load", "*.csv"),
            ("LoadProfile", "*.csv"),
            ("Load", "*.csv"),
        ],
        "solar": [
            ("Traces/solar", "*.csv"),
            ("Traces/VRE/Solar", "*.csv"),
            ("solar", "*.csv"),
            ("Solar", "*.csv"),
        ],
        "wind": [
            ("Traces/wind", "*.csv"),
            ("Traces/VRE/Wind", "*.csv"),
            ("wind", "*.csv"),
            ("Wind", "*.csv"),
        ],
        "storage_inflows": [
            ("Traces/hydro", "MonthlyNaturalInflow_*.csv"),
            ("Traces/hydro", "NaturalInflow_*.csv"),
            ("Traces/hydro", "*.csv"),
        ],
        "hydro_rating": [
            ("Traces/hydro", "HydroRating_*.csv"),
        ],
        "hydro_min_stable": [
            ("Traces/hydro", "HydroMinStable_*.csv"),
        ],
    }

    discovered: dict[str, Any] = {}
    base_dir = base_dir.resolve()

    root_candidates: list[Path] = [base_dir]
    csv_root = base_dir / "csvs_from_xml"
    if csv_root.exists():
        root_candidates.append(csv_root)
        root_candidates.extend(
            [child for child in csv_root.iterdir() if child.is_dir()]
        )

    for group_name, options in search_map.items():
        for rel_path, pattern in options:
            for root in root_candidates:
                dir_path = (root / rel_path).resolve()
                if not dir_path.exists() or not dir_path.is_dir():
                    continue
                if not any(dir_path.glob(pattern)):
                    continue
                rel_entry = dir_path.relative_to(base_dir)
                if pattern == "*.csv":
                    discovered[group_name] = rel_entry.as_posix()
                else:
                    discovered[group_name] = {
                        "path": rel_entry.as_posix(),
                        "pattern": pattern,
                    }
                logger.debug(
                    "Trace group '%s' auto-detected at %s (pattern=%s)",
                    group_name,
                    dir_path,
                    pattern,
                )
                break
            if group_name in discovered:
                break

    if "demand" not in discovered:
        msg = (
            "Unable to auto-detect demand traces for investment-period aggregation. "
            "Expected directories such as 'Traces/demand' or 'demand/'."
        )
        raise FileNotFoundError(msg)

    return discovered


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
    use_investment_periods: bool = False,
    investment_manifest: str | None = None,
    investment_group: str | None = None,
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
        use_investment_periods=use_investment_periods,
        investment_manifest=investment_manifest,
        investment_group=investment_group,
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
    use_investment_periods: bool = False,
    investment_manifest: str | None = None,
    investment_group_rating: str | None = None,
    investment_group_min_stable: str | None = None,
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
        rating_group = (
            investment_group_rating or investment_group_min_stable or "hydro_rating"
        )
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
            use_investment_periods=use_investment_periods,
            investment_manifest=investment_manifest,
            investment_group=rating_group,
        )
        summary["rating"] = rating_summary

    # Load Min Stable Level profiles (p_min_pu) for must-run constraints
    if load_min_stable:
        min_group = (
            investment_group_min_stable or investment_group_rating or "hydro_min_stable"
        )
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
            use_investment_periods=use_investment_periods,
            investment_manifest=investment_manifest,
            investment_group=min_group,
        )
        summary["min_stable"] = min_summary

    return {"load_hydro_dispatch": summary}


def add_storage_inflows_step(
    network: pypsa.Network,
    csv_dir: str | Path,
    inflow_path: str | Path,
    use_investment_periods: bool = False,
    investment_manifest: str | None = None,
    investment_group: str | None = None,
) -> dict:
    """Step: Add natural inflow time series to storage units (hydro)."""
    summary = add_storage_inflows_csv(
        network=network,
        csv_dir=csv_dir,
        inflow_path=inflow_path,
        use_investment_periods=use_investment_periods,
        investment_manifest=investment_manifest,
        investment_group=investment_group,
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
    use_investment_periods: bool = False,
    investment_manifest: str | None = None,
    investment_group: str | None = None,
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
        use_investment_periods=use_investment_periods,
        investment_manifest=investment_manifest,
        investment_group=investment_group,
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
    period: int | str | None = None,
    use_investment_periods: bool | None = None,
    solver_config: dict | None = None,
) -> dict:
    """Step: Run PyPSA network optimization."""
    network.consistency_check()

    logger.info(
        "Optimisation setup: %d buses, %d generators, %d loads, %d links, %d storage units, %d snapshots",
        len(network.buses),
        len(network.generators),
        len(network.loads),
        len(network.links),
        len(network.storage_units),
        len(network.snapshots),
    )

    snapshots = network.snapshots
    if year is not None:
        timestamps = get_snapshot_timestamps(network.snapshots)
        mask = timestamps.year == year
        snapshots = network.snapshots[mask]

    period_filter: int | None = None
    if period is not None:
        label_map = getattr(network, "investment_period_label_map", {})
        if (
            not use_investment_periods
            or not label_map
            or not isinstance(network.snapshots, pd.MultiIndex)
            or "period" not in network.snapshots.names
        ):
            logger.info(
                "Ignoring period filter %s; investment periods disabled or unavailable.",
                period,
            )
        else:
            if isinstance(period, str):
                for pid, label in label_map.items():
                    if str(label) == period:
                        period_filter = int(pid)
                        break
                else:
                    try:
                        period_filter = int(period)
                    except ValueError as exc:
                        msg = (
                            f"Period '{period}' not recognised. Available labels: "
                            f"{sorted(str(label) for label in label_map.values())}"
                        )
                        raise ValueError(msg) from exc
            else:
                period_filter = int(period)

            if period_filter is not None:
                period_levels = network.snapshots.get_level_values("period")
                if period_filter not in set(period_levels):
                    msg = (
                        f"Period '{period_filter}' not present in network snapshots. "
                        f"Available: {sorted(set(period_levels))}"
                    )
                    raise ValueError(msg)
                snapshots = network.snapshots[period_levels == period_filter]

    logger.info(
        "Snapshots selected for optimisation: %d (year filter=%s, period filter=%s)",
        len(snapshots),
        year,
        period_filter,
    )

    class NoSnapshotsSelectedError(ValueError):
        def __init__(self):
            msg = "No snapshots selected for optimisation; check filters."
            super().__init__(msg)

    if len(snapshots) == 0:
        msg = "No snapshots selected for optimisation; check filters."
        raise NoSnapshotsSelectedError()

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

    logger.info("Solver config: %s", solver_config)
    logger.info("Calling network.optimize ...")

    res = network.optimize(snapshots=snapshots, **solver_config)
    return {
        "optimize": {
            "solve": res[0],
            "status": res[1],
            "snapshots_count": len(snapshots),
            "year_filter": year,
            "period_filter": period_filter,
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
