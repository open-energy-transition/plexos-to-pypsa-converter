"""Workflow builder that assembles step definitions from detected features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from plexos_to_pypsa_converter.workflow.detection import ModelFeatures


def _path_parent_or_self(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return paths[0].parent if paths[0].is_file() else paths[0]


def build_workflow_steps(
    model_id: str,
    features: ModelFeatures,
    *,
    enable_vre: bool = True,
    enable_hydro_dispatch: bool = True,
    enable_hydro_inflows: bool = True,
    enable_units: bool = True,
    enable_outages: bool = True,
    enable_slack_generators: bool = True,
    enable_save: bool = True,
    optimize: bool = False,
    optimize_year: int | None = None,
    solver_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create a list of workflow steps based on detected assets and toggles."""
    steps: list[dict[str, Any]] = []

    # Always start by creating the model (CSV path provided by run_model_workflow)
    steps.append({"name": "create_model", "params": {"use_csv": True}})

    if enable_vre and features.vre_profile_files:
        profiles_path = _path_parent_or_self(features.vre_profile_files)
        steps.append(
            {
                "name": "load_vre_profiles",
                "params": {
                    "profiles_path": str(profiles_path) if profiles_path else None,
                    "csv_dir": None,
                    "property_name": "Rating",
                    "target_property": "p_max_pu",
                    "target_type": "generators_t",
                    "apply_mode": "replace",
                    "scenario": 1,
                    "generator_filter": "vre_only",
                    "carrier_mapping": {"Wind": "wind", "Solar": "solar"},
                },
            }
        )

    if enable_hydro_dispatch and features.hydro_dispatch_files:
        hydro_dispatch_dir = _path_parent_or_self(features.hydro_dispatch_files)
        steps.append(
            {
                "name": "load_hydro_dispatch",
                "params": {
                    "profiles_path": str(hydro_dispatch_dir)
                    if hydro_dispatch_dir
                    else None,
                    "csv_dir": None,
                    "scenario": "Value",
                    "generator_filter": "hydro_only",
                },
            }
        )

    if enable_hydro_inflows and features.hydro_inflow_files:
        inflow_dir = _path_parent_or_self(features.hydro_inflow_files)
        steps.append(
            {
                "name": "add_storage_inflows",
                "params": {
                    "inflow_path": str(inflow_dir) if inflow_dir else None,
                    "csv_dir": None,
                },
            }
        )

    if enable_units and features.units_files:
        steps.append(
            {
                "name": "apply_generator_units",
                "params": {"csv_dir": None},
            }
        )

    if enable_outages:
        if features.units_out_dir:
            steps.append(
                {
                    "name": "load_monthly_outages",
                    "params": {
                        "units_out_dir": str(features.units_out_dir),
                        "csv_dir": None,
                        "generator_filter": "exclude_vre",
                        "ramp_aware": True,
                    },
                }
            )
        elif features.outage_files or features.has_outage_properties:
            outage_dir = _path_parent_or_self(features.outage_files)
            steps.append(
                {
                    "name": "parse_outages",
                    "params": {
                        "csv_dir": str(outage_dir) if outage_dir else None,
                        "include_explicit": True,
                        "include_forced": True,
                        "include_maintenance": True,
                        "generator_filter": "exclude_vre",
                        "random_seed": 42,
                    },
                }
            )

    if enable_slack_generators:
        steps.append({"name": "add_slack", "params": {}})

    if optimize:
        steps.append(
            {
                "name": "optimize",
                "params": {
                    "year": optimize_year,
                    "solver_config": solver_config,
                },
            }
        )

    if enable_save:
        steps.append({"name": "save_network", "params": {}})

    return steps
