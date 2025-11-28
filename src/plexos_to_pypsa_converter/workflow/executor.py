"""Workflow executor for running model processing pipelines.

This module provides the main run_model_workflow() function that executes
a declarative workflow defined in the registry.
"""

import inspect
from pathlib import Path

import pypsa

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.utils.model_paths import get_model_directory
from plexos_to_pypsa_converter.workflow.builder import build_workflow_steps
from plexos_to_pypsa_converter.workflow.detection import detect_model_features
from plexos_to_pypsa_converter.workflow.steps import STEP_REGISTRY


def _auto_select_csv_dir(base_path: Path) -> Path | None:
    """Recursively select a CSV directory when only a single subdirectory exists."""
    if not base_path.exists() or not base_path.is_dir():
        return None

    if any(base_path.glob("*.csv")):
        return base_path

    # Ignore hidden/system folders such as .DS_Store
    subdirs = [
        p for p in base_path.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]

    if len(subdirs) == 1:
        nested = _auto_select_csv_dir(subdirs[0])
        return nested or subdirs[0]

    # If multiple subdirectories exist (e.g., auto-export created region folders),
    # prefer a directory named "WECC" (common for CAISO IRP exports), otherwise
    # pick the first in sorted order to provide a deterministic choice.
    if not subdirs:
        return None

    preferred = [p for p in subdirs if p.name.lower() == "wecc"]
    target = preferred[0] if preferred else sorted(subdirs)[0]
    nested = _auto_select_csv_dir(target)
    return nested or target

    return None


def _resolve_csv_dir_path(
    model_dir: Path,
    workflow: dict,
    descriptor: dict,
) -> Path:
    """Determine the CSV directory for workflow steps."""
    csv_dir_override = descriptor.get("csv_dir")
    if csv_dir_override:
        return Path(csv_dir_override)

    csv_dir_pattern = workflow.get("csv_dir_pattern") or "csvs_from_xml"
    pattern_path = Path(csv_dir_pattern)

    if pattern_path.is_absolute():
        candidate = pattern_path
    else:
        candidate = (model_dir / pattern_path).resolve()

    if candidate.exists() and candidate.is_dir():
        auto_selected = _auto_select_csv_dir(candidate)
        if auto_selected:
            return auto_selected

    return candidate


def run_model_workflow(
    model_id: str,
    workflow_overrides: dict | None = None,
    *,
    model_descriptor: dict | None = None,
    model_dir_override: str | Path | None = None,
    registry_model_id: str | None = None,
    solve: bool = False,
    optimize: bool | None = None,
    optimize_year: int | None = None,
    enable_vre: bool = True,
    enable_hydro_dispatch: bool = True,
    enable_hydro_inflows: bool = True,
    enable_units: bool = True,
    enable_outages: bool = True,
    enable_slack_generators: bool = True,
    enable_save: bool = True,
    force_demand_strategy: str | None = None,
    solver_name: str = "highs",
    solver_options: dict | None = None,
    auto_workflow: bool = True,
    **step_overrides,
) -> tuple[pypsa.Network, dict]:
    """Execute model processing workflow from registry definition.

    This function orchestrates the execution of a multi-step workflow defined in
    the model registry, handling parameter injection, path resolution, and
    summary aggregation. By default it runs only the conversion steps. Pass
    ``solve=True`` to allow the ``optimize`` step to execute.
    """
    descriptor = model_descriptor or MODEL_REGISTRY.get(model_id) or {}
    if not descriptor and workflow_overrides is None and not auto_workflow:
        available = ", ".join(MODEL_REGISTRY.keys())
        msg = (
            f"Model '{model_id}' not found in registry and no workflow overrides were provided. "
            "Pass `workflow_overrides` or `model_descriptor` for custom models. "
            f"Available registry models: [{available}]"
        )
        raise ValueError(msg)

    processing_workflow = descriptor.get("processing_workflow")
    if processing_workflow is None and workflow_overrides is None and not auto_workflow:
        msg = (
            f"Model '{model_id}' does not have a processing_workflow defined. "
            "Provide one via the registry or the model_descriptor argument."
        )
        raise ValueError(msg)

    workflow = workflow_overrides or processing_workflow or {}

    effective_registry_model_id = registry_model_id or descriptor.get(
        "registry_model_id"
    )

    # Resolve base directories (allow descriptor / override to set custom paths)
    if model_dir_override is not None:
        model_dir = Path(model_dir_override)
    elif descriptor.get("model_dir"):
        model_dir = Path(descriptor["model_dir"])
    else:
        try:
            model_dir = get_model_directory(model_id)
        except FileNotFoundError as exc:
            msg = (
                f"Model directory for '{model_id}' could not be resolved. "
                "Provide `model_dir_override` or include `model_dir` in the model descriptor."
            )
            raise FileNotFoundError(msg) from exc

    csv_dir: Path = _resolve_csv_dir_path(model_dir, workflow, descriptor)

    profiles_path = Path(
        descriptor.get("profiles_path", model_dir),
    )
    inflow_path = Path(descriptor.get("inflow_path", model_dir))

    # Build units_out_dir if pattern or explicit path is specified
    units_out_dir_override = descriptor.get("units_out_dir")
    if units_out_dir_override:
        units_out_dir = Path(units_out_dir_override)
    else:
        units_out_dir_pattern = workflow.get("units_out_dir_pattern")
        units_out_dir = (
            Path(model_dir) / units_out_dir_pattern
            if units_out_dir_pattern
            else Path(model_dir)
        )

    context = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "csv_dir": str(csv_dir),
        "profiles_path": str(profiles_path),
        "inflow_path": str(inflow_path),
        "units_out_dir": str(units_out_dir),
    }
    if effective_registry_model_id:
        context["registry_model_id"] = effective_registry_model_id
    if force_demand_strategy:
        context["force_demand_strategy"] = force_demand_strategy

    def parse_step_overrides(step_overrides: dict) -> dict[str, dict]:
        parsed = {}
        for key, value in step_overrides.items():
            if "__" in key:
                step, param = key.split("__", 1)
                parsed.setdefault(step, {})[param] = value
        return parsed

    parsed_overrides = parse_step_overrides(step_overrides)

    # Build workflow steps automatically when requested or when registry steps are absent
    # Only auto-build when no processing_workflow was supplied
    if (
        auto_workflow
        and workflow_overrides is None
        and descriptor.get("processing_workflow") is None
    ):
        features = detect_model_features(model_dir, csv_dir)
        effective_optimize = optimize if optimize is not None else solve
        solver_config = {"solver_name": solver_name}
        if solver_options:
            solver_config["solver_options"] = solver_options
        workflow_steps = build_workflow_steps(
            model_id=model_id,
            features=features,
            enable_vre=enable_vre,
            enable_hydro_dispatch=enable_hydro_dispatch,
            enable_hydro_inflows=enable_hydro_inflows,
            enable_units=enable_units,
            enable_outages=enable_outages,
            enable_slack_generators=enable_slack_generators,
            enable_save=enable_save,
            optimize=effective_optimize,
            optimize_year=optimize_year,
            solver_config=solver_config,
        )
        workflow = {"steps": workflow_steps, "solver_config": solver_config}

    network: pypsa.Network | None = None
    aggregated_summary: dict = {}
    steps = workflow.get("steps", [])
    effective_optimize = optimize if optimize is not None else solve
    if force_demand_strategy:
        for step in steps:
            if step.get("name") == "create_model":
                step.setdefault("params", {})["demand_assignment_strategy"] = (
                    force_demand_strategy
                )
                break

    if not effective_optimize:
        steps = [step for step in steps if step.get("name") != "optimize"]
    print(
        f"Running workflow for model: {model_id}\nModel directory: {model_dir}\nWorkflow steps: {len(steps)}\n"
    )
    for step_idx, step_def in enumerate(steps, 1):
        step_name = step_def["name"]
        step_params = step_def.get("params", {}).copy()
        condition = step_def.get("condition")
        if condition and not _evaluate_condition(condition, context):
            print(
                f"Step {step_idx}/{len(steps)}: {step_name} (skipped - condition not met)"
            )
            continue
        if step_name not in STEP_REGISTRY:
            msg = f"Unknown workflow step: {step_name}. Available steps: {list(STEP_REGISTRY.keys())}"
            raise ValueError(msg)
        if step_name in parsed_overrides:
            step_params.update(parsed_overrides[step_name])
        step_fn = STEP_REGISTRY[step_name]
        step_params = _inject_context(step_params, context, step_fn)
        print(f"Step {step_idx}/{len(steps)}: {step_name}")
        try:
            if step_name == "create_model":
                network, step_summary = step_fn(**step_params)
                aggregated_summary.update(step_summary)
                # After create_model, CSV exports may have been generated; refresh csv_dir
                # to pick up nested export paths (e.g., csvs_from_xml/WECC).
                refreshed_csv_dir = _resolve_csv_dir_path(
                    model_dir, workflow, descriptor
                )
                context["csv_dir"] = str(refreshed_csv_dir)
            elif step_name == "optimize":
                if "solver_config" not in step_params:
                    step_params["solver_config"] = workflow.get("solver_config")
                step_summary = step_fn(network=network, **step_params)
                aggregated_summary.update(step_summary)
            else:
                if network is None:
                    msg = f"Step '{step_name}' requires a network, but create_model has not been called yet. Ensure 'create_model' is the first step in the workflow."
                    raise RuntimeError(msg)  # noqa: TRY301
                step_summary = step_fn(network=network, **step_params)
                aggregated_summary.update(step_summary)
            print(f"{step_name} completed\n")
        except Exception as e:
            print(f"{step_name} failed: {e}\n")
            raise
    print(f"Workflow complete for model: {model_id}\n")
    return network, aggregated_summary


def _inject_context(params: dict, context: dict, step_fn: callable) -> dict:
    """Inject context variables into step parameters.

    Uses function signature inspection to only inject parameters that the
    step function actually accepts, preventing TypeErrors from unexpected
    keyword arguments.

    Args:
        params: Step parameters dict from registry
        context: Context variables dict (model_id, csv_dir, profiles_path, inflow_path, units_out_dir)
        step_fn: The step function to call (used to inspect signature)

    Returns:
        Updated parameters dict with appropriate context variables injected
    """
    injected = params.copy()

    # Get the function's parameter names
    sig = inspect.signature(step_fn)
    accepted_params = set(sig.parameters.keys())

    # Check if function accepts **kwargs
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    # Standard context variables that might be injected
    context_vars = [
        "model_id",
        "model_dir",
        "csv_dir",
        "profiles_path",
        "inflow_path",
        "units_out_dir",
        "registry_model_id",
    ]

    for key in context_vars:
        if (
            key in context
            and (key in accepted_params or has_var_keyword)
            and (key not in injected or injected[key] is None)
        ):
            injected[key] = context[key]

    return injected


def _evaluate_condition(condition: str, context: dict) -> bool:
    """Evaluate a simple condition string.

    Currently supports basic equality checks like: model_id == 'sem-2024-2032'

    Args:
        condition: Condition string to evaluate
        context: Context variables for evaluation

    Returns:
        True if condition is met, False otherwise
    """
    # Very basic implementation - could be expanded later
    # For now, only support model_id equality checks
    if "model_id ==" in condition:
        target_model = condition.split("model_id ==")[1].strip().strip("'\"")
        return context.get("model_id") == target_model

    # Default to True if we don't understand the condition
    return True
