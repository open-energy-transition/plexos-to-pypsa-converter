"""Workflow executor for running model processing pipelines.

This module provides the main run_model_workflow() function that executes
a declarative workflow defined in the registry.
"""

import inspect

import pypsa

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.utils.model_paths import get_model_directory
from plexos_to_pypsa_converter.workflow.steps import STEP_REGISTRY


def run_model_workflow(
    model_id: str,
    workflow_overrides: dict | None = None,
    *,
    solve: bool = False,
    **step_overrides,
) -> tuple[pypsa.Network, dict]:
    """Execute model processing workflow from registry definition.

    This function orchestrates the execution of a multi-step workflow defined in
    the model registry, handling parameter injection, path resolution, and
    summary aggregation. By default it runs only the conversion steps. Pass
    ``solve=True`` to allow the ``optimize`` step to execute.
    """
    if model_id not in MODEL_REGISTRY:
        msg = f"Model '{model_id}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)
    model_meta = MODEL_REGISTRY[model_id]
    if "processing_workflow" not in model_meta:
        msg = f"Model '{model_id}' does not have a processing_workflow defined. This model may not support the workflow system yet."
        raise ValueError(msg)
    workflow = workflow_overrides or model_meta["processing_workflow"]
    model_dir = get_model_directory(model_id)
    csv_dir_pattern = workflow.get("csv_dir_pattern", "csvs_from_xml")
    csv_dir = model_dir / csv_dir_pattern

    # Build units_out_dir if pattern is specified
    units_out_dir_pattern = workflow.get("units_out_dir_pattern")
    units_out_dir = (
        model_dir / units_out_dir_pattern if units_out_dir_pattern else model_dir
    )

    context = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "csv_dir": str(csv_dir),
        "profiles_path": str(model_dir),
        "inflow_path": str(model_dir),
        "units_out_dir": str(units_out_dir),
    }

    def parse_step_overrides(step_overrides: dict) -> dict[str, dict]:
        parsed = {}
        for key, value in step_overrides.items():
            if "__" in key:
                step, param = key.split("__", 1)
                parsed.setdefault(step, {})[param] = value
        return parsed

    parsed_overrides = parse_step_overrides(step_overrides)

    network: pypsa.Network | None = None
    aggregated_summary: dict = {}
    steps = workflow.get("steps", [])
    if not solve:
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
        "csv_dir",
        "profiles_path",
        "inflow_path",
        "units_out_dir",
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
