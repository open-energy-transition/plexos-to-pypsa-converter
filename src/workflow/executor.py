"""Workflow executor for running model processing pipelines.

This module provides the main run_model_workflow() function that executes
a declarative workflow defined in the registry.
"""

import inspect

import pypsa

from db.registry import MODEL_REGISTRY
from utils.model_paths import get_model_directory
from workflow.steps import STEP_REGISTRY


def run_model_workflow(
    model_id: str,
    workflow_overrides: dict | None = None,
    **step_overrides,
) -> tuple[pypsa.Network, dict]:
    """Execute model processing workflow from registry definition.

    This function orchestrates the execution of a multi-step workflow defined
    in the model registry, handling parameter injection, path resolution,
    and summary aggregation.

    Args:
        model_id: Model identifier from registry (e.g., "sem-2024-2032")
        workflow_overrides: Optional dict to completely override workflow definition
        **step_overrides: Override parameters for specific steps using syntax:
            step_name__param_name=value (e.g., optimize__year=2024)

    Returns:
        (network, aggregated_summary) tuple where:
            - network: Final PyPSA network after all steps
            - aggregated_summary: Dict containing summaries from all steps

    Raises:
        ValueError: If model_id not in registry or workflow step not found
        KeyError: If required workflow configuration is missing

    Example:
        # Run standard workflow
        >>> network, summary = run_model_workflow("sem-2024-2032")

        # Override optimization year
        >>> network, summary = run_model_workflow(
        ...     "sem-2024-2032",
        ...     optimize__year=2024
        ... )

        # Completely custom workflow
        >>> network, summary = run_model_workflow(
        ...     "sem-2024-2032",
        ...     workflow_overrides={
        ...         "steps": [
        ...             {"name": "create_model", "params": {}},
        ...             {"name": "optimize", "params": {"year": 2023}}
        ...         ]
        ...     }
        ... )
    """
    # Validate model exists in registry
    if model_id not in MODEL_REGISTRY:
        msg = f"Model '{model_id}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)

    model_meta = MODEL_REGISTRY[model_id]

    # Check if model has processing workflow defined
    if "processing_workflow" not in model_meta:
        msg = f"Model '{model_id}' does not have a processing_workflow defined. This model may not support the workflow system yet."
        raise ValueError(msg)

    # Load workflow definition (use override if provided)
    workflow = workflow_overrides or model_meta["processing_workflow"]

    # Resolve model paths
    model_dir = get_model_directory(model_id)
    csv_dir_pattern = workflow.get("csv_dir_pattern", "csvs_from_xml")
    # Resolve csv_dir
    csv_dir = model_dir / csv_dir_pattern
    # Build context for parameter injection
    context = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "csv_dir": str(csv_dir),
        "profiles_path": str(model_dir),
        "inflow_path": str(model_dir),
    }

    # Parse step-specific overrides from kwargs
    parsed_overrides = {}
    for key, value in step_overrides.items():
        if "__" in key:
            step_name, param_name = key.split("__", 1)
            if step_name not in parsed_overrides:
                parsed_overrides[step_name] = {}
            parsed_overrides[step_name][param_name] = value
        else:
            # Direct parameter (applies to all steps?)
            # For now, ignore - could support global params later
            pass

    # Execute workflow steps
    network = None
    aggregated_summary = {}

    print(f"Running workflow for model: {model_id}")
    print(f"Model directory: {model_dir}")
    print(f"Workflow steps: {len(workflow.get('steps', []))}\n")

    for step_idx, step_def in enumerate(workflow.get("steps", []), 1):
        step_name = step_def["name"]
        step_params = step_def.get("params", {}).copy()

        # Check if step should be skipped based on condition
        condition = step_def.get("condition")
        if condition and not _evaluate_condition(condition, context):
            print(
                f"⏭️  Step {step_idx}/{len(workflow['steps'])}: {step_name} (skipped - condition not met)"
            )
            continue

        # Validate step exists
        if step_name not in STEP_REGISTRY:
            msg = f"Unknown workflow step: {step_name}. Available steps: {list(STEP_REGISTRY.keys())}"
            raise ValueError(msg)

        # Apply step-specific overrides
        if step_name in parsed_overrides:
            step_params.update(parsed_overrides[step_name])

        # Get step function
        step_fn = STEP_REGISTRY[step_name]

        # Inject context parameters (only those the step function accepts)
        step_params = _inject_context(step_params, context, step_fn)

        # Execute step
        print(f"▶️  Step {step_idx}/{len(workflow['steps'])}: {step_name}")

        try:
            # Special handling for create_model (returns network, summary)
            if step_name == "create_model":
                network, step_summary = step_fn(**step_params)
                aggregated_summary.update(step_summary)
            # Special handling for optimize (doesn't return summary)
            elif step_name == "optimize":
                # Inject solver_config from workflow if not in params
                if "solver_config" not in step_params:
                    step_params["solver_config"] = workflow.get("solver_config")
                step_summary = step_fn(network=network, **step_params)
                aggregated_summary.update(step_summary)
            # All other steps require network and return summary
            else:
                if network is None:
                    msg = f"Step '{step_name}' requires a network, but create_model has not been called yet. Ensure 'create_model' is the first step in the workflow."
                    raise RuntimeError(msg)  # noqa: TRY301 # TODO: Fix type checker
                step_summary = step_fn(network=network, **step_params)
                aggregated_summary.update(step_summary)

            print(f"   ✓ {step_name} completed\n")

        except Exception as e:
            print(f"   ✗ {step_name} failed: {e}\n")
            raise

    print(f"✅ Workflow complete for model: {model_id}\n")

    return network, aggregated_summary


def _inject_context(params: dict, context: dict, step_fn: callable) -> dict:
    """Inject context variables into step parameters.

    Uses function signature inspection to only inject parameters that the
    step function actually accepts, preventing TypeErrors from unexpected
    keyword arguments.

    Args:
        params: Step parameters dict from registry
        context: Context variables dict (model_id, csv_dir, profiles_path, inflow_path)
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
    context_vars = ["model_id", "csv_dir", "profiles_path", "inflow_path"]

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
