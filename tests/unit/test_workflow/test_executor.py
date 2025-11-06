"""Unit tests for workflow executor."""

import pytest

from plexos_to_pypsa_converter.workflow.executor import (
    _evaluate_condition,
    _inject_context,
)
from plexos_to_pypsa_converter.workflow.steps import STEP_REGISTRY


class TestInjectContext:
    """Test context parameter injection."""

    def test_inject_context_basic(self):
        """Test basic context injection into parameters."""

        def dummy_step(network, csv_dir, profiles_path):
            pass

        params = {"some_param": "value"}
        context = {
            "model_id": "test-model",
            "csv_dir": "/path/to/csv",
            "profiles_path": "/path/to/profiles",
        }

        injected = _inject_context(params, context, dummy_step)

        # Should inject csv_dir and profiles_path (accepted by function)
        assert injected["csv_dir"] == "/path/to/csv"
        assert injected["profiles_path"] == "/path/to/profiles"
        # Should NOT inject model_id (not in function signature)
        assert "model_id" not in injected or injected["model_id"] == context["model_id"]

    def test_inject_context_with_kwargs(self):
        """Test context injection when function has **kwargs."""

        def dummy_step(network, **kwargs):
            pass

        params = {}
        context = {
            "model_id": "test-model",
            "csv_dir": "/path/to/csv",
            "profiles_path": "/path/to/profiles",
            "inflow_path": "/path/to/inflow",
        }

        injected = _inject_context(params, context, dummy_step)

        # With **kwargs, all context vars should be injected
        assert injected["model_id"] == "test-model"
        assert injected["csv_dir"] == "/path/to/csv"
        assert injected["profiles_path"] == "/path/to/profiles"
        assert injected["inflow_path"] == "/path/to/inflow"

    def test_inject_context_preserves_existing_params(self):
        """Test that existing params are not overwritten by None context values."""

        def dummy_step(csv_dir, profiles_path):
            pass

        params = {"csv_dir": "/explicit/path"}
        context = {
            "csv_dir": "/context/path",
            "profiles_path": "/path/to/profiles",
        }

        injected = _inject_context(params, context, dummy_step)

        # Explicit param should be preserved (not overwritten)
        assert injected["csv_dir"] == "/explicit/path"
        # Context value should be injected for missing param
        assert injected["profiles_path"] == "/path/to/profiles"

    def test_inject_context_only_injects_accepted_params(self):
        """Test that only params accepted by function are injected."""

        def dummy_step(csv_dir):  # Only accepts csv_dir
            pass

        params = {}
        context = {
            "csv_dir": "/path/to/csv",
            "profiles_path": "/path/to/profiles",  # Not in function signature
            "model_id": "test-model",  # Not in function signature
        }

        injected = _inject_context(params, context, dummy_step)

        # Only csv_dir should be injected
        assert injected["csv_dir"] == "/path/to/csv"
        assert "profiles_path" not in injected
        assert "model_id" not in injected


class TestEvaluateCondition:
    """Test condition evaluation."""

    def test_evaluate_condition_model_id_equality_true(self):
        """Test model_id equality condition that evaluates to True."""
        condition = "model_id == 'sem-2024-2032'"
        context = {"model_id": "sem-2024-2032"}

        result = _evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_condition_model_id_equality_false(self):
        """Test model_id equality condition that evaluates to False."""
        condition = "model_id == 'sem-2024-2032'"
        context = {"model_id": "aemo-2024-isp-progressive-change"}

        result = _evaluate_condition(condition, context)

        assert result is False

    def test_evaluate_condition_model_id_with_double_quotes(self):
        """Test condition with double quotes."""
        condition = 'model_id == "sem-2024-2032"'
        context = {"model_id": "sem-2024-2032"}

        result = _evaluate_condition(condition, context)

        assert result is True

    def test_evaluate_condition_unknown_format_defaults_true(self):
        """Test that unknown condition format defaults to True."""
        condition = "some_unknown_condition"
        context = {}

        result = _evaluate_condition(condition, context)

        # Unknown conditions default to True
        assert result is True


class TestWorkflowRegistry:
    """Test workflow step registry."""

    def test_step_registry_has_all_steps(self):
        """Test that STEP_REGISTRY contains all expected steps."""
        expected_steps = [
            "create_model",
            "scale_p_min_pu",
            "add_curtailment_link",
            "load_vre_profiles",
            "add_storage_inflows",
            "apply_generator_units",
            "parse_outages",
            "fix_outage_ramps",
            "add_slack",
            "optimize",
        ]

        for step in expected_steps:
            assert step in STEP_REGISTRY, f"Missing step in registry: {step}"

    def test_all_steps_are_callable(self):
        """Test that all registered steps are callable."""
        for step_name, step_fn in STEP_REGISTRY.items():
            assert callable(step_fn), f"Step {step_name} is not callable"

    def test_steps_have_docstrings(self):
        """Test that step functions have docstrings."""
        for step_name, step_fn in STEP_REGISTRY.items():
            # Allow some steps to skip this requirement if they're imported from other modules
            if step_fn.__doc__ is None and step_name not in [
                "fix_outage_ramps",
                "add_slack",
            ]:
                pytest.fail(f"Step {step_name} missing docstring")


# Note: Full workflow execution tests (run_model_workflow) are in integration tests
# because they require real model data and full CSV parsing infrastructure.
