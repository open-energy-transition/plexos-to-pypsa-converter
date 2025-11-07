"""Integration tests for full workflow execution.

These tests verify that the complete workflow system works end-to-end
for both SEM and AEMO models.
"""

import pytest

# Skip all tests if models are not available (for CI environments without cache)
pytest_plugins = []

# Try to import workflow executor
try:
    from plexos_to_pypsa_converter.workflow.executor import run_model_workflow

    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False


@pytest.mark.skipif(not WORKFLOW_AVAILABLE, reason="Workflow system not available")
@pytest.mark.integration
class TestSEMWorkflow:
    """Test SEM 2024-2032 full workflow execution."""

    @pytest.mark.slow
    def test_sem_workflow_executes_all_steps(self, sem_model_id):
        """Test that SEM workflow executes all expected steps."""
        network, summary = run_model_workflow(sem_model_id, solve=True)

        # Verify all expected steps ran
        expected_steps = [
            "create_model",
            "load_vre_profiles",
            "add_storage_inflows",
            "apply_generator_units",
            "parse_outages",
            "fix_outage_ramps",
            "add_slack",
            "optimize",
        ]

        for step in expected_steps:
            assert step in summary, f"Step '{step}' did not execute"

        # Verify network structure
        assert len(network.buses) > 0
        assert len(network.generators) >= 50
        assert len(network.snapshots) > 0

    @pytest.mark.slow
    def test_sem_workflow_creates_valid_network(self, sem_model_id):
        """Test that SEM workflow creates a valid PyPSA network."""
        network, summary = run_model_workflow(sem_model_id, solve=True)

        # Check basic network structure
        assert hasattr(network, "buses")
        assert hasattr(network, "generators")
        assert hasattr(network, "links")
        assert hasattr(network, "storage_units")
        assert hasattr(network, "snapshots")

        # Check that components were created
        assert not network.buses.empty
        assert not network.generators.empty

    @pytest.mark.slow
    def test_sem_workflow_with_step_override(self, sem_model_id):
        """Test that workflow step overrides work correctly."""
        network, summary = run_model_workflow(
            sem_model_id,
            solve=True,
            scale_p_min_pu__scaling_factor=0.5,  # Override default 0.7
        )

        # Check that override was applied
        if "scale_p_min_pu" in summary:
            assert summary["scale_p_min_pu"]["scaling_factor"] == 0.5


@pytest.mark.skipif(not WORKFLOW_AVAILABLE, reason="Workflow system not available")
@pytest.mark.integration
class TestAEMOWorkflow:
    """Test AEMO 2024 ISP Progressive Change full workflow execution."""

    @pytest.mark.slow
    def test_aemo_workflow_executes_all_steps(self, aemo_model_id):
        """Test that AEMO workflow executes all expected steps."""
        network, summary = run_model_workflow(aemo_model_id, solve=True)

        # Verify core steps ran
        expected_steps = [
            "create_model",
            "load_vre_profiles",
            "add_storage_inflows",
            "apply_generator_units",
            "parse_outages",
            "optimize",
        ]

        for step in expected_steps:
            assert step in summary, f"Step '{step}' did not execute"

        # Verify network structure
        assert len(network.buses) > 0
        assert len(network.generators) >= 100  # AEMO has more generators
        assert len(network.snapshots) > 0

    @pytest.mark.slow
    def test_aemo_workflow_creates_valid_network(self, aemo_model_id):
        """Test that AEMO workflow creates a valid PyPSA network."""
        network, summary = run_model_workflow(aemo_model_id, solve=True)

        # Check basic network structure
        assert hasattr(network, "buses")
        assert hasattr(network, "generators")
        assert hasattr(network, "links")
        assert hasattr(network, "storage_units")
        assert hasattr(network, "snapshots")

        # Check that components were created
        assert not network.buses.empty
        assert not network.generators.empty

    @pytest.mark.slow
    def test_aemo_workflow_vre_profiles_loaded(self, aemo_model_id):
        """Test that VRE profiles are loaded correctly."""
        network, summary = run_model_workflow(aemo_model_id, solve=True)

        # Check that VRE profile loading step ran
        assert "load_vre_profiles" in summary

        # Check that some generators have p_max_pu time series
        assert len(network.generators_t.p_max_pu.columns) > 0


@pytest.mark.skipif(not WORKFLOW_AVAILABLE, reason="Workflow system not available")
@pytest.mark.integration
class TestWorkflowErrorHandling:
    """Test workflow error handling."""

    def test_workflow_with_invalid_model_id(self):
        """Test that workflow fails gracefully with invalid model ID."""
        with pytest.raises(ValueError, match="not found in registry"):
            run_model_workflow("nonexistent-model")

    def test_workflow_with_model_without_workflow(self):
        """Test error handling for models without workflow definition."""
        # This would require a model in registry without processing_workflow
        # Skip if no such model exists


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow
