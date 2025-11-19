"""Unit tests for model registry."""

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.workflow.steps import STEP_REGISTRY


class TestModelRegistry:
    """Test MODEL_REGISTRY structure and content."""

    def test_registry_not_empty(self):
        """Test that registry contains models."""
        assert len(MODEL_REGISTRY) > 0, "MODEL_REGISTRY is empty"

    def test_all_models_have_required_fields(self):
        """Test that all models have required metadata fields."""
        required_fields = ["name", "source", "xml_filename", "model_type"]

        for model_id, config in MODEL_REGISTRY.items():
            for field in required_fields:
                assert field in config, (
                    f"Model {model_id} missing required field: {field}"
                )

    def test_model_types_are_valid(self):
        """Test that all model_type values are valid."""
        valid_types = ["electricity", "multi_sector_gas_electric", "multi_sector_flow"]

        for model_id, config in MODEL_REGISTRY.items():
            model_type = config["model_type"]
            assert model_type in valid_types, (
                f"Model {model_id} has invalid model_type: {model_type}"
            )

    def test_sem_model_exists(self):
        """Test that SEM model is in registry."""
        assert "sem-2024-2032" in MODEL_REGISTRY

    def test_aemo_model_exists(self):
        """Test that AEMO Progressive Change model is in registry."""
        assert "aemo-2024-isp-progressive-change" in MODEL_REGISTRY


class TestWorkflowDefinitions:
    """Test workflow definitions in MODEL_REGISTRY."""

    def test_workflow_steps_reference_valid_step_names(self):
        """Test that workflow step names exist in STEP_REGISTRY."""
        for model_id, config in MODEL_REGISTRY.items():
            if "processing_workflow" not in config:
                continue

            workflow = config["processing_workflow"]
            if "steps" not in workflow:
                continue

            for step_def in workflow["steps"]:
                step_name = step_def["name"]
                assert step_name in STEP_REGISTRY, (
                    f"Model {model_id} references unknown step: {step_name}"
                )

    def test_workflow_steps_have_name_field(self):
        """Test that all workflow steps have 'name' field."""
        for model_id, config in MODEL_REGISTRY.items():
            if "processing_workflow" not in config:
                continue

            workflow = config["processing_workflow"]
            if "steps" not in workflow:
                continue

            for idx, step_def in enumerate(workflow["steps"]):
                assert "name" in step_def, (
                    f"Model {model_id} step {idx} missing 'name' field"
                )

    def test_sem_workflow_definition(self):
        """Test SEM workflow definition is complete."""
        sem_config = MODEL_REGISTRY["sem-2024-2032"]

        assert "processing_workflow" in sem_config
        workflow = sem_config["processing_workflow"]

        assert "steps" in workflow
        assert len(workflow["steps"]) > 0

        # Check for key steps
        step_names = [step["name"] for step in workflow["steps"]]
        assert "create_model" in step_names
        assert "load_vre_profiles" in step_names
        assert "optimize" in step_names

    def test_aemo_workflow_definition(self):
        """Test AEMO workflow definition is complete."""
        aemo_config = MODEL_REGISTRY["aemo-2024-isp-progressive-change"]

        assert "processing_workflow" in aemo_config
        workflow = aemo_config["processing_workflow"]

        assert "steps" in workflow
        assert len(workflow["steps"]) > 0

        # Check for key steps
        step_names = [step["name"] for step in workflow["steps"]]
        assert "create_model" in step_names
        assert "load_vre_profiles" in step_names
        assert "optimize" in step_names


class TestDefaultConfigurations:
    """Test default configurations in MODEL_REGISTRY."""

    def test_electricity_models_have_default_config(self):
        """Test that electricity models have default_config."""
        for model_id, config in MODEL_REGISTRY.items():
            if config["model_type"] == "electricity":
                # Electricity models should have some configuration
                # (either default_config or processing_workflow)
                has_config = (
                    "default_config" in config or "processing_workflow" in config
                )
                assert has_config, f"Electricity model {model_id} missing configuration"

    def test_sem_default_config(self):
        """Test SEM default configuration."""
        sem_config = MODEL_REGISTRY["sem-2024-2032"]

        assert "default_config" in sem_config
        default = sem_config["default_config"]

        # SEM uses per_node strategy with target node
        assert "demand_assignment_strategy" in default
        assert default["demand_assignment_strategy"] == "per_node"

    def test_aemo_default_config(self):
        """Test AEMO default configuration."""
        aemo_config = MODEL_REGISTRY["aemo-2024-isp-progressive-change"]

        assert "default_config" in aemo_config
        default = aemo_config["default_config"]

        # AEMO uses per_node strategy
        assert "demand_assignment_strategy" in default
        assert default["demand_assignment_strategy"] == "per_node"
