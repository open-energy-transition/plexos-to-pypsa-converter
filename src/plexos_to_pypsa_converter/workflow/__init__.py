"""Workflow system for declarative model processing pipelines.

This module provides a registry-driven workflow system that eliminates code
duplication in example scripts by defining processing steps and configurations
centrally in the model registry.

Main entry point:
    run_model_workflow() - Execute a model's processing workflow from registry

Example:
    >>> from plexos_to_pypsa_converter.workflow import run_model_workflow
    >>> network, summary = run_model_workflow("sem-2024-2032")
"""

from plexos_to_pypsa_converter.workflow.executor import run_model_workflow
from plexos_to_pypsa_converter.workflow.filters import (
    FILTER_PRESETS,
    resolve_filter_preset,
)
from plexos_to_pypsa_converter.workflow.steps import STEP_REGISTRY

__all__ = [
    "run_model_workflow",
    "STEP_REGISTRY",
    "FILTER_PRESETS",
    "resolve_filter_preset",
]
