"""Model registry and path utilities.

This module provides model metadata and path resolution for supported PLEXOS models.
Model data should be placed in src/examples/data/.

Each model can optionally include a "recipe" - a list of instructions for automatically
downloading, extracting, and organizing model files. Recipes are executed when a model
is requested but not found locally (if auto_download is enabled).

Recipe Instruction Types:
- download: Download file from URL
- extract: Extract archive to target directory
- move: Move files/directories (supports glob patterns)
- copy: Copy files/directories
- rename: Rename file/directory
- delete: Delete files/directories
- create_dir: Create directory
- flatten: Flatten nested directory structure
- validate: Validate model installation
- manual: Display manual download instructions

See recipe_executor.py for detailed instruction schema and examples.
"""

import warnings
from pathlib import Path

from src.db.registry import MODEL_REGISTRY
from src.utils.model_paths import find_model_xml


def get_model_xml_path(model_id: str) -> Path | None:
    """Get XML path for a model, or None if not found.

    Parameters
    ----------
    model_id : str
        Model identifier from MODEL_REGISTRY

    Returns
    -------
    Path or None
        Path to XML file if model data exists in src/examples/data/, None otherwise

    Raises
    ------
    ValueError
        If model_id is not in MODEL_REGISTRY

    Examples
    --------
    >>> xml_path = get_model_xml_path("aemo-2024-isp-progressive")
    >>> if xml_path:
    ...     print(f"Found at: {xml_path}")
    ... else:
    ...     print("Model not downloaded. Please download to src/examples/data/")
    """
    if model_id not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        error_msg = (
            f"Unknown model ID: {model_id}. Available models: {available_models}"
        )
        raise ValueError(error_msg)

    xml_filename = MODEL_REGISTRY[model_id].get("xml_filename")
    return find_model_xml(model_id, xml_filename)


# Backwards compatibility: Provide INPUT_XMLS but with deprecation warning
def _get_input_xmls() -> dict:
    """Provide INPUT_XMLS dictionary with deprecation warning.

    DEPRECATED: Get INPUT_XMLS dictionary. Use get_model_xml_path() instead.
    """
    warnings.warn(
        "INPUT_XMLS is deprecated and will be removed in a future version. "
        "Use get_model_xml_path(model_id) instead, which searches for models "
        "in src/examples/data/.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = {}
    for model_id in MODEL_REGISTRY:
        xml_path = get_model_xml_path(model_id)
        if xml_path is not None:
            result[model_id] = str(xml_path)
    return result


# Provide INPUT_XMLS for backwards compatibility
INPUT_XMLS = property(lambda self: _get_input_xmls())
