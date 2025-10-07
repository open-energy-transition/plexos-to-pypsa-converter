"""Model registry and path utilities.

This module provides model metadata and path resolution for supported PLEXOS models.
Model data should be placed in src/examples/data/.
"""

import warnings
from pathlib import Path

from src.utils.model_paths import find_model_xml

# Model metadata registry
MODEL_REGISTRY = {
    "aemo-2024-green": {
        "name": "AEMO 2024 ISP - Green Energy Exports",
        "source": "AEMO",
        "xml_filename": "2024 ISP Green Energy Exports Model.xml",
    },
    "aemo-2024-isp-progressive": {
        "name": "AEMO 2024 ISP - Progressive Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Progressive Change Model.xml",
    },
    "aemo-2024-isp-step": {
        "name": "AEMO 2024 ISP - Step Change",
        "source": "AEMO",
        "xml_filename": "2024 ISP Step Change Model.xml",
    },
    "caiso-irp23": {
        "name": "CAISO IRP 2023 Stochastic",
        "source": "CAISO",
        "xml_filename": "CAISOIRP23Stochastic 20240517.xml",
    },
    "caiso-sa25": {
        "name": "CAISO 2025 Summer Assessment",
        "source": "CAISO",
        "xml_filename": "CAISOSA25 20250505.xml",
    },
    "nrel-118": {
        "name": "NREL Extended IEEE 118-bus",
        "source": "NREL",
        "xml_filename": "mti-118-mt-da-rt-reserves-all-generators.xml",
    },
    "sem-2024-2032": {
        "name": "SEM 2024-2032 Validation Model",
        "source": "SEM",
        "xml_filename": "PUBLIC Validation 2024-2032 Model 2025-03-14.xml",
    },
    "marei-eu": {
        "name": "European Power & Gas Model",
        "source": "UCC",
        "xml_filename": "European Integrated Power & Gas Model.xml",
    },
    "plexos-world-2015": {
        "name": "PLEXOS-World 2015 Gold V1.1",
        "source": "UCC",
        "xml_filename": "PLEXOS-World 2015 Gold V1.1.xml",
    },
    "plexos-world-spatial": {
        "name": "PLEXOS-World Spatial Resolution",
        "source": "UCC",
        "xml_filename": "PLEXOS-World Spatial Resolution Case Study (Second Journal Submission version).xml",
    },
    "plexos-message": {
        "name": "MESSAGEix-GLOBIOM",
        "source": "UCC",
        "xml_filename": "H2_Global_MESSAGEix_EN_NPi2020_500.xml",
    },
}


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
