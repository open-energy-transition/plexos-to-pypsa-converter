"""Model path resolution for PLEXOS-to-PyPSA converter.

This module handles finding model data in src/examples/data/ and provides
a consistent interface for model path lookup.

If a model is not found locally and auto-download is enabled (default),
the system will attempt to download and install the model using its recipe
definition from MODEL_REGISTRY.
"""

import os
from pathlib import Path

from src.db.models import MODEL_REGISTRY
from src.utils.recipe_executor import RecipeExecutor


def get_data_root() -> Path:
    """Get the root directory for model data.

    Returns
    -------
    Path
        Path to src/examples/data/
    """
    # Assumes this file is in src/utils/
    return Path(__file__).parent.parent / "examples" / "data"


def find_model_xml(
    model_id: str, xml_filename: str | None = None, auto_download: bool = True
) -> Path | None:
    """Find XML file for a given model in src/examples/data/.

    If the model is not found locally and auto_download is enabled, this function
    will attempt to download and install the model using its recipe definition.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g., 'aemo-2024-isp-progressive')
    xml_filename : str, optional
        Specific XML filename to look for. If None, searches for any .xml file.
    auto_download : bool, default True
        If True, automatically download missing models using their recipe.
        Can be disabled globally with environment variable: PLEXOS_AUTO_DOWNLOAD=0

    Returns
    -------
    Path or None
        Path to XML file if found, None otherwise

    Examples
    --------
    >>> xml_path = find_model_xml("aemo-2024-isp-progressive")
    >>> if xml_path:
    ...     print(f"Found at: {xml_path}")
    ... else:
    ...     print("Model not available")

    >>> # Disable auto-download for this call
    >>> xml_path = find_model_xml("model-id", auto_download=False)
    """
    # First try to find existing model
    xml_path = _find_existing_model(model_id, xml_filename)
    if xml_path is not None:
        return xml_path

    # Model not found - check if auto-download is enabled
    auto_download = auto_download and _auto_download_enabled()

    if auto_download and _has_recipe(model_id) and _execute_recipe(model_id):
        # Recipe succeeded - try to find model again
        return _find_existing_model(model_id, xml_filename)

    return None


def _find_existing_model(model_id: str, xml_filename: str | None = None) -> Path | None:
    """Find existing model in src/examples/data/.

    This is a helper function that looks for models without triggering downloads.

    Parameters
    ----------
    model_id : str
        Model identifier
    xml_filename : str, optional
        Specific XML filename to look for

    Returns
    -------
    Path or None
        Path to XML file if found, None otherwise
    """
    data_root = get_data_root()

    # Model ID to subdirectory mapping
    # Note: The directory structure uses model IDs as top-level folder names
    model_dirs = {
        "aemo-2024-green": "aemo-2024-green",
        "aemo-2024-isp-progressive": "aemo-2024-isp-progressive",
        "aemo-2024-isp-step": "aemo-2024-isp-step",
        "caiso-irp23": "caiso-irp23",
        "caiso-sa25": "caiso-sa25",
        "nrel-118": "nrel-118",
        "sem-2024-2032": "sem-2024-2032",
        "marei-eu": "marei-eu",
        "plexos-world-2015": "plexos-world-2015",
        "plexos-world-spatial": "plexos-world-spatial",
        "plexos-message": "plexos-message",
    }

    if model_id not in model_dirs:
        return None

    model_dir = data_root / model_dirs[model_id]

    if not model_dir.exists():
        return None

    # Special case: plexos-message has XML in a subdirectory
    if model_id == "plexos-message":
        subdir = model_dir / "MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link-main"
        if subdir.exists():
            model_dir = subdir

    # If specific filename provided, look for it
    if xml_filename:
        xml_path = model_dir / xml_filename
        return xml_path if xml_path.exists() else None

    # Otherwise, search for any .xml file (excluding PLEXOS_*.xml config files)
    xml_files = [
        f for f in model_dir.rglob("*.xml") if not f.name.startswith("PLEXOS_")
    ]
    return xml_files[0] if xml_files else None


def _auto_download_enabled() -> bool:
    """Check if auto-download is enabled via environment variable.

    Returns
    -------
    bool
        True if auto-download is enabled, False otherwise

    Notes
    -----
    Auto-download can be disabled by setting environment variable:
    PLEXOS_AUTO_DOWNLOAD=0
    """
    return os.getenv("PLEXOS_AUTO_DOWNLOAD", "1") != "0"


def _has_recipe(model_id: str) -> bool:
    """Check if model has a recipe defined in MODEL_REGISTRY.

    Parameters
    ----------
    model_id : str
        Model identifier

    Returns
    -------
    bool
        True if model has a recipe, False otherwise
    """
    return "recipe" in MODEL_REGISTRY.get(model_id, {})


def _execute_recipe(model_id: str) -> bool:
    """Execute recipe for a model.

    Parameters
    ----------
    model_id : str
        Model identifier

    Returns
    -------
    bool
        True if recipe executed successfully, False otherwise
    """
    model_config = MODEL_REGISTRY.get(model_id)
    if not model_config or "recipe" not in model_config:
        return False

    recipe = model_config["recipe"]
    model_dir = get_data_root() / model_id

    executor = RecipeExecutor(model_id, model_dir, verbose=True)

    try:
        return executor.execute_recipe(recipe)
    except Exception as e:
        print(f"\n❌ Failed to download model '{model_id}': {e}")
        print("You may need to download this model manually.")
        print(f"Place model files in: {model_dir}\n")
        return False


def get_model_directory(model_id: str) -> Path | None:
    """Get the directory containing model data.

    Parameters
    ----------
    model_id : str
        Model identifier

    Returns
    -------
    Path or None
        Directory path if found, None otherwise

    Examples
    --------
    >>> model_dir = get_model_directory("aemo-2024-isp-progressive")
    >>> if model_dir:
    ...     demand_path = model_dir / "Traces" / "demand"
    """
    xml_path = find_model_xml(model_id)
    return xml_path.parent if xml_path else None
