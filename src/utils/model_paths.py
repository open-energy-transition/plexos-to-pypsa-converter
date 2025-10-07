"""Model path resolution for PLEXOS-to-PyPSA converter.

This module handles finding model data in src/examples/data/ and provides
a consistent interface for model path lookup.
"""

from pathlib import Path


def get_data_root() -> Path:
    """Get the root directory for model data.

    Returns
    -------
    Path
        Path to src/examples/data/
    """
    # Assumes this file is in src/utils/
    return Path(__file__).parent.parent / "examples" / "data"


def find_model_xml(model_id: str, xml_filename: str | None = None) -> Path | None:
    """Find XML file for a given model in src/examples/data/.

    Parameters
    ----------
    model_id : str
        Model identifier (e.g., 'aemo-2024-isp-progressive')
    xml_filename : str, optional
        Specific XML filename to look for. If None, searches for any .xml file.

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
    ...     print("Model not downloaded yet")
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
