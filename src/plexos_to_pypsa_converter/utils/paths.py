"""Cross-platform path handling utilities.

This module provides consistent path handling across Windows, Mac, and Linux systems,
particularly for dealing with PLEXOS database paths that may contain Windows-style backslashes.
"""

import os
from pathlib import Path


def normalize_path(path: str) -> str:
    """Normalize a path string to use the current OS path separators.

    Converts Windows backslashes to forward slashes or appropriate OS separators.

    Parameters
    ----------
    path : str
        Path string that may contain Windows-style backslashes

    Returns
    -------
    str
        Normalized path string using current OS separators
    """
    if not path:
        return path

    # Replace Windows backslashes with forward slashes, then normalize
    normalized = path.replace("\\", "/")
    return os.path.normpath(normalized)


def extract_filename(path: str) -> str:
    """Extract just the filename from a path, handling cross-platform separators.

    Parameters
    ----------
    path : str
        Full or relative path that may contain Windows or Unix separators

    Returns
    -------
    str
        Just the filename portion of the path
    """
    if not path:
        return path

    # Normalize the path first, then extract filename using pathlib
    normalized = normalize_path(path)
    return Path(normalized).name


def safe_join(*paths: str) -> str:
    """Safely join path components, normalizing each component first.

    Parameters
    ----------
    *paths : str
        Path components to join

    Returns
    -------
    str
        Joined and normalized path
    """
    if not paths:
        return ""

    # Normalize each component before joining
    normalized_paths = [normalize_path(p) for p in paths if p]
    if not normalized_paths:
        return ""
    result = Path(normalized_paths[0])
    for path in normalized_paths[1:]:
        result = result / path
    return str(result)


def contains_path_pattern(text: str, pattern: str) -> bool:
    """Check if text contains a path pattern, handling cross-platform separators.

    This is useful for checking if a text field contains references to specific
    directory patterns regardless of the separator style used.

    Parameters
    ----------
    text : str
        Text to search in
    pattern : str
        Path pattern to search for (e.g., "Traces/solar/")

    Returns
    -------
    bool
        True if the pattern is found (with any separator style)
    """
    if not text or not pattern:
        return False

    # Normalize both text and pattern to forward slashes for comparison
    norm_text = text.replace("\\", "/")
    norm_pattern = pattern.replace("\\", "/")

    return norm_pattern in norm_text


def get_parent_directory(file_path: str | Path) -> str:
    """Get the parent directory of a file path, handling cross-platform paths.

    Parameters
    ----------
    file_path : str or Path
        Path to a file

    Returns
    -------
    str
        Parent directory path
    """
    if isinstance(file_path, str):
        file_path = Path(normalize_path(file_path))
    elif isinstance(file_path, Path):
        # Ensure the path uses current OS separators
        file_path = Path(normalize_path(str(file_path)))

    return str(file_path.parent)


def resolve_relative_path(base_dir: str, relative_path: str) -> str:
    """Resolve a relative path against a base directory, handling cross-platform paths.

    Parameters
    ----------
    base_dir : str
        Base directory path
    relative_path : str
        Relative path that may contain Windows or Unix separators

    Returns
    -------
    str
        Absolute path
    """
    base_path = Path(normalize_path(base_dir))
    rel_path = normalize_path(relative_path)

    return str(base_path / rel_path)
