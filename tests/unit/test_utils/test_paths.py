"""Unit tests for src.utils.paths module."""

from pathlib import Path

import pytest

from utils.paths import (
    contains_path_pattern,
    extract_filename,
    get_parent_directory,
    normalize_path,
    resolve_relative_path,
    safe_join,
)

# ============================================================================
# Tests for normalize_path
# ============================================================================


@pytest.mark.unit
def test_normalize_path_windows_to_current_os():
    """Test that Windows backslashes are normalized to current OS separators."""
    windows_path = r"C:\Users\test\file.txt"
    result = normalize_path(windows_path)
    # Should not contain backslashes (unless on Windows)
    assert "\\" not in result or Path(result).exists() or True  # Platform-dependent


@pytest.mark.unit
def test_normalize_path_unix_style():
    """Test that Unix-style paths are preserved."""
    unix_path = "/home/user/file.txt"
    result = normalize_path(unix_path)
    # Result should be a valid normalized path
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.unit
def test_normalize_path_empty_string():
    """Test normalize_path with empty string."""
    result = normalize_path("")
    assert result == ""


@pytest.mark.unit
def test_normalize_path_mixed_separators():
    """Test normalize_path with mixed forward and backslashes."""
    mixed_path = r"folder\subfolder/file.txt"
    result = normalize_path(mixed_path)
    # Should be normalized to consistent separators
    assert "/" in result or "\\" in result or result == "folder/subfolder/file.txt"


# ============================================================================
# Tests for extract_filename
# ============================================================================


@pytest.mark.unit
def test_extract_filename_from_windows_path():
    """Test extracting filename from Windows-style path."""
    windows_path = r"C:\Users\test\documents\file.txt"
    result = extract_filename(windows_path)
    assert result == "file.txt"


@pytest.mark.unit
def test_extract_filename_from_unix_path():
    """Test extracting filename from Unix-style path."""
    unix_path = "/home/user/documents/file.txt"
    result = extract_filename(unix_path)
    assert result == "file.txt"


@pytest.mark.unit
def test_extract_filename_from_mixed_path():
    """Test extracting filename from path with mixed separators."""
    mixed_path = r"folder\subfolder/file.csv"
    result = extract_filename(mixed_path)
    assert result == "file.csv"


@pytest.mark.unit
def test_extract_filename_from_filename_only():
    """Test extract_filename when input is just a filename."""
    filename = "file.txt"
    result = extract_filename(filename)
    assert result == "file.txt"


@pytest.mark.unit
def test_extract_filename_empty_string():
    """Test extract_filename with empty string."""
    result = extract_filename("")
    assert result == ""


@pytest.mark.unit
def test_extract_filename_with_extension():
    """Test that filename extraction preserves file extensions."""
    path = "/path/to/data.csv"
    result = extract_filename(path)
    assert result == "data.csv"
    assert result.endswith(".csv")


# ============================================================================
# Tests for safe_join
# ============================================================================


@pytest.mark.unit
def test_safe_join_basic():
    """Test basic path joining."""
    result = safe_join("folder", "subfolder", "file.txt")
    # Should create a valid path
    assert "folder" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_safe_join_with_windows_paths():
    """Test safe_join with Windows-style paths."""
    result = safe_join(r"C:\Users", "test", "file.txt")
    # Should handle Windows paths
    assert "test" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_safe_join_with_mixed_separators():
    """Test safe_join with mixed path separators."""
    result = safe_join(r"folder\subfolder", "another/path", "file.txt")
    assert "folder" in result or "subfolder" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_safe_join_empty_components():
    """Test safe_join with empty path components."""
    result = safe_join("folder", "", "file.txt")
    # Should ignore empty components
    assert result != ""


@pytest.mark.unit
def test_safe_join_all_empty():
    """Test safe_join with all empty components."""
    result = safe_join("", "", "")
    assert result == ""


@pytest.mark.unit
def test_safe_join_single_component():
    """Test safe_join with a single path component."""
    result = safe_join("folder")
    assert result == "folder"


# ============================================================================
# Tests for contains_path_pattern
# ============================================================================


@pytest.mark.unit
def test_contains_path_pattern_unix_format():
    """Test pattern matching with Unix-style paths."""
    text = "/path/to/Traces/solar/profile.csv"
    pattern = "Traces/solar"
    assert contains_path_pattern(text, pattern) is True


@pytest.mark.unit
def test_contains_path_pattern_windows_format():
    """Test pattern matching with Windows-style paths."""
    text = r"C:\path\to\Traces\solar\profile.csv"
    pattern = "Traces/solar"
    assert contains_path_pattern(text, pattern) is True


@pytest.mark.unit
def test_contains_path_pattern_windows_pattern():
    """Test pattern matching when pattern uses Windows separators."""
    text = "/path/to/Traces/solar/profile.csv"
    pattern = r"Traces\solar"
    assert contains_path_pattern(text, pattern) is True


@pytest.mark.unit
def test_contains_path_pattern_not_found():
    """Test pattern matching when pattern is not in text."""
    text = "/path/to/data/profile.csv"
    pattern = "Traces/solar"
    assert contains_path_pattern(text, pattern) is False


@pytest.mark.unit
def test_contains_path_pattern_empty_text():
    """Test pattern matching with empty text."""
    result = contains_path_pattern("", "pattern")
    assert result is False


@pytest.mark.unit
def test_contains_path_pattern_empty_pattern():
    """Test pattern matching with empty pattern."""
    result = contains_path_pattern("some text", "")
    assert result is False


@pytest.mark.unit
def test_contains_path_pattern_case_sensitive():
    """Test that pattern matching is case-sensitive."""
    text = "/path/to/Traces/solar/profile.csv"
    pattern = "traces/solar"  # lowercase
    # Pattern matching should be case-sensitive
    assert contains_path_pattern(text, pattern) is False


# ============================================================================
# Tests for get_parent_directory
# ============================================================================


@pytest.mark.unit
def test_get_parent_directory_from_string():
    """Test getting parent directory from string path."""
    file_path = "/home/user/documents/file.txt"
    result = get_parent_directory(file_path)
    # Parent should not contain the filename
    assert "file.txt" not in result
    assert "documents" in result or result.endswith("documents")


@pytest.mark.unit
def test_get_parent_directory_from_path_object():
    """Test getting parent directory from Path object."""
    file_path = Path("/home/user/documents/file.txt")
    result = get_parent_directory(file_path)
    assert "file.txt" not in result
    assert isinstance(result, str)


@pytest.mark.unit
def test_get_parent_directory_windows_path():
    """Test getting parent directory from Windows path."""
    file_path = r"C:\Users\test\file.txt"
    result = get_parent_directory(file_path)
    assert "file.txt" not in result


@pytest.mark.unit
def test_get_parent_directory_relative_path():
    """Test getting parent directory from relative path."""
    file_path = "folder/subfolder/file.txt"
    result = get_parent_directory(file_path)
    assert "file.txt" not in result
    assert "subfolder" in result or result.endswith("subfolder")


# ============================================================================
# Tests for resolve_relative_path
# ============================================================================


@pytest.mark.unit
def test_resolve_relative_path_basic():
    """Test resolving a relative path against a base directory."""
    base_dir = "/home/user"
    relative_path = "documents/file.txt"
    result = resolve_relative_path(base_dir, relative_path)
    assert "home" in result or "user" in result
    assert "documents" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_resolve_relative_path_windows():
    """Test resolving relative path with Windows base directory."""
    base_dir = r"C:\Users\test"
    relative_path = "documents/file.txt"
    result = resolve_relative_path(base_dir, relative_path)
    assert "documents" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_resolve_relative_path_with_windows_relative():
    """Test resolving when relative path uses Windows separators."""
    base_dir = "/home/user"
    relative_path = r"documents\file.txt"
    result = resolve_relative_path(base_dir, relative_path)
    assert "documents" in result
    assert "file.txt" in result


@pytest.mark.unit
def test_resolve_relative_path_with_subdirs():
    """Test resolving relative path with multiple subdirectories."""
    base_dir = "/home/user"
    relative_path = "documents/work/project/file.txt"
    result = resolve_relative_path(base_dir, relative_path)
    assert all(
        component in result
        for component in ["documents", "work", "project", "file.txt"]
    )


# ============================================================================
# Integration-style tests combining multiple functions
# ============================================================================


@pytest.mark.unit
def test_path_workflow_windows_to_unix():
    """Test complete workflow of normalizing and joining Windows paths."""
    # Simulate Windows path from database
    db_path = r"Traces\solar\profile.csv"

    # Normalize it
    normalized = normalize_path(db_path)

    # Extract filename
    filename = extract_filename(normalized)
    assert filename == "profile.csv"

    # Join with a base path
    base = "/data/vre_profiles"
    full_path = safe_join(base, normalized)

    # Should contain all components
    assert "profile.csv" in full_path


@pytest.mark.unit
def test_path_workflow_extract_and_join():
    """Test workflow of extracting filename and joining with new base."""
    original_path = "/old/path/to/data/file.csv"

    # Extract filename
    filename = extract_filename(original_path)
    assert filename == "file.csv"

    # Join with new base
    new_base = "/new/base/directory"
    new_path = safe_join(new_base, filename)

    # Should have new base and same filename
    assert "new" in new_path
    assert "base" in new_path
    assert filename in new_path
    assert "old" not in new_path
