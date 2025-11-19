"""Data discovery functions for PLEXOS database.

This module provides functions to automatically discover data file paths
and dependencies from PLEXOS database, enabling data-driven model creation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum

from plexos_to_pypsa_converter.utils.paths import resolve_relative_path


@dataclass
class DataFileInfo:
    """Information about a discovered data file."""

    object_name: str
    object_class: str
    property_name: str
    file_path: str
    file_type: str  # 'demand', 'vre', 'hydro', 'timeslice', 'other'


def extract_file_paths_from_property(prop: dict) -> list[str]:
    """Extract file paths from a property's texts field.

    Parameters
    ----------
    prop : dict
        Property dictionary from PlexosDB with 'texts' field

    Returns
    -------
    List[str]
        List of extracted file paths
    """
    file_paths: list[str] = []
    texts = prop.get("texts", "")

    if not texts:
        return file_paths

    # Pattern 1: Storage/Generator style with "Data File." prefix
    if "Data File." in texts:
        try:
            # Extract path after "Data File."
            remaining = texts.split("Data File.")[1]
            # Split on common delimiters to isolate the file path
            path = remaining.split()[0].strip()  # Take first word/path
            # Remove any trailing characters or additional formatting
            path = path.split("\n")[0].split("\t")[0].split(",")[0].strip()
            if path:
                file_paths.append(path)
        except (IndexError, AttributeError):
            pass

    # Pattern 2: Node "Filename" properties - direct file path in texts field
    elif prop.get("property") == "Filename" and texts.strip():
        try:
            # Direct file path, just clean it up
            path = texts.strip()
            path = path.split("\n")[0].split("\t")[0].split(",")[0].strip()
            if path and (
                "\\" in path or "/" in path
            ):  # Ensure it looks like a file path
                file_paths.append(path)
        except (IndexError, AttributeError):
            pass

    return file_paths


def categorize_file_by_path(
    file_path: str, object_class: str = "", property_name: str = ""
) -> str:
    """Categorize a file by its path pattern and source object context.

    Parameters
    ----------
    file_path : str
        The file path to categorize
    object_class : str, optional
        The PLEXOS object class that references this file
    property_name : str, optional
        The property name that contains this file reference

    Returns
    -------
    str
        File type: 'demand', 'vre', 'hydro', 'timeslice', 'other'
    """
    path_lower = file_path.lower()

    # For demand files, prioritize Node objects and filter out load_subtractor files
    if any(pattern in path_lower for pattern in ["demand", "load"]):
        # Skip Generator-sourced demand files (typically load_subtractor files that don't exist)
        if object_class == "Generator":
            return "other"  # Treat as 'other' to exclude from demand category
        # Prioritize Node Filename properties (actual demand files)
        elif (
            object_class == "Node"
            and property_name == "Filename"
            or any(pattern in path_lower for pattern in ["demand"])
        ):
            return "demand"
        else:
            return "other"
    elif any(pattern in path_lower for pattern in ["solar", "wind", "vre"]):
        return "vre"
    elif any(pattern in path_lower for pattern in ["hydro", "inflow"]):
        return "hydro"
    elif "timeslice" in path_lower:
        return "timeslice"
    else:
        return "other"


def discover_data_files_for_class(
    db: PlexosDB, class_enum: ClassEnum
) -> list[DataFileInfo]:
    """Discover all data file references for objects of a specific class.

    Parameters
    ----------
    db : PlexosDB
        PlexosDB instance
    class_enum : ClassEnum
        PLEXOS class to search (e.g., ClassEnum.Generator)

    Returns
    -------
    List[DataFileInfo]
        List of discovered data file information
    """
    discovered_files = []

    try:
        # Get all objects of this class
        objects = db.list_objects_by_class(class_enum)

        for obj_name in objects:
            try:
                # Get properties for this object
                properties = db.get_object_properties(class_enum, obj_name)

                for prop in properties:
                    # Extract file paths from this property
                    file_paths = extract_file_paths_from_property(prop)

                    for file_path in file_paths:
                        file_type = categorize_file_by_path(
                            file_path,
                            object_class=class_enum.name,
                            property_name=prop.get("property", ""),
                        )

                        data_file_info = DataFileInfo(
                            object_name=obj_name,
                            object_class=class_enum.name,
                            property_name=prop.get("property", ""),
                            file_path=file_path,
                            file_type=file_type,
                        )
                        discovered_files.append(data_file_info)

            except Exception as e:
                print(
                    f"Error processing object {obj_name} in class {class_enum.name}: {e}"
                )
                continue

    except Exception as e:
        print(f"Error discovering data files for class {class_enum.name}: {e}")

    return discovered_files


def discover_all_model_data_files(db: PlexosDB) -> dict[str, list[DataFileInfo]]:
    """Discover all data file references across all relevant PLEXOS classes.

    Parameters
    ----------
    db : PlexosDB
        PlexosDB instance

    Returns
    -------
    Dict[str, List[DataFileInfo]]
        Dictionary mapping file types to lists of DataFileInfo objects
    """
    all_files = []

    # Classes that commonly reference data files
    relevant_classes = [
        ClassEnum.Generator,
        ClassEnum.Storage,
        ClassEnum.Node,  # Demand files are found as Node properties, not separate Load class
        ClassEnum.Region,
    ]

    # Note: DataFile class objects don't have proper property structure for file discovery
    # File references are found through other object classes' properties instead

    for class_enum in relevant_classes:
        try:
            class_files = discover_data_files_for_class(db, class_enum)
            all_files.extend(class_files)
        except AttributeError as e:
            print(
                f"Warning: ClassEnum.{class_enum.name} not available in this PLEXOS version: {e}"
            )
            continue
        except Exception as e:
            print(f"Error discovering data files for class {class_enum.name}: {e}")
            continue

    # Group by file type
    files_by_type: dict[str, list[DataFileInfo]] = {
        "demand": [],
        "vre": [],
        "hydro": [],
        "timeslice": [],
        "other": [],
    }

    for file_info in all_files:
        files_by_type[file_info.file_type].append(file_info)

    return files_by_type


def resolve_relative_paths(
    files_by_type: dict[str, list[DataFileInfo]], main_directory: str
) -> dict[str, list[str]]:
    """Resolve relative file paths against a main directory and validate file existence.

    Parameters
    ----------
    files_by_type : Dict[str, List[DataFileInfo]]
        Dictionary of discovered files by type
    main_directory : str
        Main model directory to resolve paths against

    Returns
    -------
    Dict[str, List[str]]
        Dictionary mapping file types to lists of absolute paths (existing files only)
    """
    resolved_paths: dict[str, list[str]] = {
        "demand": [],
        "vre": [],
        "hydro": [],
        "timeslice": [],
        "other": [],
    }

    for file_type, file_infos in files_by_type.items():
        existing_files = []
        missing_files = []

        for file_info in file_infos:
            # Use cross-platform path resolution
            absolute_path = resolve_relative_path(main_directory, file_info.file_path)

            # Check if file exists, especially important for demand files
            if Path(absolute_path).exists():
                existing_files.append(absolute_path)
            else:
                missing_files.append(
                    {
                        "path": absolute_path,
                        "object": file_info.object_name,
                        "class": file_info.object_class,
                        "property": file_info.property_name,
                    }
                )

        resolved_paths[file_type] = existing_files

        # Report missing files for demand category (most critical)
        if file_type == "demand" and missing_files:
            print(f"Warning: {len(missing_files)} demand files not found on disk:")
            for missing in missing_files[:5]:  # Show first 5 missing files
                print(
                    f"  {missing['class']} '{missing['object']}' -> {missing['path']}"
                )
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
            print(f"Found {len(existing_files)} valid demand files")

    return resolved_paths


def infer_model_structure(
    files_by_type: dict[str, list[DataFileInfo]],
) -> dict[str, str]:
    """Infer model structure and file organization patterns.

    Parameters
    ----------
    files_by_type : Dict[str, List[DataFileInfo]]
        Dictionary of discovered files by type

    Returns
    -------
    Dict[str, str]
        Dictionary with inferred structure information
    """
    structure_info = {
        "demand_pattern": "unknown",
        "vre_pattern": "unknown",
        "hydro_pattern": "unknown",
        "has_timeslice": "no",
    }

    # Analyze demand file patterns
    demand_files = files_by_type.get("demand", [])
    if demand_files:
        # Check if demand files are per-node or aggregated
        demand_objects = {f.object_name for f in demand_files}
        if len(demand_objects) > 5:  # Many objects with demand files
            structure_info["demand_pattern"] = "per_node"
        else:
            structure_info["demand_pattern"] = "aggregated"

    # Check for VRE profiles
    vre_files = files_by_type.get("vre", [])
    if vre_files:
        structure_info["vre_pattern"] = "profiles_available"

    # Check for hydro inflows
    hydro_files = files_by_type.get("hydro", [])
    if hydro_files:
        structure_info["hydro_pattern"] = "inflows_available"

    # Check for timeslice data
    timeslice_files = files_by_type.get("timeslice", [])
    if timeslice_files:
        structure_info["has_timeslice"] = "yes"

    return structure_info


def discover_model_paths(db: PlexosDB, main_directory: str) -> dict[str, Any]:
    """Discover all model data dependencies from database.

    Parameters
    ----------
    db : PlexosDB
        PlexosDB instance
    main_directory : str
        Main model directory

    Returns
    -------
    Dict[str, Any]
        Dictionary containing discovered paths and model structure info
    """
    print("Discovering data files from PLEXOS database...")

    # Discover all data files
    files_by_type = discover_all_model_data_files(db)

    # Print discovery summary
    for file_type, file_list in files_by_type.items():
        if file_list:
            print(f"  Found {len(file_list)} {file_type} files")

    # Resolve paths
    resolved_paths = resolve_relative_paths(files_by_type, main_directory)

    # Infer model structure
    structure_info = infer_model_structure(files_by_type)

    # Determine appropriate paths for setup_network
    setup_paths = {}

    # Demand path - use the directory containing the actual demand files
    if resolved_paths["demand"]:
        # Get the parent directory of the first demand file (they should all be in the same directory)
        first_demand_file = Path(resolved_paths["demand"][0])
        demand_dir = str(first_demand_file.parent)
        setup_paths["demand_source"] = demand_dir
        setup_paths["snapshots_source"] = demand_dir

    # VRE profiles path - use main directory to access both solar and wind subdirectories
    if resolved_paths["vre"]:
        setup_paths["vre_profiles_path"] = main_directory

    # Hydro inflows path - use the directory containing the actual hydro files
    if resolved_paths["hydro"]:
        # Get the parent directory of the first hydro file (they should all be in the same directory)
        first_hydro_file = Path(resolved_paths["hydro"][0])
        hydro_dir = str(first_hydro_file.parent)
        setup_paths["inflow_path"] = hydro_dir

    # Timeslice CSV - use first timeslice file (absolute path)
    if resolved_paths["timeslice"]:
        setup_paths["timeslice_csv"] = resolved_paths["timeslice"][0]

    return {
        "discovered_files": files_by_type,
        "resolved_paths": resolved_paths,
        "structure_info": structure_info,
        "setup_paths": setup_paths,
    }
