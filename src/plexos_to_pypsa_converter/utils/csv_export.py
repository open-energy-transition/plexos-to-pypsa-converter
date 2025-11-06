"""Utility for auto-exporting CSVs from PLEXOS XML using COAD."""

import datetime
import logging
from pathlib import Path

from coad.COAD import COAD
from coad.export_plexos_model import get_all_objects, write_object_report

logger = logging.getLogger(__name__)


def export_csvs_from_xml(
    xml_file: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Export CSVs from PLEXOS XML file using COAD.

    This function exports all objects from all systems in the PLEXOS model
    to CSV files organized by system name.

    Parameters
    ----------
    xml_file : str | Path
        Path to PLEXOS XML file
    output_dir : str | Path, optional
        Directory to write CSVs to. If None, creates 'csvs_from_xml'
        subdirectory next to the XML file.

    Returns
    -------
    Path
        Path to the directory containing exported CSVs

    Examples
    --------
    >>> xml_file = "models/sem-2024/model.xml"
    >>> csv_dir = export_csvs_from_xml(xml_file)
    >>> print(f"CSVs exported to: {csv_dir}")
    CSVs exported to: models/sem-2024/csvs_from_xml
    """
    xml_file = Path(xml_file)

    if output_dir is None:
        output_dir = xml_file.parent / "csvs_from_xml"
    else:
        output_dir = Path(output_dir)

    logger.info("Exporting CSVs from XML using COAD...")
    logger.info(f"  Source: {xml_file}")
    logger.info(f"  Destination: {output_dir}")

    # Load PLEXOS XML with COAD
    try:
        c = COAD(str(xml_file))
    except Exception:
        logger.exception("Failed to load XML with COAD")
        raise

    # Get all system names
    try:
        system_names = c.list("System")
    except Exception as e:
        logger.warning(f"Could not list systems: {e}. Will export all objects.")
        system_names = []

    if not system_names:
        logger.info("  No systems found in XML. Exporting all objects...")
        # If no systems, export everything to root output_dir
        _export_system_csvs(
            c,
            system_name=None,
            output_dir=output_dir,
            xml_file=xml_file,
        )
    else:
        logger.info(f"  Found {len(system_names)} system(s): {system_names}")

        # Export each system to its own subdirectory
        for system_name in system_names:
            logger.info(f"  Exporting system: {system_name}...")
            try:
                coad_obj = c["System"][system_name]
                system_output_dir = output_dir / system_name

                _export_system_csvs(
                    c,
                    system_name=system_name,
                    output_dir=system_output_dir,
                    xml_file=xml_file,
                    coad_obj=coad_obj,
                )
            except Exception:
                logger.exception(f"Failed to export system {system_name}")
                raise

    logger.info(f"CSV export complete: {output_dir}")
    return output_dir


def _export_system_csvs(
    coad: COAD,
    system_name: str | None,
    output_dir: Path,
    xml_file: Path,
    coad_obj=None,
) -> None:
    """Export CSVs for a single system or all objects if no system specified."""
    # Get all objects for this system (or all objects if no system)
    try:
        if coad_obj is not None:
            all_objs = get_all_objects(coad_obj.coad)
        else:
            # Export all objects from entire model
            all_objs = get_all_objects(coad)
    except Exception:
        logger.exception("Failed to get objects")
        raise

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export objects to CSV
    try:
        if coad_obj is not None:
            write_object_report(
                coad_obj,
                interesting_objs=all_objs,
                folder=str(output_dir),
            )
        else:
            # Export from root COAD object
            write_object_report(
                coad,
                interesting_objs=all_objs,
                folder=str(output_dir),
            )
    except Exception:
        logger.exception("Failed to write CSV reports")
        raise

    # Write README
    _write_export_readme(output_dir, system_name, xml_file)


def _write_export_readme(
    output_dir: Path,
    system_name: str | None,
    xml_file: Path,
) -> None:
    """Write README documenting the CSV export."""
    readme_path = output_dir / "README.md"

    system_info = (
        f"System: {system_name}" if system_name else "All objects (no system filtering)"
    )

    with readme_path.open("w") as f:
        f.write(
            f"""# Auto-Generated COAD CSV Export

This folder contains CSVs auto-generated from the PLEXOS XML model.

## Export Information

- {system_info}
- XML file: {xml_file.resolve()}
- Export script: src/utils/csv_export.py (via create_model() factory)
- Generated: {datetime.datetime.now().isoformat()}

## CSV Files

The CSVs in this directory represent all PLEXOS objects and their properties
exported from the XML file using the COAD library.

### Key Files

- `Generator.csv` - All generator objects with properties
- `Node.csv` - All node (bus) objects
- `Line.csv` - All transmission line objects
- `Storage.csv` - All storage objects
- `Fuel.csv` - All fuel/carrier objects
- `Battery.csv` - Battery storage objects (if present)
- And other class CSVs depending on model content

### Time-Varying Properties

If the COAD export includes time-varying properties, you should see:
- `Time varying properties.csv` - Properties with date ranges and timeslice associations

Note: Older COAD versions may not export this file. If missing, the CSV conversion
will fall back to static properties only.

## Usage

These CSVs can be used by the plexos-pypsa converter with the CSV-based approach:

```python
from plexos_to_pypsa_converter.network.conversion import create_model

# This will use the auto-generated CSVs
network, summary = create_model("model-id", use_csv=True)
```

## Regenerating

To regenerate these CSVs, delete this directory and run `create_model()` again.
The CSVs will be automatically re-exported from the XML.
"""
        )


def check_csv_export_exists(model_dir: str | Path) -> Path | None:
    """Check if CSV export already exists for a model.

    Parameters
    ----------
    model_dir : str | Path
        Model directory to check

    Returns
    -------
    Path or None
        Path to CSV export directory if it exists and has Generator.csv,
        None otherwise.
    """
    model_dir = Path(model_dir)
    csv_dir = model_dir / "csvs_from_xml"

    if not csv_dir.exists():
        return None

    # Check for Generator.csv as a marker that export completed successfully
    # (could be in root or in a system subdirectory)
    if (csv_dir / "Generator.csv").exists():
        return csv_dir

    # Check for system subdirectories
    for subdir in csv_dir.iterdir():
        if subdir.is_dir() and (subdir / "Generator.csv").exists():
            return subdir

    return None
