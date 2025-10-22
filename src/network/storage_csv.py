"""CSV-based storage functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based functions in storage.py.
These functions read from COAD CSV exports instead of querying the SQLite database.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from db.csv_readers import (
    find_bus_for_object_csv,
    find_bus_for_storage_via_generators_csv,
    get_property_from_static_csv,
    load_static_properties,
    parse_numeric_value,
)
from network.generators_csv import _discover_datafile_mappings
from network.storage import create_inflow_from_timeslice_array, load_inflow_from_file
from utils.paths import extract_filename

logger = logging.getLogger(__name__)


# PLEXOS Storage Model Constants (same as storage.py)
PLEXOS_STORAGE_MODEL = {
    "AUTO": 0,
    "ENERGY": 1,  # Storage volumes in GWh
    "LEVEL": 2,  # Storage units in height above sea-level
    "VOLUME": 3,  # Storage volumes in CMD (metric) or AF (imperial)
}

PLEXOS_END_EFFECTS = {
    "AUTOMATIC": 0,
    "FREE": 1,  # End volume set freely by optimization
    "RECYCLE": 2,  # End volume = Initial volume (cyclic)
}


def detect_storage_model_type(storage_df: pd.DataFrame, storage_name: str) -> str:
    """Detect the PLEXOS storage model type from CSV properties.

    Returns one of: 'energy', 'level', 'volume', 'battery', 'unknown'
    """

    def get_prop_value(name: str) -> str | None:
        return get_property_from_static_csv(storage_df, storage_name, name)

    # Check for explicit model type indicators
    model_value = get_prop_value("Model")
    if model_value is not None:
        try:
            model_int = int(model_value)
            if model_int == PLEXOS_STORAGE_MODEL["ENERGY"]:
                return "energy"
            elif model_int == PLEXOS_STORAGE_MODEL["LEVEL"]:
                return "level"
            elif model_int == PLEXOS_STORAGE_MODEL["VOLUME"]:
                return "volume"
        except (ValueError, TypeError):
            pass

    # Check for Units field to identify model type
    units = get_prop_value("Units")
    if units:
        units_lower = str(units).lower()
        if "gwh" in units_lower or "mwh" in units_lower:
            return "energy"
        elif "m" in units_lower or "ft" in units_lower or "level" in units_lower:
            return "level"
        elif "cmd" in units_lower or "af" in units_lower or "cumec" in units_lower:
            return "volume"

    # Check for battery-specific properties in column names
    if (
        "Charge Efficiency" in storage_df.columns
        or "Discharge Efficiency" in storage_df.columns
    ):
        return "battery"

    # Fallback: check if Max Volume is in reasonable range for energy
    max_volume = get_prop_value("Max Volume")
    if max_volume:
        try:
            max_val = float(max_volume)
            if max_val < 10000:  # Likely energy in GWh
                return "energy"
            else:  # Likely volume in CMD/AF
                return "volume"
        except (ValueError, TypeError):
            pass

    return "unknown"


def get_end_effects_method(storage_df: pd.DataFrame, storage_name: str) -> str:
    """Extract End Effects Method from CSV properties."""
    value = get_property_from_static_csv(storage_df, storage_name, "End Effects Method")

    if value is not None:
        # Handle numeric values (PLEXOS enum)
        if isinstance(value, int | float):
            value_int = int(value)
            if value_int == PLEXOS_END_EFFECTS["FREE"]:
                return "free"
            elif value_int == PLEXOS_END_EFFECTS["RECYCLE"]:
                return "recycle"

        # Handle string values
        if isinstance(value, str):
            value_lower = value.lower()
            if "free" in value_lower:
                return "free"
            elif "recycle" in value_lower or "cyclic" in value_lower:
                return "recycle"

    # Default
    return "free"


def parse_storage_volume_energy_conversion(
    storage_df: pd.DataFrame, storage_name: str
) -> float:
    """Parse the conversion factor between volume and energy for hydro storage.

    For volume-based storage models, PLEXOS needs a conversion factor to translate
    storage volumes (CMD or AF) into energy (MWh). This is typically defined using
    Generator efficiency, head height, or explicit energy/volume ratios.

    Returns
    -------
    float
        Conversion factor (MWh per unit volume), or 1.0 if not applicable
    """
    # For now, return 1.0 as default
    # Future: parse "Energy Rating" or calculate from generator efficiency + head
    return 1.0


def port_storage_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
    node_df: pd.DataFrame | None = None,
    generator_df: pd.DataFrame | None = None,
) -> None:
    """Port storage units from COAD CSV exports to PyPSA network.

    This is a CSV-based alternative to port_storage() that reads from CSV files
    instead of querying PlexosDB.

    Currently supports:
    - Battery storage from Battery.csv (fully implemented)
    - Hydro pumped storage from Storage.csv (TEMPORARILY DISABLED)

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add storage units to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    node_df : pd.DataFrame, optional
        Pre-loaded Node.csv data (if None, will load from csv_dir)
    generator_df : pd.DataFrame, optional
        Pre-loaded Generator.csv data (if None, will load from csv_dir)
    """
    csv_dir = Path(csv_dir)

    # Load CSVs if not provided
    if node_df is None:
        node_df = load_static_properties(csv_dir, "Node")

    if generator_df is None:
        generator_df = load_static_properties(csv_dir, "Generator")

    # ===== PROCESS PLEXOS BATTERY CLASS OBJECTS =====
    battery_csv_path = csv_dir / "Battery.csv"

    if battery_csv_path.exists():
        logger.info("Loading batteries from Battery.csv...")
        battery_df = load_static_properties(csv_dir, "Battery")

        if not battery_df.empty:
            _port_batteries_from_csv(network, battery_df, node_df, generator_df)
        else:
            logger.info("Battery.csv is empty, skipping battery import")
    else:
        logger.info("Battery.csv not found, skipping battery import")

    # ===== PROCESS PLEXOS STORAGE CLASS OBJECTS =====
    # TEMPORARILY DISABLED: Pumped hydro storage porting
    # These are complex hydro storage systems from Storage.csv that need careful modeling
    # with HEAD/TAIL pairs, natural inflows, volume tracking, etc.
    # For now, skip them to avoid infeasibility issues and focus on batteries only.
    logger.info(
        "Skipping PLEXOS Storage class objects (pumped hydro) - focusing on Battery class only"
    )


def _port_batteries_from_csv(
    network: pypsa.Network,
    battery_df: pd.DataFrame,
    node_df: pd.DataFrame,
    generator_df: pd.DataFrame,
) -> None:
    """Port battery storage units from Battery.csv to PyPSA network.

    Handles PLEXOS battery modeling conventions including HEAD/TAIL pairs
    and generator-based connections.
    """
    # Track HEAD/TAIL pairs
    head_batteries = {}
    tail_batteries = {}
    standalone_batteries = []

    # Classify batteries
    for battery_name in battery_df.index:
        if battery_name.endswith("_H") or battery_name.endswith("HEAD"):
            base_name = (
                battery_name[:-2] if battery_name.endswith("_H") else battery_name[:-4]
            )
            head_batteries[base_name] = battery_name
        elif battery_name.endswith("_T") or battery_name.endswith("TAIL"):
            base_name = (
                battery_name[:-2] if battery_name.endswith("_T") else battery_name[:-4]
            )
            tail_batteries[base_name] = battery_name
        else:
            standalone_batteries.append(battery_name)

    # Process HEAD/TAIL pairs
    added_count = 0
    skipped_batteries = []

    for base_name in head_batteries:
        if base_name not in tail_batteries:
            logger.warning(
                f"Found HEAD battery without TAIL: {head_batteries[base_name]}"
            )
            skipped_batteries.append(f"{head_batteries[base_name]} (missing TAIL pair)")
            continue

        head_name = head_batteries[base_name]
        tail_name = tail_batteries[base_name]

        try:
            # Extract properties from HEAD and TAIL
            max_volume_head = get_property_from_static_csv(
                battery_df, head_name, "Max Volume"
            )
            max_volume_tail = get_property_from_static_csv(
                battery_df, tail_name, "Max Volume"
            )
            max_power_head = get_property_from_static_csv(
                battery_df, head_name, "Max Capacity"
            )
            max_power_tail = get_property_from_static_csv(
                battery_df, tail_name, "Max Capacity"
            )

            # Parse values
            try:
                max_energy = float(max_volume_tail) if max_volume_tail else 0.0
                max_power_charge = (
                    float(max_power_head) if max_power_head else max_energy
                )
                max_power_discharge = (
                    float(max_power_tail) if max_power_tail else max_energy
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse battery properties for {base_name}, using defaults"
                )
                max_energy = 0.0
                max_power_charge = 0.0
                max_power_discharge = 0.0

            # Get efficiency
            charge_eff = get_property_from_static_csv(
                battery_df, head_name, "Efficiency"
            )
            discharge_eff = get_property_from_static_csv(
                battery_df, tail_name, "Efficiency"
            )

            try:
                efficiency_store = (
                    float(charge_eff) / 100.0 if charge_eff else 0.95
                )  # Convert % to fraction
                efficiency_dispatch = (
                    float(discharge_eff) / 100.0 if discharge_eff else 0.95
                )
            except (ValueError, TypeError):
                efficiency_store = 0.95
                efficiency_dispatch = 0.95

            # Calculate max hours (duration)
            if max_power_discharge > 0:
                max_hours = max_energy / max_power_discharge
            else:
                max_hours = 4.0  # Default 4-hour battery

            # Find bus for battery (try multiple approaches)
            bus = None
            connection_method = None
            primary_generator = None

            # Approach 1: Check if HEAD/TAIL has a node assignment
            bus = find_bus_for_object_csv(node_df, "Battery", head_name, fallback=None)
            if bus:
                connection_method = "direct_node"
            else:
                bus = find_bus_for_object_csv(
                    node_df, "Battery", tail_name, fallback=None
                )
                if bus:
                    connection_method = "direct_node"

            # Approach 2: Find via associated generators
            if bus is None:
                (
                    bus,
                    primary_generator,
                ) = find_bus_for_storage_via_generators_csv(
                    generator_df, node_df, head_name
                )
                if bus:
                    connection_method = "via_generator"

            if bus is None:
                (
                    bus,
                    primary_generator,
                ) = find_bus_for_storage_via_generators_csv(
                    generator_df, node_df, tail_name
                )
                if bus:
                    connection_method = "via_generator"

            if bus is None:
                logger.warning(f"Could not find bus for battery {base_name}, skipping")
                skipped_batteries.append(f"{base_name} (no bus found)")
                continue

            # Determine carrier
            carrier = "battery"  # Default carrier for batteries

            # Add to PyPSA network as StorageUnit
            network.add(
                "StorageUnit",
                base_name,
                bus=bus,
                carrier=carrier,
                p_nom=max_power_discharge,  # Discharge power capacity
                p_nom_extendable=False,
                max_hours=max_hours,
                efficiency_store=efficiency_store,
                efficiency_dispatch=efficiency_dispatch,
                cyclic_state_of_charge=True,  # Batteries typically cyclic
                state_of_charge_initial=0.5,  # Start at 50% SOC
            )

            added_count += 1

            logger.info(
                f"Added battery {base_name}: {max_power_discharge:.1f} MW, {max_hours:.1f} hours, "
                f"bus={bus}, carrier={carrier}"
            )
            if connection_method == "via_generator":
                logger.info(
                    f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
                )

        except Exception as e:
            logger.warning(f"Error processing battery {base_name}: {e}")
            skipped_batteries.append(f"{base_name} (error: {e})")

    # Summary
    logger.info(f"Batteries added: {added_count}")
    logger.info(f"Batteries skipped: {len(skipped_batteries)}")

    if skipped_batteries:
        logger.info(f"Skipped batteries: {skipped_batteries}")


def add_storage_inflows_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    inflow_path: str | Path,
) -> dict:
    """Add inflow time series for hydro storage units to PyPSA network from CSV data.

    This is a CSV-based alternative to add_hydro_inflows() that reads from COAD CSV
    exports instead of querying PlexosDB. Supports multiple inflow data formats:
    - Data file references (Natural Inflow.Data File column)
    - Timeslice arrays (Natural Inflow.Timeslice column)
    - Static values (Natural Inflow column)

    The function auto-discovers storage→inflow linkages from Storage.csv and Data File.csv,
    loads the appropriate inflow data, and applies it to network.storage_units_t.inflow.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network with storage units already added
    csv_dir : str | Path
        Directory containing COAD CSV exports (must include Storage.csv and Data File.csv)
    inflow_path : str | Path
        Base directory containing inflow CSV files referenced in Data File.csv

    Returns
    -------
    dict
        Summary statistics: {
            "storage_units_with_inflows": int,
            "storage_units_without_inflows": int,
            "failed_storage_units": list[str],
            "inflow_sources": {
                "data_file": int,
                "timeslice_array": int,
                "static_value": int
            }
        }

    Examples
    --------
    >>> # AEMO model with Data File references
    >>> summary = add_storage_inflows_csv(
    ...     network=network,
    ...     csv_dir="src/examples/data/aemo-2024-isp-progressive-change/csvs_from_xml/NEM",
    ...     inflow_path="src/examples/data/aemo-2024-isp-progressive-change"
    ... )
    >>> print(f"Storage units with inflows: {summary['storage_units_with_inflows']}")

    >>> # CAISO model with timeslice arrays
    >>> summary = add_storage_inflows_csv(
    ...     network=network,
    ...     csv_dir="src/examples/data/caiso-irp23/csvs_from_xml/WECC",
    ...     inflow_path="src/examples/data/caiso-irp23"
    ... )
    """
    csv_dir = Path(csv_dir)
    inflow_path = Path(inflow_path)

    # Initialize statistics
    stats = {
        "storage_units_with_inflows": 0,
        "storage_units_without_inflows": 0,
        "failed_storage_units": [],
        "inflow_sources": {
            "data_file": 0,
            "timeslice_array": 0,
            "static_value": 0,
        },
    }

    # Load Storage.csv
    storage_df = load_static_properties(csv_dir, "Storage")
    if storage_df.empty:
        logger.warning("Storage.csv not found or empty. Cannot load inflows.")
        return stats

    # Discover data file mappings (for Data File references)
    datafile_to_csv = _discover_datafile_mappings(csv_dir)

    # Get network snapshots for time series alignment
    snapshots = pd.DatetimeIndex(network.snapshots)

    # Process each storage unit in the network
    for storage_name in network.storage_units.index:
        try:
            # Check if storage exists in Storage.csv
            if storage_name not in storage_df.index:
                logger.debug(
                    f"Storage unit {storage_name} not found in Storage.csv, skipping inflow"
                )
                stats["storage_units_without_inflows"] += 1
                continue

            inflow_data = None
            inflow_source = None

            # Priority 1: Try Data File reference (Natural Inflow.Data File column)
            datafile_ref = get_property_from_static_csv(
                storage_df, storage_name, "Natural Inflow.Data File"
            )

            if datafile_ref:
                # Strip "Data File." prefix if present
                if datafile_ref.startswith("Data File."):
                    datafile_obj = datafile_ref[len("Data File.") :]
                else:
                    datafile_obj = datafile_ref

                # Look up CSV filename in Data File.csv mappings
                if datafile_obj in datafile_to_csv:
                    csv_filename = datafile_to_csv[datafile_obj]
                    clean_filename = extract_filename(csv_filename.strip())

                    # Load inflow from file
                    inflow_data = load_inflow_from_file(
                        clean_filename, str(inflow_path), snapshots
                    )

                    if inflow_data is not None:
                        inflow_source = "data_file"
                        logger.info(
                            f"Loaded inflow data for {storage_name} from data file: {clean_filename}"
                        )
                else:
                    logger.warning(
                        f"Data file object '{datafile_obj}' not found in Data File.csv for {storage_name}"
                    )

            # Priority 2: Try Timeslice array (Natural Inflow.Timeslice column)
            if inflow_data is None:
                timeslice_array = get_property_from_static_csv(
                    storage_df, storage_name, "Natural Inflow.Timeslice"
                )

                if timeslice_array:
                    try:
                        # Parse array value
                        values = parse_numeric_value(timeslice_array, strategy="array")

                        if values and isinstance(values, list) and len(values) > 1:
                            inflow_data = create_inflow_from_timeslice_array(
                                values, snapshots
                            )
                            inflow_source = "timeslice_array"
                            logger.info(
                                f"Created inflow data for {storage_name} from timeslice array ({len(values)} values)"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Error parsing timeslice array for {storage_name}: {e}"
                        )

            # Priority 3: Try static value (Natural Inflow column)
            if inflow_data is None:
                static_inflow = get_property_from_static_csv(
                    storage_df, storage_name, "Natural Inflow"
                )

                if static_inflow:
                    try:
                        inflow_value = float(static_inflow)
                        if inflow_value > 0:
                            # Create constant inflow series
                            inflow_data = pd.Series(inflow_value, index=snapshots)
                            inflow_source = "static_value"
                            logger.info(
                                f"Set constant inflow {inflow_value} for {storage_name}"
                            )
                    except (ValueError, TypeError):
                        logger.debug(
                            f"Could not parse static Natural Inflow value for {storage_name}: {static_inflow}"
                        )

            # Apply inflow data to network
            if inflow_data is not None:
                network.storage_units_t.inflow[storage_name] = inflow_data
                stats["storage_units_with_inflows"] += 1
                stats["inflow_sources"][inflow_source] += 1
            else:
                logger.debug(f"No inflow data found for storage unit {storage_name}")
                stats["storage_units_without_inflows"] += 1

        except Exception as e:
            logger.warning(f"Error processing inflows for {storage_name}: {e}")
            stats["failed_storage_units"].append(f"{storage_name} ({e})")

    # Log summary
    logger.info(
        f"Added inflow data for {stats['storage_units_with_inflows']} storage units"
    )
    logger.info(f"  - From data files: {stats['inflow_sources']['data_file']}")
    logger.info(
        f"  - From timeslice arrays: {stats['inflow_sources']['timeslice_array']}"
    )
    logger.info(f"  - From static values: {stats['inflow_sources']['static_value']}")
    logger.info(
        f"Storage units without inflows: {stats['storage_units_without_inflows']}"
    )

    if stats["failed_storage_units"]:
        logger.warning(
            f"Failed to process {len(stats['failed_storage_units'])} storage units: "
            f"{stats['failed_storage_units']}"
        )

    return stats
