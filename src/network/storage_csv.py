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
)

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
            value = int(value)
            return {0: "automatic", 1: "free", 2: "recycle"}.get(value, "automatic")

        # Handle string values
        elif isinstance(value, str):
            value_lower = value.lower()
            if "free" in value_lower:
                return "free"
            elif "recycle" in value_lower:
                return "recycle"
            elif "automatic" in value_lower or "auto" in value_lower:
                return "automatic"

    return "automatic"  # Default


def convert_plexos_volume_to_energy(
    volume: float,
    model_type: str,
    units: str | None = None,
) -> tuple[float, str]:
    """Convert PLEXOS volume/capacity to PyPSA energy units (MWh).

    Returns (energy_mwh, conversion_method_used)
    """
    if model_type == "energy":
        # Direct energy conversion
        if units and "gwh" in str(units).lower():
            return volume * 1000, "direct_gwh_to_mwh"  # GWh to MWh
        else:
            return volume, "direct_mwh"  # Assume MWh

    elif model_type == "volume":
        # Volume to energy conversion (simplified approximation)
        if units and "cmd" in str(units).lower():
            return volume * 0.05, "volume_cmd_approx"
        elif units and ("af" in str(units).lower() or "acre" in str(units).lower()):
            return volume * 0.005, "volume_af_approx"
        else:
            return volume * 0.01, "volume_unknown_approx"

    elif model_type == "level":
        # Level-based storage cannot be directly converted
        logger.warning(
            "Level-based storage model detected but cannot convert to energy without reservoir area data"
        )
        return volume, "level_unsupported"

    elif model_type == "battery":
        # Battery storage - likely already in energy units
        return volume, "battery_direct"

    else:
        # Unknown model type - assume energy units
        return volume, "unknown_assume_energy"


def detect_pumped_hydro_pairs_csv(
    storage_df: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Detect HEAD/TAIL pairs for pumped hydro storage from CSV.

    Returns dict mapping storage names to their pumped hydro configuration.
    """
    pumped_hydro_pairs = {}

    # Group storage units by potential pairs
    head_units = []
    tail_units = []

    for storage_name in storage_df.index:
        try:
            unit_name_lower = storage_name.lower()

            # Get category from CSV if available
            category = get_property_from_static_csv(
                storage_df, storage_name, "Category"
            )
            category_lower = str(category).lower() if category else ""

            if (
                "head" in unit_name_lower
                or "head" in category_lower
                or "upper" in unit_name_lower
            ):
                head_units.append(storage_name)
            elif (
                "tail" in unit_name_lower
                or "tail" in category_lower
                or "lower" in unit_name_lower
            ):
                tail_units.append(storage_name)

        except Exception as e:
            logger.warning(
                f"Error checking storage unit {storage_name} for pumped hydro: {e}"
            )
            continue

    # Try to pair HEAD and TAIL units
    for head_unit in head_units:
        # Look for corresponding tail unit
        head_base = (
            head_unit.lower().replace("head", "").replace("upper", "").strip("_- ")
        )

        for tail_unit in tail_units:
            tail_base = (
                tail_unit.lower().replace("tail", "").replace("lower", "").strip("_- ")
            )

            if (
                head_base == tail_base
                or head_base in tail_base
                or tail_base in head_base
            ):
                pair_name = f"{head_unit}_{tail_unit}"
                pumped_hydro_pairs[pair_name] = {
                    "head": head_unit,
                    "tail": tail_unit,
                    "type": "pumped_hydro_pair",
                }
                break

    # Mark remaining units as standalone
    all_paired = set()
    for pair_info in pumped_hydro_pairs.values():
        all_paired.add(pair_info["head"])
        all_paired.add(pair_info["tail"])

    for storage_name in storage_df.index:
        if storage_name not in all_paired:
            pumped_hydro_pairs[storage_name] = {"type": "standalone_storage"}

    return pumped_hydro_pairs


def add_storage_csv(
    network: pypsa.Network, csv_dir: str | Path, timeslice_csv: str | None = None
) -> None:
    """Enhanced function to add PLEXOS storage units and batteries from CSV.

    This is the CSV-based version of add_storage() from storage.py.

    Supports different PLEXOS storage representations:
    - PLEXOS Storage class: Energy, Volume, Level-based, Pumped hydro pairs
    - PLEXOS Battery class: Battery storage systems

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add storage to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice CSV file (for future use)
    """
    csv_dir = Path(csv_dir)

    # ===== PROCESS PLEXOS STORAGE CLASS OBJECTS =====
    logger.info("Processing PLEXOS Storage class objects from CSV...")

    storage_df = load_static_properties(csv_dir, "Storage")

    if storage_df.empty:
        logger.info("No Storage.csv found or no storage units in CSV")
        storage_units = []
    else:
        storage_units = list(storage_df.index)
        logger.info(f"Found {len(storage_units)} PLEXOS storage units in CSV")

        # Get unique categories for carriers
        storage_categories = {}
        unique_storage_carriers = set()

        for storage_name in storage_units:
            category = get_property_from_static_csv(
                storage_df, storage_name, "Category"
            )
            carrier = str(category) if category else "hydro"
            storage_categories[storage_name] = carrier
            unique_storage_carriers.add(carrier)

        logger.info(f"Storage carriers found: {sorted(unique_storage_carriers)}")

        # Ensure all storage carriers exist in network
        for carrier in unique_storage_carriers:
            if carrier not in network.carriers.index:
                network.add("Carrier", name=carrier)
                logger.info(f"Added storage carrier: {carrier}")

        # Detect pumped hydro pairs
        storage_pairs = detect_pumped_hydro_pairs_csv(storage_df)
        logger.info(
            f"Detected storage configuration: "
            f"{len([p for p in storage_pairs.values() if p['type'] == 'pumped_hydro_pair'])} pumped hydro pairs, "
            f"{len([p for p in storage_pairs.values() if p['type'] == 'standalone_storage'])} standalone units"
        )

        added_storage_units = 0
        skipped_storage_units = []

        # Process each storage configuration
        for config_name, config in storage_pairs.items():
            try:
                if config["type"] == "pumped_hydro_pair":
                    # Handle pumped hydro HEAD/TAIL pair
                    head_carrier = storage_categories.get(
                        config["head"], "pumped_hydro"
                    )
                    tail_carrier = storage_categories.get(
                        config["tail"], "pumped_hydro"
                    )
                    success = port_pumped_hydro_pair_csv(
                        network,
                        csv_dir,
                        config["head"],
                        config["tail"],
                        head_carrier,
                        tail_carrier,
                    )
                    if success:
                        added_storage_units += 1
                        logger.info(
                            f"Added pumped hydro pair: {config['head']} + {config['tail']}"
                        )
                    else:
                        skipped_storage_units.extend([config["head"], config["tail"]])

                elif config["type"] == "standalone_storage":
                    # Handle standalone storage unit
                    storage_carrier = storage_categories.get(config_name, "hydro")
                    success = port_standalone_storage_csv(
                        network, csv_dir, config_name, storage_carrier
                    )
                    if success:
                        added_storage_units += 1
                        logger.info(f"Added standalone storage: {config_name}")
                    else:
                        skipped_storage_units.append(config_name)

            except Exception:
                logger.exception(
                    f"Error processing storage configuration {config_name}"
                )
                if config["type"] == "pumped_hydro_pair":
                    skipped_storage_units.extend([config["head"], config["tail"]])
                else:
                    skipped_storage_units.append(config_name)

        logger.info(
            f"Successfully added {added_storage_units} PLEXOS Storage configurations"
        )
        if skipped_storage_units:
            logger.warning(
                f"Skipped {len(skipped_storage_units)} PLEXOS Storage units: "
                f"{skipped_storage_units[:10]}{'...' if len(skipped_storage_units) > 10 else ''}"
            )

    # ===== PROCESS PLEXOS BATTERY CLASS OBJECTS =====
    logger.info("Processing PLEXOS Battery class objects from CSV...")

    initial_storage_count = len(network.storage_units)

    # Call port_batteries
    port_batteries_csv(network, csv_dir, timeslice_csv)

    final_storage_count = len(network.storage_units)
    added_battery_units = (
        final_storage_count - initial_storage_count - len(storage_units)
    )

    logger.info(f"Successfully added {added_battery_units} PLEXOS Battery units")

    # ===== SUMMARY =====
    total_added = len(network.storage_units) - initial_storage_count
    logger.info(f"Total storage units added to network: {total_added}")


def port_standalone_storage_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    storage_name: str,
    carrier: str = "hydro",
) -> bool:
    """Port a standalone PLEXOS storage unit to PyPSA from CSV.

    This is the CSV-based version of port_standalone_storage() from storage.py.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add storage to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    storage_name : str
        Name of the storage unit
    carrier : str, optional
        Carrier type for the storage unit (from category), defaults to "hydro"

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    csv_dir = Path(csv_dir)

    try:
        storage_df = load_static_properties(csv_dir, "Storage")

        if storage_name not in storage_df.index:
            logger.warning(f"Storage {storage_name} not found in Storage.csv")
            return False

        def get_prop_float(name: str) -> float | None:
            val = get_property_from_static_csv(storage_df, storage_name, name)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
            return None

        # Detect storage model type
        model_type = detect_storage_model_type(storage_df, storage_name)
        logger.info(f"Detected model type '{model_type}' for storage {storage_name}")

        # Find connected bus
        bus = find_bus_for_object_csv(storage_df, storage_name)
        connection_method = "direct"
        primary_generator = None

        if bus is None:
            # Try finding bus via generators
            generator_df = load_static_properties(csv_dir, "Generator")
            bus, primary_generator = find_bus_for_storage_via_generators_csv(
                storage_name, storage_df, generator_df
            )
            connection_method = "via_generator"

        if bus is None:
            logger.warning(f"No connected bus found for storage {storage_name}")
            return False

        # Extract basic properties
        initial_volume = get_prop_float("Initial Volume") or 0.0
        min_volume = get_prop_float("Min Volume") or 0.0
        max_volume = get_prop_float("Max Volume") or 1000.0
        units = get_property_from_static_csv(storage_df, storage_name, "Units")

        # Convert volumes to energy (MWh)
        max_energy, _ = convert_plexos_volume_to_energy(max_volume, model_type, units)
        min_energy, _ = convert_plexos_volume_to_energy(min_volume, model_type, units)
        initial_energy, _ = convert_plexos_volume_to_energy(
            initial_volume, model_type, units
        )

        # Get End Effects Method
        end_effects = get_end_effects_method(storage_df, storage_name)
        cyclic_state_of_charge = end_effects == "recycle"

        # Override carrier if model type suggests battery
        if model_type == "battery" and carrier == "hydro":
            carrier = "battery"

        # Calculate power capacity
        if model_type == "battery":
            default_hours = 4.0
        else:
            default_hours = 8.0

        p_nom = max_energy / default_hours if max_energy > 0 else 100.0
        max_hours = max_energy / p_nom if p_nom > 0 else default_hours

        # Calculate state of charge parameters
        if max_energy > 0:
            state_of_charge_initial = initial_energy / max_energy
        else:
            state_of_charge_initial = 0.5

        # Add storage unit to network
        network.add(
            "StorageUnit",
            storage_name,
            bus=bus,
            carrier=carrier,
            p_nom=p_nom,
            max_hours=max_hours,
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
            state_of_charge_initial=state_of_charge_initial,
            cyclic_state_of_charge=cyclic_state_of_charge,
        )

        logger.info(
            f"Added storage {storage_name}: {p_nom:.1f} MW, {max_hours:.1f} hours, "
            f"model={model_type}, carrier={carrier}"
        )
        if connection_method == "via_generator":
            logger.info(
                f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
            )

    except Exception:
        logger.exception(f"Error porting standalone storage {storage_name}")
        return False
    else:
        return True


def port_pumped_hydro_pair_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    head_name: str,
    tail_name: str,
    head_carrier: str = "pumped_hydro",
    tail_carrier: str = "pumped_hydro",
) -> bool:
    """Port a PLEXOS pumped hydro HEAD/TAIL pair to PyPSA from CSV.

    This is the CSV-based version of port_pumped_hydro_pair() from storage.py.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add storage to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    head_name : str
        Name of the HEAD storage unit
    tail_name : str
        Name of the TAIL storage unit
    head_carrier : str, optional
        Carrier for HEAD storage, defaults to "pumped_hydro"
    tail_carrier : str, optional
        Carrier for TAIL storage, defaults to "pumped_hydro"

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    csv_dir = Path(csv_dir)

    try:
        storage_df = load_static_properties(csv_dir, "Storage")

        if head_name not in storage_df.index or tail_name not in storage_df.index:
            logger.warning(
                f"HEAD {head_name} or TAIL {tail_name} not found in Storage.csv"
            )
            return False

        def get_prop_float_for(name: str, storage: str) -> float | None:
            val = get_property_from_static_csv(storage_df, storage, name)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
            return None

        # Find connected bus
        head_bus = find_bus_for_object_csv(storage_df, head_name)
        tail_bus = find_bus_for_object_csv(storage_df, tail_name)
        bus = head_bus or tail_bus
        connection_method = "direct"
        primary_generator = None

        if bus is None:
            # Try via generators
            generator_df = load_static_properties(csv_dir, "Generator")
            bus, primary_generator = find_bus_for_storage_via_generators_csv(
                head_name, storage_df, generator_df
            )
            connection_method = "via_generator"

            if bus is None:
                bus, primary_generator = find_bus_for_storage_via_generators_csv(
                    tail_name, storage_df, generator_df
                )
                connection_method = "via_generator"

        if bus is None:
            logger.warning(
                f"No connected bus found for pumped hydro pair {head_name}/{tail_name}"
            )
            return False

        # Get volumes for both reservoirs
        head_max = get_prop_float_for("Max Volume", head_name) or 0.0
        head_initial = get_prop_float_for("Initial Volume", head_name) or 0.0

        # Detect model types
        head_model_type = detect_storage_model_type(storage_df, head_name)

        # Convert to energy (use head reservoir as primary)
        max_energy, _ = convert_plexos_volume_to_energy(head_max, head_model_type)
        initial_energy, _ = convert_plexos_volume_to_energy(
            head_initial, head_model_type
        )

        # Calculate power capacity
        default_hours = 8.0
        p_nom = max_energy / default_hours if max_energy > 0 else 200.0
        max_hours = max_energy / p_nom if p_nom > 0 else default_hours

        # Initial state of charge
        state_of_charge_initial = initial_energy / max_energy if max_energy > 0 else 0.5

        # Pumped hydro efficiency
        efficiency = 0.85

        # Determine carrier
        carrier = (
            head_carrier
            if head_carrier != "pumped_hydro"
            else (tail_carrier if tail_carrier != "pumped_hydro" else "pumped_hydro")
        )

        # Create combined name
        system_name = f"{head_name.replace('HEAD', '').replace('Upper', '').strip('_- ')}_PumpedHydro"
        if not system_name or system_name == "_PumpedHydro":
            system_name = f"{head_name}_{tail_name}"

        # Add as single StorageUnit
        network.add(
            "StorageUnit",
            system_name,
            bus=bus,
            carrier=carrier,
            p_nom=p_nom,
            max_hours=max_hours,
            efficiency_store=efficiency,
            efficiency_dispatch=efficiency,
            state_of_charge_initial=state_of_charge_initial,
            cyclic_state_of_charge=True,
        )

        logger.info(
            f"Added pumped hydro system {system_name}: {p_nom:.1f} MW, {max_hours:.1f} hours, "
            f"efficiency={efficiency:.1%}, carrier={carrier}"
        )
        logger.info(f"  Combined from HEAD: {head_name} + TAIL: {tail_name}")
        if connection_method == "via_generator":
            logger.info(
                f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
            )

    except Exception:
        logger.exception(f"Error porting pumped hydro pair {head_name}/{tail_name}")
        return False
    else:
        return True


def port_batteries_csv(
    network: pypsa.Network, csv_dir: str | Path, timeslice_csv: str | None = None
) -> None:
    """Add PLEXOS batteries as PyPSA StorageUnit components from CSV.

    This is the CSV-based version of port_batteries() from storage.py.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add batteries to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice CSV file (for future use)
    """
    csv_dir = Path(csv_dir)

    logger.info("Adding batteries from CSV...")

    battery_df = load_static_properties(csv_dir, "Battery")

    if battery_df.empty:
        logger.info("No Battery.csv found or no batteries in CSV")
        return

    logger.info(f"Found {len(battery_df)} batteries in CSV")

    # Collect unique battery categories for carriers
    battery_carriers = set()
    for battery_name in battery_df.index:
        category = get_property_from_static_csv(battery_df, battery_name, "Category")
        carrier = str(category) if category else "battery"
        battery_carriers.add(carrier)

    logger.info(f"Battery carriers found: {sorted(battery_carriers)}")

    # Ensure all battery carriers exist in network
    for carrier in battery_carriers:
        if carrier not in network.carriers.index:
            network.add("Carrier", name=carrier)
            logger.info(f"Added carrier: {carrier}")

    skipped_batteries = []
    added_count = 0

    def get_property_value(battery_name: str, property_name: str, default=None):
        """Extract property value by name."""
        val = get_property_from_static_csv(battery_df, battery_name, property_name)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        return default

    for battery_name in battery_df.index:
        try:
            logger.debug(f"Processing battery: {battery_name}")

            # Determine carrier from category
            category = get_property_from_static_csv(
                battery_df, battery_name, "Category"
            )
            carrier = str(category) if category else "battery"

            # Find connected bus
            bus = find_bus_for_object_csv(battery_df, battery_name)
            connection_method = "direct"
            primary_generator = None

            if bus is None:
                # Try via generators
                generator_df = load_static_properties(csv_dir, "Generator")
                bus, primary_generator = find_bus_for_storage_via_generators_csv(
                    battery_name, battery_df, generator_df
                )
                connection_method = "via_generator"

            if bus is None:
                logger.warning(f"No connected bus found for battery {battery_name}")
                skipped_batteries.append(f"{battery_name} (no bus)")
                continue

            # Extract Max Power (required)
            max_power = get_property_value(battery_name, "Max Power")
            if max_power is None or max_power <= 0:
                logger.warning(f"No valid 'Max Power' for battery {battery_name}")
                skipped_batteries.append(f"{battery_name} (no Max Power)")
                continue

            # Extract volume properties
            max_volume = get_property_value(
                battery_name, "Max Volume"
            ) or get_property_value(battery_name, "Max SoC")
            initial_volume = get_property_value(
                battery_name, "Initial Volume"
            ) or get_property_value(battery_name, "Initial SoC", 0.0)
            min_volume = get_property_value(
                battery_name, "Min Volume"
            ) or get_property_value(battery_name, "Min SoC", 0.0)

            # Calculate max_hours
            if max_volume is not None and max_volume > 0:
                max_hours = max_volume / max_power
            else:
                logger.warning(
                    f"No valid 'Max Volume/SoC' for battery {battery_name}, using default 4 hours"
                )
                max_hours = 4.0

            # Extract efficiencies
            efficiency_store = get_property_value(
                battery_name, "Charge Efficiency", 0.9
            )
            efficiency_dispatch = get_property_value(
                battery_name, "Discharge Efficiency", 0.9
            )

            # Convert from percentage if needed
            if efficiency_store > 1.0:
                efficiency_store = efficiency_store / 100.0
            if efficiency_dispatch > 1.0:
                efficiency_dispatch = efficiency_dispatch / 100.0

            # Extract lifetime
            lifetime = get_property_value(battery_name, "Technical Life")
            if lifetime is None:
                lifetime = get_property_value(battery_name, "Economic Life")

            # Create storage unit entry
            storage_unit_data = {
                "bus": bus,
                "carrier": carrier,
                "p_nom": max_power,
                "max_hours": max_hours,
                "efficiency_store": efficiency_store,
                "efficiency_dispatch": efficiency_dispatch,
                "state_of_charge_initial": initial_volume if initial_volume else 0.0,
                "state_of_charge_min": min_volume if min_volume else 0.0,
            }

            if lifetime is not None:
                storage_unit_data["lifetime"] = lifetime

            # Add to network
            network.add("StorageUnit", battery_name, **storage_unit_data)
            added_count += 1

            logger.info(
                f"Added battery {battery_name}: {max_power:.1f} MW, {max_hours:.1f} hours, "
                f"bus={bus}, carrier={carrier}"
            )
            if connection_method == "via_generator":
                logger.info(
                    f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
                )

        except Exception as e:
            logger.warning(f"Error processing battery {battery_name}: {e}")
            skipped_batteries.append(f"{battery_name} (error: {e})")

    # Summary
    logger.info(f"Batteries added: {added_count}")
    logger.info(f"Batteries skipped: {len(skipped_batteries)}")

    if skipped_batteries:
        logger.info(f"Skipped batteries: {skipped_batteries}")
