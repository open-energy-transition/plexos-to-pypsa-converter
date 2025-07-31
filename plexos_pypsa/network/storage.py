import logging
import os
from typing import Any, Dict, Optional, Tuple

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import (
    find_bus_for_object,
    find_bus_for_storage_via_generators,
)
from plexos_pypsa.network.costs import (
    set_battery_marginal_costs,
    set_capital_costs_generic,
)

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# PLEXOS Storage Model Constants
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


def detect_storage_model_type(props: list) -> str:
    """
    Detect the PLEXOS storage model type from storage properties.

    Returns one of: 'energy', 'level', 'volume', 'battery', 'unknown'
    """

    # Helper to get property value by name
    def get_prop_value(name: str) -> Optional[str]:
        for p in props:
            if p.get("property") == name:
                return p.get("value")
        return None

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
        units_lower = units.lower()
        if "gwh" in units_lower or "mwh" in units_lower:
            return "energy"
        elif "m" in units_lower or "ft" in units_lower or "level" in units_lower:
            return "level"
        elif "cmd" in units_lower or "af" in units_lower or "cumec" in units_lower:
            return "volume"

    # Check for battery-specific properties or categories
    for p in props:
        prop_name = p.get("property", "").lower()
        if any(term in prop_name for term in ["battery", "charge", "discharge"]):
            return "battery"

    # Fallback: check if Max Volume is in reasonable range for energy (assume GWh if < 10000)
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


def get_end_effects_method(props: list) -> str:
    """Extract End Effects Method from storage properties."""
    for p in props:
        if p.get("property") == "End Effects Method":
            value = p.get("value")

            # Handle numeric values (PLEXOS enum)
            if isinstance(value, (int, float)):
                value = int(value)
                if value == 0:
                    return "free"
                elif value == 1:
                    return "recycle"
                elif value == 2:
                    return "automatic"
                else:
                    return "automatic"  # Default for unknown numeric values

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
    units: Optional[str] = None,
    additional_props: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Convert PLEXOS volume/capacity to PyPSA energy units (MWh).

    Returns (energy_mwh, conversion_method_used)
    """
    if model_type == "energy":
        # Direct energy conversion
        if units and "gwh" in units.lower():
            return volume * 1000, "direct_gwh_to_mwh"  # GWh to MWh
        else:
            return volume, "direct_mwh"  # Assume MWh

    elif model_type == "volume":
        # Volume to energy conversion (simplified approximation)
        # This requires assumptions about water density, head height, turbine efficiency
        # For now, use approximate conversion factors

        if units and "cmd" in units.lower():
            # CMD (cumec-days) to MWh - approximate conversion
            # 1 CMD ≈ 0.01-0.1 MWh depending on head (using 0.05 as middle estimate)
            return volume * 0.05, "volume_cmd_approx"
        elif units and ("af" in units.lower() or "acre" in units.lower()):
            # Acre-feet to MWh - approximate conversion
            # 1 AF ≈ 0.001-0.01 MWh depending on head (using 0.005 as estimate)
            return volume * 0.005, "volume_af_approx"
        else:
            # Unknown volume units - use conservative estimate
            return volume * 0.01, "volume_unknown_approx"

    elif model_type == "level":
        # Level-based storage cannot be directly converted without reservoir area
        # Return a warning value and method
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


def detect_pumped_hydro_pairs(
    storage_units: list, db: PlexosDB
) -> Dict[str, Dict[str, str]]:
    """
    Detect HEAD/TAIL pairs for pumped hydro storage.

    Returns dict mapping storage names to their pumped hydro configuration.
    """
    pumped_hydro_pairs = {}

    # Group storage units by potential pairs
    head_units = []
    tail_units = []

    for unit_name in storage_units:
        try:
            props = db.get_object_properties(ClassEnum.Storage, unit_name)

            # Check category or name for HEAD/TAIL indicators
            unit_name_lower = unit_name.lower()
            category = ""

            # Get category from properties if available
            for p in props:
                if p.get("property") == "Category":
                    category = p.get("value", "")
                    break

            category_lower = category.lower()

            if (
                "head" in unit_name_lower
                or "head" in category_lower
                or "upper" in unit_name_lower
            ):
                head_units.append(unit_name)
            elif (
                "tail" in unit_name_lower
                or "tail" in category_lower
                or "lower" in unit_name_lower
            ):
                tail_units.append(unit_name)

        except Exception as e:
            logger.warning(
                f"Error checking storage unit {unit_name} for pumped hydro: {e}"
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

    for unit_name in storage_units:
        if unit_name not in all_paired:
            pumped_hydro_pairs[unit_name] = {"type": "standalone_storage"}

    return pumped_hydro_pairs


def add_storage(network: Network, db: PlexosDB, timeslice_csv=None) -> None:
    """
    Enhanced function to add PLEXOS storage units and batteries to PyPSA network.

    Supports different PLEXOS storage representations:
    - PLEXOS Storage class: Energy-based (GWh/MWh), Volume-based (CMD/AF), Level-based (height), Pumped hydro (HEAD/TAIL pairs)
    - PLEXOS Battery class: Battery storage systems with charge/discharge properties

    Parameters
    ----------
    network : Network
        The PyPSA network to add storage to
    db : PlexosDB
        The PLEXOS database
    timeslice_csv : str, optional
        Path to timeslice CSV file for time-dependent properties
    """

    # ===== PROCESS PLEXOS STORAGE CLASS OBJECTS =====
    logger.info("Processing PLEXOS Storage class objects...")

    # Get all storage objects with their categories using SQL
    storage_query = """
        SELECT 
            o.name AS storage_name,
            o.object_id,
            c.name AS category_name
        FROM t_object o
        JOIN t_class cl ON o.class_id = cl.class_id
        LEFT JOIN t_category c ON o.category_id = c.category_id
        WHERE cl.name = 'Storage'
    """

    storage_results = db.query(storage_query)
    logger.info(f"Found {len(storage_results)} PLEXOS storage units")

    # Create mapping of storage names to categories for later use
    storage_categories = {}
    unique_storage_carriers = set()

    if not storage_results:
        logger.info("No PLEXOS Storage units found in database")
        storage_units = []
    else:
        for storage_name, object_id, category_name in storage_results:
            carrier = category_name if category_name else "hydro"
            storage_categories[storage_name] = carrier
            unique_storage_carriers.add(carrier)

        logger.info(f"Storage carriers found: {sorted(unique_storage_carriers)}")

        # Ensure all storage carriers exist in the network
        for carrier in unique_storage_carriers:
            if carrier not in network.carriers.index:
                network.add("Carrier", name=carrier)
                logger.info(f"Added storage carrier: {carrier}")

        # Get list of storage unit names for further processing
        storage_units = [row[0] for row in storage_results]  # Extract storage names

    # Detect pumped hydro pairs
    storage_pairs = detect_pumped_hydro_pairs(storage_units, db)
    logger.info(
        f"Detected storage configuration: {len([p for p in storage_pairs.values() if p['type'] == 'pumped_hydro_pair'])} pumped hydro pairs, {len([p for p in storage_pairs.values() if p['type'] == 'standalone_storage'])} standalone units"
    )

    added_storage_units = 0
    skipped_storage_units = []

    # Process each storage configuration
    for config_name, config in storage_pairs.items():
        try:
            if config["type"] == "pumped_hydro_pair":
                # Handle pumped hydro HEAD/TAIL pair
                head_carrier = storage_categories.get(config["head"], "pumped_hydro")
                tail_carrier = storage_categories.get(config["tail"], "pumped_hydro")
                success = port_pumped_hydro_pair(
                    network,
                    db,
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
                success = port_standalone_storage(
                    network, db, config_name, storage_carrier
                )
                if success:
                    added_storage_units += 1
                    logger.info(f"Added standalone storage: {config_name}")
                else:
                    skipped_storage_units.append(config_name)

        except Exception as e:
            logger.error(f"Error processing storage configuration {config_name}: {e}")
            if config["type"] == "pumped_hydro_pair":
                skipped_storage_units.extend([config["head"], config["tail"]])
            else:
                skipped_storage_units.append(config_name)

    logger.info(
        f"Successfully added {added_storage_units} PLEXOS Storage configurations to network"
    )
    if skipped_storage_units:
        logger.warning(
            f"Skipped {len(skipped_storage_units)} PLEXOS Storage units: {skipped_storage_units[:10]}{'...' if len(skipped_storage_units) > 10 else ''}"
        )

    # ===== PROCESS PLEXOS BATTERY CLASS OBJECTS =====
    logger.info("Processing PLEXOS Battery class objects...")

    # Store initial battery count to track what was added
    initial_storage_count = len(network.storage_units)

    # Call the existing port_batteries function
    port_batteries(network, db, timeslice_csv)

    # Calculate batteries added
    final_storage_count = len(network.storage_units)
    added_battery_units = (
        final_storage_count - initial_storage_count - added_storage_units
    )

    logger.info(
        f"Successfully added {added_battery_units} PLEXOS Battery units to network"
    )

    # ===== SUMMARY =====
    total_added = added_storage_units + added_battery_units
    logger.info(
        f"Total storage units added to network: {total_added} ({added_storage_units} Storage class + {added_battery_units} Battery class)"
    )


def port_standalone_storage(
    network: Network, db: PlexosDB, storage_name: str, carrier: str = "hydro"
) -> bool:
    """
    Port a standalone PLEXOS storage unit to PyPSA StorageUnit.

    Parameters
    ----------
    network : Network
        The PyPSA network to add storage to
    db : PlexosDB
        The PLEXOS database
    storage_name : str
        Name of the storage unit
    carrier : str, optional
        Carrier type for the storage unit (from PLEXOS category), defaults to "hydro"

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        props = db.get_object_properties(ClassEnum.Storage, storage_name)

        # Helper to get property value by name
        def get_prop_value(name: str) -> Optional[str]:
            for p in props:
                if p.get("property") == name:
                    return p.get("value")
            return None

        def get_prop_float(name: str) -> Optional[float]:
            val = get_prop_value(name)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
            return None

        # Detect storage model type
        model_type = detect_storage_model_type(props)
        logger.info(f"Detected model type '{model_type}' for storage {storage_name}")

        # Find connected bus - try direct connection first, then via generators
        bus = find_bus_for_object(db, storage_name, ClassEnum.Storage)
        connection_method = "direct"
        primary_generator = None

        if bus is None:
            # Try finding bus via connected generators (common for hydro storage)
            bus, primary_generator = find_bus_for_storage_via_generators(
                db, storage_name
            )
            connection_method = "via_generator"

        if bus is None:
            logger.warning(
                f"No connected bus found for storage {storage_name} (tried direct and generator connections)"
            )
            return False

        # Extract basic properties
        initial_volume = get_prop_float("Initial Volume") or 0.0
        min_volume = get_prop_float("Min Volume") or 0.0
        max_volume = get_prop_float("Max Volume") or 1000.0  # Default 1000 MWh
        units = get_prop_value("Units")

        # Convert volumes to energy (MWh)
        max_energy, max_method = convert_plexos_volume_to_energy(
            max_volume, model_type, units
        )
        min_energy, min_method = convert_plexos_volume_to_energy(
            min_volume, model_type, units
        )
        initial_energy, initial_method = convert_plexos_volume_to_energy(
            initial_volume, model_type, units
        )

        # Get End Effects Method for cyclic state of charge
        end_effects = get_end_effects_method(props)
        cyclic_state_of_charge = end_effects == "recycle"

        # Use the passed carrier (from PLEXOS category)
        # Note: carrier parameter already contains the category-based carrier
        # Only override if model_type suggests it's a battery
        if model_type == "battery" and carrier == "hydro":
            carrier = "battery"

        # Calculate power capacity (assume reasonable power/energy ratio if not specified)
        # For hydro: typically 6-12 hours, for batteries: 2-4 hours
        if model_type == "battery":
            default_hours = 4.0
        else:
            default_hours = 8.0

        p_nom = max_energy / default_hours if max_energy > 0 else 100.0  # MW
        max_hours = max_energy / p_nom if p_nom > 0 else default_hours

        # Calculate state of charge parameters (as percentages)
        if max_energy > 0:
            state_of_charge_initial = initial_energy / max_energy
            # Don't set min state of charge if it's zero (let PyPSA handle it)
            state_of_charge_min = min_energy / max_energy if min_energy > 0 else 0.0
        else:
            state_of_charge_initial = 0.5  # 50% default
            state_of_charge_min = 0.0

        # Add storage unit to network using PyPSA's add method
        network.add(
            "StorageUnit",
            storage_name,
            bus=bus,
            carrier=carrier,
            p_nom=p_nom,
            max_hours=max_hours,
            efficiency_store=0.9,  # Default efficiency
            efficiency_dispatch=0.9,
            state_of_charge_initial=state_of_charge_initial,
            cyclic_state_of_charge=cyclic_state_of_charge,
        )

        # Set min state of charge if it's greater than 0
        if state_of_charge_min > 0:
            # This would require using Store + Link for more complex constraints
            logger.info(
                f"Storage {storage_name} has minimum state of charge {state_of_charge_min:.2%}, but StorageUnit doesn't support this constraint directly"
            )

        # Enhanced logging with connection information
        if connection_method == "via_generator":
            logger.info(
                f"Added storage {storage_name}: {p_nom:.1f} MW, {max_hours:.1f} hours, model={model_type}, carrier={carrier}"
            )
            logger.info(
                f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
            )
        else:
            logger.info(
                f"Added storage {storage_name}: {p_nom:.1f} MW, {max_hours:.1f} hours, model={model_type}, carrier={carrier}"
            )
            logger.info(f"  Connection: direct to bus '{bus}'")

        return True

    except Exception as e:
        logger.error(f"Error porting standalone storage {storage_name}: {e}")
        return False


def port_pumped_hydro_pair(
    network: Network,
    db: PlexosDB,
    head_name: str,
    tail_name: str,
    head_carrier: str = "pumped_hydro",
    tail_carrier: str = "pumped_hydro",
) -> bool:
    """
    Port a PLEXOS pumped hydro HEAD/TAIL pair to PyPSA StorageUnit.

    Combines the two reservoirs into a single StorageUnit representing the system.

    Parameters
    ----------
    network : Network
        The PyPSA network to add storage to
    db : PlexosDB
        The PLEXOS database
    head_name : str
        Name of the HEAD storage unit
    tail_name : str
        Name of the TAIL storage unit
    head_carrier : str, optional
        Carrier for HEAD storage (from PLEXOS category), defaults to "pumped_hydro"
    tail_carrier : str, optional
        Carrier for TAIL storage (from PLEXOS category), defaults to "pumped_hydro"

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Get properties for both head and tail
        head_props = db.get_object_properties(ClassEnum.Storage, head_name)
        tail_props = db.get_object_properties(ClassEnum.Storage, tail_name)

        def get_prop_float_from_props(props: list, name: str) -> Optional[float]:
            for p in props:
                if p.get("property") == name:
                    try:
                        return float(p.get("value"))
                    except (ValueError, TypeError):
                        pass
            return None

        # Find connected bus (try direct first, then via generators)
        head_bus = find_bus_for_object(db, head_name, ClassEnum.Storage)
        tail_bus = find_bus_for_object(db, tail_name, ClassEnum.Storage)
        primary_generator = None
        connection_method = "direct"

        # Use head bus as primary, or tail if head not found
        bus = head_bus or tail_bus

        if bus is None:
            # Try finding bus via connected generators for head storage
            bus, primary_generator = find_bus_for_storage_via_generators(db, head_name)
            connection_method = "via_generator"

            if bus is None:
                # Try finding bus via connected generators for tail storage
                bus, primary_generator = find_bus_for_storage_via_generators(
                    db, tail_name
                )
                connection_method = "via_generator"

        if bus is None:
            logger.warning(
                f"No connected bus found for pumped hydro pair {head_name}/{tail_name} (tried direct and generator connections)"
            )
            return False

        # Get volumes for both reservoirs
        head_max = get_prop_float_from_props(head_props, "Max Volume") or 0.0
        head_min = get_prop_float_from_props(head_props, "Min Volume") or 0.0
        head_initial = get_prop_float_from_props(head_props, "Initial Volume") or 0.0

        tail_max = get_prop_float_from_props(tail_props, "Max Volume") or 0.0
        tail_min = get_prop_float_from_props(tail_props, "Min Volume") or 0.0
        tail_initial = get_prop_float_from_props(tail_props, "Initial Volume") or 0.0

        # Detect model types
        head_model_type = detect_storage_model_type(head_props)
        tail_model_type = detect_storage_model_type(tail_props)

        # Convert to energy (use head reservoir as primary)
        max_energy, conversion_method = convert_plexos_volume_to_energy(
            head_max, head_model_type
        )
        initial_energy, _ = convert_plexos_volume_to_energy(
            head_initial, head_model_type
        )

        # For pumped hydro, the effective storage is the usable capacity of the upper reservoir
        # The lower reservoir acts as a source/sink

        # Calculate power capacity (typical pumped hydro: 6-10 hours)
        default_hours = 8.0
        p_nom = max_energy / default_hours if max_energy > 0 else 200.0  # MW
        max_hours = max_energy / p_nom if p_nom > 0 else default_hours

        # Initial state of charge
        state_of_charge_initial = initial_energy / max_energy if max_energy > 0 else 0.5

        # Pumped hydro typically has high efficiency (80-90%)
        efficiency = 0.85  # Round-trip efficiency ~85%

        # Determine carrier type: prefer HEAD carrier, fallback to TAIL, then default
        carrier = (
            head_carrier
            if head_carrier != "pumped_hydro"
            else (tail_carrier if tail_carrier != "pumped_hydro" else "pumped_hydro")
        )

        # Create combined name for the pumped hydro system
        system_name = f"{head_name.replace('HEAD', '').replace('Upper', '').strip('_- ')}_PumpedHydro"
        if not system_name or system_name == "_PumpedHydro":
            system_name = f"{head_name}_{tail_name}"

        # Add as single StorageUnit representing the pumped hydro system
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
        )  # Pumped hydro typically cyclic

        logger.info(
            f"Added pumped hydro system {system_name}: {p_nom:.1f} MW, {max_hours:.1f} hours, efficiency={efficiency:.1%}, carrier={carrier}"
        )
        logger.info(
            f"  Combined from HEAD: {head_name} ({head_max:.1f}, {head_carrier}) + TAIL: {tail_name} ({tail_max:.1f}, {tail_carrier})"
        )
        if connection_method == "via_generator":
            logger.info(
                f"  Connection: via generator '{primary_generator}' to bus '{bus}'"
            )
        else:
            logger.info(f"  Connection: direct to bus '{bus}'")

        return True

    except Exception as e:
        logger.error(f"Error porting pumped hydro pair {head_name}/{tail_name}: {e}")
        return False


def add_hydro_inflows(network: Network, db: PlexosDB, path: str):
    """
    Enhanced function to add inflow time series for hydro storage units to PyPSA network.

    Supports multiple inflow data formats:
    - Data file references in Natural Inflow property
    - Timeslice arrays in Natural Inflow property
    - Monthly/daily inflow profiles

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        The Plexos database containing storage unit data.
    path : str
        Path to the folder containing inflow profile files.
    """
    added_inflows = 0

    for storage_unit in network.storage_units.index:
        try:
            # Find original PLEXOS storage names that correspond to this PyPSA storage unit
            # Handle both standalone and pumped hydro systems
            original_names = []

            if "_PumpedHydro" in storage_unit:
                # This is a pumped hydro system - check both HEAD and TAIL
                base_name = storage_unit.replace("_PumpedHydro", "")
                potential_names = [
                    f"{base_name}HEAD",
                    f"{base_name}TAIL",
                    f"HEAD{base_name}",
                    f"TAIL{base_name}",
                    f"{base_name} HEAD",
                    f"{base_name} TAIL",
                ]

                # Check which ones exist in the database
                all_storage_names = db.list_objects_by_class(ClassEnum.Storage)
                for name in potential_names:
                    if name in all_storage_names:
                        original_names.append(name)
            else:
                # Standalone storage - use the name directly
                original_names = [storage_unit]

            # Check each original storage name for inflow data
            inflow_added = False
            for orig_name in original_names:
                try:
                    # Try Storage class first, then Battery class
                    props = None
                    try:
                        props = db.get_object_properties(ClassEnum.Storage, orig_name)
                        object_class = "Storage"
                    except:
                        try:
                            props = db.get_object_properties(
                                ClassEnum.Battery, orig_name
                            )
                            object_class = "Battery"
                        except:
                            continue

                    if props:
                        inflow_data = process_storage_inflows(
                            props, path, network.snapshots
                        )

                        if inflow_data is not None:
                            # Add inflows to the PyPSA storage unit
                            network.storage_units_t.inflow[storage_unit] = inflow_data
                            logger.info(
                                f"Added inflow data for storage {storage_unit} from {orig_name} ({object_class} class)"
                            )
                            added_inflows += 1
                            inflow_added = True
                            break  # Use first successful inflow data

                except Exception as e:
                    logger.warning(f"Error processing inflows for {orig_name}: {e}")
                    continue

            if not inflow_added:
                logger.info(f"No inflow data found for storage {storage_unit}")

        except Exception as e:
            logger.error(f"Error adding inflows for storage unit {storage_unit}: {e}")

    logger.info(f"Added inflow data for {added_inflows} storage units")


def process_storage_inflows(
    props: list, inflow_path: str, snapshots: pd.DatetimeIndex
) -> Optional[pd.Series]:
    """
    Process PLEXOS storage inflow data from properties.

    Returns inflow time series aligned with network snapshots, or None if no inflow data.
    """
    from plexos_pypsa.utils.paths import extract_filename
    
    # Look for Natural Inflow property with Data File reference
    for prop in props:
        if prop.get("property") == "Natural Inflow":
            # Check if it's a data file reference
            if "Data File" in prop.get("texts", ""):
                filename = (
                    prop["texts"].split("Data File.")[1]
                    if "Data File." in prop["texts"]
                    else None
                )
                if filename:
                    # Extract just the filename to avoid double path issues
                    clean_filename = extract_filename(filename.strip())
                    return load_inflow_from_file(clean_filename, inflow_path, snapshots)
    
    # Also check for Filename property (common for hydro inflow files)
    for prop in props:
        if prop.get("property") == "Filename":
            filename = prop.get("texts", "").strip()
            if filename:
                # Extract just the filename to avoid double path issues
                clean_filename = extract_filename(filename)
                return load_inflow_from_file(clean_filename, inflow_path, snapshots)

            # Check if it's a timeslice array (list of values)
            value = prop.get("value")
            if value and isinstance(value, (list, str)):
                try:
                    if isinstance(value, str):
                        # Parse string representation of list
                        import ast

                        value = (
                            ast.literal_eval(value)
                            if value.startswith("[")
                            else [float(value)]
                        )

                    if isinstance(value, list) and len(value) > 1:
                        return create_inflow_from_timeslice_array(value, snapshots)

                except Exception as e:
                    logger.warning(f"Error parsing inflow timeslice array: {e}")

    return None


def load_inflow_from_file(
    filename: str, inflow_path: str, snapshots: pd.DatetimeIndex
) -> Optional[pd.Series]:
    """Load inflow data from file and align with snapshots."""
    from plexos_pypsa.utils.paths import safe_join
    
    try:
        # Use cross-platform path joining
        file_path = safe_join(inflow_path, filename)

        if not os.path.exists(file_path):
            logger.warning(f"Inflow file not found: {file_path}")
            return None

        # Read the inflow file
        df = pd.read_csv(file_path)

        # Try different date column combinations
        if "Year" in df.columns and "Month" in df.columns:
            if "Day" in df.columns:
                # Daily data
                df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
            else:
                # Monthly data - assume first day of month
                df["Day"] = 1
                df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            logger.warning(f"Cannot parse date columns in inflow file: {file_path}")
            return None

        # Find inflow column
        inflow_col = None
        for col in ["Inflows", "Inflow", "Natural Inflow", "Flow"]:
            if col in df.columns:
                inflow_col = col
                break

        if inflow_col is None:
            logger.warning(f"Cannot find inflow data column in file: {file_path}")
            return None

        # Create time series
        inflow_series = df.set_index("date")[inflow_col]

        # Resample to match network snapshots
        inflows_resampled = inflow_series.reindex(snapshots, method="ffill")

        # If data is daily but snapshots are hourly, distribute evenly
        if len(snapshots) > len(inflow_series):
            time_instances_per_day = (
                snapshots.to_series()
                .groupby(snapshots.to_series().dt.date)
                .size()
                .iloc[0]
            )
            inflows_resampled = inflows_resampled / time_instances_per_day

        return inflows_resampled

    except Exception as e:
        logger.error(f"Error loading inflow file {filename}: {e}")
        return None


def create_inflow_from_timeslice_array(
    values: list, snapshots: pd.DatetimeIndex
) -> pd.Series:
    """Create inflow time series from timeslice array values."""
    try:
        # Convert to numeric values
        numeric_values = [float(v) for v in values]

        # Create repeating pattern across snapshots
        # Assume values represent monthly or seasonal patterns
        if len(numeric_values) == 12:
            # Monthly pattern - repeat for each month
            inflow_data = []
            for snapshot in snapshots:
                month_idx = snapshot.month - 1  # 0-indexed
                inflow_data.append(numeric_values[month_idx])
        elif len(numeric_values) == 4:
            # Seasonal pattern - map to months
            season_mapping = [
                0,
                0,
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                0,
            ]  # Dec-Feb=0, Mar-May=1, etc.
            inflow_data = []
            for snapshot in snapshots:
                season_idx = season_mapping[snapshot.month - 1]
                inflow_data.append(numeric_values[season_idx])
        else:
            # Repeat values cyclically
            inflow_data = [
                numeric_values[i % len(numeric_values)] for i in range(len(snapshots))
            ]

        return pd.Series(inflow_data, index=snapshots)

    except Exception as e:
        logger.error(f"Error creating inflow from timeslice array: {e}")
        return pd.Series(0.0, index=snapshots)  # Return zero inflows as fallback


def port_batteries(network: Network, db: PlexosDB, timeslice_csv=None):
    """
    Comprehensive function to add PLEXOS batteries as PyPSA StorageUnit components.

    This function converts PLEXOS Battery class objects to PyPSA StorageUnit components
    with the following property mappings:
    - Node -> bus (using find_bus_for_object)
    - Max Power -> p_nom
    - Max SoC/Max Volume -> max_hours (calculated as Max Volume / Max Power)
    - Initial SoC/Initial Volume -> state_of_charge_initial (in MWh)
    - Min SoC/Min Volume -> state_of_charge_min
    - Charge Efficiency -> efficiency_store
    - Discharge Efficiency -> efficiency_dispatch
    - Technical Life (preferred) / Economic Life -> lifetime

    Parameters
    ----------
    network : Network
        The PyPSA network to add batteries to.
    db : PlexosDB
        The Plexos database containing battery data.
    timeslice_csv : str, optional
        Path to timeslice CSV file (for future time-dependent properties).

    Notes
    -----
    - Batteries without a connected bus will be skipped
    - Missing properties will use reasonable defaults where possible
    - Technical Life is preferred over Economic Life if both are available
    """
    print("Adding batteries to network...")

    # Get all battery objects with their categories using SQL
    battery_query = """
        SELECT 
            o.name AS battery_name,
            o.object_id,
            c.name AS category_name
        FROM t_object o
        JOIN t_class cl ON o.class_id = cl.class_id
        LEFT JOIN t_category c ON o.category_id = c.category_id
        WHERE cl.name = 'Battery'
    """

    battery_results = db.query(battery_query)
    print(f"  Found {len(battery_results)} batteries in database")

    if not battery_results:
        print("  No batteries found in database")
        return

    # Collect unique battery categories for carrier validation
    battery_carriers = set()
    for battery_name, object_id, category_name in battery_results:
        carrier = category_name if category_name else "battery"
        battery_carriers.add(carrier)

    print(f"  Battery carriers found: {sorted(battery_carriers)}")

    # Ensure all battery carriers exist in the network
    for carrier in battery_carriers:
        if carrier not in network.carriers.index:
            network.add("Carrier", name=carrier)
            print(f"  Added carrier: {carrier}")

    # Track skipped batteries for reporting
    skipped_batteries = []
    added_count = 0

    def get_property_value(props, property_name, default=None):
        """Helper function to extract property value by name."""
        for prop in props:
            if prop["property"] == property_name:
                try:
                    return (
                        float(prop["value"]) if prop["value"] is not None else default
                    )
                except (ValueError, TypeError):
                    return default
        return default

    for battery_name, object_id, category_name in battery_results:
        try:
            print(f"  Processing battery: {battery_name}")

            # Determine the carrier from the category
            carrier = category_name if category_name else "battery"

            # Find the connected bus using find_bus_for_object
            bus = find_bus_for_object(db, battery_name, ClassEnum.Battery)
            primary_generator = None
            connection_method = "direct"

            if bus is None:
                # Try finding bus via connected generators (similar to storage units)
                bus, primary_generator = find_bus_for_storage_via_generators(
                    db, battery_name
                )
                connection_method = "via_generator"

            if bus is None:
                print(
                    f"    Warning: No connected bus found for battery {battery_name} (tried direct and generator connections)"
                )
                skipped_batteries.append(f"{battery_name} (no bus)")
                continue

            # Get all properties for this battery
            # TODO: get_object_properties should handle both Battery and Generator classes
            try:
                props = db.get_object_properties(ClassEnum.Battery, battery_name)
            except KeyError:
                props = db.get_object_properties(ClassEnum.Generator, battery_name)

            # Extract Max Power (required for p_nom)
            max_power = get_property_value(props, "Max Power")
            if max_power is None or max_power <= 0:
                print(
                    f"    Warning: No valid 'Max Power' found for battery {battery_name}"
                )
                skipped_batteries.append(f"{battery_name} (no Max Power)")
                continue

            # Extract volume properties (try different naming conventions)
            max_volume = get_property_value(props, "Max Volume") or get_property_value(
                props, "Max SoC"
            )
            initial_volume = get_property_value(
                props, "Initial Volume"
            ) or get_property_value(props, "Initial SoC", 0.0)
            min_volume = get_property_value(props, "Min Volume") or get_property_value(
                props, "Min SoC", 0.0
            )

            # Calculate max_hours (Max Volume / Max Power)
            if max_volume is not None and max_volume > 0:
                max_hours = max_volume / max_power
            else:
                print(
                    f"    Warning: No valid 'Max Volume/SoC' found for battery {battery_name}, using default 4 hours"
                )
                max_hours = 4.0  # Default to 4-hour battery

            # Extract efficiency properties
            efficiency_store = get_property_value(
                props, "Charge Efficiency", 0.9
            )  # Default 90%
            efficiency_dispatch = get_property_value(
                props, "Discharge Efficiency", 0.9
            )  # Default 90%

            # Convert efficiencies from percentage to decimal if needed
            if efficiency_store > 1.0:
                efficiency_store = efficiency_store / 100.0
            if efficiency_dispatch > 1.0:
                efficiency_dispatch = efficiency_dispatch / 100.0

            # Extract lifetime (prefer Technical Life over Economic Life)
            lifetime = get_property_value(props, "Technical Life")
            if lifetime is None:
                lifetime = get_property_value(props, "Economic Life")

            # Create the storage unit entry
            storage_unit_data = {
                "bus": bus,
                "carrier": carrier,
                "p_nom": max_power,
                "max_hours": max_hours,
                "efficiency_store": efficiency_store,
                "efficiency_dispatch": efficiency_dispatch,
                "state_of_charge_initial": initial_volume
                if initial_volume is not None
                else 0.0,
                "state_of_charge_min": min_volume if min_volume is not None else 0.0,
            }

            # Add lifetime if available
            if lifetime is not None:
                storage_unit_data["lifetime"] = lifetime

            # Add the battery to the network using PyPSA's add method
            network.add("StorageUnit", battery_name, **storage_unit_data)
            added_count += 1

            # Enhanced logging with connection information
            if connection_method == "via_generator":
                print(
                    f"    Added battery {battery_name}: {max_power:.1f} MW, {max_hours:.1f} hours, bus={bus}, carrier={carrier}"
                )
                print(
                    f"      Connection: via generator '{primary_generator}' to bus '{bus}'"
                )
            else:
                print(
                    f"    Added battery {battery_name}: {max_power:.1f} MW, {max_hours:.1f} hours, bus={bus}, carrier={carrier}"
                )

        except Exception as e:
            print(f"    Error processing battery {battery_name}: {e}")
            skipped_batteries.append(f"{battery_name} (error: {e})")

    # Summary reporting
    print("\nBattery processing complete:")
    print(f"  Batteries added: {added_count}")
    print(f"  Batteries skipped: {len(skipped_batteries)}")

    if skipped_batteries:
        print("  Skipped batteries:")
        for skipped in skipped_batteries:
            print(f"    - {skipped}")

    # Set capital costs for all added batteries
    if added_count > 0:
        print("\nSetting battery capital costs...")
        set_capital_costs_generic(network, db, "StorageUnit", ClassEnum.Battery)

        print("Setting battery marginal costs...")
        set_battery_marginal_costs(network, db, timeslice_csv)
