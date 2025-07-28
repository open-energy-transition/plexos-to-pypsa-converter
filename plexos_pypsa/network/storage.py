import logging
import os

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def add_storage(network: Network, db: PlexosDB) -> None:
    """Adds storage units from PLEXOS input db to a PyPSA network."""

    storage_units = db.list_objects_by_class(ClassEnum.Storage)
    logger.info(f"Found {len(storage_units)} storage units in db")

    skipped_units = []

    for su in storage_units:
        try:
            props = db.get_object_properties(ClassEnum.Storage, su)

            # Helper to look up a property by name
            def get_prop(name):
                for p in props:
                    if p["property"] == name:
                        return float(p["value"])
                return None

            # Infer the bus from memberships
            memberships = db.get_memberships_system(su, object_class=ClassEnum.Storage)
            bus = next(
                (
                    m["name"]
                    for m in memberships
                    if m["collection_name"]
                    in {"Storage From", "Head Storage", "Tail Storage", "Storage To"}
                ),
                None,
            )
            if bus is None:
                logger.warning(f"Skipping {su}: no connected bus found via memberships")
                skipped_units.append(su)
                continue

            # Add storage unit to the PyPSA network
            network.storage_units.loc[su] = {
                "bus": bus,
                "carrier": "hydro",  # NOTE: can't find non-hydro storage in AEMO?
                "name": su,
            }

        except Exception as e:
            logger.error(f"Failed to add storage unit {su}: {e}")
            skipped_units.append(su)

    logger.info(
        f"Added {len(storage_units) - len(skipped_units)} storage units to network"
    )
    if skipped_units:
        logger.info(f"Skipped {len(skipped_units)} storage units: {skipped_units}")


def add_hydro_inflows(network: Network, db: PlexosDB, path: str):
    """
    Adds inflow time series for hydro storage units to the PyPSA network.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        The Plexos database containing storage unit data.
    path : str
        Path to the folder containing inflow profile files.
    """
    for storage_unit in network.storage_units.index:
        # Retrieve storage unit properties from the database
        props = db.get_object_properties(ClassEnum.Storage, storage_unit)

        # Check if the storage unit has a hydro inflow profile
        filename = next(
            (
                p["texts"]
                for p in props
                if "Traces\\hydro\\MonthlyNaturalInflow" in p["texts"]
            ),
            None,
        )

        if filename:
            file_path = os.path.join(path, filename.replace("\\", os.sep))

            try:
                # Read the inflow profile file
                df = pd.read_csv(file_path)

                # Create a date column using Year, Month, and Day
                df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

                # Set the date column as the index and drop unnecessary columns
                daily_inflows = df.set_index("date")["Inflows"]

                # Get the network's snapshots
                snapshots = network.snapshots

                # Resample daily inflows to match the network's snapshots
                inflows_resampled = daily_inflows.reindex(snapshots, method="ffill")

                # Detect the number of time instances per day
                time_instances_per_day = (
                    snapshots.to_series()
                    .groupby(snapshots.to_series().dt.date)
                    .size()
                    .iloc[0]
                )

                # Evenly divide daily inflows across the time instances per day
                inflows_scaled = inflows_resampled / time_instances_per_day

                # Add the inflows as a time series to the storage unit
                network.storage_units_t.inflow[storage_unit] = inflows_scaled

                print(
                    f"Added hydro inflow profile for storage unit {storage_unit} from {filename}"
                )

            except Exception as e:
                print(
                    f"Failed to process inflow profile for storage unit {storage_unit}: {e}"
                )
        else:
            # If the storage unit does not have a hydro inflow profile, skip it
            print(
                f"Storage unit {storage_unit} does not have a hydro inflow profile. Skipping."
            )


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

    # Get all battery objects from the database
    batteries = db.list_objects_by_class(ClassEnum.Battery)
    print(f"  Found {len(batteries)} batteries in database")

    if not batteries:
        print("  No batteries found in database")
        return

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

    for battery in batteries:
        try:
            print(f"  Processing battery: {battery}")

            # Find the connected bus using find_bus_for_object
            bus = find_bus_for_object(db, battery, ClassEnum.Battery)
            if bus is None:
                print(f"    Warning: No connected bus found for battery {battery}")
                skipped_batteries.append(f"{battery} (no bus)")
                continue

            # Get all properties for this battery
            try:
                props = db.get_object_properties(ClassEnum.Battery, battery)
            except KeyError:
                props = db.get_object_properties(ClassEnum.Generator, battery)

            # Extract Max Power (required for p_nom)
            max_power = get_property_value(props, "Max Power")
            if max_power is None or max_power <= 0:
                print(f"    Warning: No valid 'Max Power' found for battery {battery}")
                skipped_batteries.append(f"{battery} (no Max Power)")
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
                    f"    Warning: No valid 'Max Volume/SoC' found for battery {battery}, using default 4 hours"
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
                "carrier": "battery",
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

            # Add the battery to the network
            network.storage_units.loc[battery] = storage_unit_data
            added_count += 1

            print(
                f"    Added battery {battery}: {max_power:.1f} MW, {max_hours:.1f} hours, bus={bus}"
            )

        except Exception as e:
            print(f"    Error processing battery {battery}: {e}")
            skipped_batteries.append(f"{battery} (error: {e})")

    # Summary reporting
    print("\nBattery processing complete:")
    print(f"  Batteries added: {added_count}")
    print(f"  Batteries skipped: {len(skipped_batteries)}")

    if skipped_batteries:
        print("  Skipped batteries:")
        for skipped in skipped_batteries:
            print(f"    - {skipped}")
