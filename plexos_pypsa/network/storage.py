import logging
import os

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

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
            # If the storage unit does not have a hydro inflow profile,      it
            print(
                f"Storage unit {storage_unit} does not have a hydro inflow profile. Skipping."
            )
