from typing import Any

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum

from plexos_to_pypsa_converter.db.timeslice_parser import (
    load_and_parse_timeslice_patterns,
)


def find_bus_for_object(
    db: PlexosDB, object_name: str, object_class: ClassEnum
) -> str | None:
    """Find the associated bus (Node) for a given object (e.g., Generator, Storage).

    Parameters
    ----------
        db: PlexosDB instance
        object_name: Name of the object (e.g., generator name)
        object_class: ClassEnum for the object (e.g., ClassEnum.Generator)

    Returns
    -------
        The name of the associated Node (bus), or None if not found.
    """
    try:
        memberships = db.get_memberships_system(object_name, object_class=object_class)
    except Exception as e:
        print(f"Error finding memberships for {object_name}: {e}")
        return None

    def find_node_from_memberships(memberships: list) -> str | None:
        """Given a list of membership dicts, return the child_object_name of the first Node found."""
        for m in memberships:
            # Check if the child is a Node
            if m.get("child_class_name") == "Node":
                return m.get("child_object_name")
            # Or if the parent is a Node (less common, but possible)
            if m.get("parent_class_name") == "Node":
                return m.get("parent_object_name")
        return None

    # First pass: direct relationship to Node
    node_name = find_node_from_memberships(memberships)
    if node_name:
        return node_name

    # Second pass: indirect match via collection name (common for Storage)
    for m in memberships:
        if "node" in m.get("collection_name", "").lower():
            return m.get("name")
    print(f"No associated bus found for {object_name}")
    print(f"No associated bus found for {object_name}")
    return None


def find_bus_for_storage_via_generators(
    db: PlexosDB, storage_name: str
) -> tuple[str, str] | None:
    """Find bus for storage unit by checking connected generators.

    PLEXOS storage units are often connected to generators as "Head Storage" or "Tail Storage"
    rather than directly to nodes. This function finds the bus by looking up connected generators.

    Parameters
    ----------
    db : PlexosDB
        PlexosDB instance
    storage_name : str
        Name of the storage unit

    Returns
    -------
    tuple
        (bus_name, primary_generator_name) or None if not found
    """
    # Query database directly for generator-storage relationships
    # Based on earlier analysis: generators are parents, storage are children
    query = """
        SELECT
            po.name as parent_object_name,
            pc.name as parent_class_name,
            co.name as child_object_name,
            cc.name as child_class_name,
            col.name as collection_name
        FROM t_membership m
        JOIN t_object po ON m.parent_object_id = po.object_id
        JOIN t_class pc ON po.class_id = pc.class_id
        JOIN t_object co ON m.child_object_id = co.object_id
        JOIN t_class cc ON co.class_id = cc.class_id
        JOIN t_collection col ON m.collection_id = col.collection_id
        WHERE (pc.name = 'Generator' AND co.name = ? AND cc.name = 'Storage' AND col.name LIKE '%Storage%')
    """

    try:
        # Look for generators that have this storage unit as Head/Tail Storage
        rows = db.query(query, [storage_name])
        connected_generators = []
        for row in rows:
            gen_name = row[0]  # parent_object_name (the generator)
            collection_name = row[4]  # collection_name
            connected_generators.append((gen_name, collection_name))

    except Exception as e:
        print(f"Error querying generator-storage relationships for {storage_name}: {e}")
        return None

    if not connected_generators:
        print(f"No connected generators found for storage {storage_name}")
        return None

    print(
        f"Storage {storage_name} connected to generators: {[gen for gen, _ in connected_generators]}"
    )

    # Find bus for connected generators (try in order)
    for gen_name, collection_name in connected_generators:
        gen_bus = find_bus_for_object(db, gen_name, ClassEnum.Generator)
        if gen_bus:
            print(
                f"Storage {storage_name}: found bus '{gen_bus}' via generator '{gen_name}' ({collection_name})"
            )
            return gen_bus, gen_name

    # No successful bus connection found
    generator_names = [gen for gen, _ in connected_generators]
    print(
        f"Could not find bus for storage {storage_name} via any of its generators: {generator_names}"
    )
    return None


def get_emission_rates(
    db: PlexosDB, generator_name: str
) -> dict[str, tuple[float, str]]:
    """Extract emission rate properties for a specific generator.

    Parameters
    ----------
        db: PlexosDB instance
        generator_name: name of the generator (str)

    Returns
    -------
        Dictionary of emission properties {emission_type: rate}
    """
    emission_props: dict[str, tuple[float, str]] = {}
    try:
        properties = db.get_object_properties(ClassEnum.Generator, generator_name)
    except Exception as e:
        print(f"Error retrieving properties for {generator_name}: {e}")
        return emission_props

    for prop in properties:
        tag = prop.get("tag", "").lower()
        if "emission" in tag or any(
            pollutant in tag for pollutant in ["co2", "so2", "nox", "Co2"]
        ):
            emission_type = prop.get("property").lower()
            emission_value = prop.get("value")
            unit = prop.get("unit")
            emission_props[emission_type] = (emission_value, unit)

    return emission_props


def find_fuel_for_generator(db: PlexosDB, generator_name: str) -> str | None:
    """Find the associated fuel for a given generator by searching its memberships.

    Parameters
    ----------
        db: PlexosDB instance
        generator_name: Name of the generator (str)

    Returns
    -------
        The name of the associated Fuel, or None if not found.
    """
    try:
        memberships = db.get_memberships_system(
            generator_name, object_class=ClassEnum.Generator
        )
    except Exception as e:
        print(f"Error finding memberships for {generator_name}: {e}")
        return None

    for m in memberships:
        if m.get("class") == "Fuel":
            return m.get("name")
    print(f"No associated fuel found for {generator_name}")
    return None


def read_timeslice_activity(
    timeslice_csv: str,
    snapshots: pd.DatetimeIndex | Any,
    trading_periods_per_day: int = 24,
) -> pd.DataFrame:
    """Read a timeslice CSV and return a parsed DataFrame.

    Supports two formats:
    1. Activity timeseries (AEMO format): DATETIME, NAME, TIMESLICE (-1=active, 0=inactive)
    2. Pattern definitions (SEM/CAISO format): object, category, Include(text) with patterns

    Parameters
    ----------
        timeslice_csv: Path to the timeslice CSV file
        snapshots: pd.DatetimeIndex of model snapshots
        trading_periods_per_day: Number of trading periods per day (for P symbol in patterns)

    Returns
    -------
        pd.DataFrame with index=snapshots, columns=timeslice names, values=True/False for activity

    Notes
    -----
    The function automatically detects the format based on column names:
    - If DATETIME/NAME/TIMESLICE columns exist -> activity timeseries format
    - If object/Include(text) columns exist -> pattern definition format
    """
    df = pd.read_csv(timeslice_csv)

    # Detect format based on columns
    columns_lower = [col.lower() for col in df.columns]

    # Format 1: Activity timeseries (DATETIME, NAME, TIMESLICE)
    if (
        "datetime" in columns_lower
        and "name" in columns_lower
        and "timeslice" in columns_lower
    ):
        return _read_timeslice_activity_timeseries(df, snapshots)

    # Format 2: Pattern definitions (object, Include(text))
    elif "object" in df.columns and any("include" in col.lower() for col in df.columns):
        return _read_timeslice_activity_patterns(df, snapshots, trading_periods_per_day)

    else:
        msg = f"Unknown timeslice CSV format. Columns: {df.columns.tolist()}"
        raise ValueError(msg)


def _read_timeslice_activity_timeseries(
    df: pd.DataFrame, snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """Parse timeslice activity timeseries format (AEMO-style).

    Format: DATETIME, NAME, TIMESLICE (-1=active, 0=inactive)
    """
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], dayfirst=True)
    timeslice_names = df["NAME"].unique()
    activity = pd.DataFrame(False, index=snapshots, columns=timeslice_names)

    for ts_name in timeslice_names:
        ts_df = df[df["NAME"] == ts_name].sort_values("DATETIME")
        # Find the last TIMESLICE setting before the first snapshot
        before_first = ts_df[ts_df["DATETIME"] < snapshots[0]]
        if not before_first.empty:
            last_setting = before_first.iloc[-1]["TIMESLICE"]
            activity.loc[:, ts_name] = last_setting == -1
        # Apply all changes at or after the first snapshot
        for _, row in ts_df[ts_df["DATETIME"] >= snapshots[0]].iterrows():
            datetime = row["DATETIME"]
            mask = activity.index >= datetime
            if row["TIMESLICE"] == -1:  # Active
                activity.loc[mask, ts_name] = True
            elif row["TIMESLICE"] == 0:  # Inactive
                activity.loc[mask, ts_name] = False

    return activity


def _read_timeslice_activity_patterns(
    df: pd.DataFrame, snapshots: pd.DatetimeIndex, trading_periods_per_day: int = 24
) -> pd.DataFrame:
    """Parse timeslice pattern definitions (SEM/CAISO-style).

    Format: object, category, Include(text) with PLEXOS pattern strings
    """
    activity = load_and_parse_timeslice_patterns(df, snapshots, trading_periods_per_day)

    return activity


def get_dataid_timeslice_map(db: PlexosDB) -> dict[int, list[str]]:
    """Return a dict mapping data_id to timeslice object_id(s) (class_id=76) via the tag table.

    Parameters
    ----------
        db: PlexosDB instance

    Returns
    -------
        dict mapping data_id to list of timeslice names
    """
    # Query for all data_id, tag_object_id pairs where tag_object_id is a timeslice
    query = """
        SELECT t.data_id, t.object_id, o.name
        FROM t_tag t
        JOIN t_object o ON t.object_id = o.object_id
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.class_id = 76
    """
    rows = db.query(query)
    # Map data_id to timeslice name(s)
    dataid_to_timeslice: dict[int, list[str]] = {}
    for data_id, _object_id, timeslice_name in rows:
        dataid_to_timeslice.setdefault(data_id, []).append(timeslice_name)
    return dataid_to_timeslice


def get_property_active_mask(
    row: pd.Series,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None = None,
    dataid_to_timeslice: dict[int, list[str]] | None = None,
) -> pd.Series:
    """Return a boolean mask indicating the periods (snapshots) during which a property entry is considered active.

    Parameters
    ----------
        row: A row from the properties DataFrame (with 'from', 'to', 'data_id' fields)
        snapshots: pd.DatetimeIndex of model snapshots
        timeslice_activity: pd.DataFrame of timeslice activity (index=snapshots, columns
            =timeslice names, values=True/False for activity)
        dataid_to_timeslice: dict mapping data_id to list of timeslice names

    Returns
    -------
        pd.Series of booleans indexed by snapshots, indicating if the property is active based on its date range and optional timeslice activity.
    """
    mask = pd.Series(True, index=snapshots)
    # Date logic
    if pd.notnull(row.get("from")):
        mask &= snapshots >= row["from"]
    if pd.notnull(row.get("to")):
        mask &= snapshots <= row["to"]

    # Timeslice logic
    if (
        timeslice_activity is not None
        and dataid_to_timeslice is not None
        and row.get("data_id") in dataid_to_timeslice
    ):
        for ts in dataid_to_timeslice[row["data_id"]]:
            if ts in timeslice_activity.columns:
                ts_mask = timeslice_activity[ts]
                # If t_date_from is present, only active after that date
                if pd.notnull(row.get("from")):
                    ts_mask = ts_mask & (snapshots >= row["from"])
                # If t_date_to is present, only active before or at that date
                if pd.notnull(row.get("to")):
                    ts_mask = ts_mask & (snapshots <= row["to"])
                mask &= ts_mask
            elif ts.startswith("M") and ts[1:].isdigit() and 1 <= int(ts[1:]) <= 12:
                month = int(ts[1:])
                ts_mask = pd.Series(snapshots.month == month, index=snapshots)
                if pd.notnull(row.get("from")):
                    ts_mask = ts_mask & (snapshots >= row["from"])
                if pd.notnull(row.get("to")):
                    ts_mask = ts_mask & (snapshots <= row["to"])
                mask &= ts_mask
            else:
                mask &= True
    return mask
