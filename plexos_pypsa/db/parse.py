import pandas as pd
from plexosdb.enums import ClassEnum  # type: ignore


def find_bus_for_object(db, object_name, object_class):
    """
    Finds the associated bus (Node) for a given object (e.g., Generator, Storage).

    Parameters:
        db: PlexosDB instance
        object_name: Name of the object (e.g., generator name)
        object_class: ClassEnum for the object (e.g., ClassEnum.Generator)

    Returns:
        The name of the associated Node (bus), or None if not found.
    """
    try:
        memberships = db.get_memberships_system(object_name, object_class=object_class)
    except Exception as e:
        print(f"Error finding memberships for {object_name}: {e}")
        return None

    # First pass: direct relationship to Node
    for m in memberships:
        if m.get("class") == "Node":
            return m.get("name")

    # Second pass: indirect match via collection name (common for Storage)
    for m in memberships:
        if "node" in m.get("collection_name", "").lower():
            return m.get("name")

    # Third pass: check child memberships if available (less common)
    try:
        child_memberships = db.get_child_memberships(
            object_name, object_class=object_class
        )
        for cm in child_memberships:
            if cm.get("class") == "Node":
                return cm.get("name")
    except Exception as e:
        print(f"Error finding child memberships for {object_name}: {e}")

    print(f"No associated bus found for {object_name}")
    return None


def get_emission_rates(db, generator_name):
    """
    Extracts emission rate properties for a specific generator.

    Parameters:
        db: PlexosDB instance
        generator_name: name of the generator (str)

    Returns:
        Dictionary of emission properties {emission_type: rate}
    """
    emission_props = {}
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


def find_fuel_for_generator(db, generator_name):
    """
    Finds the associated fuel for a given generator by searching its memberships.

    Parameters:
        db: PlexosDB instance
        generator_name: Name of the generator (str)

    Returns:
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
    # print(f"No associated fuel found for {generator_name}")
    return None


def read_timeslice_activity(timeslice_csv, snapshots):
    """
    Reads a timeslice CSV and returns a DataFrame indexed by snapshots, columns=timeslice names, values=True/False for activity.
    Assumes the CSV has columns: DATETIME, NAME, TIMESLICE (where TIMESLICE is -1 for active, 0 for inactive).
    Ensures that if the last setting before the first snapshot is present, it is carried over to the first snapshot.
    """

    df = pd.read_csv(timeslice_csv)
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


def get_dataid_timeslice_map(db):
    """
    Returns a dict mapping data_id to timeslice object_id(s) (class_id=76) via the tag table.
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
    dataid_to_timeslice = {}
    for data_id, object_id, timeslice_name in rows:
        dataid_to_timeslice.setdefault(data_id, []).append(timeslice_name)
    return dataid_to_timeslice


def get_property_active_mask(
    row, snapshots, timeslice_activity=None, dataid_to_timeslice=None
):
    """
    Returns a boolean mask indicating the periods (snapshots) during which a property entry is considered active,
    based on its date range and optional timeslice activity.

    Parameters
    ----------
    row : dict or pandas.Series
        A mapping containing at least the keys "from", "to", and optionally "data_id".
        - "from": The start date (inclusive) when the property becomes active. Can be None or NaT for no lower bound.
        - "to": The end date (inclusive) when the property is no longer active. Can be None or NaT for no upper bound.
        - "data_id": Identifier used to look up timeslice information (optional).

    snapshots : pandas.DatetimeIndex or pandas.Series
        The time periods over which to evaluate the property's activity status.

    timeslice_activity : pandas.DataFrame, optional
        A DataFrame where columns are timeslice labels and the index matches `snapshots`.
        Each column is a boolean Series indicating whether the timeslice is active at each snapshot.

    dataid_to_timeslice : dict, optional
        Mapping from data_id values to lists of timeslice labels (strings).
        Used to determine which timeslices are relevant for the given property.

    Returns
    -------
    mask : pandas.Series (bool)
        Boolean Series indexed by `snapshots`, where True indicates the property is active at that snapshot,
        and False otherwise.

    Notes
    -----
    - The mask is first filtered by the "from" and "to" date range, if provided.
    - If timeslice information is available for the property's data_id, the mask is further filtered:
        - For each relevant timeslice:
            - If the timeslice exists in `timeslice_activity`, its boolean mask is applied.
            - If the timeslice is of the form "M1".."M12", it is interpreted as a month and the mask is True for that month.
            - Unknown timeslices are treated as always active (no additional filtering).
    - If no timeslice information is provided, only the date range is used.
    row, snapshots, timeslice_activity=None, dataid_to_timeslice=None
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
                mask &= timeslice_activity[ts]
            elif ts.startswith("M") and ts[1:].isdigit() and 1 <= int(ts[1:]) <= 12:
                # Month timeslice (M1..M12): active for the full month
                month = int(ts[1:])
                mask &= snapshots.month == month
            else:
                # Unknown timeslice: assume always active
                mask &= True
    return mask


# gen_name = "MURRAY10"
# emissions = get_emission_rates(prog_db, gen_name)
# print(f"Emission rates for {gen_name}:")
# for k, v in emissions.items():
#     print(f"  {k}: {v[0]} {v[1]}")
