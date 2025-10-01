import logging
from typing import Any

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.db.parse import (
    get_dataid_timeslice_map,
    get_property_active_mask,
    read_timeslice_activity,
)

logger = logging.getLogger(__name__)


def parse_fuel_prices(
    db: PlexosDB, network: pypsa.Network, timeslice_csv: str | None = None
) -> pd.DataFrame:
    """Parse fuel prices from the PlexosDB using SQL and return a DataFrame with index=network.snapshots, columns=carrier names.

    Applies price values according to these rules:
    - If a property is linked to a timeslice, use the timeslice activity to set the property for the relevant snapshots (takes precedence over t_date_from/t_date_to).
    - If "Price" property is present (possibly time-dependent), use it as the fuel price.
    - If no price is available, fallback to 0.0.

    This is how the time-dependent prices are handled:
    - "t_date_from" Only: Value is effective from this date onward, until superseded.
    - "t_date_to" Only: Value applies up to and including this date, starting from simulation start or last defined "Date From".
    - Both: Value applies within the defined date range.
    - If timeslice exists, use the timeslice timeseries file to determine when the property is active.
    - If the timeslice doesn't exist in the timeslice file but has names like M1, M2, M3, .. M12, assume it is active for the full month.

    Parameters
    ----------
    db : PlexosDB
        The Plexos database containing fuel/carrier data.
    network : Network
        The PyPSA network containing carriers.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent prices.

    Returns
    -------
    pandas.DataFrame
        DataFrame with index=network.snapshots, columns=carrier names, values=fuel prices.
    """
    snapshots = network.snapshots
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)
    dataid_to_timeslice = (
        get_dataid_timeslice_map(db) if timeslice_csv is not None else {}
    )

    # 1. Get fuel/carrier object_id and name
    fuel_query = """
        SELECT o.object_id, o.name
        FROM t_object o
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.name = 'Fuel'
    """
    fuel_rows = db.query(fuel_query)
    fuel_df = pd.DataFrame(fuel_rows, columns=["object_id", "carrier"])

    # Only keep carriers that are in the network
    if hasattr(network, "carriers") and not network.carriers.empty:
        fuel_df = fuel_df[fuel_df["carrier"].isin(network.carriers.index)]
    else:
        # If no carriers in network yet, keep all fuels
        logger.info("No carriers found in network, processing all fuels from database")

    # 2. Get data_id for each fuel (child_object_id) from t_membership
    membership_query = """
        SELECT m.parent_object_id AS parent_object_id, m.child_object_id AS child_object_id, m.membership_id AS membership_id
        FROM t_membership m
        JOIN t_object p ON m.parent_object_id = p.object_id
        JOIN t_class pc ON p.class_id = pc.class_id
        JOIN t_object c ON m.child_object_id = c.object_id
        JOIN t_class cc ON c.class_id = cc.class_id
    """
    membership_rows = db.query(membership_query)
    membership_df = pd.DataFrame(
        membership_rows,
        columns=["parent_object_id", "child_object_id", "membership_id"],
    )

    # 3. Get property values for each data_id (Price), with t_date_from/to,
    # including both entries with and without a tag/timeslice (class_id=76).
    prop_query = """
        SELECT
            d.data_id,
            d.membership_id,
            p.name as property,
            d.value,
            df.date as t_date_from,
            dt.date as t_date_to,
            o.name as timeslice
        FROM t_data d
        JOIN t_property p ON d.property_id = p.property_id
        LEFT JOIN t_date_from df ON d.data_id = df.data_id
        LEFT JOIN t_date_to dt ON d.data_id = dt.data_id
        LEFT JOIN t_tag t ON d.data_id = t.data_id
        LEFT JOIN t_object o ON t.object_id = o.object_id AND o.class_id = 76
        WHERE p.name = 'Price'
    """
    prop_rows = db.query(prop_query)

    prop_df = pd.DataFrame(
        prop_rows,
        columns=[
            "data_id",
            "membership_id",
            "property",
            "value",
            "t_date_from",
            "t_date_to",
            "timeslice",
        ],
    )

    # 4. Merge: fuel object_id -> data_id -> property info
    merged = pd.merge(membership_df, prop_df, on="membership_id", how="inner")
    merged = pd.merge(
        merged, fuel_df, left_on="child_object_id", right_on="object_id", how="inner"
    )

    def build_fuel_price_timeseries(
        merged: pd.DataFrame,
        fuel_df: pd.DataFrame,
        snapshots: pd.DatetimeIndex | Any,
        timeslice_activity: pd.DataFrame | None = None,
        dataid_to_timeslice: dict | None = None,
    ) -> pd.DataFrame:
        """For each fuel/carrier, create a time series for price using Price property, with timeslice and date logic.
        Returns a DataFrame: index=snapshots, columns=carrier names, values=prices.
        """
        fuel_series = {}
        for fuel in fuel_df["carrier"]:
            props = merged[merged["carrier"] == fuel]

            # Build property entries
            property_entries = []
            for _, p in props.iterrows():
                entry = {
                    "property": p["property"],
                    "value": float(p["value"]),
                    "from": pd.to_datetime(p["t_date_from"])
                    if pd.notnull(p["t_date_from"])
                    else None,
                    "to": pd.to_datetime(p["t_date_to"])
                    if pd.notnull(p["t_date_to"])
                    else None,
                    "data_id": p["data_id"],
                }
                if dataid_to_timeslice and p["data_id"] in dataid_to_timeslice:
                    entry["timeslices"] = dataid_to_timeslice[p["data_id"]]
                property_entries.append(entry)
            if property_entries:
                prop_df_entries = pd.DataFrame(property_entries)
            else:
                # Create empty DataFrame with expected columns
                prop_df_entries = pd.DataFrame(
                    columns=["property", "value", "from", "to", "data_id", "timeslices"]
                )

            # Helper to build a time series for a property
            def build_ts(
                prop_name: str, entries: pd.DataFrame, fallback: float | None = None
            ) -> pd.Series:
                ts = pd.Series(index=snapshots, dtype=float)
                already_set = pd.Series(False, index=snapshots)

                # Get all rows for this property
                prop_rows = entries[entries["property"] == prop_name]

                # Time-specific entries first (these take precedence)
                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if is_time_specific:
                        mask = get_property_active_mask(
                            row, snapshots, timeslice_activity, dataid_to_timeslice
                        )
                        # Only override where not already set, or where overlapping (newer overrides older)
                        to_set = mask & (~already_set | mask)
                        ts.loc[to_set] = row["value"]
                        already_set |= mask

                # Non-time-specific entries fill remaining unset values
                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if not is_time_specific:
                        ts.loc[ts.isnull()] = row["value"]

                if fallback is not None:
                    ts = ts.fillna(fallback)
                return ts

            # Build time series for Price
            price_ts = build_ts("Price", prop_df_entries, fallback=0.0)
            fuel_series[fuel] = price_ts

        # Concatenate all fuel Series into a single DataFrame
        if fuel_series:
            result = pd.concat(fuel_series, axis=1)
            result.index = snapshots
        else:
            # Return empty DataFrame with correct structure if no fuels found
            result = pd.DataFrame(index=snapshots)

        return result

    # Build the price timeseries for each fuel/carrier
    price_timeseries = build_fuel_price_timeseries(
        merged, fuel_df, snapshots, timeslice_activity, dataid_to_timeslice
    )

    return price_timeseries
