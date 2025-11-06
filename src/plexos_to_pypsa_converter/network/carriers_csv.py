"""CSV-based carrier/fuel functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based functions in carriers.py.
These functions read from COAD CSV exports instead of querying the SQLite database.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.db.csv_readers import (
    ensure_datetime,
    get_dataid_timeslice_map_csv,
    load_static_properties,
    load_time_varying_properties,
)
from plexos_to_pypsa_converter.db.parse import (
    get_property_active_mask,
    read_timeslice_activity,
)

logger = logging.getLogger(__name__)


def parse_fuel_prices_csv(
    csv_dir: str | Path, network: pypsa.Network, timeslice_csv: str | None = None
) -> pd.DataFrame:
    """Parse fuel prices from COAD CSV exports and return DataFrame with price time series.

    This is the CSV-based version of parse_fuel_prices() from carriers.py.

    Applies price values according to these rules:
    - If a property is linked to a timeslice, use the timeslice activity to set the property
      for the relevant snapshots (takes precedence over date_from/date_to).
    - If "Price" property is present (possibly time-dependent), use it as the fuel price.
    - If no price is available, fallback to 0.0.

    Time-dependent prices handling:
    - "date_from" Only: Value is effective from this date onward, until superseded.
    - "date_to" Only: Value applies up to and including this date.
    - Both: Value applies within the defined date range.
    - If timeslice exists, use the timeslice timeseries file to determine when property is active.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    network : pypsa.Network
        The PyPSA network containing carriers.
    timeslice_csv : str, optional
        Path to the timeslice activity CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=network.snapshots, columns=carrier names, values=fuel prices.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add carriers to network ...
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> prices = parse_fuel_prices_csv(csv_dir, network, "timeslice.csv")
    """
    snapshots = network.snapshots

    # Load timeslice activity if provided
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)

    # Load static fuel properties
    fuel_df = load_static_properties(csv_dir, "Fuel")

    # Only keep fuels that are in the network carriers
    if hasattr(network, "carriers") and not network.carriers.empty:
        fuel_df = fuel_df[fuel_df.index.isin(network.carriers.index)]
    else:
        logger.info("No carriers found in network, processing all fuels from CSV")

    if fuel_df.empty:
        logger.warning("No fuels found in CSV that match network carriers")
        return pd.DataFrame(index=snapshots, dtype=float)

    # Load time-varying properties for fuel prices
    time_varying = load_time_varying_properties(csv_dir)

    if time_varying.empty:
        logger.info(
            "No time-varying properties CSV found. Using static properties only."
        )
        # Build simple price series from static properties
        return _build_static_fuel_prices(fuel_df, snapshots)

    # Filter to Fuel class and Price property
    fuel_time_varying = time_varying[
        (time_varying["class"] == "Fuel") & (time_varying["property"] == "Price")
    ].copy()

    # Build data_id to timeslice mapping
    dataid_to_timeslice = {}
    if timeslice_csv is not None:
        dataid_to_timeslice = get_dataid_timeslice_map_csv(fuel_time_varying)

    # Build fuel price time series
    price_timeseries = _build_fuel_price_timeseries_csv(
        fuel_time_varying=fuel_time_varying,
        fuel_df=fuel_df,
        snapshots=snapshots,
        timeslice_activity=timeslice_activity,
        dataid_to_timeslice=dataid_to_timeslice,
    )

    return price_timeseries


def _build_static_fuel_prices(
    fuel_df: pd.DataFrame, snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """Build fuel price DataFrame from static properties when no time-varying data available.

    Uses Price from static CSV or defaults to 0.0 for all snapshots.
    """
    fuel_series = {}

    for fuel in fuel_df.index:
        # Try to get static price
        if "Price" in fuel_df.columns:
            static_price = fuel_df.loc[fuel, "Price"]
            if pd.notna(static_price):
                try:
                    price_value = float(static_price)
                    fuel_series[fuel] = pd.Series(
                        price_value, index=snapshots, dtype=float
                    )
                    continue
                except (ValueError, TypeError):
                    pass

        # Default to 0.0 if no price found
        fuel_series[fuel] = pd.Series(0.0, index=snapshots, dtype=float)

    result = pd.DataFrame(fuel_series, index=snapshots)
    return result


def _build_fuel_price_timeseries_csv(
    fuel_time_varying: pd.DataFrame,
    fuel_df: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None = None,
    dataid_to_timeslice: dict | None = None,
) -> pd.DataFrame:
    """Build fuel price time series using time-varying CSV data.

    Parameters
    ----------
    fuel_time_varying : pd.DataFrame
        Time-varying properties filtered to Fuel class and Price property
    fuel_df : pd.DataFrame
        Static fuel properties
    snapshots : pd.DatetimeIndex
        Model time snapshots
    timeslice_activity : pd.DataFrame, optional
        Timeslice activity matrix
    dataid_to_timeslice : dict, optional
        Mapping from data_id to timeslice names

    Returns
    -------
    pd.DataFrame
        Fuel price time series with index=snapshots, columns=fuel/carrier names
    """
    fuel_series = {}

    for fuel in fuel_df.index:
        # Get properties for this fuel from time-varying data
        fuel_props = fuel_time_varying[fuel_time_varying["object"] == fuel].copy()

        # Build property entries
        property_entries = []
        for _, p in fuel_props.iterrows():
            entry = {
                "property": p["property"],
                "value": float(p["value"]),
                "from": ensure_datetime(p["date_from"]),
                "to": ensure_datetime(p["date_to"]),
                "data_id": int(p["data_id"]) if pd.notnull(p["data_id"]) else None,
            }
            property_entries.append(entry)

        if not property_entries:
            # No time-varying properties for this fuel, use default
            fuel_series[fuel] = pd.Series(0.0, index=snapshots, dtype=float)
            continue

        prop_df_entries = pd.DataFrame(property_entries)

        # Build time series for Price
        ts = _build_price_ts(
            prop_df_entries,
            snapshots,
            timeslice_activity,
            dataid_to_timeslice,
            fallback=0.0,
        )

        fuel_series[fuel] = ts

    # Concatenate all fuel Series into a single DataFrame
    if fuel_series:
        result = pd.DataFrame(fuel_series, index=snapshots)
    else:
        result = pd.DataFrame(index=snapshots)

    return result


def _build_price_ts(
    entries: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None,
    dataid_to_timeslice: dict | None,
    fallback: float | None = None,
) -> pd.Series:
    """Build a time series for Price property with date/timeslice logic.

    Parameters
    ----------
    entries : pd.DataFrame
        Property entries with columns: property, value, from, to, data_id
    snapshots : pd.DatetimeIndex
        Time snapshots
    timeslice_activity : pd.DataFrame, optional
        Timeslice activity matrix
    dataid_to_timeslice : dict, optional
        Mapping from data_id to timeslice names
    fallback : float, optional
        Fallback value for unset snapshots

    Returns
    -------
    pd.Series
        Time series for the property
    """
    ts = pd.Series(index=snapshots, dtype=float)

    # Get all Price rows
    price_rows = entries[entries["property"] == "Price"].copy()

    if price_rows.empty:
        if fallback is not None:
            ts[:] = fallback
        return ts

    # Track which snapshots have been set
    already_set = pd.Series(False, index=snapshots)

    # Time-specific entries first (these take precedence)
    for _, row in price_rows.iterrows():
        is_time_specific = (
            pd.notnull(row.get("from"))
            or pd.notnull(row.get("to"))
            or (dataid_to_timeslice and row["data_id"] in dataid_to_timeslice)
        )

        if is_time_specific:
            mask = get_property_active_mask(
                row, snapshots, timeslice_activity, dataid_to_timeslice
            )
            # Only override where not already set, or where overlapping
            to_set = mask & (~already_set | mask)
            ts.loc[to_set] = row["value"]
            already_set |= mask

    # Non-time-specific entries fill remaining unset values
    for _, row in price_rows.iterrows():
        is_time_specific = (
            pd.notnull(row.get("from"))
            or pd.notnull(row.get("to"))
            or (dataid_to_timeslice and row["data_id"] in dataid_to_timeslice)
        )

        if not is_time_specific:
            ts.loc[ts.isnull()] = row["value"]

    if fallback is not None:
        ts = ts.fillna(fallback)

    return ts
