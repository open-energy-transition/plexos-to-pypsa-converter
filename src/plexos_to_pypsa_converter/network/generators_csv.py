"""CSV-based generator functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based functions in generators.py.
These functions read from COAD CSV exports instead of querying the SQLite database.
"""

import ast
import logging
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pypsa import Network

from plexos_to_pypsa_converter.db.csv_readers import (
    ensure_datetime,
    find_bus_for_object_csv,
    find_fuel_for_generator_csv,
    get_dataid_timeslice_map_csv,
    get_property_from_static_csv,
    list_objects_by_class_csv,
    load_static_properties,
    load_time_varying_properties,
    parse_numeric_value,
    read_plexos_input_csv,
)
from plexos_to_pypsa_converter.db.parse import (
    get_property_active_mask,
    read_timeslice_activity,
)
from plexos_to_pypsa_converter.network.carriers_csv import parse_fuel_prices_csv
from plexos_to_pypsa_converter.network.costs_csv import set_capital_costs_generic_csv
from plexos_to_pypsa_converter.utils.paths import (
    contains_path_pattern,
    extract_filename,
    safe_join,
)

logger = logging.getLogger(__name__)


def parse_generator_ratings_csv(
    csv_dir: str | Path, network: Network, timeslice_csv: str | None = None
) -> pd.DataFrame:
    """Parse generator ratings from COAD CSV exports and return DataFrame with p_max_pu time series.

    This is the CSV-based version of parse_generator_ratings() from generators.py.

    Applies rating values according to these rules:
    - If a property is linked to a timeslice, use the timeslice activity to set the property
      for the relevant snapshots (takes precedence over t_date_from/t_date_to).
    - If "Rating" is present (possibly time-dependent), use it as p_max_pu (normalized by Max Capacity).
    - Otherwise, if "Rating Factor" is present (possibly time-dependent), use it as p_max_pu
      (Rating Factor is a percentage of Max Capacity).
    - If neither, fallback to "Max Capacity".

    Time-dependent ratings handling:
    - "date_from" Only: Value is effective from this date onward, until superseded.
    - "date_to" Only: Value applies up to and including this date.
    - Both: Value applies within the defined date range.
    - If timeslice exists, use the timeslice timeseries file to determine when property is active.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    network : Network
        PyPSA network with generators already added
    timeslice_csv : str, optional
        Path to the timeslice activity CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with index=network.snapshots, columns=generator names, values=p_max_pu

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators to network ...
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> ratings = parse_generator_ratings_csv(csv_dir, network, "timeslice.csv")
    >>> network.generators_t.p_max_pu = ratings
    """
    snapshots = network.snapshots

    # Load timeslice activity if provided
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)

    # Load static generator properties
    generator_df = load_static_properties(csv_dir, "Generator")

    # Only keep generators that are in the network
    generator_df = generator_df[generator_df.index.isin(network.generators.index)]

    if generator_df.empty:
        logger.warning("No generators found in CSV that match network generators")
        return pd.DataFrame(
            index=snapshots, columns=network.generators.index, dtype=float
        )

    # Load time-varying properties for Rating, Rating Factor, and Max Capacity
    time_varying = load_time_varying_properties(csv_dir)

    if time_varying.empty:
        logger.info(
            "No time-varying properties CSV found. Using static properties only."
        )
        # Build simple p_max_pu from static properties
        return _build_static_ratings(generator_df, network)

    # Filter to Generator class and relevant properties
    gen_time_varying = time_varying[
        (time_varying["class"] == "Generator")
        & (time_varying["property"].isin(["Rating", "Rating Factor", "Max Capacity"]))
    ].copy()

    # Build data_id to timeslice mapping
    dataid_to_timeslice = {}
    if timeslice_csv is not None:
        dataid_to_timeslice = get_dataid_timeslice_map_csv(gen_time_varying)

    # Build p_max_pu time series for each generator
    p_max_pu_timeseries = _build_generator_p_max_pu_timeseries_csv(
        gen_time_varying=gen_time_varying,
        generator_df=generator_df,
        network=network,
        snapshots=snapshots,
        timeslice_activity=timeslice_activity,
        dataid_to_timeslice=dataid_to_timeslice,
    )

    return p_max_pu_timeseries


def parse_generator_min_stable_levels_csv(
    csv_dir: str | Path, network: Network, timeslice_csv: str | None = None
) -> pd.DataFrame:
    """Parse generator minimum stable levels from COAD CSV exports and return DataFrame with p_min_pu time series.

    This is the CSV-based version of parse_generator_min_stable_levels() from generators.py.

    Applies minimum generation constraints according to these rules:
    - If "Min Stable Factor" is present (percentage), use it as p_min_pu (Min Stable Factor / 100)
    - If "Min Stable Level" is present (MW), use it as p_min_pu (Min Stable Level / p_nom)
    - If "Min Pump Load" is present (for storage), use it as p_min_pu (Min Pump Load / p_nom)
    - If none present, fallback to p_min_pu = 0.0 (no minimum constraint)

    Time-dependent handling:
    - "date_from" Only: Value is effective from this date onward, until superseded.
    - "date_to" Only: Value applies up to and including this date.
    - Both: Value applies within the defined date range.
    - If timeslice exists, use the timeslice timeseries file to determine when property is active.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    network : Network
        PyPSA network with generators already added
    timeslice_csv : str, optional
        Path to the timeslice activity CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with index=network.snapshots, columns=generator names, values=p_min_pu

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators to network ...
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> min_levels = parse_generator_min_stable_levels_csv(csv_dir, network, "timeslice.csv")
    >>> network.generators_t.p_min_pu = min_levels
    """
    snapshots = network.snapshots

    # Load timeslice activity if provided
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)

    # Load static generator properties
    generator_df = load_static_properties(csv_dir, "Generator")

    # Only keep generators that are in the network
    generator_df = generator_df[generator_df.index.isin(network.generators.index)]

    if generator_df.empty:
        logger.warning("No generators found in CSV that match network generators")
        return pd.DataFrame(
            index=snapshots, columns=network.generators.index, dtype=float
        )

    # Load time-varying properties for Min Stable properties
    time_varying = load_time_varying_properties(csv_dir)

    if time_varying.empty:
        logger.info(
            "No time-varying properties CSV found. Using static properties only (defaulting to p_min_pu=0.0)."
        )
        # Build simple p_min_pu from static properties (all zeros)
        gen_series = {}
        for gen in generator_df.index:
            if gen in network.generators.index:
                gen_series[gen] = pd.Series(0.0, index=snapshots, dtype=float)
        result = pd.DataFrame(gen_series, index=snapshots)
        return result

    # Filter to Generator class and relevant properties
    gen_time_varying = time_varying[
        (time_varying["class"] == "Generator")
        & (
            time_varying["property"].isin(
                ["Min Stable Level", "Min Stable Factor", "Min Pump Load"]
            )
        )
    ].copy()

    # Build data_id to timeslice mapping
    dataid_to_timeslice = {}
    if timeslice_csv is not None:
        dataid_to_timeslice = get_dataid_timeslice_map_csv(gen_time_varying)

    # Build p_min_pu time series for each generator
    p_min_pu_timeseries = _build_generator_p_min_pu_timeseries_csv(
        gen_time_varying=gen_time_varying,
        generator_df=generator_df,
        network=network,
        snapshots=snapshots,
        timeslice_activity=timeslice_activity,
        dataid_to_timeslice=dataid_to_timeslice,
    )

    return p_min_pu_timeseries


def _build_static_ratings(generator_df: pd.DataFrame, network: Network) -> pd.DataFrame:
    """Build p_max_pu DataFrame from static properties when no time-varying data available.

    Uses Rating or Max Capacity from static CSV to determine p_max_pu = 1.0 for all snapshots.
    """
    snapshots = network.snapshots
    gen_series = {}

    for gen in generator_df.index:
        if gen in network.generators.index:
            # All generators get p_max_pu = 1.0 when using static properties only
            gen_series[gen] = pd.Series(1.0, index=snapshots, dtype=float)

    result = pd.DataFrame(gen_series, index=snapshots)
    return result


def _build_generator_p_max_pu_timeseries_csv(
    gen_time_varying: pd.DataFrame,
    generator_df: pd.DataFrame,
    network: Network,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None = None,
    dataid_to_timeslice: dict | None = None,
) -> pd.DataFrame:
    """Build p_max_pu time series for each generator using time-varying CSV data.

    Parameters
    ----------
    gen_time_varying : pd.DataFrame
        Time-varying properties filtered to Generator class
    generator_df : pd.DataFrame
        Static generator properties
    network : Network
        PyPSA network
    snapshots : pd.DatetimeIndex
        Model time snapshots
    timeslice_activity : pd.DataFrame, optional
        Timeslice activity matrix
    dataid_to_timeslice : dict, optional
        Mapping from data_id to timeslice names

    Returns
    -------
    pd.DataFrame
        p_max_pu time series with index=snapshots, columns=generator names
    """
    gen_series = {}

    for gen in generator_df.index:
        if gen not in network.generators.index:
            continue

        # Get properties for this generator from time-varying data
        gen_props = gen_time_varying[gen_time_varying["object"] == gen].copy()

        # Build property entries
        property_entries = []
        for _, p in gen_props.iterrows():
            entry = {
                "property": p["property"],
                "value": float(p["value"]),
                "from": ensure_datetime(p["date_from"]),
                "to": ensure_datetime(p["date_to"]),
                "data_id": int(p["data_id"]) if pd.notnull(p["data_id"]) else None,
            }
            property_entries.append(entry)

        if property_entries:
            prop_df_entries = pd.DataFrame(property_entries)
        else:
            # No time-varying properties for this generator, use default
            gen_series[gen] = pd.Series(1.0, index=snapshots, dtype=float)
            continue

        # Helper to build a time series for a property
        def build_ts(
            prop_name: str, entries: pd.DataFrame, fallback: float | None = None
        ) -> pd.Series:
            ts = pd.Series(index=snapshots, dtype=float)
            prop_rows = entries[entries["property"] == prop_name].copy()

            if prop_rows.empty:
                if fallback is not None:
                    ts[:] = fallback
                return ts

            # Sort by from date (NaT values first)
            prop_rows["from_sort"] = pd.to_datetime(prop_rows["from"])
            prop_rows = prop_rows.sort_values("from_sort", na_position="first")

            # Track which snapshots have been set
            already_set = pd.Series(False, index=snapshots)

            # Time-specific entries first (these take precedence)
            for _, row in prop_rows.iterrows():
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
            for _, row in prop_rows.iterrows():
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

        # Get Max Capacity for this generator (from static or time-varying)
        maxcap = None
        maxcap_entries = prop_df_entries[prop_df_entries["property"] == "Max Capacity"]
        if not maxcap_entries.empty:
            maxcap = maxcap_entries.iloc[0]["value"]
        else:
            # Try static CSV
            maxcap = get_property_from_static_csv(generator_df, gen, "Max Capacity")

        # Build time series for Rating and Rating Factor
        rating_ts = build_ts("Rating", prop_df_entries)
        rating_factor_ts = build_ts("Rating Factor", prop_df_entries)

        # Load Min Stable Level for validation (data quality check)
        # Need to ensure Rating >= Min Stable Level for physical feasibility
        min_stable_level_ts = build_ts("Min Stable Level", prop_df_entries)

        # If no time-varying Min Stable Level, check static CSV as fallback
        if min_stable_level_ts.isnull().all():
            static_min_level = get_property_from_static_csv(
                generator_df, gen, "Min Stable Level"
            )
            if static_min_level is not None:
                parsed_val = parse_numeric_value(static_min_level, use_first=True)
                if parsed_val is not None:
                    min_stable_level_ts = pd.Series(parsed_val, index=snapshots)

        # Validate Rating against Min Stable Level (data quality check)
        # When Rating < Min Stable Level, generator cannot physically operate at that rating
        # Treat this as data error and ignore Rating (fallback to Max Capacity instead)
        if rating_ts.notnull().any() and min_stable_level_ts.notnull().any():
            invalid_mask = (
                rating_ts.notnull()
                & min_stable_level_ts.notnull()
                & (rating_ts < min_stable_level_ts)
            )

            if invalid_mask.any():
                num_invalid = invalid_mask.sum()
                min_rating = rating_ts[invalid_mask].min()
                max_min_stable = min_stable_level_ts[invalid_mask].max()

                logger.warning(
                    f"Generator '{gen}': Rating < Min Stable Level for {num_invalid} timesteps "
                    f"(Rating: {min_rating:.2f} MW < Min Stable: {max_min_stable:.2f} MW). "
                    f"Ignoring Rating and using Max Capacity instead (data quality issue)."
                )

                # Clear invalid Rating values (will fallback to p_max_pu=1.0)
                rating_ts[invalid_mask] = np.nan

        # Get p_nom for scaling Rating
        p_nom = None
        if gen in network.generators.index and "p_nom" in network.generators.columns:
            p_nom = network.generators.loc[gen, "p_nom"]
        if p_nom is None or (isinstance(p_nom, float) and np.isnan(p_nom)):
            p_nom = maxcap

        # Build final p_max_pu time series
        ts = pd.Series(index=snapshots, dtype=float)

        if p_nom:
            # Use Rating Factor if present (convert from percentage)
            mask_rf = rating_factor_ts.notnull()
            ts[mask_rf] = rating_factor_ts[mask_rf] / 100.0

            # Where Rating Factor is not present, use Rating if present
            mask_rating = ts.isnull() & rating_ts.notnull()
            ts[mask_rating] = rating_ts[mask_rating] / p_nom

        # Where neither is present, fallback to 1.0
        ts = ts.fillna(1.0)
        gen_series[gen] = ts

    # Concatenate all generator Series into a single DataFrame
    if gen_series:
        result = pd.DataFrame(gen_series, index=snapshots)
    else:
        result = pd.DataFrame(index=snapshots)

    return result


def _build_generator_p_min_pu_timeseries_csv(
    gen_time_varying: pd.DataFrame,
    generator_df: pd.DataFrame,
    network: Network,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None = None,
    dataid_to_timeslice: dict | None = None,
) -> pd.DataFrame:
    """Build p_min_pu time series for each generator using time-varying CSV data.

    Parameters
    ----------
    gen_time_varying : pd.DataFrame
        Time-varying properties filtered to Generator class
    generator_df : pd.DataFrame
        Static generator properties
    network : Network
        PyPSA network
    snapshots : pd.DatetimeIndex
        Model time snapshots
    timeslice_activity : pd.DataFrame, optional
        Timeslice activity matrix
    dataid_to_timeslice : dict, optional
        Mapping from data_id to timeslice names

    Returns
    -------
    pd.DataFrame
        p_min_pu time series with index=snapshots, columns=generator names
    """
    gen_series = {}

    for gen in generator_df.index:
        if gen not in network.generators.index:
            continue

        # Get properties for this generator from time-varying data
        gen_props = gen_time_varying[gen_time_varying["object"] == gen].copy()

        # Build property entries
        property_entries = []
        for _, p in gen_props.iterrows():
            entry = {
                "property": p["property"],
                "value": float(p["value"]),
                "from": ensure_datetime(p["date_from"]),
                "to": ensure_datetime(p["date_to"]),
                "data_id": int(p["data_id"]) if pd.notnull(p["data_id"]) else None,
            }
            property_entries.append(entry)

        if property_entries:
            prop_df_entries = pd.DataFrame(property_entries)
        else:
            # No time-varying properties - create empty DataFrame
            # Will check static CSV properties as fallback
            prop_df_entries = pd.DataFrame(
                columns=["property", "value", "from", "to", "data_id"]
            )

        # Helper to build a time series for a property
        def build_ts(
            prop_name: str, entries: pd.DataFrame, fallback: float | None = None
        ) -> pd.Series:
            ts = pd.Series(index=snapshots, dtype=float)
            prop_rows = entries[entries["property"] == prop_name].copy()

            if prop_rows.empty:
                if fallback is not None:
                    ts[:] = fallback
                return ts

            # Sort by from date (NaT values first)
            prop_rows["from_sort"] = pd.to_datetime(prop_rows["from"])
            prop_rows = prop_rows.sort_values("from_sort", na_position="first")

            # Track which snapshots have been set
            already_set = pd.Series(False, index=snapshots)

            # Time-specific entries first (these take precedence)
            for _, row in prop_rows.iterrows():
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
            for _, row in prop_rows.iterrows():
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

        # Get Max Capacity for this generator (from static or time-varying)
        maxcap = None
        maxcap_entries = prop_df_entries[prop_df_entries["property"] == "Max Capacity"]
        if not maxcap_entries.empty:
            maxcap = maxcap_entries.iloc[0]["value"]
        else:
            # Try static CSV
            maxcap = get_property_from_static_csv(generator_df, gen, "Max Capacity")

        # Build time series for Min Stable properties
        min_stable_level_ts = build_ts("Min Stable Level", prop_df_entries)
        min_stable_factor_ts = build_ts("Min Stable Factor", prop_df_entries)
        min_pump_load_ts = build_ts("Min Pump Load", prop_df_entries)

        # If no time-varying data, try static CSV as fallback
        if min_stable_level_ts.isnull().all():
            static_min_level = get_property_from_static_csv(
                generator_df, gen, "Min Stable Level"
            )
            if static_min_level is not None:
                parsed_val = parse_numeric_value(static_min_level, use_first=True)
                if parsed_val is not None:
                    min_stable_level_ts = pd.Series(parsed_val, index=snapshots)

        if min_stable_factor_ts.isnull().all():
            static_min_factor = get_property_from_static_csv(
                generator_df, gen, "Min Stable Factor"
            )
            if static_min_factor is not None:
                parsed_val = parse_numeric_value(static_min_factor, use_first=True)
                if parsed_val is not None:
                    min_stable_factor_ts = pd.Series(parsed_val, index=snapshots)

        if min_pump_load_ts.isnull().all():
            static_pump_load = get_property_from_static_csv(
                generator_df, gen, "Min Pump Load"
            )
            if static_pump_load is not None:
                parsed_val = parse_numeric_value(static_pump_load, use_first=True)
                if parsed_val is not None:
                    min_pump_load_ts = pd.Series(parsed_val, index=snapshots)

        # Get p_nom for scaling Min Stable Level and Min Pump Load
        p_nom = None
        if gen in network.generators.index and "p_nom" in network.generators.columns:
            p_nom = network.generators.loc[gen, "p_nom"]
        if p_nom is None or (isinstance(p_nom, float) and np.isnan(p_nom)):
            p_nom = maxcap

        # Build final p_min_pu time series
        ts = pd.Series(index=snapshots, dtype=float)

        if p_nom:
            # Priority 1: Use Min Stable Factor if present (convert from percentage)
            mask_factor = min_stable_factor_ts.notnull()
            ts[mask_factor] = min_stable_factor_ts[mask_factor] / 100.0

            # Priority 2: Use Min Stable Level if Min Stable Factor not present
            mask_level = ts.isnull() & min_stable_level_ts.notnull()
            ts[mask_level] = min_stable_level_ts[mask_level] / p_nom

            # Priority 3: Use Min Pump Load if neither above is present
            mask_pump = ts.isnull() & min_pump_load_ts.notnull()
            ts[mask_pump] = min_pump_load_ts[mask_pump] / p_nom

        # Where none is present, fallback to 0.0
        ts = ts.fillna(0.0)
        gen_series[gen] = ts

    # Concatenate all generator Series into a single DataFrame
    if gen_series:
        result = pd.DataFrame(gen_series, index=snapshots)
    else:
        result = pd.DataFrame(index=snapshots)

    return result


def add_generators_csv(
    network: Network,
    csv_dir: str | Path,
    generators_as_links: bool = False,
    fuel_bus_prefix: str = "fuel_",
):
    """Add generators from COAD CSV exports to a PyPSA network.

    This is the CSV-based version of add_generators() from generators.py.

    Parameters
    ----------
    network : Network
        The PyPSA network to which the generators will be added.
    csv_dir : str | Path
        Directory containing COAD CSV exports
    generators_as_links : bool, optional
        If True, represent conventional generators as Links connecting fuel buses to electric buses.
    fuel_bus_prefix : str, optional
        Prefix for fuel bus names when generators_as_links=True.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> add_generators_csv(network, csv_dir)
    """
    # Load static generator properties
    generator_df = load_static_properties(csv_dir, "Generator")

    if generator_df.empty:
        logger.warning(f"No generators found in {csv_dir}")
        return

    empty_generators = []
    skipped_generators = []
    generators = list_objects_by_class_csv(generator_df)

    # Load time-varying properties to check for capacity expansions
    try:
        time_varying_props = load_time_varying_properties(csv_dir)
    except Exception:
        time_varying_props = pd.DataFrame()

    # Get model start date from network snapshots (if available)
    model_start_date = (
        network.snapshots.min()
        if hasattr(network, "snapshots") and len(network.snapshots) > 0
        else None
    )

    for gen in generators:
        # Check for time-varying Max Capacity with dates (capacity expansions)
        p_max = None
        has_expansion = False

        if not time_varying_props.empty:
            max_cap_entries = time_varying_props[
                (time_varying_props["class"] == "Generator")
                & (time_varying_props["object"] == gen)
                & (time_varying_props["property"] == "Max Capacity")
            ].copy()

            # Filter entries with dates
            if not max_cap_entries.empty:
                max_cap_entries["date_from"] = pd.to_datetime(
                    max_cap_entries["date_from"], errors="coerce"
                )
                dated_entries = max_cap_entries[max_cap_entries["date_from"].notna()]

                if not dated_entries.empty:
                    has_expansion = (
                        len(dated_entries) > 1
                    )  # Multiple dated entries = expansion
                    dated_entries = dated_entries.sort_values("date_from")

                    # Use capacity at model start date
                    if model_start_date is not None:
                        # Find the entry that applies at model start
                        applicable = dated_entries[
                            dated_entries["date_from"] <= model_start_date
                        ]
                        if not applicable.empty:
                            p_max = float(applicable.iloc[-1]["value"])
                        else:
                            # All dates are in the future - use first entry
                            p_max = float(dated_entries.iloc[0]["value"])
                    else:
                        # No snapshots yet - use first dated entry
                        p_max = float(dated_entries.iloc[0]["value"])

                    if has_expansion:
                        future_caps = dated_entries[
                            dated_entries["date_from"]
                            > (model_start_date or pd.Timestamp.min)
                        ]
                        if not future_caps.empty:
                            print(
                                f"Note: {gen} has capacity expansion schedule "
                                f"({len(dated_entries)} stages, currently using {p_max:.2f} MW). "
                                f"Full expansion modeling not yet implemented."
                            )

        # Fallback to static Max Capacity if no time-varying found
        if p_max is None:
            p_max_raw = get_property_from_static_csv(generator_df, gen, "Max Capacity")

            if p_max_raw is None:
                print(f"Warning: 'Max Capacity' not found for {gen}. No p_nom set.")
                p_max = None
            else:
                # Handle cases where Max Capacity might be a list (e.g., "['55.197', '62.606']")
                # For capacity, we sum multiple values if present (e.g., multiple units)
                p_max = parse_numeric_value(p_max_raw, use_first=False)

        # Extract generator properties
        prop_map = {
            "Min Capacity": "p_nom_min",
            "Start Cost": "start_up_cost",
            "Shutdown Cost": "shut_down_cost",
            "Min Up Time": "min_up_time",
            "Min Down Time": "min_down_time",
            "Max Ramp Up": "ramp_limit_up",
            "Max Ramp Down": "ramp_limit_down",
            "Ramp Up Rate": "ramp_limit_start_up",
            "Ramp Down Rate": "ramp_limit_start_down",
            "Technical Life": "lifetime",
        }

        gen_attrs = {}
        for prop, attr in prop_map.items():
            val = get_property_from_static_csv(generator_df, gen, prop)
            if val is not None:
                parsed_val = parse_numeric_value(val, use_first=True)
                if parsed_val is not None:
                    gen_attrs[attr] = parsed_val

        # Max Ramp Up and Max Ramp Down are in MW/min, so to convert to ramp_limit_up and ramp_limit_down:
        # multiply by 60 and divide by p_nom
        if "ramp_limit_up" in gen_attrs and p_max > 0:
            gen_attrs["ramp_limit_up"] = (
                gen_attrs["ramp_limit_up"] * 60.0 / p_max
            )  # per hour
        if "ramp_limit_down" in gen_attrs and p_max > 0:
            gen_attrs["ramp_limit_down"] = (
                gen_attrs["ramp_limit_down"] * 60.0 / p_max
            )  # per hour

        # Find associated bus/node
        bus = find_bus_for_object_csv(generator_df, gen)
        if bus is None:
            print(f"Warning: No associated bus found for generator {gen}")
            skipped_generators.append(gen)
            continue

        # Find associated fuel/carrier
        carrier = find_fuel_for_generator_csv(generator_df, gen)

        # Determine if this is a conventional generator
        conventional_fuels = [
            "Natural Gas",
            "Natural Gas CCGT",
            "Natural Gas OCGT",
            "Coal",
            "Hard Coal",
            "Lignite",
            "Nuclear",
            "Oil",
            "Biomass",
            "Biomass Waste",
            "Solids Fired",
            "Gas",
        ]
        is_conventional = carrier in conventional_fuels if carrier else False

        # Add generator to network
        if generators_as_links and is_conventional and carrier:
            # Create fuel bus if it doesn't exist
            fuel_carrier = (
                carrier.replace(" CCGT", "").replace(" OCGT", "").replace(" Waste", "")
            )
            fuel_bus_name = (
                f"{fuel_bus_prefix}{fuel_carrier.replace(' ', '_')}"
                if fuel_bus_prefix
                else f"{fuel_carrier.replace(' ', '_')}"
            )

            if fuel_bus_name not in network.buses.index:
                if fuel_carrier not in network.carriers.index:
                    network.add("Carrier", fuel_carrier)
                network.add("Bus", fuel_bus_name, carrier=fuel_carrier)

            # Create generator-link (fuel bus -> electric bus)
            link_name = f"gen_link_{gen}"
            if link_name not in network.links.index:
                # Set efficiency based on technology
                efficiency = 0.40  # Default thermal efficiency
                if "CCGT" in carrier:
                    efficiency = 0.55
                elif "OCGT" in carrier:
                    efficiency = 0.35
                elif "Nuclear" in carrier:
                    efficiency = 0.33
                elif "Coal" in carrier or "Lignite" in carrier:
                    efficiency = 0.40
                elif "Biomass" in carrier:
                    efficiency = 0.30

                # Create link
                if p_max is not None:
                    network.add(
                        "Link",
                        link_name,
                        bus0=fuel_bus_name,
                        bus1=bus,
                        p_nom=p_max,
                        efficiency=efficiency,
                        carrier="conversion",
                    )
                else:
                    network.add(
                        "Link",
                        link_name,
                        bus0=fuel_bus_name,
                        bus1=bus,
                        efficiency=efficiency,
                        carrier="conversion",
                    )

                # Set ramp limits
                for attr, val in gen_attrs.items():
                    if attr in ["ramp_limit_up", "ramp_limit_down"]:
                        network.links.loc[link_name, attr] = val

                # Add infinite fuel supply generator
                fuel_supply_name = f"fuel_supply_{fuel_carrier.replace(' ', '_')}"
                if fuel_supply_name not in network.generators.index:
                    network.add(
                        "Generator",
                        fuel_supply_name,
                        bus=fuel_bus_name,
                        p_nom=99999,
                        carrier=fuel_carrier,
                        marginal_cost=0.1,
                    )
        # Add as standard generator
        elif p_max is not None:
            if carrier is not None:
                network.add("Generator", gen, bus=bus, p_nom=p_max, carrier=carrier)
            else:
                network.add("Generator", gen, bus=bus, p_nom=p_max)

            for attr, val in gen_attrs.items():
                network.generators.loc[gen, attr] = val
        elif carrier is not None:
            network.add("Generator", gen, bus=bus, carrier=carrier)
        else:
            network.add("Generator", gen, bus=bus)

    # Report skipped generators
    if empty_generators:
        print(f"\nSkipped {len(empty_generators)} generators with no properties:")
        for g in empty_generators:
            print(f"  - {g}")

    if skipped_generators:
        print(f"\nSkipped {len(skipped_generators)} generators with no associated bus:")
        for g in skipped_generators:
            print(f"  - {g}")


def set_capacity_ratings_csv(
    network: Network, csv_dir: str | Path, timeslice_csv: str | None = None
):
    """Set the capacity ratings for generators using COAD CSV exports.

    This is the CSV-based version of set_capacity_ratings() from generators.py.

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to the timeslice CSV file

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> set_capacity_ratings_csv(network, csv_dir, "timeslice.csv")
    """
    # Get the generator ratings from CSV (already p_max_pu)
    generator_ratings = parse_generator_ratings_csv(csv_dir, network, timeslice_csv)

    # Only keep generators present in both network and generator_ratings
    valid_gens = [
        gen for gen in network.generators.index if gen in generator_ratings.columns
    ]
    missing_gens = [
        gen for gen in network.generators.index if gen not in generator_ratings.columns
    ]

    # Assign all columns at once
    if valid_gens:
        network.generators_t.p_max_pu.loc[:, valid_gens] = generator_ratings[
            valid_gens
        ].copy()

    # Warn about missing generators
    for gen in missing_gens:
        print(f"Warning: Generator {gen} not found in ratings DataFrame.")


def set_min_stable_levels_csv(
    network: Network, csv_dir: str | Path, timeslice_csv: str | None = None
):
    """Set the minimum stable levels (p_min_pu) for generators using COAD CSV exports.

    This is the CSV-based version of set_min_stable_levels() from generators.py.

    This function retrieves minimum generation constraints from the CSV exports
    (Min Stable Level, Min Stable Factor, Min Pump Load) and sets the `p_min_pu`
    time series for generators.

    This ensures thermal generators with minimum load constraints (e.g., 60% min load)
    have those constraints properly represented in PyPSA, which is critical for:
    - Realistic unit commitment modeling
    - Proper handling of generator outages (outages scale both p_max_pu and p_min_pu)
    - Avoiding infeasibility when generators cannot be fully turned off

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to the timeslice CSV file

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> set_min_stable_levels_csv(network, csv_dir, "timeslice.csv")

    Notes
    -----
    Property precedence (first match wins):
    1. Min Stable Factor (percentage) -> p_min_pu = value / 100
    2. Min Stable Level (MW) -> p_min_pu = value / p_nom
    3. Min Pump Load (MW, for storage) -> p_min_pu = value / p_nom
    4. Fallback: p_min_pu = 0.0 (no minimum constraint)
    """
    # Get the generator minimum stable levels from CSV
    generator_min_levels = parse_generator_min_stable_levels_csv(
        csv_dir, network, timeslice_csv
    )

    if generator_min_levels.empty:
        print(
            "No minimum stable levels found. All generators can operate down to 0 MW."
        )
        return

    # Only keep generators present in both network and generator_min_levels
    valid_gens = [
        gen for gen in network.generators.index if gen in generator_min_levels.columns
    ]
    missing_gens = [
        gen
        for gen in network.generators.index
        if gen not in generator_min_levels.columns
    ]

    # Assign all columns at once to avoid fragmentation
    if valid_gens:
        network.generators_t.p_min_pu.loc[:, valid_gens] = generator_min_levels[
            valid_gens
        ].copy()

    # Report statistics
    nonzero_gens = [gen for gen in valid_gens if (generator_min_levels[gen] > 0).any()]

    print(
        f"Set p_min_pu for {len(valid_gens)} generators ({len(nonzero_gens)} with nonzero minimum)"
    )

    # Warn about missing generators (optional - usually expected for VRE)
    if missing_gens and len(missing_gens) < 10:
        for gen in missing_gens:
            logger.debug(f"Generator {gen} not found in min stable levels DataFrame.")


def validate_and_fix_generator_constraints(
    network: Network, verbose: bool = True
) -> dict:
    """Validate and fix generator constraints where p_min_pu > p_max_pu.

    When available capacity (p_max_pu) is less than minimum stable level (p_min_pu),
    the generator physically cannot operate. This typically happens when:
    - Rating Factor reduces capacity below Min Stable Level
    - Rating only exists in static CSV but not time-varying
    - Max Capacity < Min Stable Level (data quality issue)

    In these cases, we relax p_min_pu to 0, allowing the generator to be turned off
    or operate at reduced capacity. This preserves feasibility while acknowledging
    that the unit cannot meet its normal minimum operating constraints.

    Parameters
    ----------
    network : Network
        PyPSA network with generators
    verbose : bool, default True
        Print summary of fixes

    Returns
    -------
    dict
        Summary with list of fixed generators

    Examples
    --------
    >>> # After setting all generator properties
    >>> validate_and_fix_generator_constraints(network)
    Fixed 7 generators with p_min_pu > p_max_pu constraints
      (Generators cannot meet min stable level at reduced capacity)
    """
    fixed_generators = []

    for gen in network.generators.index:
        if gen not in network.generators_t.p_max_pu.columns:
            continue
        if gen not in network.generators_t.p_min_pu.columns:
            continue

        p_max = network.generators_t.p_max_pu[gen]
        p_min = network.generators_t.p_min_pu[gen]

        # Check for infeasibility
        infeasible_mask = p_min > p_max

        if infeasible_mask.any():
            num_infeasible = infeasible_mask.sum()
            max_violation = (p_min - p_max)[infeasible_mask].max()

            logger.warning(
                f"Generator '{gen}': p_min_pu > p_max_pu for {num_infeasible} timesteps "
                f"(max violation: {max_violation:.4f}). "
                f"Setting p_min_pu = 0 for infeasible periods (unit cannot meet min stable level)."
            )

            # Fix: Set p_min_pu = 0 where infeasible
            network.generators_t.p_min_pu.loc[infeasible_mask, gen] = 0.0
            fixed_generators.append(gen)

    if verbose and fixed_generators:
        print(
            f"Fixed {len(fixed_generators)} generators with p_min_pu > p_max_pu constraints"
        )
        print("  (Generators cannot meet min stable level at reduced capacity)")

    return {"fixed_generators": fixed_generators}


def set_generator_efficiencies_csv(
    network: Network, csv_dir: str | Path, use_incr: bool = True
):
    """Set the efficiency for each generator using COAD CSV exports.

    This is the CSV-based version of set_generator_efficiencies() from generators.py.

    The efficiency is calculated as:
        efficiency = (p_nom / fuel) * 3.6
    where:
        fuel = hr_base + (hr_inc * p_nom)

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    use_incr : bool, optional
        Whether to use Heat Rate Incr values. Default True.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> set_generator_efficiencies_csv(network, csv_dir)
    """
    generator_df = load_static_properties(csv_dir, "Generator")

    efficiencies = []

    for gen in network.generators.index:
        p_nom = (
            network.generators.at[gen, "p_nom"]
            if "p_nom" in network.generators.columns
            else None
        )

        if p_nom is None or np.isnan(p_nom):
            efficiencies.append(np.nan)
            print(f"Warning: 'p_nom' not found for {gen}. No efficiency set.")
            continue

        # Get Heat Rate Base
        hr_base = get_property_from_static_csv(generator_df, gen, "Heat Rate Base")

        if hr_base is None:
            efficiencies.append(1)
            continue

        hr_base = parse_numeric_value(hr_base, use_first=True)
        if hr_base is None:
            efficiencies.append(1)
            continue

        # Get Heat Rate Incr values (can be stored as list-like string in CSV)
        hr_incr_value = get_property_from_static_csv(
            generator_df, gen, "Heat Rate Incr"
        )
        hr_incs = []

        if hr_incr_value is not None:
            # Parse if it's a string representation of a list
            if isinstance(hr_incr_value, str) and "[" in hr_incr_value:
                try:
                    hr_incs = [float(x) for x in ast.literal_eval(hr_incr_value)]
                except Exception:
                    try:
                        hr_incs = [float(hr_incr_value)]
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    hr_incs = [float(hr_incr_value)]
                except (ValueError, TypeError):
                    pass

        # Calculate efficiency
        if not use_incr or not hr_incs:
            if not use_incr and hr_incs:
                print(
                    f"Heat Rate Incr found for {gen}. Only Heat Rate Base will be used."
                )
            fuel = hr_base
        elif len(hr_incs) == 1:
            fuel = hr_base + hr_incs[0] * p_nom
        else:
            n = len(hr_incs)
            p_seg = p_nom / n
            fuel = hr_base + sum(hr * p_seg for hr in hr_incs)

        efficiency = (p_nom / fuel) * 3.6 if fuel else 1

        if efficiency > 1:
            print(
                f"   - Warning: Calculated efficiency > 1 for {gen} (efficiency={efficiency:.3f})"
            )

        efficiencies.append(efficiency)

    network.generators["efficiency"] = efficiencies


def set_vre_profiles_csv(network: Network, csv_dir: str | Path, vre_profiles_path: str):
    """Add time series profiles for solar and wind generators using COAD CSV exports.

    This is the CSV-based version of set_vre_profiles() from generators.py.

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    vre_profiles_path : str
        Path to the folder containing generation profile files

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> set_vre_profiles_csv(network, csv_dir, "models/sem-2024/Traces/")
    """
    generator_df = load_static_properties(csv_dir, "Generator")

    def _raise_unsupported_resolution(filename: str):
        """Raise unsupported resolution error."""
        msg = f"Unsupported resolution in file: {filename}"
        raise ValueError(msg)

    dispatch_dict = {}

    for gen in network.generators.index:
        # Skip Adelaide_Desal_FFP
        if gen == "Adelaide_Desal_FFP":
            print(f"Skipping generator {gen}")
            continue

        # Look for profile filename in generator properties
        # Common columns that might contain file paths
        filename = None
        for col in ["Rating", "Rating.Variable", "Rating.Data File", "Data File"]:
            if col in generator_df.columns:
                val = get_property_from_static_csv(generator_df, gen, col)
                if val and (
                    contains_path_pattern(str(val), "Traces/solar/")
                    or contains_path_pattern(str(val), "Traces/wind/")
                ):
                    filename = str(val)
                    break

        if not filename:
            continue

        profile_type = (
            "solar" if contains_path_pattern(filename, "Traces/solar/") else "wind"
        )

        # Extract just the filename
        clean_filename = extract_filename(filename.strip())
        file_path = safe_join(vre_profiles_path, "Traces", profile_type, clean_filename)

        # Set carrier
        carrier_value = "Solar" if profile_type == "solar" else "Wind"
        network.generators.at[gen, "carrier"] = carrier_value

        try:
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])
            df.columns = pd.Index(
                [
                    str(int(col))
                    if col.strip().isdigit()
                    and col not in {"Year", "Month", "Day", "datetime"}
                    else col
                    for col in df.columns
                ]
            )

            non_date_columns = [
                col
                for col in df.columns
                if col not in {"Year", "Month", "Day", "datetime"}
            ]

            if len(non_date_columns) == 24:
                resolution = 60
            elif len(non_date_columns) == 48:
                resolution = 30
            else:
                _raise_unsupported_resolution(filename)

            df_long = df.melt(
                id_vars=["datetime"],
                value_vars=non_date_columns,
                var_name="time",
                value_name="cf",
            )

            if resolution == 60:
                df_long["time"] = pd.to_timedelta(
                    (df_long["time"].astype(int) - 1) * 60, unit="m"
                )
            elif resolution == 30:
                df_long["time"] = pd.to_timedelta(
                    (df_long["time"].astype(int) - 1) * 30, unit="m"
                )

            df_long["series"] = df_long["datetime"].dt.floor("D") + df_long["time"]
            df_long.set_index("series", inplace=True)
            df_long.drop(columns=["datetime", "time"], inplace=True)

            # Get original p_max_pu for the generator
            p_max_pu = network.generators_t.p_max_pu[gen].copy()

            # Align index
            dispatch = df_long["cf"].reindex(p_max_pu.index).fillna(0) * p_max_pu

            # Collect dispatch series
            dispatch_dict[gen] = dispatch

            print(
                f" - Added {profile_type} profile for generator {gen} from {filename}"
            )

        except Exception as e:
            print(f"Failed to process profile for generator {gen}: {e}")

    # Assign all dispatch columns at once
    if dispatch_dict:
        dispatch_df = pd.DataFrame(
            dispatch_dict, index=network.generators_t.p_max_pu.index
        )
        network.generators_t.p_max_pu.loc[:, dispatch_df.columns] = dispatch_df
        network.generators_t.p_min_pu.loc[:, dispatch_df.columns] = dispatch_df


# =============================================================================
# Generic Profile Loading Functions (Data File.csv Auto-Discovery)
# =============================================================================


def _discover_datafile_mappings(csv_dir: str | Path) -> dict[str, str]:
    """Build mapping from Data File objects to CSV filenames.

    Parses Data File.csv to extract the relationship between data file object names
    and their corresponding CSV file paths.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports (must include Data File.csv)

    Returns
    -------
    dict[str, str]
        Mapping of {datafile_object_name: csv_filename}
        Example: {"StochasticWindNI": "NI Wind_5base years - 2018-2033.csv"}

    Examples
    --------
    >>> mappings = _discover_datafile_mappings("csvs_from_xml/SEM Forecast model")
    >>> mappings["StochasticWindNI"]
    'NI Wind_5base years - 2018-2033.csv'
    """
    data_file_path = Path(csv_dir) / "Data File.csv"

    if not data_file_path.exists():
        logger.warning(f"Data File.csv not found at {data_file_path}")
        return {}

    try:
        data_file_df = pd.read_csv(data_file_path)
    except Exception:
        logger.exception("Failed to read Data File.csv")
        return {}

    profile_mapping = {}

    for _, row in data_file_df.iterrows():
        obj_name = row.get("object")
        filename_text = row.get("Filename(text)")

        if obj_name and filename_text and pd.notna(filename_text):
            # Keep the relative path from Filename(text) column
            # Examples:
            # - "CSV Files\NI Wind_5base years - 2018-2033.csv" (SEM - just filename)
            # - "Traces\solar\Adelaide_Desal_FFP_RefYear4006.csv" (AEMO - preserve subdirectory)
            # Convert backslashes to forward slashes for cross-platform compatibility
            filename = str(filename_text).replace("\\", "/")

            profile_mapping[obj_name] = filename

    logger.info(f"Discovered {len(profile_mapping)} data file mappings")
    return profile_mapping


def _find_generators_with_datafile(
    generator_df: pd.DataFrame, property_name: str
) -> dict[str, str]:
    """Find generators that reference Data Files for a given property.

    Searches Generator.csv for the "{property_name}.Data File" column to identify
    which generators have external data files associated with specific properties.

    Parameters
    ----------
    generator_df : pd.DataFrame
        DataFrame from load_static_properties(csv_dir, "Generator")
    property_name : str
        PLEXOS property name (e.g., "Rating", "Fixed Load", "Max Capacity")

    Returns
    -------
    dict[str, str]
        Mapping of {generator_name: datafile_object_name} for generators with data files
        Example: {"Wind NI -- All": "StochasticWindNI"}

    Examples
    --------
    >>> gen_df = load_static_properties("csvs_from_xml/SEM Forecast model", "Generator")
    >>> refs = _find_generators_with_datafile(gen_df, "Rating")
    >>> refs.get("Wind NI -- All")
    'StochasticWindNI'
    """
    datafile_column = f"{property_name}.Data File"

    if datafile_column not in generator_df.columns:
        logger.warning(
            f"Column '{datafile_column}' not found in Generator.csv. "
            f"No generators have {property_name} data files."
        )
        return {}

    gen_to_datafile = {}

    for gen_name in generator_df.index:
        datafile_ref = generator_df.at[gen_name, datafile_column]

        if pd.notna(datafile_ref) and datafile_ref != "":
            gen_to_datafile[gen_name] = str(datafile_ref)

    logger.info(
        f"Found {len(gen_to_datafile)} generators with '{property_name}.Data File' references"
    )
    return gen_to_datafile


def load_data_file_profiles_csv(
    network: Network,
    csv_dir: str | Path,
    profiles_path: str | Path,
    property_name: str,
    target_property: str,
    target_type: str = "generators_t",
    apply_mode: str = "replace",
    scenario: str | int = "1",
    generator_filter: Callable | None = None,
    carrier_mapping: dict[str, str] | None = None,
    value_scaling: float = 1.0,
    manual_mappings: dict[str, str] | None = None,
) -> dict:
    """Load time series profiles from Data File.csv mappings and apply to network.

    This is a fully generic function that auto-discovers generator->profile linkages
    from PLEXOS Data File.csv metadata and applies loaded data to PyPSA network properties
    with configurable mapping options.

    The function performs these steps:
    1. Parse Data File.csv to map data file objects -> CSV filenames
    2. Parse Generator.csv to find generators with "{property_name}.Data File" references
       (or use manual_mappings as fallback if auto-discovery fails)
    3. Create missing carriers if carrier_mapping is provided
    4. Load and apply profile data to specified PyPSA properties

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports (must contain Data File.csv and Generator.csv)
    profiles_path : str | Path
        Base directory containing profile CSV files referenced in Data File.csv
    property_name : str
        PLEXOS property name to look for (e.g., "Rating", "Fixed Load", "Max Capacity").
        Function searches for "{property_name}.Data File" column in Generator.csv.
    target_property : str
        PyPSA property to set (e.g., "p_max_pu", "p_set", "p_nom")
    target_type : str, default "generators_t"
        Where to apply data:
        - "generators_t": Time-varying properties (network.generators_t[target_property])
        - "generators": Static properties (network.generators[target_property])
    apply_mode : str, default "replace"
        How to apply loaded data:
        - "replace": Overwrite existing values
        - "multiply": Multiply by existing values (for capacity factors on top of ratings)
        - "set_both_min_max": Set both p_max_pu and p_min_pu (for must-run profiles)
        - "add": Add to existing values
        - "mask": Multiply as availability mask (0-1 values)
    scenario : str | int, default "1"
        Which stochastic scenario to use if data has multiple scenarios
    generator_filter : callable, optional
        Function taking generator name and returning True to process, False to skip.
        Example: lambda gen: "Wind" in gen or "Solar" in gen
    carrier_mapping : dict, optional
        Mapping to set generator carriers based on keywords in generator or file names.
        Example: {"Wind": "Wind", "Solar": "Solar"}
        Keys are matched case-insensitively against generator names and data file names.
        If carriers don't exist in network, they will be automatically created.
    value_scaling : float, default 1.0
        Scaling factor for values (e.g., 0.01 to convert percentage to fraction)
    manual_mappings : dict[str, str], optional
        Manual fallback mappings when auto-discovery fails. Maps generator names to
        data file object names. Used when CSV export lacks "{property_name}.Data File" column.
        Example: {"Wind NI -- All": "StochasticWindNI", "Wind ROI": "StochasticWindROI"}

    Returns
    -------
    dict
        Summary: {
            "processed_generators": int,
            "skipped_generators": list[str],
            "failed_generators": list[str],
            "applied_to": str (target_type.target_property),
            "mode": str (apply_mode)
        }

    Examples
    --------
    Load VRE capacity factor profiles (SEM model with manual mappings fallback):
    >>> sem_vre_mappings = {
    ...     "Wind NI -- All": "StochasticWindNI",
    ...     "Wind ROI": "StochasticWindROI",
    ...     "Wind Offshore": "StochasticWindOffshore",
    ...     "Wind Offshore -- Arklow Phase 1": "StochasticWindROI",
    ...     "Solar NI -- All": "StochasticSolarNI",
    ...     "Solar ROI": "StochasticSolarROI",
    ... }
    >>> load_data_file_profiles_csv(
    ...     network=network,
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     profiles_path="CSV Files",
    ...     property_name="Rating",
    ...     target_property="p_max_pu",
    ...     target_type="generators_t",
    ...     apply_mode="multiply",
    ...     scenario="1",
    ...     generator_filter=lambda gen: "Wind" in gen or "Solar" in gen,
    ...     carrier_mapping={"Wind": "Wind", "Solar": "Solar"},
    ...     value_scaling=0.01,  # Convert percentage to fraction
    ...     manual_mappings=sem_vre_mappings,  # Fallback for incomplete CSV export
    ... )

    Load hydro dispatch schedule (CAISO model):
    >>> load_data_file_profiles_csv(
    ...     network=network,
    ...     csv_dir="csvs_from_xml/WECC",
    ...     profiles_path="FixedDispatch",
    ...     property_name="Fixed Load",
    ...     target_property="p_set",
    ...     target_type="generators_t",
    ...     apply_mode="replace",
    ... )

    Load generator outage/availability:
    >>> load_data_file_profiles_csv(
    ...     network=network,
    ...     csv_dir="csvs_from_xml/WECC",
    ...     profiles_path="Units Out",
    ...     property_name="Rating",
    ...     target_property="p_max_pu",
    ...     apply_mode="mask",  # Availability factor 0-1
    ... )
    """
    csv_dir = Path(csv_dir)
    profiles_path = Path(profiles_path)

    # Step 1: Discover data file mappings
    datafile_to_csv = _discover_datafile_mappings(csv_dir)

    if not datafile_to_csv:
        logger.warning("No data file mappings found. Cannot load profiles.")
        return {
            "processed_generators": 0,
            "skipped_generators": [],
            "failed_generators": [],
            "applied_to": f"{target_type}.{target_property}",
            "mode": apply_mode,
        }

    # Step 2: Find generators with data file references
    generator_df = load_static_properties(csv_dir, "Generator")

    if generator_df.empty:
        logger.warning("No Generator.csv found. Cannot load profiles.")
        return {
            "processed_generators": 0,
            "skipped_generators": [],
            "failed_generators": [],
            "applied_to": f"{target_type}.{target_property}",
            "mode": apply_mode,
        }

    gen_to_datafile = _find_generators_with_datafile(generator_df, property_name)

    if not gen_to_datafile:
        # Try manual mappings as fallback
        if manual_mappings:
            logger.info(
                f"Auto-discovery found no generators with '{property_name}.Data File' references. "
                f"Using manual_mappings fallback with {len(manual_mappings)} mappings."
            )
            gen_to_datafile = manual_mappings
        else:
            logger.warning(
                f"No generators found with '{property_name}.Data File' references, "
                f"and no manual_mappings provided."
            )
            return {
                "processed_generators": 0,
                "skipped_generators": [],
                "failed_generators": [],
                "applied_to": f"{target_type}.{target_property}",
                "mode": apply_mode,
            }

    # Step 3: Ensure carriers exist in network (if carrier_mapping provided)
    if carrier_mapping:
        # Collect all carriers that will be used
        carriers_to_create = set(carrier_mapping.values())

        # Create missing carriers
        for carrier in carriers_to_create:
            if carrier not in network.carriers.index:
                network.add("Carrier", carrier)
                logger.info(f"Created carrier: {carrier}")

    # Step 4: Load and apply profiles
    profile_dict = {}
    skipped_generators = []
    failed_generators = []

    for gen_name, datafile_obj in gen_to_datafile.items():
        # Apply filter if provided
        if generator_filter and not generator_filter(gen_name):
            skipped_generators.append(gen_name)
            continue

        # Check if generator exists in network
        if gen_name not in network.generators.index:
            logger.debug(f"Generator {gen_name} not in network, skipping")
            skipped_generators.append(gen_name)
            continue

        # Look up CSV filename
        # Try exact match first, then try without "Data File." prefix if present
        csv_filename = None
        if datafile_obj in datafile_to_csv:
            csv_filename = datafile_to_csv[datafile_obj]
        elif datafile_obj.startswith("Data File."):
            # Strip "Data File." prefix and try again
            stripped_name = datafile_obj.replace("Data File.", "", 1)
            if stripped_name in datafile_to_csv:
                csv_filename = datafile_to_csv[stripped_name]

        if csv_filename is None:
            logger.warning(
                f"No CSV file mapping for data file '{datafile_obj}' (gen: {gen_name})"
            )
            skipped_generators.append(gen_name)
            continue
        csv_path = profiles_path / csv_filename

        if not csv_path.exists():
            logger.warning(
                f"Profile CSV not found: {csv_path} (gen: {gen_name}, datafile: {datafile_obj})"
            )
            failed_generators.append(gen_name)
            continue

        # Set carrier if mapping provided
        if carrier_mapping:
            for keyword, carrier in carrier_mapping.items():
                if (
                    keyword.lower() in gen_name.lower()
                    or keyword.lower() in csv_filename.lower()
                ):
                    network.generators.at[gen_name, "carrier"] = carrier
                    break

        # Load profile using existing reader
        try:
            profile_df = read_plexos_input_csv(
                csv_path,
                scenario=scenario,
                snapshots=network.snapshots,  # Enable tiling for annual profiles
                interpolation_method="linear",  # Linear interpolation for sparse data
            )

            # Extract values (handle both single column and multi-column formats)
            if "value" in profile_df.columns:
                profile_series = profile_df["value"]
            else:
                # Use first column
                profile_series = profile_df.iloc[:, 0]

            # Apply scaling
            if value_scaling != 1.0:
                profile_series = profile_series * value_scaling

            # Store for later application
            profile_dict[gen_name] = profile_series

            logger.info(
                f"Loaded profile for {gen_name} from {csv_filename} (scenario {scenario})"
            )

        except Exception:
            logger.exception(f"Failed to load profile for {gen_name}")
            failed_generators.append(gen_name)
            traceback.print_exc()

    # Step 4: Apply profiles to network based on target_type and apply_mode
    if not profile_dict:
        logger.warning("No profiles were successfully loaded")
        return {
            "processed_generators": 0,
            "skipped_generators": skipped_generators,
            "failed_generators": failed_generators,
            "applied_to": f"{target_type}.{target_property}",
            "mode": apply_mode,
        }

    if target_type == "generators_t":
        # Time-varying property
        snapshots = network.snapshots

        # Initialize DataFrame if needed
        if not hasattr(network.generators_t, target_property):
            network.generators_t[target_property] = pd.DataFrame(
                index=snapshots, columns=network.generators.index, dtype=float
            )

        # Apply profiles based on mode
        for gen_name, profile_series in profile_dict.items():
            # Align to network snapshots
            aligned_profile = profile_series.reindex(snapshots).fillna(0)

            if apply_mode == "replace":
                network.generators_t[target_property][gen_name] = aligned_profile

            elif apply_mode == "multiply":
                # Multiply by existing values
                existing = network.generators_t[target_property][gen_name]
                network.generators_t[target_property][gen_name] = (
                    existing * aligned_profile
                )

            elif apply_mode == "set_both_min_max":
                # Set both p_max_pu and p_min_pu (for must-run/dispatch profiles)
                network.generators_t.p_max_pu[gen_name] = aligned_profile
                network.generators_t.p_min_pu[gen_name] = aligned_profile

            elif apply_mode == "add":
                existing = network.generators_t[target_property][gen_name]
                network.generators_t[target_property][gen_name] = (
                    existing + aligned_profile
                )

            elif apply_mode == "mask":
                # Multiply as availability mask (0-1 values)
                existing = network.generators_t[target_property][gen_name]
                network.generators_t[target_property][gen_name] = (
                    existing * aligned_profile
                )

            else:
                logger.error(f"Unknown apply_mode: {apply_mode}")
                msg = "apply_mode must be one of: replace, multiply, set_both_min_max, add, mask"
                raise ValueError(msg)

    elif target_type == "generators":
        # Static property - use mean or first value
        for gen_name, profile_series in profile_dict.items():
            # For static properties, use the mean of the time series
            static_value = profile_series.mean()

            if apply_mode == "replace":
                network.generators.at[gen_name, target_property] = static_value
            elif apply_mode == "multiply":
                existing = network.generators.at[gen_name, target_property]
                network.generators.at[gen_name, target_property] = (
                    existing * static_value
                )
            elif apply_mode == "add":
                existing = network.generators.at[gen_name, target_property]
                network.generators.at[gen_name, target_property] = (
                    existing + static_value
                )
            else:
                logger.warning(
                    f"apply_mode '{apply_mode}' not fully supported for static properties, using 'replace'"
                )
                network.generators.at[gen_name, target_property] = static_value

    else:
        msg = "target_type must be 'generators_t' or 'generators'"
        raise ValueError(msg)

    processed_count = len(profile_dict)
    logger.info(
        f"Successfully applied {processed_count} profiles to {target_type}.{target_property} (mode: {apply_mode})"
    )

    return {
        "processed_generators": processed_count,
        "skipped_generators": skipped_generators,
        "failed_generators": failed_generators,
        "applied_to": f"{target_type}.{target_property}",
        "mode": apply_mode,
    }


# =============================================================================
# Convenience Wrapper Functions
# =============================================================================


def load_vre_profiles_csv(
    network: Network,
    csv_dir: str | Path,
    profiles_path: str | Path,
    scenario: str | int = "1",
    value_scaling: float = 0.01,
    set_min_pu: bool = True,
) -> dict:
    """Load VRE capacity factor profiles using Data File.csv mappings.

    Convenience wrapper around load_data_file_profiles_csv() for loading VRE
    (wind/solar) capacity factor profiles. Automatically filters to Wind/Solar
    generators and applies profiles as multiplicative capacity factors.

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports (must contain Data File.csv and Generator.csv)
    profiles_path : str | Path
        Base directory containing VRE profile CSV files
    scenario : str | int, default "1"
        Which stochastic scenario to use if data has multiple scenarios
    value_scaling : float, default 0.01
        Scaling factor for profile values (default 0.01 converts percentage to fraction)
    set_min_pu : bool, default True
        Also set p_min_pu to match p_max_pu (makes VRE must-run at capacity factor)

    Returns
    -------
    dict
        Summary with processed/skipped/failed generator counts

    Examples
    --------
    >>> load_vre_profiles_csv(
    ...     network=network,
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     profiles_path="CSV Files",
    ...     scenario="1",
    ... )
    """
    summary = load_data_file_profiles_csv(
        network=network,
        csv_dir=csv_dir,
        profiles_path=profiles_path,
        property_name="Rating",
        target_property="p_max_pu",
        target_type="generators_t",
        apply_mode="multiply",
        scenario=scenario,
        generator_filter=lambda gen: "Wind" in gen or "Solar" in gen,
        carrier_mapping={"Wind": "Wind", "Solar": "Solar"},
        value_scaling=value_scaling,
    )

    # Also set p_min_pu for VRE generators if requested
    if set_min_pu and summary["processed_generators"] > 0:
        for gen in network.generators.index:
            if (
                "Wind" in gen or "Solar" in gen
            ) and gen in network.generators_t.p_max_pu.columns:
                network.generators_t.p_min_pu[gen] = network.generators_t.p_max_pu[gen]

    return summary


def load_hydro_dispatch_csv(
    network: Network,
    csv_dir: str | Path,
    profiles_path: str | Path,
    scenario: str | int = "1",
    generator_filter: Callable | None = None,
) -> dict:
    """Load hydro fixed dispatch schedules using Data File.csv mappings.

    Convenience wrapper around load_data_file_profiles_csv() for loading hydro
    generator dispatch schedules (Fixed Load property). Sets p_set time series
    for hydro generators with fixed dispatch profiles.

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports (must contain Data File.csv and Generator.csv)
    profiles_path : str | Path
        Base directory containing fixed dispatch CSV files
    scenario : str | int, default "1"
        Which scenario to use if data has multiple scenarios
    generator_filter : callable, optional
        Function taking generator name and returning True to process, False to skip.
        Default None processes all generators with "Fixed Load.Data File" references.
        Example: lambda gen: "Hydro" in gen

    Returns
    -------
    dict
        Summary with processed/skipped/failed generator counts

    Examples
    --------
    >>> load_hydro_dispatch_csv(
    ...     network=network,
    ...     csv_dir="csvs_from_xml/WECC",
    ...     profiles_path="FixedDispatch",
    ...     generator_filter=lambda gen: "Hydro" in gen,
    ... )
    """
    return load_data_file_profiles_csv(
        network=network,
        csv_dir=csv_dir,
        profiles_path=profiles_path,
        property_name="Fixed Load",
        target_property="p_set",
        target_type="generators_t",
        apply_mode="replace",
        scenario=scenario,
        generator_filter=generator_filter,
        carrier_mapping={"Hydro": "Hydro", "Water": "Hydro"},
        value_scaling=1.0,
    )


def build_units_timeseries(
    generator_name: str,
    static_units_value: str | float | None,
    time_varying_df: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
) -> pd.Series:
    """Build Units time series for a single generator.

    Combines static Units value with time-varying Units data to create a complete
    time series showing when generator is built, operating, and retired.

    Parameters
    ----------
    generator_name : str
        Name of the generator
    static_units_value : str | float | None
        Static Units value from Generator.csv (may be array like "['0', '101.4']")
    time_varying_df : pd.DataFrame
        Time-varying properties filtered for this generator and Units property
    snapshots : pd.DatetimeIndex
        Network snapshots to align time series to

    Returns
    -------
    pd.Series
        Units time series indexed by snapshots, showing Units value at each time

    Examples
    --------
    >>> # Generator starts at 0, comes online at 101.4 MW in 2026, retires in 2048
    >>> units_ts = build_units_timeseries(
    ...     "Aramara Solar Farm",
    ...     "['0', '101.4', '0']",
    ...     time_varying_units_df,
    ...     snapshots
    ... )
    """
    # Parse static Units value to get maximum/reference value
    static_units = None
    if static_units_value is not None:
        static_units = parse_numeric_value(static_units_value, strategy="max")

    # Get time-varying entries for this generator
    gen_units_tv = time_varying_df[time_varying_df["object"] == generator_name].copy()

    if gen_units_tv.empty and static_units is None:
        # No Units data at all, assume always 1 unit
        logger.debug(f"No Units data for {generator_name}, assuming Units=1")
        return pd.Series(1.0, index=snapshots)

    # Separate entries with and without dates
    entries_with_dates = gen_units_tv[
        gen_units_tv["date_from"].notna() | gen_units_tv["date_to"].notna()
    ].copy()
    entries_without_dates = gen_units_tv[
        gen_units_tv["date_from"].isna() & gen_units_tv["date_to"].isna()
    ].copy()

    # Determine default/initial Units value
    # Logic distinguishes retirements (static_units > 0) from new builds (static_units = 0/None)
    # - Retirements: Use static_units as default, future entry changes to 0
    # - New builds: Default to 0, future entry brings capacity online
    if not entries_with_dates.empty:
        # Sort to find earliest entry
        entries_with_dates_sorted = entries_with_dates.sort_values("date_from")
        first_entry_date = entries_with_dates_sorted.iloc[0]["date_from"]

        if pd.notna(first_entry_date) and pd.Timestamp(first_entry_date) > snapshots[0]:
            # First entry is in the future
            if static_units is not None and static_units > 0:
                # Facility exists (static_units > 0), use it as default
                # Future entry represents a change/retirement
                default_units = static_units
            else:
                # No existing capacity, future entry is a new build
                default_units = 0.0
        else:
            # First entry is at/before simulation start - use static or 1.0
            default_units = static_units if static_units is not None else 1.0
    else:
        # No dated entries - use static Units or 1.0
        default_units = static_units if static_units is not None else 1.0

    # If there are entries without dates (CAISO pattern), use those as defaults
    if not entries_without_dates.empty:
        # Take the first entry without dates as the default
        default_value = entries_without_dates.iloc[0]["value"]
        try:
            default_units = float(default_value)
        except (ValueError, TypeError):
            logger.warning(
                f"Could not parse default Units value '{default_value}' for {generator_name}, using {default_units}"
            )

    # Initialize Units time series with default value
    units_ts = pd.Series(default_units, index=snapshots)

    # Apply chronological changes from dated entries
    if not entries_with_dates.empty:
        # Sort by date_from
        entries_with_dates = entries_with_dates.sort_values("date_from")

        for _, row in entries_with_dates.iterrows():
            units_value = row["value"]
            date_from = row["date_from"]
            date_to = row["date_to"]

            try:
                units_float = float(units_value)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not parse Units value '{units_value}' for {generator_name}, skipping"
                )
                continue

            # Determine time range for this Units value
            if pd.notna(date_from):
                start_date = pd.Timestamp(date_from)
            else:
                # No start date, apply from beginning
                start_date = snapshots[0]

            if pd.notna(date_to):
                end_date = pd.Timestamp(date_to)
            else:
                # No end date, apply until end of simulation
                end_date = snapshots[-1]

            # Apply Units value to snapshots in this range
            mask = (snapshots >= start_date) & (snapshots <= end_date)
            units_ts.loc[mask] = units_float

    return units_ts


def apply_generator_units_timeseries_csv(
    network: Network,
    csv_dir: str | Path,
    generator_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Apply time-varying Units to scale generator capacity and handle retirements.

    WARNING:  **CRITICAL EXECUTION ORDER**:
    This function MUST be called AFTER VRE profiles are loaded via
    load_data_file_profiles_csv(). If called before VRE loading, Units
    adjustments will be overwritten.

    **Correct order:**
    1. create_model() - creates network
    2. load_data_file_profiles_csv() - loads VRE profiles
    3. apply_generator_units_timeseries_csv() - applies Units (THIS FUNCTION)

    **Dispatch mode support:**
    For dispatch mode (p_min_pu = p_max_pu), this function multiplies BOTH
    p_min_pu and p_max_pu to preserve the dispatch constraint while applying
    retirement/build schedules and capacity scaling.

    This function handles:
    - Generator retirements (Units -> 0)
    - New builds coming online (Units 0 -> N)
    - Capacity scaling for multi-unit facilities (Units > 1)
    - Partial retirements (Units N -> M where M < N)

    The implementation:
    1. Parses static Units from Generator.csv
    2. Loads time-varying Units from Time varying properties.csv
    3. Sets p_nom to maximum capacity (p_nom * max_units)
    4. Applies time-varying capacity via p_max_pu and p_min_pu multipliers

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added and VRE profiles loaded
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added and VRE profiles loaded
    csv_dir : str | Path
        Directory containing COAD CSV exports
    generator_filter : callable, optional
        Function taking generator name and returning True to process. If None, all generators are processed.

    Returns
    -------
    dict
        Summary with statistics:
        - total_generators: Number of generators processed
        - generators_with_units_data: Generators with Units time series
        - generators_with_retirements: Generators that retire
        - generators_with_new_builds: Generators that come online
        - generators_with_scaling: Generators with Units > 1
        - p_nom_adjusted: Generators with p_nom scaled
        - p_max_pu_adjusted: Generators with p_max_pu multiplied
        - p_min_pu_adjusted: Generators with p_min_pu multiplied

    Examples
    --------
    >>> # Load VRE profiles first
    >>> vre_summary = load_data_file_profiles_csv(
    ...     network, csv_dir, apply_mode="set_both_min_max", ...
    ... )
    >>> # Then apply Units
    >>> summary = apply_generator_units_timeseries_csv(network, csv_dir, generator_filter=lambda gen: gen.startswith("Wind"))
    >>> print(f"Processed {summary['generators_with_units_data']} generators with Units data")
    >>> print(f"  - {summary['generators_with_retirements']} with retirements")
    >>> print(f"  - {summary['generators_with_new_builds']} with new builds")
    """
    csv_dir = Path(csv_dir)
    snapshots = network.snapshots

    logger.info(
        "Applying generator Units time series for capacity scaling and retirements..."
    )

    # Load static generator properties
    generator_df = load_static_properties(csv_dir, "Generator")
    if generator_df.empty:
        logger.warning(f"No generators found in {csv_dir}")
        return {
            "total_generators": 0,
            "generators_with_units_data": 0,
            "generators_with_retirements": 0,
            "generators_with_new_builds": 0,
            "generators_with_scaling": 0,
        }

    # Load time-varying Units data
    time_varying = load_time_varying_properties(
        csv_dir, class_name="Generator", property_name="Units"
    )

    if time_varying.empty:
        logger.info("No time-varying Units data found. Using static Units only.")

    # Initialize statistics
    stats = {
        "total_generators": len(network.generators),
        "generators_with_units_data": 0,
        "generators_with_retirements": 0,
        "generators_with_new_builds": 0,
        "generators_with_scaling": 0,
        "p_nom_adjusted": 0,
        "p_max_pu_adjusted": 0,
        "p_min_pu_adjusted": 0,
    }

    # Process each generator in network
    for gen in network.generators.index:
        if generator_filter is not None and not generator_filter(gen):
            continue
        # Get static Units value
        static_units_str = get_property_from_static_csv(generator_df, gen, "Units")

        # Check if generator has any Units data
        has_time_varying = (
            gen in time_varying["object"].values if not time_varying.empty else False
        )
        has_static = static_units_str is not None and static_units_str != ""

        if not has_time_varying and not has_static:
            # No Units data, skip
            continue

        # Build Units time series
        units_ts = build_units_timeseries(
            gen, static_units_str, time_varying, snapshots
        )

        # Get max Units value (determines p_nom)
        max_units = units_ts.max()
        min_units = units_ts.min()

        if max_units == 0:
            # Generator never operates, set to 0 capacity
            logger.debug(f"{gen}: Units always 0, setting p_nom=0")
            network.generators.loc[gen, "p_nom"] = 0.0
            stats["p_nom_adjusted"] += 1
            continue

        # Scale p_nom by max Units
        original_p_nom = network.generators.loc[gen, "p_nom"]
        new_p_nom = original_p_nom * max_units
        network.generators.loc[gen, "p_nom"] = new_p_nom

        # Calculate units multiplier (relative to max)
        units_multiplier = units_ts / max_units

        # Apply to p_max_pu
        if gen in network.generators_t.p_max_pu.columns:
            existing_p_max_pu = network.generators_t.p_max_pu[gen]

            # Check if existing p_max_pu is meaningful (non-zero VRE profile)
            # If it's all zeros or all ones (default), replace instead of multiply
            if existing_p_max_pu.max() == 0 or (
                existing_p_max_pu.min() == 1.0 and existing_p_max_pu.max() == 1.0
            ):
                # Replace with units_multiplier
                if gen == "BARCALDN":
                    logger.info(
                        f"DEBUG {gen}: REPLACE path - p_max_pu is default (all 0 or all 1)"
                    )
                network.generators_t.p_max_pu[gen] = units_multiplier
            else:
                # Multiply existing p_max_pu (preserves VRE profiles)
                if gen == "BARCALDN":
                    logger.info(
                        f"DEBUG {gen}: MULTIPLY path - p_max_pu has meaningful values"
                    )
                result = existing_p_max_pu * units_multiplier
                network.generators_t.p_max_pu[gen] = result

            stats["p_max_pu_adjusted"] += 1

        else:
            # Create new p_max_pu time series
            network.generators_t.p_max_pu[gen] = units_multiplier
            stats["p_max_pu_adjusted"] += 1

        # Apply to p_min_pu (critical for dispatch mode: p_min_pu = p_max_pu)
        # Must multiply p_min_pu to preserve dispatch constraint after Units applied
        if gen in network.generators_t.p_min_pu.columns:
            existing_p_min_pu = network.generators_t.p_min_pu[gen]

            # Check if p_min_pu has meaningful values (dispatch mode)
            if existing_p_min_pu.max() > 0:
                if existing_p_min_pu.max() == 0 or (
                    existing_p_min_pu.min() == 1.0 and existing_p_min_pu.max() == 1.0
                ):
                    # Replace with units_multiplier
                    network.generators_t.p_min_pu[gen] = units_multiplier
                else:
                    # Multiply existing p_min_pu (preserves dispatch profiles)
                    network.generators_t.p_min_pu[gen] = (
                        existing_p_min_pu * units_multiplier
                    )

                stats["p_min_pu_adjusted"] += 1

        # Update statistics
        stats["generators_with_units_data"] += 1

        if max_units > 1:
            stats["generators_with_scaling"] += 1

        if min_units == 0 and max_units > 0:
            # Generator comes online or retires
            if units_ts.iloc[0] == 0 and units_ts.iloc[-1] > 0:
                stats["generators_with_new_builds"] += 1
            elif units_ts.iloc[0] > 0 and units_ts.iloc[-1] == 0:
                stats["generators_with_retirements"] += 1
            elif (units_ts == 0).any() and (units_ts > 0).any():
                # Has both periods of operation and non-operation
                stats["generators_with_new_builds"] += 1
                stats["generators_with_retirements"] += 1

        stats["p_nom_adjusted"] += 1

        # Log notable cases
        if max_units > 1:
            logger.debug(
                f"{gen}: Scaled capacity {original_p_nom:.1f} -> {new_p_nom:.1f} MW (Units: {max_units})"
            )
        if min_units == 0 and max_units > 0:
            logger.debug(
                f"{gen}: Time-varying operation (Units: {min_units} -> {max_units})"
            )

    # Log summary
    logger.info("Generator Units processing complete:")
    logger.info(f"  Total generators: {stats['total_generators']}")
    logger.info(f"  Generators with Units data: {stats['generators_with_units_data']}")
    logger.info(
        f"    - With capacity scaling (Units > 1): {stats['generators_with_scaling']}"
    )
    logger.info(f"    - With new builds: {stats['generators_with_new_builds']}")
    logger.info(f"    - With retirements: {stats['generators_with_retirements']}")
    logger.info(f"  p_nom adjusted: {stats['p_nom_adjusted']} generators")
    logger.info(
        f"  p_max_pu time series adjusted: {stats['p_max_pu_adjusted']} generators"
    )
    logger.info(
        f"  p_min_pu time series adjusted: {stats['p_min_pu_adjusted']} generators"
    )

    return stats


def set_capital_costs_csv(network: Network, csv_dir: str | Path):
    """Set the capital_cost for each generator using COAD CSV exports.

    This is the CSV-based version of set_capital_costs() from generators.py.

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    """
    set_capital_costs_generic_csv(network, csv_dir, "Generator")


def set_marginal_costs_csv(
    network: Network, csv_dir: str | Path, timeslice_csv: str | None = None
):
    """Set the marginal costs for generators using COAD CSV exports.

    This is the CSV-based version of set_marginal_costs() from generators.py.

    The marginal cost is calculated as:
    marginal_cost = (fuel_price * heat_rate_inc) + vo_m_charge

    Parameters
    ----------
    network : Network
        The PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to the timeslice CSV file

    Examples
    --------
    >>> network = pypsa.Network()
    >>> # ... add generators ...
    >>> set_marginal_costs_csv(network, csv_dir, "timeslice.csv")
    """
    # Get fuel prices for all carriers
    fuel_prices = parse_fuel_prices_csv(csv_dir, network, timeslice_csv)

    if fuel_prices.empty:
        logger.warning(
            "No fuel prices found. Cannot set marginal costs for thermal generators."
        )

    generator_df = load_static_properties(csv_dir, "Generator")
    snapshots = network.snapshots
    marginal_costs_dict = {}
    skipped_generators = []

    for gen in network.generators.index:
        # Get VO&M Charge first (applies to all generator types)
        vo_m_charge_value = get_property_from_static_csv(
            generator_df, gen, "VO&M Charge"
        )

        if vo_m_charge_value is None:
            vo_m_charge = 0.0
        else:
            vo_m_charge = parse_numeric_value(vo_m_charge_value, use_first=True)
            if vo_m_charge is None:
                vo_m_charge = 0.0

        # Find the generator's carrier/fuel
        carrier = None
        if "carrier" in network.generators.columns and pd.notna(
            network.generators.loc[gen, "carrier"]
        ):
            carrier = network.generators.loc[gen, "carrier"]
        else:
            carrier = find_fuel_for_generator_csv(generator_df, gen)

        # Check if this is a fuel-burning generator
        has_fuel = (
            carrier is not None
            and not fuel_prices.empty
            and carrier in fuel_prices.columns
        )

        if not has_fuel:
            # Non-fuel generators (hydro, wind, solar, storage, etc.)
            # Marginal cost is just VO&M charge (typically 0)
            marginal_cost_ts = pd.Series(vo_m_charge, index=snapshots)
            marginal_costs_dict[gen] = marginal_cost_ts
            logger.debug(f"Generator {gen}: no fuel, marginal_cost={vo_m_charge}")
            continue

        # Fuel-burning generator - need heat rate and fuel price
        fuel_price_ts = fuel_prices[carrier]

        if fuel_price_ts.isna().all():
            # Fuel exists but no prices - use VO&M only
            marginal_cost_ts = pd.Series(vo_m_charge, index=snapshots)
            marginal_costs_dict[gen] = marginal_cost_ts
            logger.debug(
                f"Generator {gen}: fuel {carrier} has no prices, using VO&M only"
            )
            continue

        # Get Heat Rate Inc values
        hr_incr_value = get_property_from_static_csv(
            generator_df, gen, "Heat Rate Incr"
        )
        heat_rate_inc_values = []

        if hr_incr_value is not None:
            # Parse if it's a string representation of a list
            if isinstance(hr_incr_value, str) and "[" in hr_incr_value:
                try:
                    heat_rate_inc_values = [
                        float(x) for x in ast.literal_eval(hr_incr_value)
                    ]
                except Exception:
                    try:
                        heat_rate_inc_values = [float(hr_incr_value)]
                    except (ValueError, TypeError):
                        pass
            else:
                try:
                    heat_rate_inc_values = [float(hr_incr_value)]
                except (ValueError, TypeError):
                    pass

        # Calculate average heat rate inc
        if heat_rate_inc_values:
            heat_rate_inc = sum(heat_rate_inc_values) / len(heat_rate_inc_values)
        else:
            # Has fuel but no heat rate - unusual, but default to VO&M only
            logger.warning(
                f"Generator {gen} has fuel {carrier} but no Heat Rate Inc, using VO&M only"
            )
            marginal_cost_ts = pd.Series(vo_m_charge, index=snapshots)
            marginal_costs_dict[gen] = marginal_cost_ts
            continue

        # Calculate marginal cost time series: (fuel_price * heat_rate) + VO&M
        marginal_cost_ts = (fuel_price_ts * heat_rate_inc) + vo_m_charge
        marginal_costs_dict[gen] = marginal_cost_ts

        logger.debug(
            f"Generator {gen}: heat_rate_inc={heat_rate_inc}, vo_m_charge={vo_m_charge}, carrier={carrier}"
        )

    # Create DataFrame from marginal costs dictionary
    if marginal_costs_dict:
        marginal_costs_df = pd.DataFrame(marginal_costs_dict, index=snapshots)

        # Only keep generators present in both network and marginal_costs_df
        valid_gens = [
            gen for gen in network.generators.index if gen in marginal_costs_df.columns
        ]

        # Assign time series to network
        if not hasattr(network.generators_t, "marginal_cost"):
            network.generators_t["marginal_cost"] = pd.DataFrame(
                index=snapshots, columns=network.generators.index, dtype=float
            )

        network.generators_t.marginal_cost.loc[:, valid_gens] = marginal_costs_df[
            valid_gens
        ].copy()

        # Report success
        successful_gens = len(valid_gens)
        print(f"Successfully set marginal costs for {successful_gens} generators")
    else:
        print("No generators had complete data for marginal cost calculation")

    # Report skipped generators
    if skipped_generators:
        print(
            f"Skipped adding marginal costs for {len(skipped_generators)} generators due to missing cost properties:"
        )
        for gen in skipped_generators:
            print(f"  - {gen}")


def reassign_generators_to_node(network: Network, target_node: str):
    """Reassign all generators to a specific node.

    This function doesn't use database, so it's the same as generators.py version.
    Included here for completeness of the CSV-based API.

    Parameters
    ----------
    network : Network
        The PyPSA network containing generators.
    target_node : str
        Name of the node to assign all generators to.

    Returns
    -------
    dict
        Summary information about the reassignment.
    """
    if target_node not in network.buses.index:
        msg = f"Target node '{target_node}' not found in network buses"
        raise ValueError(msg)

    original_assignments = network.generators["bus"].copy()
    unique_original_buses = original_assignments.unique()

    # Reassign all generators to the target node
    network.generators["bus"] = target_node

    reassigned_count = len(network.generators)
    print(f"Reassigned {reassigned_count} generators to node '{target_node}'")
    print(
        f"  - Originally spread across {len(unique_original_buses)} buses: {list(unique_original_buses)[:5]}{'...' if len(unique_original_buses) > 5 else ''}"
    )

    return {
        "reassigned_count": reassigned_count,
        "target_node": target_node,
        "original_buses": list(unique_original_buses),
        "original_assignments": original_assignments,
    }


def port_generators_csv(
    network: Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
    vre_profiles_path: str | None = None,
    target_node: str | None = None,
    generators_as_links: bool = False,
    fuel_bus_prefix: str = "fuel_",
):
    """Comprehensive function to add generators and set all their properties using COAD CSV exports.

    This is the CSV-based version of port_generators() from generators.py.

    This function combines all generator-related operations:
    - Adds generators from CSV
    - Sets capacity ratings (p_max_pu)
    - Sets generator efficiencies
    - Sets capital costs
    - Sets marginal costs (time-dependent)
    - Sets VRE profiles for solar and wind generators
    - Optionally converts conventional generators to fuel->electric links
    - Optionally reassigns all generators to a specific node

    Parameters
    ----------
    network : Network
        The PyPSA network to which generators will be added.
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    vre_profiles_path : str, optional
        Path to the folder containing VRE generation profile files.
    target_node : str, optional
        If specified, all generators will be reassigned to this node after setup.
    generators_as_links : bool, optional
        If True, represent conventional generators as Links. Default False.
    fuel_bus_prefix : str, optional
        Prefix for fuel bus names when generators_as_links=True. Default "fuel_".

    Returns
    -------
    dict or None
        If target_node is specified, returns summary information about reassignment.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> port_generators_csv(network, csv_dir,
    ...                     timeslice_csv="timeslice.csv",
    ...                     vre_profiles_path="models/sem-2024/")
    """
    print("Starting generator porting process (CSV-based)...")

    # Step 1: Add generators
    if generators_as_links:
        print("1. Adding generators (with generators-as-links conversion)...")
    else:
        print("1. Adding generators...")
    add_generators_csv(
        network,
        csv_dir,
        generators_as_links=generators_as_links,
        fuel_bus_prefix=fuel_bus_prefix,
    )

    # Step 2: Set capacity ratings (p_max_pu)
    print("2. Setting capacity ratings...")
    set_capacity_ratings_csv(network, csv_dir, timeslice_csv=timeslice_csv)

    # Step 2b: Set minimum stable levels (p_min_pu)
    print("2b. Setting minimum stable levels (p_min_pu)...")
    set_min_stable_levels_csv(network, csv_dir, timeslice_csv=timeslice_csv)

    # Step 2c: Validate and fix generator constraints
    print("2c. Validating generator constraints...")
    validate_and_fix_generator_constraints(network, verbose=True)

    # Step 3: Set generator efficiencies
    print("3. Setting generator efficiencies...")
    set_generator_efficiencies_csv(network, csv_dir, use_incr=True)

    # Step 4: Set capital costs
    print("4. Setting capital costs...")
    set_capital_costs_csv(network, csv_dir)

    # Step 5: Set marginal costs (time-dependent)
    print("5. Setting marginal costs...")
    set_marginal_costs_csv(network, csv_dir, timeslice_csv=timeslice_csv)

    # Step 6: Set VRE profiles (if path provided)
    if vre_profiles_path:
        print("6. Setting VRE profiles...")
        set_vre_profiles_csv(network, csv_dir, vre_profiles_path)
    else:
        print("6. Skipping VRE profiles (no path provided)")

    # Step 7: Reassign generators to target node if specified
    if target_node:
        print(f"7. Reassigning generators to node '{target_node}'...")
        return reassign_generators_to_node(network, target_node)
    else:
        print("7. Skipping generator reassignment (no target node specified)")

    print(f"Generator porting complete! Added {len(network.generators)} generators.")
