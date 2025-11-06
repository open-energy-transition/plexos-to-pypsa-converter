"""Helper functions for generator Units time series implementation.

This temporary file will be integrated into generators_csv.py.
"""

import logging
from pathlib import Path

import pandas as pd
from pypsa import Network

from plexos_to_pypsa_converter.db.csv_readers import (
    get_property_from_static_csv,
    load_static_properties,
    load_time_varying_properties,
    parse_numeric_value,
)

logger = logging.getLogger(__name__)


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
) -> dict:
    """Apply time-varying Units to scale generator capacity and handle retirements.

    This function handles:
    - Generator retirements (Units -> 0)
    - New builds coming online (Units 0 -> N)
    - Capacity scaling for multi-unit facilities (Units > 1)
    - Partial retirements (Units N -> M where M < N)

    The implementation:
    1. Parses static Units from Generator.csv
    2. Loads time-varying Units from Time varying properties.csv
    3. Sets p_nom to maximum capacity (p_nom * max_units)
    4. Applies time-varying capacity via p_max_pu multiplier
    5. Applies same multiplier to p_min_pu (if exists) for dispatch mode consistency

    **Dispatch Mode Handling:**
    For thermal generators with minimum stable levels (p_min_pu time series),
    the units multiplier is applied to BOTH p_min_pu and p_max_pu to maintain
    consistent dispatch ranges during retirements and commissioning.

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Returns
    -------
    dict
        Summary with statistics:
        - total_generators: Number of generators processed
        - generators_with_units_data: Generators with Units time series
        - generators_with_retirements: Generators that retire
        - generators_with_new_builds: Generators that come online
        - generators_with_scaling: Generators with Units > 1
        - p_min_pu_adjusted: Generators with p_min_pu scaled (dispatch mode)

    Examples
    --------
    >>> summary = apply_generator_units_timeseries_csv(network, "csvs_from_xml/NEM")
    >>> print(f"Processed {summary['generators_with_units_data']} generators with Units data")
    >>> print(f"  - {summary['generators_with_retirements']} with retirements")
    >>> print(f"  - {summary['generators_with_new_builds']} with new builds")
    >>> print(f"  - {summary['p_min_pu_adjusted']} with dispatch mode bounds adjusted")
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
    }

    # Process each generator in network
    for gen in network.generators.index:
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

        # Apply to p_max_pu (VRE profiles, rating factors, retirements)
        if gen in network.generators_t.p_max_pu.columns:
            # Multiply existing p_max_pu (preserves VRE profiles)
            network.generators_t.p_max_pu[gen] *= units_multiplier
            stats["p_max_pu_adjusted"] += 1
        else:
            # Create new p_max_pu time series
            network.generators_t.p_max_pu[gen] = units_multiplier
            stats["p_max_pu_adjusted"] += 1

        # Apply to p_min_pu (dispatch mode, min stable levels)
        # This ensures thermal generators with minimum load constraints
        # have their bounds scaled consistently during retirements/new builds
        if gen in network.generators_t.p_min_pu.columns:
            network.generators_t.p_min_pu[gen] *= units_multiplier
            if "p_min_pu_adjusted" not in stats:
                stats["p_min_pu_adjusted"] = 0
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
    if stats.get("p_min_pu_adjusted", 0) > 0:
        logger.info(
            f"  p_min_pu time series adjusted: {stats['p_min_pu_adjusted']} generators (dispatch mode)"
        )

    return stats
