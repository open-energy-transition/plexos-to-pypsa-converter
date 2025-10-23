"""Generator outage modeling for PLEXOS to PyPSA conversion.

This module provides functionality to model generator outages in a way that
approximates PLEXOS behavior while remaining compatible with PyPSA.

PLEXOS uses three types of outages:
1. Forced Outages: Random outages simulated via Monte Carlo
2. Maintenance Outages: Scheduled via PASA optimization
3. Explicit Outages: User-specified dates via "Units Out" property

This module provides:
- Loading pre-computed outage schedules (e.g., CAISO UnitsOut data)
- Parsing explicit outages from PLEXOS properties
- Simplified Monte Carlo and maintenance scheduling
- Expected value fallback (deterministic derating)
- Generalized application of outage schedules to PyPSA networks

All outages are represented as time series modifications to p_max_pu and p_min_pu.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pypsa import Network

from db.csv_readers import (
    ensure_datetime,
    get_property_from_static_csv,
    load_static_properties,
    load_time_varying_properties,
    parse_numeric_value,
    read_plexos_input_csv,
)

logger = logging.getLogger(__name__)


def _raise_invalid_apply_mode(mode: str) -> None:
    """Raise ValueError for invalid apply_mode."""
    msg = f"apply_mode must be 'multiply' or 'replace', got '{mode}'"
    raise ValueError(msg)


def _calculate_capacity_factor(
    p_nom: float | None,
    outage_rating: float | None,
    outage_factor: float | None = None,
) -> float:
    """Calculate capacity factor during outage.

    PLEXOS Priority:
    1. Outage Factor (if provided) - overrides Outage Rating
    2. Outage Rating (absolute MW unavailable)
    3. Default: Full outage (CF=0.0)

    Parameters
    ----------
    p_nom : float | None
        Nominal generator capacity (MW) from network.generators
    outage_rating : float | None
        Capacity unavailable during outage (MW)
    outage_factor : float | None
        Proportion of capacity available during outage (0-1)
        In PLEXOS: Outage Factor = % of capacity that remains AVAILABLE
        Example: 0.9 (90%) means 90% available, 10% unavailable
        Note: Outage Factor overrides Outage Rating if both defined

    Returns
    -------
    float
        Capacity factor during outage (0.0 = full outage, 1.0 = fully available)

    Examples
    --------
    >>> # Full outage with Outage Factor
    >>> _calculate_capacity_factor(100, None, 0.0)
    0.0

    >>> # Partial outage: 33.7% available during outage
    >>> _calculate_capacity_factor(100, None, 0.337)
    0.337

    >>> # Outage Rating: 10 MW unavailable from 100 MW
    >>> _calculate_capacity_factor(100, 10, None)
    0.9

    >>> # Outage Factor takes precedence over Outage Rating
    >>> _calculate_capacity_factor(100, 50, 0.8)
    0.8
    """
    # Priority 1: Outage Factor (if provided)
    # In PLEXOS: Outage Factor is the proportion AVAILABLE during outage
    if outage_factor is not None:
        # Clamp to valid range [0, 1]
        return max(0.0, min(1.0, outage_factor))

    # Priority 2: Outage Rating (absolute MW unavailable)
    if outage_rating is not None and p_nom is not None and p_nom > 0:
        # Calculate available capacity
        # Outage Rating = capacity unavailable
        # So available = (p_nom - outage_rating) / p_nom
        return max(0.0, (p_nom - outage_rating) / p_nom)

    # Default: Full outage
    return 0.0


@dataclass
class OutageEvent:
    """Represents a single generator outage event.

    Attributes
    ----------
    generator : str
        Name of the generator experiencing the outage
    start : pd.Timestamp
        Start time of the outage (inclusive)
    end : pd.Timestamp
        End time of the outage (exclusive)
    outage_type : str
        Type of outage: "forced", "maintenance", or "explicit"
    capacity_factor : float
        Capacity factor during outage (0.0 = full outage, 0.5 = 50% derated)
    scenario : str | int | None
        Scenario identifier for stochastic outages (optional)
    """

    generator: str
    start: pd.Timestamp
    end: pd.Timestamp
    outage_type: str
    capacity_factor: float = 0.0
    scenario: str | int | None = None

    def __post_init__(self) -> None:
        """Validate outage event parameters."""
        if self.capacity_factor < 0.0 or self.capacity_factor > 1.0:
            msg = f"capacity_factor must be in [0, 1], got {self.capacity_factor}"
            raise ValueError(msg)

        if self.outage_type not in ["forced", "maintenance", "explicit"]:
            msg = f"outage_type must be 'forced', 'maintenance', or 'explicit', got {self.outage_type}"
            raise ValueError(msg)

        if self.end <= self.start:
            msg = f"end time ({self.end}) must be after start time ({self.start})"
            raise ValueError(msg)


def build_outage_schedule(
    outage_events: list[OutageEvent],
    snapshots: pd.DatetimeIndex,
    generators: list[str] | None = None,
    default_capacity_factor: float = 1.0,
) -> pd.DataFrame:
    """Build time series schedule from outage events.

    Converts a list of OutageEvent objects into a PyPSA-compatible p_max_pu
    time series DataFrame. Multiple outage events for the same generator and
    time period are combined using minimum capacity factor (worst case).

    Parameters
    ----------
    outage_events : list[OutageEvent]
        List of outage events to apply
    snapshots : pd.DatetimeIndex
        Time index for the schedule
    generators : list[str], optional
        List of generators to include in schedule. If None, includes all
        generators that appear in outage_events.
    default_capacity_factor : float, default 1.0
        Default capacity factor when no outage is active

    Returns
    -------
    pd.DataFrame
        Time series with index=snapshots, columns=generator names,
        values=capacity factors (0-1)

    Examples
    --------
    >>> events = [
    ...     OutageEvent(
    ...         generator="Coal1",
    ...         start=pd.Timestamp("2024-01-15"),
    ...         end=pd.Timestamp("2024-01-18"),
    ...         outage_type="maintenance",
    ...         capacity_factor=0.0,
    ...     ),
    ...     OutageEvent(
    ...         generator="Gas1",
    ...         start=pd.Timestamp("2024-02-20"),
    ...         end=pd.Timestamp("2024-02-21"),
    ...         outage_type="forced",
    ...         capacity_factor=0.0,
    ...     ),
    ... ]
    >>> snapshots = pd.date_range("2024-01-01", "2024-03-01", freq="h")
    >>> schedule = build_outage_schedule(events, snapshots)
    >>> schedule.loc["2024-01-15", "Coal1"]  # Returns 0.0 (outage)
    >>> schedule.loc["2024-01-10", "Coal1"]  # Returns 1.0 (available)
    """
    # Determine generator list
    if generators is None:
        generators = sorted({event.generator for event in outage_events})

    n_snapshots = len(snapshots)
    n_gens = len(generators)
    gen_idx = {gen: i for i, gen in enumerate(generators)}

    # Use numpy array for fast assignment
    schedule_arr = np.full((n_snapshots, n_gens), default_capacity_factor, dtype=float)

    for event in outage_events:
        g = event.generator
        if g not in gen_idx:
            logger.warning(f"Generator {g} not in schedule columns, skipping event")
            continue
        # Find affected snapshot indices (vectorized)
        mask = (snapshots >= event.start) & (snapshots < event.end)
        if not np.any(mask):
            logger.debug(
                f"Outage event for {g} ({event.start} to {event.end}) "
                f"does not overlap with any snapshots"
            )
            continue
        i = gen_idx[g]
        # Use numpy minimum for all affected rows at once
        schedule_arr[mask, i] = np.minimum(schedule_arr[mask, i], event.capacity_factor)
        logger.debug(
            f"Applied {event.outage_type} outage to {g}: "
            f"{event.start} to {event.end} (CF={event.capacity_factor})"
        )

    # Convert back to DataFrame
    schedule = pd.DataFrame(schedule_arr, index=snapshots, columns=generators)
    return schedule


def apply_outage_schedule(
    network: Network,
    outage_schedule: pd.DataFrame,
    apply_to_p_max: bool = True,
    apply_to_p_min: bool = True,
) -> dict:
    """Apply outage schedule to network generators.

    This function applies outage schedules to PyPSA networks by:
    1. Scaling p_max_pu by the outage schedule (reduces available capacity)
    2. Scaling p_min_pu by the outage schedule (maintains constraint consistency)
    3. Preventing p_min_pu > p_max_pu violations

    Note: This assumes time series p_min_pu has already been properly initialized
    from PLEXOS database properties (Min Stable Level, Min Stable Factor, etc.)
    during network setup via set_min_stable_levels() in port_generators().

    Parameters
    ----------
    network : Network
        PyPSA network with generators
    outage_schedule : pd.DataFrame
        Outage schedule from build_outage_schedule() with index=snapshots,
        columns=generator names, values=capacity factors (0-1)
    apply_to_p_max : bool, default True
        Apply schedule to p_max_pu (multiply existing values by schedule)
    apply_to_p_min : bool, default True
        Apply schedule to p_min_pu (multiply existing values by schedule)
        Maintains operational flexibility while scaling constraints

    Returns
    -------
    dict
        Summary: {
            "affected_generators": int,
            "applied_to_p_max": int,
            "applied_to_p_min": int,
        }

    Examples
    --------
    >>> # Standard usage - handles everything automatically
    >>> events = parse_explicit_outages_from_properties(csv_dir, network)
    >>> schedule = build_outage_schedule(events, network.snapshots)
    >>> summary = apply_outage_schedule(network, schedule)
    >>> print(f"Applied outages to {summary['affected_generators']} generators")
    Applied outages to 42 generators

    >>> # Custom: only apply to p_max_pu (allow free curtailment)
    >>> summary = apply_outage_schedule(network, schedule, apply_to_p_min=False)

    Notes
    -----
    Example behavior with 500 MW thermal generator (2 units, 50% min load):
    - Before: p_min_pu=0.5 (static), p_max_pu=1.0
    - One unit out (Outage Rating=250 MW) → schedule=0.5
    - After: p_min_pu=0.25 (time series, 125 MW), p_max_pu=0.5 (250 MW)
    - Maintains 50% turndown ratio while respecting reduced capacity
    """
    # Find intersection of generators in outage_schedule and network
    gens_in_p_max = list(
        set(outage_schedule.columns) & set(network.generators_t.p_max_pu.columns)
    )
    gens_in_p_min = list(
        set(outage_schedule.columns) & set(network.generators_t.p_min_pu.columns)
    )
    affected_generators = set(gens_in_p_max) | set(gens_in_p_min)

    applied_to_p_max_count = 0
    applied_to_p_min_count = 0

    # Step 1: Apply schedule to p_max_pu (vectorized)
    if apply_to_p_max and gens_in_p_max:
        network.generators_t.p_max_pu.loc[:, gens_in_p_max] = (
            network.generators_t.p_max_pu.loc[:, gens_in_p_max]
            * outage_schedule.loc[:, gens_in_p_max]
        )
        applied_to_p_max_count = len(gens_in_p_max)

    # Step 2: Apply schedule to p_min_pu (vectorized)
    if apply_to_p_min and gens_in_p_min:
        network.generators_t.p_min_pu.loc[:, gens_in_p_min] = (
            network.generators_t.p_min_pu.loc[:, gens_in_p_min]
            * outage_schedule.loc[:, gens_in_p_min]
        )
        applied_to_p_min_count = len(gens_in_p_min)

    # Log summary
    logger.info(f"Applied outage schedule to {len(affected_generators)} generators")
    if apply_to_p_max:
        logger.info(f"  p_max_pu: {applied_to_p_max_count} generators")
    if apply_to_p_min:
        logger.info(f"  p_min_pu: {applied_to_p_min_count} generators")

    return {
        "affected_generators": len(affected_generators),
        "applied_to_p_max": applied_to_p_max_count,
        "applied_to_p_min": applied_to_p_min_count,
    }


def load_precomputed_outage_schedules(
    network: Network,
    outages_path: str | Path,
    scenario: str | int = "1",
    generator_filter: Callable[[str], bool] | None = None,
    apply_mode: str = "multiply",
) -> dict:
    """Load pre-computed outage schedules from CAISO-style UnitsOut directory.

    CAISO IRP23 provides pre-computed outage schedules in the format:
    - Directory structure: Units Out/M01/, M02/, ..., M12/ (monthly folders)
    - File naming: {GeneratorName}_UnitsOut.csv
    - CSV format: Year,Month,Day,Period,1,2,3,...,N (N scenario columns)
    - Values: Binary (0=available, 1=out) for each hour and scenario

    This function discovers and loads these schedules, converting them to
    PyPSA p_max_pu time series (inverted: 1=available, 0=out).

    Parameters
    ----------
    network : Network
        PyPSA network with generators already added
    outages_path : str | Path
        Base path to UnitsOut directory (e.g., "data/caiso-irp23/Units Out")
    scenario : str | int, default "1"
        Which scenario column to load (1-indexed). For CAISO, scenarios 1-500.
    generator_filter : callable, optional
        Function taking generator name and returning True to process.
        Example: lambda gen: gen in network.generators.index
    apply_mode : str, default "multiply"
        How to apply outage schedules:
        - "multiply": Multiply existing p_max_pu (preserves VRE profiles)
        - "replace": Replace existing p_max_pu (overrides VRE profiles)

    Returns
    -------
    dict
        Summary: {
            "processed_generators": int,
            "skipped_generators": list[str],
            "failed_generators": list[str],
            "scenario": str | int,
        }

    Examples
    --------
    >>> # Load scenario 1 outages for CAISO model
    >>> summary = load_precomputed_outage_schedules(
    ...     network=network,
    ...     outages_path="src/examples/data/caiso-irp23/Units Out",
    ...     scenario="1",
    ...     generator_filter=lambda gen: gen in network.generators.index,
    ... )
    >>> print(f"Loaded outages for {summary['processed_generators']} generators")
    """
    outages_path = Path(outages_path)

    if not outages_path.exists():
        logger.error(f"Outages path not found: {outages_path}")
        return {
            "processed_generators": 0,
            "skipped_generators": [],
            "failed_generators": [],
            "scenario": scenario,
        }

    processed_generators = []
    skipped_generators = []
    failed_generators = []

    # Discover monthly subdirectories (M01, M02, ..., M12)
    monthly_dirs = sorted(
        [d for d in outages_path.iterdir() if d.is_dir() and d.name.startswith("M")]
    )

    if not monthly_dirs:
        logger.warning(f"No monthly subdirectories found in {outages_path}")
        return {
            "processed_generators": 0,
            "skipped_generators": [],
            "failed_generators": [],
            "scenario": scenario,
        }

    logger.info(f"Found {len(monthly_dirs)} monthly directories in {outages_path}")

    # Collect all UnitsOut CSV files
    outage_files = []
    for month_dir in monthly_dirs:
        csv_files = list(month_dir.glob("*_UnitsOut.csv"))
        outage_files.extend(csv_files)

    logger.info(f"Found {len(outage_files)} UnitsOut CSV files")

    # Process each file
    for csv_path in outage_files:
        # Extract generator name from filename (e.g., "Alameda1_UnitsOut.csv" -> "Alameda1")
        gen_name = csv_path.stem.replace("_UnitsOut", "")

        # Apply filter
        if generator_filter and not generator_filter(gen_name):
            skipped_generators.append(gen_name)
            continue

        # Check if generator exists in network
        if gen_name not in network.generators.index:
            logger.debug(f"Generator {gen_name} not in network, skipping")
            skipped_generators.append(gen_name)
            continue

        try:
            # Load outage schedule using existing CSV reader
            outage_df = read_plexos_input_csv(csv_path, scenario=scenario)

            # Extract availability series (invert: 0=out -> 1=available, 1=out -> 0=available)
            if "value" in outage_df.columns:
                availability_series = 1.0 - outage_df["value"]
            else:
                availability_series = 1.0 - outage_df.iloc[:, 0]

            # Align to network snapshots
            aligned_availability = availability_series.reindex(
                network.snapshots
            ).fillna(1.0)

            # Apply to network based on mode
            if apply_mode == "multiply":
                # Multiply with existing p_max_pu (preserves VRE profiles)
                existing = network.generators_t.p_max_pu[gen_name]
                network.generators_t.p_max_pu[gen_name] = (
                    existing * aligned_availability
                )

            elif apply_mode == "replace":
                # Replace existing p_max_pu
                network.generators_t.p_max_pu[gen_name] = aligned_availability

            else:
                _raise_invalid_apply_mode(apply_mode)

            processed_generators.append(gen_name)
            logger.info(
                f"Loaded outage schedule for {gen_name} (scenario {scenario}, "
                f"{availability_series.eq(0).sum()} outage hours)"
            )

        except Exception:
            logger.exception(f"Failed to load outage schedule for {gen_name}")
            failed_generators.append(gen_name)

    logger.info(
        f"Processed {len(processed_generators)} generators, "
        f"skipped {len(skipped_generators)}, "
        f"failed {len(failed_generators)}"
    )

    return {
        "processed_generators": len(processed_generators),
        "skipped_generators": skipped_generators,
        "failed_generators": failed_generators,
        "scenario": scenario,
    }


def parse_explicit_outages_from_properties(
    csv_dir: str | Path,
    network: Network,
    property_name: str = "Units Out",
    generator_filter: Callable[[str], bool] | None = None,
) -> list[OutageEvent]:
    """Parse explicit scheduled outages from PLEXOS CSV properties.

    Reads time-varying properties to find explicit outage specifications
    (e.g., "Units Out" property with date_from/date_to).

    PLEXOS "Units Out" property:
    - Value: Number of units out (can be fractional for partial outages)
    - date_from: Start date of outage (inclusive)
    - date_to: End date of outage (exclusive)

    For a generator with p_nom = 100 MW and "Units Out" = 1.0:
    - capacity_factor = 0.0 (full outage)

    For a generator with multiple units where "Units Out" = 0.5:
    - capacity_factor = 0.5 (50% of capacity available)

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    network : Network
        PyPSA network with generators
    property_name : str, default "Units Out"
        PLEXOS property name for explicit outages

    Returns
    -------
    list[OutageEvent]
        List of explicit outage events

    Examples
    --------
    >>> events = parse_explicit_outages_from_properties(
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     network=network,
    ...     property_name="Units Out",
    ... )
    >>> len(events)
    42
    """
    csv_dir = Path(csv_dir)

    # Load time-varying properties
    try:
        time_varying = load_time_varying_properties(csv_dir)
    except Exception:
        logger.exception(f"Failed to load time-varying properties from {csv_dir}")
        return []

    if time_varying.empty:
        logger.info(f"No time-varying properties found in {csv_dir}")
        return []

    # Filter to Generator class and specified property
    outage_props = time_varying[
        (time_varying["class"] == "Generator")
        & (time_varying["property"] == property_name)
    ].copy()

    if outage_props.empty:
        logger.info(
            f"No '{property_name}' properties found for generators in {csv_dir}"
        )
        return []

    logger.info(f"Found {len(outage_props)} '{property_name}' entries for generators")

    # Parse into OutageEvent objects
    outage_events = []

    for _, row in outage_props.iterrows():
        gen_name = row["object"]

        # Check if generator exists in network
        if gen_name not in network.generators.index:
            logger.debug(f"Generator {gen_name} not in network, skipping outage")
            continue

        # Apply generator filter if provided
        if generator_filter is not None and not generator_filter(gen_name):
            logger.debug(
                f"Generator {gen_name} filtered out by generator_filter, skipping outage"
            )
            continue

        # Parse dates
        start = ensure_datetime(row["date_from"])
        end = ensure_datetime(row["date_to"])

        # Validate dates
        if pd.isna(start) or pd.isna(end):
            logger.warning(
                f"Invalid dates for {gen_name} {property_name}: "
                f"start={row['date_from']}, end={row['date_to']}"
            )
            continue

        if end <= start:
            logger.warning(
                f"Invalid date range for {gen_name} {property_name}: "
                f"end ({end}) <= start ({start})"
            )
            continue

        # Parse value (number of units out)
        try:
            units_out = float(row["value"])
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid value for {gen_name} {property_name}: {row['value']}"
            )
            continue

        # Convert units_out to capacity factor
        # Assumption: "Units Out" = 1.0 means full outage (CF = 0.0)
        # For partial outages: CF = 1.0 - units_out
        capacity_factor = max(0.0, 1.0 - units_out)

        # Create outage event
        event = OutageEvent(
            generator=gen_name,
            start=start,
            end=end,
            outage_type="explicit",
            capacity_factor=capacity_factor,
        )

        outage_events.append(event)

        logger.debug(
            f"Created explicit outage for {gen_name}: {start} to {end} "
            f"(units_out={units_out}, CF={capacity_factor})"
        )

    logger.info(f"Parsed {len(outage_events)} explicit outage events")

    return outage_events


def apply_expected_outage_derating(
    network: Network,
    forced_outage_rate: float = 0.0,
    maintenance_rate: float = 0.0,
    generator_filter: Callable[[str], bool] | None = None,
) -> dict:
    """Apply deterministic expected outage derating (fallback method).

    This is the simplest approach: treats outage rates as continuous derating
    factors. Does NOT match PLEXOS behavior (which uses Monte Carlo and PASA),
    but provides a deterministic fallback when stochastic methods are not used.

    availability = (1 - forced_outage_rate) × (1 - maintenance_rate)

    WARNING: This is a simplified approximation. PLEXOS uses:
    - Monte Carlo simulation for forced outages (discrete random events)
    - PASA optimization for maintenance (discrete scheduled events)
    This function provides expected value approximation only.

    Parameters
    ----------
    network : Network
        PyPSA network with generators
    forced_outage_rate : float, default 0.0
        Forced outage rate (fraction of time, 0-1)
    maintenance_rate : float, default 0.0
        Maintenance rate (fraction of time, 0-1)
    generator_filter : callable, optional
        Function taking generator name and returning True to process

    Returns
    -------
    dict
        Summary with processed generator count

    Examples
    --------
    >>> # Apply 3% forced outage + 5% maintenance to all thermal generators
    >>> summary = apply_expected_outage_derating(
    ...     network=network,
    ...     forced_outage_rate=0.03,
    ...     maintenance_rate=0.05,
    ...     generator_filter=lambda gen: network.generators.at[gen, "carrier"] in ["Coal", "Gas"],
    ... )
    """
    if forced_outage_rate < 0.0 or forced_outage_rate > 1.0:
        msg = f"forced_outage_rate must be in [0, 1], got {forced_outage_rate}"
        raise ValueError(msg)

    if maintenance_rate < 0.0 or maintenance_rate > 1.0:
        msg = f"maintenance_rate must be in [0, 1], got {maintenance_rate}"
        raise ValueError(msg)

    # Calculate combined availability
    availability = (1.0 - forced_outage_rate) * (1.0 - maintenance_rate)

    logger.info(
        f"Applying expected outage derating: FOR={forced_outage_rate:.1%}, "
        f"MR={maintenance_rate:.1%}, availability={availability:.1%}"
    )

    processed_count = 0

    for gen in network.generators.index:
        # Apply filter if provided
        if generator_filter and not generator_filter(gen):
            continue

        # Multiply existing p_max_pu by availability factor
        if gen in network.generators_t.p_max_pu.columns:
            network.generators_t.p_max_pu[gen] *= availability
            processed_count += 1
            logger.debug(f"Applied {availability:.1%} availability to {gen}")

    logger.info(f"Applied expected outage derating to {processed_count} generators")

    return {
        "processed_generators": processed_count,
        "forced_outage_rate": forced_outage_rate,
        "maintenance_rate": maintenance_rate,
        "availability": availability,
    }


# =============================================================================
# Stochastic Outage Generation (Simplified Monte Carlo and PASA-like)
# =============================================================================


def _calculate_explicit_outage_hours(
    existing_outage_events: list[OutageEvent] | None,
    snapshots: pd.DatetimeIndex,
) -> dict[str, float]:
    """Calculate total explicit outage hours per generator from existing events.

    Helper function for stochastic generation that accounts for explicit outages.
    Computes the total duration of explicit outage events for each generator,
    which can then be deducted from FOR/MR budgets.

    Parameters
    ----------
    existing_outage_events : list[OutageEvent] | None
        List of existing explicit outage events. If None, returns empty dict.
    snapshots : pd.DatetimeIndex
        Network snapshots to constrain outage duration calculation

    Returns
    -------
    dict[str, float]
        Dictionary mapping generator name to total explicit outage hours.
        Only includes generators that have explicit outages.

    Examples
    --------
    >>> explicit_events = parse_explicit_outages_from_properties(csv_dir, network)
    >>> explicit_hours = _calculate_explicit_outage_hours(explicit_events, network.snapshots)
    >>> explicit_hours["AA1"]  # Total explicit outage hours for generator AA1
    263.5
    """
    if existing_outage_events is None or len(existing_outage_events) == 0:
        return {}

    # Calculate total outage hours per generator
    explicit_hours: dict[str, float] = {}

    for event in existing_outage_events:
        # Only count explicit outages (not forced/maintenance from previous runs)
        if event.outage_type != "explicit":
            continue

        # Calculate duration in hours, constrained to snapshots
        start = max(event.start, snapshots[0])
        end = min(event.end, snapshots[-1])

        if end <= start:
            continue  # Event outside snapshot range

        duration_hours = (end - start).total_seconds() / 3600

        # Accumulate hours for this generator
        gen = event.generator
        explicit_hours[gen] = explicit_hours.get(gen, 0.0) + duration_hours

    logger.debug(
        f"Calculated explicit outage hours for {len(explicit_hours)} generators"
    )

    return explicit_hours


def parse_generator_outage_properties_csv(
    csv_dir: str | Path,
    network: Network,
) -> pd.DataFrame:
    """Extract outage-related properties from Generator.csv.

    Parses Forced Outage Rate, Maintenance Rate, Mean Time to Repair,
    Outage Rating, and Outage Factor properties for all generators in the network.

    Outage Factor Priority:
    - If Outage Factor is present, it overrides Outage Rating (PLEXOS behavior)
    - Outage Factor represents the proportion of capacity AVAILABLE during outage
    - For time-series Outage Factor values, uses MIN for conservative outages
    - Example: Outage Factor=0.337 means 33.7% available, 66.3% unavailable

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    network : Network
        PyPSA network with generators

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [generator, forced_outage_rate, maintenance_rate,
        mean_time_to_repair, outage_rating, outage_factor]. Indexed by generator name.
        Rates and factors are converted from percentages to fractions (10.9 → 0.109).

    Examples
    --------
    >>> props = parse_generator_outage_properties_csv(
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     network=network,
    ... )
    >>> props.loc["AA1", "forced_outage_rate"]  # Returns 0.109 (10.9%)
    >>> props.loc["AGLHAL01", "outage_factor"]  # Returns 0.337 (33.7% available during outage)
    """
    csv_dir = Path(csv_dir)

    # Load static generator properties
    generator_df = load_static_properties(csv_dir, "Generator")

    if generator_df.empty:
        logger.warning(f"No generators found in {csv_dir}")
        return pd.DataFrame(
            columns=[
                "forced_outage_rate",
                "maintenance_rate",
                "mean_time_to_repair",
                "outage_rating",
                "outage_factor",
            ]
        )

    # Filter to generators in network
    generator_df = generator_df[generator_df.index.isin(network.generators.index)]

    if generator_df.empty:
        logger.warning("No generators from CSV found in network")
        return pd.DataFrame(
            columns=[
                "forced_outage_rate",
                "maintenance_rate",
                "mean_time_to_repair",
                "outage_rating",
                "outage_factor",
            ]
        )

    # Parse properties for each generator
    properties_list = []

    property_mappings = {
        "forced_outage_rate": (
            "Forced Outage Rate",
            True,
        ),  # (property_name, is_percentage)
        "maintenance_rate": ("Maintenance Rate", True),
        "mean_time_to_repair": ("Mean Time to Repair", False),
        "outage_rating": ("Outage Rating", False),
        "outage_factor": (
            "Outage Factor",
            True,
        ),  # Percentage -> fraction, use MIN for conservative outages
    }

    for gen in generator_df.index:
        gen_props = {"generator": gen}

        for prop_key, (plexos_name, is_percentage) in property_mappings.items():
            raw_value = get_property_from_static_csv(generator_df, gen, plexos_name)
            if raw_value is not None:
                # Special handling for Outage Factor: use MIN for conservative outages
                if prop_key == "outage_factor":
                    parsed_value = parse_numeric_value(raw_value, strategy="min")
                    # Log if time-series detected
                    if (
                        parsed_value is not None
                        and isinstance(raw_value, str)
                        and "[" in raw_value
                    ):
                        logger.debug(
                            f"{gen}: Outage Factor is time-varying {raw_value}, "
                            f"using min value: {parsed_value}"
                        )
                else:
                    parsed_value = parse_numeric_value(raw_value, use_first=True)

                if parsed_value is not None:
                    # Convert percentage to fraction if needed
                    gen_props[prop_key] = (
                        parsed_value / 100.0 if is_percentage else parsed_value
                    )
                else:
                    gen_props[prop_key] = None
            else:
                gen_props[prop_key] = None

        properties_list.append(gen_props)

    # Create DataFrame
    properties_df = pd.DataFrame(properties_list)
    properties_df = properties_df.set_index("generator")

    # Log statistics
    for_count = properties_df["forced_outage_rate"].notna().sum()
    mr_count = properties_df["maintenance_rate"].notna().sum()
    mttr_count = properties_df["mean_time_to_repair"].notna().sum()
    of_count = properties_df["outage_factor"].notna().sum()

    logger.info(
        f"Parsed outage properties: FOR={for_count}, MR={mr_count}, "
        f"MTTR={mttr_count}, OutageFactor={of_count} generators"
    )

    return properties_df


def generate_forced_outages_simplified(
    csv_dir: str | Path,
    network: Network,
    random_seed: int | None = None,
    generator_filter: Callable[[str], bool] | None = None,
    existing_outage_events: list[OutageEvent] | None = None,
) -> list[OutageEvent]:
    """Generate forced outage events using simplified Monte Carlo simulation.

    This function approximates PLEXOS forced outage behavior using a simplified
    stochastic approach:
    - Forced Outage Rate (FOR) determines expected outage frequency
    - Mean Time To Repair (MTTR) determines outage duration
    - Random placement throughout simulation period

    Mathematical model:
    - Expected outage hours = FOR × total_hours
    - Number of outages ≈ (FOR × total_hours) / MTTR
    - Each outage duration = MTTR hours (simplified from exponential)
    - Random uniform placement without overlap checking

    If existing_outage_events is provided, explicit outage hours are deducted
    from the FOR budget to avoid double-counting:
    - Adjusted outage hours = (FOR × total_hours) - explicit_hours
    - Only generates forced outages for remaining adjusted hours

    Limitations vs PLEXOS:
    - Fixed duration instead of exponential distribution
    - No state transitions (up/down Markov chain)
    - No correlation between generators
    - May generate overlapping outages (rare with low FOR)

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports with Generator.csv
    network : Network
        PyPSA network with generators and snapshots
    random_seed : int, optional
        Random seed for reproducibility. If None, results are non-deterministic.
    generator_filter : callable, optional
        Function taking generator name and returning True to process.
        Example: lambda gen: network.generators.at[gen, "carrier"] in ["Coal", "Gas"]
    existing_outage_events : list[OutageEvent], optional
        List of existing explicit outage events. If provided, explicit outage
        hours are deducted from FOR budget to prevent double-counting.
        Example: parse_explicit_outages_from_properties(csv_dir, network)

    Returns
    -------
    list[OutageEvent]
        List of forced outage events with outage_type="forced"

    Examples
    --------
    >>> # Generate forced outages for SEM thermal generators
    >>> events = generate_forced_outages_simplified(
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     network=network,
    ...     random_seed=42,
    ...     generator_filter=lambda gen: gen.startswith("AA"),
    ... )
    >>> len(events)
    157
    >>> events[0].outage_type
    'forced'

    >>> # Account for existing explicit outages to avoid double-counting
    >>> explicit_events = parse_explicit_outages_from_properties(csv_dir, network)
    >>> forced_events = generate_forced_outages_simplified(
    ...     csv_dir=csv_dir,
    ...     network=network,
    ...     random_seed=42,
    ...     existing_outage_events=explicit_events,  # Deducts explicit hours from FOR budget
    ... )
    """
    csv_dir = Path(csv_dir)

    # Set random seed for reproducibility
    rng = np.random.default_rng(random_seed)
    if random_seed is not None:
        logger.info(f"Using random seed: {random_seed}")

    # Calculate explicit outage hours per generator (if provided)
    explicit_hours_by_gen = _calculate_explicit_outage_hours(
        existing_outage_events, network.snapshots
    )
    if explicit_hours_by_gen:
        logger.info(
            f"Accounting for {len(explicit_hours_by_gen)} generators with explicit outages"
        )

    # Parse outage properties
    properties_df = parse_generator_outage_properties_csv(csv_dir, network)

    # Filter generators with valid FOR and MTTR
    valid_gens = properties_df[
        properties_df["forced_outage_rate"].notna()
        & properties_df["mean_time_to_repair"].notna()
        & (properties_df["forced_outage_rate"] > 0)
        & (properties_df["mean_time_to_repair"] > 0)
    ].copy()

    if valid_gens.empty:
        logger.warning("No generators with valid Forced Outage Rate and MTTR found")
        return []

    # Apply generator filter
    if generator_filter is not None:
        valid_gens = valid_gens[valid_gens.index.map(generator_filter)]

    if valid_gens.empty:
        logger.info("No generators passed filter for forced outage generation")
        return []

    logger.info(
        f"Generating forced outages for {len(valid_gens)} generators "
        f"(seed={random_seed})"
    )

    # Get simulation time range
    snapshots = network.snapshots
    start_time = snapshots[0]
    end_time = snapshots[-1]
    total_hours = len(snapshots)  # Approximate as hourly snapshots

    # Generate outage events
    outage_events = []

    for gen, row in valid_gens.iterrows():
        FOR = row["forced_outage_rate"]
        MTTR = row["mean_time_to_repair"]

        # Calculate expected outage statistics
        expected_outage_hours = FOR * total_hours

        # Deduct explicit outage hours if provided
        explicit_hours = explicit_hours_by_gen.get(str(gen), 0.0)
        adjusted_outage_hours = expected_outage_hours - explicit_hours

        if adjusted_outage_hours <= 0:
            logger.debug(
                f"{gen}: Explicit outages ({explicit_hours:.1f}h) already cover FOR budget "
                f"({expected_outage_hours:.1f}h), skipping forced outage generation"
            )
            continue

        num_outages = int(np.round(adjusted_outage_hours / MTTR))

        if num_outages == 0:
            logger.debug(
                f"{gen}: Adjusted FOR too low for outages (FOR={FOR:.2%}, "
                f"expected={expected_outage_hours:.1f}h, explicit={explicit_hours:.1f}h, "
                f"adjusted={adjusted_outage_hours:.1f}h)"
            )
            continue

        # Get p_nom for capacity factor calculation
        p_nom = (
            network.generators.at[gen, "p_nom"]
            if "p_nom" in network.generators.columns
            else None
        )
        outage_rating = row["outage_rating"]
        outage_factor = row.get("outage_factor")

        # Calculate capacity factor during outage (Outage Factor takes priority)
        capacity_factor = _calculate_capacity_factor(
            p_nom, outage_rating, outage_factor
        )

        # Log if using Outage Factor
        if outage_factor is not None:
            logger.debug(
                f"{gen}: Using Outage Factor={outage_factor:.2%} "
                f"(capacity_factor={capacity_factor:.2%} during outage)"
            )

        # Generate random outage events
        for _ in range(num_outages):
            # Random start time (uniform distribution)
            # Sample from hours 0 to (total_hours - MTTR) to avoid exceeding end
            max_start_hour = max(0, total_hours - MTTR)
            start_hour = rng.uniform(0, max_start_hour)

            # Convert to timestamp (use integer hours to avoid floating-point precision issues)
            # For long time periods, use snapshot index sampling instead
            if total_hours > 100000:  # For very long simulations
                # Sample from snapshot indices instead
                max_start_idx = max(
                    0, len(snapshots) - int(MTTR * 2)
                )  # *2 for safety margin
                start_idx = int(rng.uniform(0, max_start_idx))
                outage_start = snapshots[start_idx]
            else:
                # For shorter simulations, use timedelta
                outage_start = start_time + pd.Timedelta(hours=start_hour)

            # Duration equals MTTR hours
            outage_end = outage_start + pd.Timedelta(hours=MTTR)

            # Ensure doesn't exceed simulation period
            if outage_end > end_time:
                outage_end = end_time
                # Skip if outage would be too short (< 10% of MTTR)
                actual_duration = (outage_end - outage_start).total_seconds() / 3600
                if actual_duration < MTTR * 0.1:
                    continue

            # Create event
            event = OutageEvent(
                generator=str(gen),
                start=outage_start,
                end=outage_end,
                outage_type="forced",
                capacity_factor=capacity_factor,
            )

            outage_events.append(event)

        if explicit_hours > 0:
            logger.debug(
                f"{gen}: Generated {num_outages} forced outages "
                f"(FOR={FOR:.2%}, MTTR={MTTR:.1f}h, CF={capacity_factor:.2f}, "
                f"explicit={explicit_hours:.1f}h deducted)"
            )
        else:
            logger.debug(
                f"{gen}: Generated {num_outages} forced outages "
                f"(FOR={FOR:.2%}, MTTR={MTTR:.1f}h, CF={capacity_factor:.2f})"
            )

    # Calculate statistics
    total_events = len(outage_events)
    total_outage_hours = sum(
        (e.end - e.start).total_seconds() / 3600 for e in outage_events
    )
    avg_for = (
        total_outage_hours / (len(valid_gens) * total_hours) if total_hours > 0 else 0
    )

    logger.info(
        f"Generated {total_events} forced outage events for {len(valid_gens)} generators"
    )
    logger.info(
        f"  Total outage hours: {total_outage_hours:.1f} (avg FOR={avg_for:.2%})"
    )

    return outage_events


def schedule_maintenance_simplified(
    csv_dir: str | Path,
    network: Network,
    demand_profile: pd.Series | None = None,
    generator_filter: Callable[[str], bool] | None = None,
    min_spacing_days: int = 7,
    maintenance_window_days: int = 14,
    existing_outage_events: list[OutageEvent] | None = None,
) -> list[OutageEvent]:
    """Schedule maintenance outages in low-demand periods using heuristic.

    This function approximates PLEXOS PASA maintenance scheduling using a
    simplified demand-aware heuristic:
    - Maintenance Rate (MR) determines total maintenance hours needed
    - Maintenance scheduled during lowest-demand periods
    - Simple greedy heuristic (not optimization like PLEXOS)

    Algorithm:
    1. Calculate required maintenance hours = MR × total_hours
    2. If demand_profile provided, identify low-demand windows
    3. Schedule maintenance in lowest-demand windows
    4. Enforce minimum spacing between maintenance events
    5. Try to schedule in contiguous blocks (realistic maintenance windows)

    If existing_outage_events is provided, explicit outage hours are deducted
    from the MR budget to avoid double-counting:
    - Adjusted maintenance hours = (MR × total_hours) - explicit_hours
    - Only schedules maintenance for remaining adjusted hours

    Limitations vs PLEXOS PASA:
    - Heuristic instead of optimization
    - No reliability constraints (reserve margins, N-1, etc.)
    - No unit commitment coordination
    - No consideration of startup costs or fuel constraints

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports with Generator.csv
    network : Network
        PyPSA network with generators and snapshots
    demand_profile : pd.Series, optional
        Time series of system demand (index=snapshots, values=MW).
        If None, maintenance is scheduled uniformly throughout year.
    generator_filter : callable, optional
        Function taking generator name and returning True to process.
        Example: lambda gen: network.generators.at[gen, "carrier"] == "Nuclear"
    min_spacing_days : int, default 7
        Minimum days between maintenance events for same generator
    maintenance_window_days : int, default 14
        Preferred duration of maintenance window (days)
    existing_outage_events : list[OutageEvent], optional
        List of existing explicit outage events. If provided, explicit outage
        hours are deducted from MR budget to prevent double-counting.
        Example: parse_explicit_outages_from_properties(csv_dir, network)

    Returns
    -------
    list[OutageEvent]
        List of maintenance outage events with outage_type="maintenance"

    Examples
    --------
    >>> # Schedule maintenance for SEM thermal generators
    >>> demand = network.loads_t.p_set.sum(axis=1)  # Total system demand
    >>> events = schedule_maintenance_simplified(
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     network=network,
    ...     demand_profile=demand,
    ...     generator_filter=lambda gen: gen.startswith("AA"),
    ... )
    >>> len(events)
    47
    >>> events[0].outage_type
    'maintenance'

    >>> # Account for existing explicit outages to avoid double-counting
    >>> explicit_events = parse_explicit_outages_from_properties(csv_dir, network)
    >>> maint_events = schedule_maintenance_simplified(
    ...     csv_dir=csv_dir,
    ...     network=network,
    ...     demand_profile=demand,
    ...     existing_outage_events=explicit_events,  # Deducts explicit hours from MR budget
    ... )
    """
    csv_dir = Path(csv_dir)

    # Calculate explicit outage hours per generator (if provided)
    explicit_hours_by_gen = _calculate_explicit_outage_hours(
        existing_outage_events, network.snapshots
    )
    if explicit_hours_by_gen:
        logger.info(
            f"Accounting for {len(explicit_hours_by_gen)} generators with explicit outages"
        )

    # Parse outage properties
    properties_df = parse_generator_outage_properties_csv(csv_dir, network)

    # Filter generators with valid MR
    valid_gens = properties_df[
        properties_df["maintenance_rate"].notna()
        & (properties_df["maintenance_rate"] > 0)
    ].copy()

    if valid_gens.empty:
        logger.warning("No generators with valid Maintenance Rate found")
        return []

    # Apply generator filter
    if generator_filter is not None:
        valid_gens = valid_gens[valid_gens.index.map(generator_filter)]

    if valid_gens.empty:
        logger.info("No generators passed filter for maintenance scheduling")
        return []

    logger.info(f"Scheduling maintenance for {len(valid_gens)} generators")

    # Get simulation time range
    snapshots = network.snapshots
    start_time = snapshots[0]
    end_time = snapshots[-1]
    total_hours = len(snapshots)

    # Create demand-based priority if demand profile provided
    if demand_profile is not None:
        # Resample to daily average demand for scheduling
        daily_demand = demand_profile.resample("D").mean()
        # Lower demand = higher priority for maintenance
        demand_priority = 1.0 / (daily_demand + 1e-6)  # Avoid division by zero
        logger.info("Using demand-aware scheduling (lower demand = higher priority)")
    else:
        # Uniform priority if no demand profile
        daily_dates = pd.date_range(start_time, end_time, freq="D")
        demand_priority = pd.Series(1.0, index=daily_dates)
        logger.info("Using uniform scheduling (no demand profile provided)")

    # Normalize priority to [0, 1]
    demand_priority = (demand_priority - demand_priority.min()) / (
        demand_priority.max() - demand_priority.min() + 1e-9
    )

    outage_events = []

    for gen, row in valid_gens.iterrows():
        MR = row["maintenance_rate"]

        # Calculate required maintenance hours
        required_hours = MR * total_hours

        # Deduct explicit outage hours if provided
        explicit_hours = explicit_hours_by_gen.get(str(gen), 0.0)
        adjusted_required_hours = required_hours - explicit_hours

        if adjusted_required_hours <= 0:
            logger.debug(
                f"{gen}: Explicit outages ({explicit_hours:.1f}h) already cover MR budget "
                f"({required_hours:.1f}h), skipping maintenance scheduling"
            )
            continue

        if adjusted_required_hours < 1.0:
            logger.debug(
                f"{gen}: Adjusted MR too low for maintenance (MR={MR:.2%}, "
                f"required={required_hours:.1f}h, explicit={explicit_hours:.1f}h, "
                f"adjusted={adjusted_required_hours:.1f}h)"
            )
            continue

        # Get p_nom for capacity factor calculation
        p_nom = (
            network.generators.at[gen, "p_nom"]
            if "p_nom" in network.generators.columns
            else None
        )
        outage_rating = row["outage_rating"]
        outage_factor = row.get("outage_factor")

        # Calculate capacity factor during maintenance (Outage Factor takes priority)
        capacity_factor = _calculate_capacity_factor(
            p_nom, outage_rating, outage_factor
        )

        # Log if using Outage Factor
        if outage_factor is not None:
            logger.debug(
                f"{gen}: Using Outage Factor={outage_factor:.2%} "
                f"(capacity_factor={capacity_factor:.2%} during maintenance)"
            )

        # Schedule maintenance in blocks (using adjusted hours)
        remaining_hours = adjusted_required_hours
        scheduled_periods: list[
            tuple[pd.Timestamp, pd.Timestamp]
        ] = []  # Track scheduled dates to enforce spacing

        while remaining_hours > 0:
            # Determine block duration (prefer maintenance_window_days, but adapt)
            block_hours = min(remaining_hours, maintenance_window_days * 24)

            # Find best period for this maintenance block
            # Filter to unscheduled periods (respecting min_spacing)
            available_dates = demand_priority.index

            # Exclude periods too close to already scheduled maintenance
            for prev_start, prev_end in scheduled_periods:
                spacing_buffer = pd.Timedelta(days=min_spacing_days)
                exclude_start = prev_start - spacing_buffer
                exclude_end = prev_end + spacing_buffer
                available_dates = available_dates[
                    (available_dates < exclude_start) | (available_dates > exclude_end)
                ]

            if len(available_dates) == 0:
                logger.warning(
                    f"{gen}: Could not schedule remaining {remaining_hours:.1f}h "
                    f"(insufficient periods available)"
                )
                break

            # Find period with highest priority (lowest demand)
            block_days = int(np.ceil(block_hours / 24))
            best_score = -np.inf
            best_start_date = None

            for potential_start in available_dates:
                # Check if we have enough consecutive days
                potential_end = potential_start + pd.Timedelta(days=block_days)

                if potential_end > demand_priority.index[-1]:
                    continue  # Would exceed simulation period

                # Calculate average priority for this window
                window = demand_priority.loc[potential_start:potential_end]
                avg_priority = window.mean()

                if avg_priority > best_score:
                    best_score = avg_priority
                    best_start_date = potential_start

            if best_start_date is None:
                logger.warning(
                    f"{gen}: Could not find suitable period for {block_hours:.1f}h maintenance"
                )
                break

            # Schedule maintenance block
            maint_start = pd.Timestamp(best_start_date)
            maint_end = maint_start + pd.Timedelta(hours=block_hours)

            # Ensure doesn't exceed simulation period
            if maint_end > end_time:
                maint_end = end_time
                block_hours = (maint_end - maint_start).total_seconds() / 3600

            # Create event
            event = OutageEvent(
                generator=str(gen),
                start=maint_start,
                end=maint_end,
                outage_type="maintenance",
                capacity_factor=capacity_factor,
            )

            outage_events.append(event)
            scheduled_periods.append((maint_start, maint_end))

            remaining_hours -= block_hours

            logger.debug(
                f"{gen}: Scheduled {block_hours:.1f}h maintenance "
                f"({maint_start.date()} to {maint_end.date()})"
            )

        total_scheduled = sum(
            (end - start).total_seconds() / 3600 for start, end in scheduled_periods
        )
        if explicit_hours > 0:
            logger.debug(
                f"{gen}: Total scheduled {total_scheduled:.1f}h / {adjusted_required_hours:.1f}h adjusted "
                f"(MR={MR:.2%}, original={required_hours:.1f}h, explicit={explicit_hours:.1f}h deducted, "
                f"CF={capacity_factor:.2f})"
            )
        else:
            logger.debug(
                f"{gen}: Total scheduled {total_scheduled:.1f}h / {required_hours:.1f}h required "
                f"(MR={MR:.2%}, CF={capacity_factor:.2f})"
            )

    # Calculate statistics
    total_events = len(outage_events)
    total_maint_hours = sum(
        (e.end - e.start).total_seconds() / 3600 for e in outage_events
    )
    avg_mr = (
        total_maint_hours / (len(valid_gens) * total_hours) if total_hours > 0 else 0
    )

    logger.info(
        f"Scheduled {total_events} maintenance events for {len(valid_gens)} generators"
    )
    logger.info(
        f"  Total maintenance hours: {total_maint_hours:.1f} (avg MR={avg_mr:.2%})"
    )

    return outage_events


def generate_stochastic_outages_csv(
    csv_dir: str | Path,
    network: Network,
    include_forced: bool = True,
    include_maintenance: bool = True,
    demand_profile: pd.Series | None = None,
    random_seed: int | None = None,
    generator_filter: Callable[[str], bool] | None = None,
    existing_outage_events: list[OutageEvent] | None = None,
) -> list[OutageEvent]:
    """Generate both forced and maintenance outages from CSV properties.

    Unified interface combining forced outage generation (Monte Carlo) and
    maintenance scheduling (heuristic). Returns complete list of stochastic
    outage events that can be applied to network using build_outage_schedule().

    This is the recommended high-level function for generating stochastic
    outages from PLEXOS CSV properties.

    If existing_outage_events is provided, explicit outage hours are deducted
    from FOR/MR budgets to avoid double-counting. This is the recommended
    workflow when combining explicit and stochastic outages.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports with Generator.csv
    network : Network
        PyPSA network with generators and snapshots
    include_forced : bool, default True
        Generate forced outages using Monte Carlo simulation
    include_maintenance : bool, default True
        Schedule maintenance outages using demand-aware heuristic
    demand_profile : pd.Series, optional
        Time series of system demand for maintenance scheduling.
        If None, maintenance is scheduled uniformly.
        Example: network.loads_t.p_set.sum(axis=1)
    random_seed : int, optional
        Random seed for forced outage generation (reproducibility)
    generator_filter : callable, optional
        Function taking generator name and returning True to process.
        Applied to both forced and maintenance generation.
    existing_outage_events : list[OutageEvent], optional
        List of existing explicit outage events. If provided, explicit outage
        hours are deducted from FOR/MR budgets to prevent double-counting.
        Recommended workflow: parse explicit outages first, then generate
        stochastic outages with this parameter.

    Returns
    -------
    list[OutageEvent]
        Combined list of forced and maintenance outage events,
        sorted by start time

    Examples
    --------
    >>> # Generate all stochastic outages for SEM model
    >>> demand = network.loads_t.p_set.sum(axis=1)
    >>> events = generate_stochastic_outages_csv(
    ...     csv_dir="csvs_from_xml/SEM Forecast model",
    ...     network=network,
    ...     demand_profile=demand,
    ...     random_seed=42,
    ... )
    >>> len(events)
    204
    >>> # Count by type
    >>> forced_count = sum(1 for e in events if e.outage_type == "forced")
    >>> maint_count = sum(1 for e in events if e.outage_type == "maintenance")
    >>> print(f"Forced: {forced_count}, Maintenance: {maint_count}")
    Forced: 157, Maintenance: 47

    >>> # Recommended workflow: account for explicit outages to avoid double-counting
    >>> explicit_events = parse_explicit_outages_from_properties(csv_dir, network)
    >>> stochastic_events = generate_stochastic_outages_csv(
    ...     csv_dir=csv_dir,
    ...     network=network,
    ...     demand_profile=demand,
    ...     random_seed=42,
    ...     existing_outage_events=explicit_events,  # Deducts explicit hours from FOR/MR
    ... )
    >>> all_events = explicit_events + stochastic_events
    >>> schedule = build_outage_schedule(all_events, network.snapshots)
    """
    csv_dir = Path(csv_dir)

    # Reduce logging and avoid unnecessary list operations for performance
    forced_events = []
    maintenance_events = []

    if include_forced:
        forced_events = generate_forced_outages_simplified(
            csv_dir=csv_dir,
            network=network,
            random_seed=random_seed,
            generator_filter=generator_filter,
            existing_outage_events=existing_outage_events,
        )
    if include_maintenance:
        maintenance_events = schedule_maintenance_simplified(
            csv_dir=csv_dir,
            network=network,
            demand_profile=demand_profile,
            generator_filter=generator_filter,
            existing_outage_events=existing_outage_events,
        )
    if not forced_events and not maintenance_events:
        return []
    # Combine and sort only if needed
    all_events = forced_events + maintenance_events
    if len(all_events) > 1:
        all_events.sort(key=lambda e: e.start)
    return all_events
