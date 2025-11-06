"""CSV-based links/transmission functions for PLEXOS to PyPSA conversion.

This module provides CSV-based alternatives to the PlexosDB-based functions in links.py.
These functions read from COAD CSV exports instead of querying the SQLite database.

Multi-Value Property Handling
------------------------------
PLEXOS CSV exports may contain properties with multiple values (e.g., "['100', '150', '200']").
This module applies appropriate aggregation strategies based on power system constraints:

- **Max Flow / Max Rating**: Uses MIN strategy (most restrictive/conservative capacity)
- **Min Flow / Min Rating**: Uses MAX strategy (most restrictive constraint)
- **Max Ramp Up/Down**: Uses MIN strategy (slowest/most conservative ramp rate)
- **Reactance / Resistance**: Uses FIRST strategy (physical parameters)

See `parse_numeric_value()` in db.csv_readers for available strategies.
"""

import ast
import logging
from pathlib import Path

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.db.csv_readers import (
    ensure_datetime,
    get_dataid_timeslice_map_csv,
    get_numeric_property_from_static_csv,
    get_property_from_static_csv,
    load_static_properties,
    load_time_varying_properties,
)
from plexos_to_pypsa_converter.db.parse import (
    get_property_active_mask,
    read_timeslice_activity,
)

logger = logging.getLogger(__name__)


def _parse_node_buses(
    line_df: pd.DataFrame, line_name: str
) -> tuple[str | None, str | None]:
    """Parse bus0 and bus1 from Line.csv Node column or separate columns.

    Handles two formats:
    1. Single "Node" column with string format: "['bus1', 'bus2']"
    2. Separate "Node From" and "Node To" columns (NREL format)

    Parameters
    ----------
    line_df : pd.DataFrame
        DataFrame containing line properties
    line_name : str
        Name of the line to parse nodes for

    Returns
    -------
    tuple[str | None, str | None]
        (bus0, bus1) or (None, None) if parsing fails
    """
    # Try separate columns first (NREL format)
    node_from = get_property_from_static_csv(line_df, line_name, "Node From")
    node_to = get_property_from_static_csv(line_df, line_name, "Node To")

    if node_from and node_to:
        return str(node_from), str(node_to)

    # Try single "Node" column (most common format)
    node_value = get_property_from_static_csv(line_df, line_name, "Node")

    if not node_value:
        return None, None

    # Parse string representation of list: "['bus1', 'bus2']"
    try:
        if isinstance(node_value, str):
            nodes = ast.literal_eval(node_value)
            if isinstance(nodes, list):
                if len(nodes) == 2:
                    return str(nodes[0]), str(nodes[1])
                else:
                    logger.warning(
                        f"Line {line_name} Node column has {len(nodes)} nodes instead of 2: {nodes}"
                    )
            else:
                logger.warning(
                    f"Line {line_name} Node column parsed but not a list: {type(nodes)} = {nodes}"
                )
    except (ValueError, SyntaxError) as e:
        logger.warning(
            f"Failed to parse Node column for {line_name}: '{node_value}' - Error: {e}"
        )

    return None, None


def add_links_csv(network: pypsa.Network, csv_dir: str | Path) -> None:
    """Add transmission links to PyPSA network from COAD CSV exports.

    This is the CSV-based version of add_links() from links.py.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add links to
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Notes
    -----
    - Expects Line.csv with either:
      - Single "Node" column with format "['bus1', 'bus2']" (most models)
      - Separate "Node From" and "Node To" columns (NREL format)
    - Adds links with basic properties (bus0, bus1, ramp limits)
    - Sets carrier="AC" for all transmission links and ensures AC carrier exists
    - Does NOT set p_nom, p_min_pu, p_max_pu (use set_link_flows_csv for that)
    - Multi-value properties use MIN strategy for ramp limits (most conservative)
    """
    csv_dir = Path(csv_dir)
    line_df = load_static_properties(csv_dir, "Line")

    if line_df.empty:
        logger.info("No Line.csv found or no lines in CSV")
        return

    logger.info(f"Adding {len(line_df)} transmission links from CSV...")

    # Ensure AC carrier exists for transmission links
    if "AC" not in network.carriers.index:
        network.add("Carrier", "AC")
        logger.debug("Added AC carrier for transmission links")

    added_count = 0
    skipped_lines = []

    for line_name in line_df.index:
        try:
            # Parse buses from Node column or Node From/To columns
            bus0, bus1 = _parse_node_buses(line_df, line_name)

            if not bus0 or not bus1:
                logger.warning(f"Line {line_name} missing bus connections, skipping")
                skipped_lines.append(line_name)
                continue

            # Get ramp limits if available
            # Use "min" strategy for ramp limits - most conservative/restrictive (slowest ramp)
            max_ramp_up = get_numeric_property_from_static_csv(
                line_df, line_name, "Max Ramp Up", strategy="min"
            )
            max_ramp_down = get_numeric_property_from_static_csv(
                line_df, line_name, "Max Ramp Down", strategy="min"
            )

            # Add link to network
            link_data = {
                "bus0": bus0,
                "bus1": bus1,
                "carrier": "AC",  # Transmission lines are AC carrier
            }

            # Add ramp limits if they exist and are positive
            if max_ramp_up and max_ramp_up > 0:
                link_data["ramp_limit_up"] = max_ramp_up
            if max_ramp_down and max_ramp_down > 0:
                link_data["ramp_limit_down"] = max_ramp_down

            network.add("Link", line_name, **link_data)
            added_count += 1

            logger.debug(f"Added link {line_name} from {bus0} to {bus1}")

        except Exception as e:
            logger.warning(f"Error adding link {line_name}: {e}")
            skipped_lines.append(line_name)

    logger.info(f"Added {added_count} links to network")
    if skipped_lines:
        logger.warning(f"Skipped {len(skipped_lines)} lines: {skipped_lines[:10]}")


def add_lines_csv(network: pypsa.Network, csv_dir: str | Path) -> None:
    """Add transmission lines (AC/DC) to PyPSA network from COAD CSV exports.

    This function creates PyPSA Line components instead of Link components.
    Use this when electrical network parameters (resistance, reactance) are available.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add lines to
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Notes
    -----
    - Expects Line.csv with either:
      - Single "Node" column with format "['bus1', 'bus2']" (most models)
      - Separate "Node From" and "Node To" columns (NREL format)
    - Requires electrical parameters: "Reactance (p.u.)" and optionally "Resistance (p.u.)"
    - Adds Line components with electrical properties (bus0, bus1, x, r)
    - Does NOT set s_nom, s_min_pu, s_max_pu (use set_line_flows_csv for that)
    - If Reactance is not available, warns and skips the line
    - Multi-value properties use FIRST strategy for electrical parameters
    """
    csv_dir = Path(csv_dir)
    line_df = load_static_properties(csv_dir, "Line")

    if line_df.empty:
        logger.info("No Line.csv found or no lines in CSV")
        return

    logger.info(f"Adding {len(line_df)} transmission lines from CSV...")

    added_count = 0
    skipped_lines = []

    for line_name in line_df.index:
        try:
            # Parse buses from Node column or Node From/To columns
            bus0, bus1 = _parse_node_buses(line_df, line_name)

            if not bus0 or not bus1:
                logger.warning(f"Line {line_name} missing bus connections, skipping")
                skipped_lines.append(line_name)
                continue

            # Get electrical parameters (required for PyPSA Line)
            # Use "first" strategy for physical parameters (or could use "average")
            reactance = get_numeric_property_from_static_csv(
                line_df, line_name, "Reactance (p.u.)", strategy="first"
            )
            resistance = get_numeric_property_from_static_csv(
                line_df, line_name, "Resistance (p.u.)", strategy="first"
            )

            if not reactance:
                logger.warning(
                    f"Line {line_name} missing Reactance parameter, skipping. "
                    "Use transmission_as_lines=False for models without electrical data."
                )
                skipped_lines.append(line_name)
                continue

            # Build line data
            line_data = {
                "bus0": bus0,
                "bus1": bus1,
                "x": reactance,  # Reactance in p.u. (already parsed as float)
            }

            # Add resistance if available
            if resistance:
                line_data["r"] = (
                    resistance  # Resistance in p.u. (already parsed as float)
                )
            else:
                line_data["r"] = 0.0  # Default to zero resistance if not provided

            network.add("Line", line_name, **line_data)
            added_count += 1

            logger.debug(
                f"Added line {line_name} from {bus0} to {bus1} (x={line_data['x']:.4f}, r={line_data['r']:.4f})"
            )

        except Exception as e:
            logger.warning(f"Error adding line {line_name}: {e}")
            skipped_lines.append(line_name)

    logger.info(f"Added {added_count} lines to network")
    if skipped_lines:
        logger.warning(f"Skipped {len(skipped_lines)} lines: {skipped_lines[:10]}")


def parse_lines_flow_csv(
    csv_dir: str | Path,
    network: pypsa.Network,
    timeslice_csv: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Parse Min/Max Flow and Min/Max Rating for lines from CSV exports.

    This is the CSV-based version of parse_lines_flow() from links.py.

    Returns
    -------
    min_flow_df : pd.DataFrame
        Min flow time series (index=snapshots, columns=line names)
    max_flow_df : pd.DataFrame
        Max flow time series (index=snapshots, columns=line names)
    fallback_val_dict : dict
        Mapping of line name to fallback value for p_nom

    Notes
    -----
    - Uses time-varying properties CSV if available
    - Prefers Rating over Flow properties
    - Applies date/timeslice logic same as generators
    - Multi-value properties: MIN for max limits, MAX for min limits (conservative)
    """
    csv_dir = Path(csv_dir)
    snapshots = network.snapshots
    lines = network.links.index

    # Load timeslice activity if provided
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)

    # Load static line properties
    line_df = load_static_properties(csv_dir, "Line")

    # Load time-varying properties
    time_varying = load_time_varying_properties(csv_dir)

    if time_varying.empty:
        logger.info("No time-varying properties CSV, using static flow limits")
        return _build_static_line_flows(line_df, lines, snapshots)

    # Filter to Line class and flow/rating properties
    line_time_varying = time_varying[
        (time_varying["class"] == "Line")
        & (
            time_varying["property"].isin(
                ["Min Flow", "Max Flow", "Min Rating", "Max Rating"]
            )
        )
    ].copy()

    # Build data_id to timeslice mapping
    dataid_to_timeslice = {}
    if timeslice_csv is not None:
        dataid_to_timeslice = get_dataid_timeslice_map_csv(line_time_varying)

    # Build flow time series for each line
    min_flow_series = {}
    max_flow_series = {}
    fallback_val_dict = {}

    for line in lines:
        # Get properties for this line
        line_props = line_time_varying[line_time_varying["object"] == line].copy()

        # Build property entries
        property_entries = []
        for _, p in line_props.iterrows():
            entry = {
                "property": p["property"],
                "value": float(p["value"]),
                "from": ensure_datetime(p["date_from"]),
                "to": ensure_datetime(p["date_to"]),
                "data_id": int(p["data_id"]) if pd.notnull(p["data_id"]) else None,
            }
            property_entries.append(entry)

        if not property_entries:
            # No time-varying properties, try static
            min_flow_series[line] = pd.Series(0.0, index=snapshots, dtype=float)
            max_flow_series[line] = pd.Series(1000.0, index=snapshots, dtype=float)
            fallback_val_dict[line] = 1000.0
            continue

        prop_df_entries = pd.DataFrame(property_entries)

        # Build time series for each property
        min_flow_ts = _build_flow_ts(
            prop_df_entries,
            "Min Flow",
            snapshots,
            timeslice_activity,
            dataid_to_timeslice,
        )
        min_rating_ts = _build_flow_ts(
            prop_df_entries,
            "Min Rating",
            snapshots,
            timeslice_activity,
            dataid_to_timeslice,
        )
        max_flow_ts = _build_flow_ts(
            prop_df_entries,
            "Max Flow",
            snapshots,
            timeslice_activity,
            dataid_to_timeslice,
        )
        max_rating_ts = _build_flow_ts(
            prop_df_entries,
            "Max Rating",
            snapshots,
            timeslice_activity,
            dataid_to_timeslice,
        )

        # Prefer rating over flow
        min_final = min_rating_ts.combine_first(min_flow_ts)
        max_final = max_rating_ts.combine_first(max_flow_ts)

        # Get fallback value (first available max flow/rating)
        max_fallback = (
            max_final.dropna().iloc[0] if not max_final.dropna().empty else None
        )

        min_flow_series[line] = min_final
        max_flow_series[line] = max_final
        fallback_val_dict[line] = max_fallback

    min_flow_df = pd.DataFrame(min_flow_series, index=snapshots)
    max_flow_df = pd.DataFrame(max_flow_series, index=snapshots)

    return min_flow_df, max_flow_df, fallback_val_dict


def _build_static_line_flows(
    line_df: pd.DataFrame, lines: pd.Index, snapshots: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build flow DataFrames from static properties when no time-varying data."""
    min_flow_series = {}
    max_flow_series = {}
    fallback_val_dict = {}

    for line in lines:
        # Try to get static flow/rating values
        # Use "min" strategy for max limits (conservative - most restrictive capacity)
        # Use "max" strategy for min limits (conservative - most restrictive constraint)
        max_flow = get_numeric_property_from_static_csv(
            line_df, line, "Max Flow", strategy="min"
        )
        max_rating = get_numeric_property_from_static_csv(
            line_df, line, "Max Rating", strategy="min"
        )
        min_flow = get_numeric_property_from_static_csv(
            line_df, line, "Min Flow", strategy="max"
        )
        min_rating = get_numeric_property_from_static_csv(
            line_df, line, "Min Rating", strategy="max"
        )

        # Prefer rating over flow
        max_val = max_rating if max_rating is not None else max_flow
        min_val = min_rating if min_rating is not None else min_flow

        # Apply defaults if still None
        max_val = max_val if max_val is not None else 1000.0
        min_val = min_val if min_val is not None else 0.0

        min_flow_series[line] = pd.Series(min_val, index=snapshots, dtype=float)
        max_flow_series[line] = pd.Series(max_val, index=snapshots, dtype=float)
        fallback_val_dict[line] = max_val

    min_flow_df = pd.DataFrame(min_flow_series, index=snapshots)
    max_flow_df = pd.DataFrame(max_flow_series, index=snapshots)

    return min_flow_df, max_flow_df, fallback_val_dict


def _build_flow_ts(
    entries: pd.DataFrame,
    prop_name: str,
    snapshots: pd.DatetimeIndex,
    timeslice_activity: pd.DataFrame | None,
    dataid_to_timeslice: dict | None,
) -> pd.Series:
    """Build time series for a flow/rating property with date/timeslice logic."""
    ts = pd.Series(index=snapshots, dtype=float)

    # Get all rows for this property
    prop_rows = entries[entries["property"] == prop_name].copy()

    if prop_rows.empty:
        return ts  # Return empty series

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
            # Only override where not already set
            to_set = mask & ~already_set
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

    return ts


def set_link_flows_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
    bidirectional_links: bool = True,
) -> None:
    """Set flow limits (p_nom, p_min_pu, p_max_pu) for links from CSV.

    This is the CSV-based version of set_link_flows() from links.py.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network containing links
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice activity CSV file
    bidirectional_links : bool, default True
        If True, links with missing or zero min flows will be set to allow
        bidirectional flow (p_min_pu = -1, enabling imports/exports).
        If False, min flows of zero/NaN will remain unidirectional (p_min_pu = 0).

    Notes
    -----
    - Sets p_nom from fallback values (first available max flow/rating)
    - Sets p_min_pu and p_max_pu time series normalized by p_nom
    - Bidirectional flow allows links to flow in reverse (from bus1 to bus0)
    """
    csv_dir = Path(csv_dir)

    logger.info("Setting link flow limits from CSV...")

    min_flow_df, max_flow_df, fallback_val_dict = parse_lines_flow_csv(
        csv_dir, network, timeslice_csv
    )

    # Set p_nom from fallback values
    for line in network.links.index:
        fallback_val = fallback_val_dict.get(line, None)
        if fallback_val is not None:
            network.links.loc[line, "p_nom"] = fallback_val
            logger.debug(f"Set p_nom for link {line} to {fallback_val:.1f}")
        else:
            logger.warning(
                f"Link {line} has no fallback value for p_nom, cannot set flow limits"
            )

    # Calculate p_min_pu and p_max_pu (normalized by p_nom)
    p_min_pu = min_flow_df.divide(network.links["p_nom"], axis=1)
    p_max_pu = max_flow_df.divide(network.links["p_nom"], axis=1)

    # Replace inf/-inf with safe defaults
    # p_min_pu behavior depends on bidirectional_links flag:
    # - If True: treat missing/zero as -1 to enable bidirectional flow (imports/exports)
    # - If False: treat missing/zero as 0 to keep unidirectional flow (exports only)
    if bidirectional_links:
        p_min_pu = p_min_pu.replace([float("inf"), float("-inf")], -1).fillna(-1)
        p_min_pu[p_min_pu == 0] = -1.0  # Convert zero min flow to bidirectional
    else:
        p_min_pu = p_min_pu.replace([float("inf"), float("-inf")], 0).fillna(0)

    p_max_pu = p_max_pu.replace([float("inf"), float("-inf")], 1).fillna(1)

    # Assign to network
    network.links_t.p_min_pu = p_min_pu
    network.links_t.p_max_pu = p_max_pu

    logger.info(f"Set flow limits for {len(network.links)} links")


def set_line_flows_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
) -> None:
    """Set flow limits (s_nom, s_min_pu, s_max_pu) for PyPSA Lines from CSV.

    This is similar to set_link_flows_csv but works with Line components.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network containing lines
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice activity CSV file

    Notes
    -----
    - Sets s_nom from fallback values (first available max flow/rating)
    - Sets s_min_pu and s_max_pu time series normalized by s_nom
    - For Lines, s_nom represents nominal apparent power (MVA)
    """
    csv_dir = Path(csv_dir)

    logger.info("Setting line flow limits from CSV...")

    # Parse flows using the same logic but for network.lines
    snapshots = network.snapshots
    lines = network.lines.index

    # Load timeslice activity if provided
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)

    # Load static line properties
    line_df = load_static_properties(csv_dir, "Line")

    # Load time-varying properties
    time_varying = load_time_varying_properties(csv_dir)

    if time_varying.empty:
        logger.info("No time-varying properties CSV, using static flow limits")
        min_flow_df, max_flow_df, fallback_val_dict = _build_static_line_flows(
            line_df, lines, snapshots
        )
    else:
        # Filter to Line class and flow/rating properties
        line_time_varying = time_varying[
            (time_varying["class"] == "Line")
            & (
                time_varying["property"].isin(
                    ["Min Flow", "Max Flow", "Min Rating", "Max Rating"]
                )
            )
        ].copy()

        # Build data_id to timeslice mapping
        dataid_to_timeslice = {}
        if timeslice_csv is not None:
            dataid_to_timeslice = get_dataid_timeslice_map_csv(line_time_varying)

        # Build flow time series for each line
        min_flow_series = {}
        max_flow_series = {}
        fallback_val_dict = {}

        for line in lines:
            # Get properties for this line
            line_props = line_time_varying[line_time_varying["object"] == line].copy()

            # Build property entries
            property_entries = []
            for _, p in line_props.iterrows():
                entry = {
                    "property": p["property"],
                    "value": float(p["value"]),
                    "from": ensure_datetime(p["date_from"]),
                    "to": ensure_datetime(p["date_to"]),
                    "data_id": int(p["data_id"]) if pd.notnull(p["data_id"]) else None,
                }
                property_entries.append(entry)

            if not property_entries:
                # No time-varying properties, try static
                min_flow_series[line] = pd.Series(0.0, index=snapshots, dtype=float)
                max_flow_series[line] = pd.Series(1000.0, index=snapshots, dtype=float)
                fallback_val_dict[line] = 1000.0
                continue

            prop_df_entries = pd.DataFrame(property_entries)

            # Build time series for each property
            min_flow_ts = _build_flow_ts(
                prop_df_entries,
                "Min Flow",
                snapshots,
                timeslice_activity,
                dataid_to_timeslice,
            )
            min_rating_ts = _build_flow_ts(
                prop_df_entries,
                "Min Rating",
                snapshots,
                timeslice_activity,
                dataid_to_timeslice,
            )
            max_flow_ts = _build_flow_ts(
                prop_df_entries,
                "Max Flow",
                snapshots,
                timeslice_activity,
                dataid_to_timeslice,
            )
            max_rating_ts = _build_flow_ts(
                prop_df_entries,
                "Max Rating",
                snapshots,
                timeslice_activity,
                dataid_to_timeslice,
            )

            # Prefer rating over flow
            min_final = min_rating_ts.combine_first(min_flow_ts)
            max_final = max_rating_ts.combine_first(max_flow_ts)

            # Get fallback value (first available max flow/rating)
            max_fallback = (
                max_final.dropna().iloc[0] if not max_final.dropna().empty else None
            )

            min_flow_series[line] = min_final
            max_flow_series[line] = max_final
            fallback_val_dict[line] = max_fallback

        min_flow_df = pd.DataFrame(min_flow_series, index=snapshots)
        max_flow_df = pd.DataFrame(max_flow_series, index=snapshots)

    # Set s_nom from fallback values
    for line in network.lines.index:
        fallback_val = fallback_val_dict.get(line)
        if fallback_val is not None:
            network.lines.loc[line, "s_nom"] = fallback_val
            logger.debug(f"Set s_nom for line {line} to {fallback_val:.1f}")
        else:
            logger.warning(
                f"Line {line} has no fallback value for s_nom, cannot set flow limits"
            )

    # Calculate s_min_pu and s_max_pu (normalized by s_nom)
    s_min_pu = min_flow_df.divide(network.lines["s_nom"], axis=1)
    s_max_pu = max_flow_df.divide(network.lines["s_nom"], axis=1)

    # Replace inf/-inf with safe defaults
    s_min_pu = s_min_pu.replace([float("inf"), float("-inf")], 0).fillna(0)
    s_max_pu = s_max_pu.replace([float("inf"), float("-inf")], 1).fillna(1)

    # Assign to network
    network.lines_t.s_min_pu = s_min_pu
    network.lines_t.s_max_pu = s_max_pu

    logger.info(f"Set flow limits for {len(network.lines)} lines")


def reassign_links_to_node(network: pypsa.Network, target_node: str) -> dict:
    """Reassign all links to connect a specific node.

    This function is DB-agnostic and works with both DB and CSV approaches.
    It's copied from links.py for completeness.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network containing links
    target_node : str
        Name of the node to assign all links to

    Returns
    -------
    dict
        Summary information about the reassignment
    """
    if target_node not in network.buses.index:
        msg = f"Target node '{target_node}' not found in network buses"
        raise ValueError(msg)

    if len(network.links) == 0:
        logger.info("No links found in network. Skipping link reassignment.")
        return {
            "reassigned_count": 0,
            "target_node": target_node,
            "original_connections": [],
        }

    original_bus0 = network.links["bus0"].copy()
    original_bus1 = network.links["bus1"].copy()
    original_connections = [
        (b0, b1) for b0, b1 in zip(original_bus0, original_bus1, strict=False)
    ]
    unique_original_connections = list(set(original_connections))

    # Reassign all links to target node (creates self-loops)
    network.links["bus0"] = target_node
    network.links["bus1"] = target_node

    reassigned_count = len(network.links)
    logger.info(f"Reassigned {reassigned_count} links to node '{target_node}'")
    logger.info(f"  - Originally {len(unique_original_connections)} unique connections")
    logger.info(
        f"  - All links now connect {target_node} to {target_node} (self-loops)"
    )

    return {
        "reassigned_count": reassigned_count,
        "target_node": target_node,
        "original_connections": unique_original_connections,
    }


def port_links_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
    target_node: str | None = None,
    bidirectional_links: bool = True,
) -> dict | None:
    """Comprehensive function to add links and set all properties from CSV.

    This is the CSV-based version of port_links() from links.py.

    Combines all link-related operations:
    - Adds transmission links from CSV
    - Sets link flow limits (p_min_pu, p_max_pu)
    - Optionally reassigns all links to a specific node

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add links to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice activity CSV file
    target_node : str, optional
        If specified, all links will be reassigned to this node
    bidirectional_links : bool, default True
        If True, links with missing or zero min flows will be set to allow
        bidirectional flow (p_min_pu = -1, enabling imports/exports).
        If False, min flows of zero/NaN will remain unidirectional (p_min_pu = 0).

    Returns
    -------
    dict or None
        If target_node is specified, returns reassignment summary

    Examples
    --------
    >>> network = pypsa.Network()
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> port_links_csv(network, csv_dir)

    # With node reassignment for aggregated demand
    >>> port_links_csv(network, csv_dir, target_node="Load_Aggregate")
    """
    logger.info("Starting CSV-based link porting process...")

    # Step 1: Add links
    logger.info("1. Adding links from CSV...")
    add_links_csv(network, csv_dir)

    # Step 2: Set link flow limits
    logger.info("2. Setting link flow limits...")
    set_link_flows_csv(
        network,
        csv_dir,
        timeslice_csv=timeslice_csv,
        bidirectional_links=bidirectional_links,
    )

    # Step 3: Reassign links to target node if specified
    if target_node:
        logger.info(f"3. Reassigning links to target node: {target_node}")
        return reassign_links_to_node(network, target_node)
    else:
        logger.info("3. Skipping link reassignment (no target node specified)")

    logger.info(f"Link porting complete! Added {len(network.links)} links.")
    return None


def port_transmission_csv(
    network: pypsa.Network,
    csv_dir: str | Path,
    timeslice_csv: str | None = None,
    target_node: str | None = None,
    transmission_as_lines: bool = False,
) -> dict | None:
    """Comprehensive function to add transmission (Lines or Links) from CSV.

    This function routes to either Lines or Links based on the transmission_as_lines parameter.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add transmission to
    csv_dir : str | Path
        Directory containing COAD CSV exports
    timeslice_csv : str, optional
        Path to timeslice activity CSV file
    target_node : str, optional
        If specified and using Links, all links will be reassigned to this node
    transmission_as_lines : bool, default False
        If True, adds PyPSA Line components (requires electrical parameters: reactance, resistance)
        If False, adds PyPSA Link components (only requires flow limits)

    Returns
    -------
    dict or None
        If target_node is specified (Links mode only), returns reassignment summary

    Notes
    -----
    - **Use Links (transmission_as_lines=False)** when:
      - Model has operational/economic parameters (Max Flow, costs) but no electrical data
      - Most PLEXOS models fall into this category (SEM, CAISO, MaREI, etc.)
    - **Use Lines (transmission_as_lines=True)** when:
      - Model has electrical network parameters (Reactance, Resistance)
      - Example: NREL 118-bus model with impedance data
    - Lines are added with electrical properties (x, r)
    - Links are added as generic flow components with ramp limits

    Examples
    --------
    >>> network = pypsa.Network()
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> port_transmission_csv(network, csv_dir)  # Uses Links (default)

    >>> # For electrical model with reactance/resistance data
    >>> port_transmission_csv(network, csv_dir, transmission_as_lines=True)

    >>> # With node reassignment (Links mode only)
    >>> port_transmission_csv(network, csv_dir, target_node="Load_Aggregate")
    """
    if transmission_as_lines:
        logger.info("Using PyPSA Lines for transmission (electrical network model)...")

        # Step 1: Add lines
        logger.info("1. Adding lines from CSV...")
        add_lines_csv(network, csv_dir)

        # Step 2: Set line flow limits
        logger.info("2. Setting line flow limits...")
        set_line_flows_csv(network, csv_dir, timeslice_csv=timeslice_csv)

        # Note: target_node reassignment only makes sense for Links
        if target_node:
            logger.warning(
                f"target_node='{target_node}' specified but transmission_as_lines=True. "
                "Node reassignment is not applicable to PyPSA Line components."
            )

        logger.info(f"Transmission porting complete! Added {len(network.lines)} lines.")
        return None

    else:
        logger.info(
            "Using PyPSA Links for transmission (operational/economic model)..."
        )
        return port_links_csv(network, csv_dir, timeslice_csv, target_node)
