"""PyPSA Lines from PLEXOS Line objects.

This module handles conversion of PLEXOS Line objects to PyPSA Line components
with proper electrical impedance properties. PLEXOS Line objects typically
have capacity properties (Max Flow, Min Flow) but lack electrical impedance
data (resistance, reactance), so this module adds realistic synthetic values.
"""

import logging
from typing import Any

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from network.links import parse_lines_flow

logger = logging.getLogger(__name__)


def add_lines_from_plexos(
    network: Network, db: PlexosDB, default_s_nom: float = 1000.0
) -> int:
    """Add transmission lines to PyPSA network from PLEXOS Line objects.

    This function:
    1. Retrieves PLEXOS Line objects from the database
    2. Extracts capacity from Max Flow/Max Rating properties
    3. Gets node connectivity from Line memberships
    4. Adds realistic electrical impedance values (r, x)
    5. Creates PyPSA Line components

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network to add lines to
    db : PlexosDB
        PLEXOS database containing Line objects
    default_s_nom : float, default 1000.0
        Default line capacity (MW) if no Max Flow found

    Returns
    -------
    int
        Number of lines added to the network

    Notes
    -----
    - Only adds lines where both connected nodes exist in network.buses
    - Uses realistic transmission line impedance ratios (X/R ≈ 3-10)
    - Capacity comes from Max Flow or Max Rating properties
    - Line names are preserved from PLEXOS model

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB.from_xml("model.xml")
    >>> lines_added = add_lines_from_plexos(network, db)
    >>> print(f"Added {lines_added} transmission lines")
    """
    lines_added = 0

    try:
        # Get all PLEXOS Line objects
        line_objects = db.list_objects_by_class(ClassEnum.Line)
        logger.info(f"Found {len(line_objects)} PLEXOS Line objects")

        for line_name in line_objects:
            # Get line properties for capacity
            props = db.get_object_properties(ClassEnum.Line, line_name)

            # Get connected nodes from memberships
            memberships = db.get_memberships_system(
                line_name, object_class=ClassEnum.Line
            )

            node_from = None
            node_to = None
            for membership in memberships:
                if membership.get("collection_name") == "Node From":
                    node_from = membership.get("name")
                elif membership.get("collection_name") == "Node To":
                    node_to = membership.get("name")

            # Only add line if both nodes exist in network
            if not (
                node_from
                and node_to
                and node_from in network.buses.index
                and node_to in network.buses.index
            ):
                logger.debug(
                    f"Skipping line {line_name}: missing nodes {node_from}, {node_to}"
                )
                continue

            # Extract capacity from PLEXOS properties
            s_nom = _extract_line_capacity(props, default_s_nom)

            # Add realistic electrical impedance
            r, x = _calculate_line_impedance(line_name, node_from, node_to)

            # Add PyPSA Line component
            if line_name not in network.lines.index:
                network.add(
                    "Line",
                    line_name,
                    bus0=node_from,
                    bus1=node_to,
                    s_nom=abs(s_nom),
                    r=r,
                    x=x,
                )
                lines_added += 1
                logger.debug(
                    f"Added line {line_name}: {node_from}-{node_to}, "
                    f"s_nom={s_nom:.1f}MW, r={r:.4f}, x={x:.4f}"
                )
            else:
                logger.warning(f"Line {line_name} already exists, skipping")

    except Exception:
        logger.exception("Error adding lines from PLEXOS")
        raise

    logger.info(f"Added {lines_added} transmission lines from PLEXOS model")
    return lines_added


def _extract_line_capacity(props: list, default: float) -> float:
    """Extract line capacity from PLEXOS properties."""
    # Priority order: Max Rating > Max Flow > default
    max_rating = None
    max_flow = None

    for prop in props:
        prop_name = prop.get("property", "")
        prop_value = prop.get("value")

        if prop_name == "Max Rating" and prop_value is not None:
            try:
                max_rating = float(prop_value)
            except (ValueError, TypeError):
                pass
        elif prop_name == "Max Flow" and prop_value is not None:
            try:
                max_flow = float(prop_value)
            except (ValueError, TypeError):
                pass

    # Use max rating if available, otherwise max flow, otherwise default
    if max_rating is not None:
        return max_rating
    elif max_flow is not None:
        return max_flow
    else:
        return default


def _calculate_line_impedance(
    line_name: str, node_from: str, node_to: str
) -> tuple[float, float]:
    """Calculate realistic electrical impedance for transmission line.

    Returns (resistance, reactance) based on typical transmission line characteristics.
    Uses line name and node names to estimate line characteristics.
    """
    # Base impedance values for typical transmission lines
    # These are per-unit values scaled for typical line lengths
    base_r = 0.01  # Base resistance
    base_x = 0.03  # Base reactance (X/R ≈ 3 for transmission lines)

    # Scaling factors based on line/node naming patterns
    length_factor = 1.0
    voltage_factor = 1.0

    # Estimate line length from node names (country codes, regions)
    if _is_international_line(node_from, node_to):
        length_factor = 2.0  # International lines typically longer
    elif _is_long_distance_line(line_name):
        length_factor = 1.5  # Long domestic lines

    # Estimate voltage level (affects impedance)
    if _is_high_voltage_line(line_name):
        voltage_factor = 0.5  # Higher voltage = lower impedance per MW
    elif _is_low_voltage_line(line_name):
        voltage_factor = 1.5  # Lower voltage = higher impedance per MW

    # Calculate final impedance values
    r = base_r * length_factor * voltage_factor
    x = base_x * length_factor * voltage_factor

    return r, x


def _is_international_line(node_from: str, node_to: str) -> bool:
    """Check if line connects different countries based on node names."""
    # Extract potential country codes (first 2-3 characters)
    from_country = node_from[:2] if len(node_from) >= 2 else ""
    to_country = node_to[:2] if len(node_to) >= 2 else ""

    # Common European country code patterns
    country_codes = [
        "AT",
        "BE",
        "BG",
        "CH",
        "CZ",
        "DE",
        "DK",
        "ES",
        "FI",
        "FR",
        "GB",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LU",
        "NL",
        "NO",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    ]

    return (
        from_country in country_codes
        and to_country in country_codes
        and from_country != to_country
    )


def _is_long_distance_line(line_name: str) -> bool:
    """Check if line name suggests long distance transmission."""
    line_lower = line_name.lower()
    long_distance_indicators = ["long", "inter", "trans", "corridor", "trunk"]
    return any(indicator in line_lower for indicator in long_distance_indicators)


def _is_high_voltage_line(line_name: str) -> bool:
    """Check if line name suggests high voltage (>400kV)."""
    line_lower = line_name.lower()
    # Look for voltage indicators
    hv_indicators = ["765", "500kv", "400kv", "ehv", "uhv"]
    return any(indicator in line_lower for indicator in hv_indicators)


def _is_low_voltage_line(line_name: str) -> bool:
    """Check if line name suggests lower transmission voltage (<220kV)."""
    line_lower = line_name.lower()
    # Look for lower voltage indicators
    lv_indicators = ["138", "115", "69kv", "sub", "distribution"]
    return any(indicator in line_lower for indicator in lv_indicators)


def set_line_flow_limits(
    network: Network, db: PlexosDB, timeslice_csv: str | None = None
) -> None:
    """Set time-varying flow limits for PyPSA lines based on PLEXOS data.

    This function sets s_min_pu and s_max_pu based on Min Flow/Max Flow
    properties from PLEXOS, allowing for time-varying transmission limits.

    Parameters
    ----------
    network : pypsa.Network
        Network containing lines to set limits for
    db : PlexosDB
        PLEXOS database with line properties
    timeslice_csv : str, optional
        Path to timeslice CSV for time-dependent properties

    Notes
    -----
    - Uses existing parse_lines_flow function from links.py
    - Sets s_min_pu and s_max_pu instead of p_min_pu and p_max_pu
    - Handles time-varying limits if timeslice data available
    """
    if len(network.lines) == 0:
        logger.info("No lines in network, skipping flow limit setup")
        return

    try:
        # Create a mock network object with lines as links for parsing
        class MockNetwork:
            def __init__(self, real_network):
                self.snapshots = real_network.snapshots
                # Create mock links dataframe from lines
                mock_links_data = [
                    {"name": line_name} for line_name in real_network.lines.index
                ]
                self.links = pd.DataFrame(mock_links_data).set_index("name")

        mock_net = MockNetwork(network)

        # Parse flow data using existing logic
        min_flow_df, max_flow_df, fallback_dict = parse_lines_flow(
            db, mock_net, timeslice_csv
        )

        # Apply flow limits to lines (convert from p_*_pu to s_*_pu)
        lines_with_limits = 0

        for line_name in network.lines.index:
            if line_name in fallback_dict and fallback_dict[line_name] is not None:
                # Min flow limits (typically negative or zero)
                if line_name in min_flow_df.columns:
                    min_series = min_flow_df[line_name]
                    s_min_pu = min_series / network.lines.at[line_name, "s_nom"]
                    # Clamp to reasonable values
                    s_min_pu = s_min_pu.clip(lower=-1.0, upper=0.0).fillna(0.0)
                    network.lines_t.s_min_pu[line_name] = s_min_pu

                # Max flow limits (typically positive)
                if line_name in max_flow_df.columns:
                    max_series = max_flow_df[line_name]
                    s_max_pu = max_series / network.lines.at[line_name, "s_nom"]
                    # Clamp to reasonable values
                    s_max_pu = s_max_pu.clip(lower=0.0, upper=1.0).fillna(1.0)
                    network.lines_t.s_max_pu[line_name] = s_max_pu

                lines_with_limits += 1

        logger.info(f"Set flow limits for {lines_with_limits} lines")

    except ImportError:
        logger.warning("Could not import parse_lines_flow, skipping flow limits")
    except Exception:
        logger.exception("Error setting line flow limits")


def port_lines(
    network: Network,
    db: PlexosDB,
    default_s_nom: float = 1000.0,
    timeslice_csv: str | None = None,
) -> dict[str, Any]:
    """Complete line porting process from PLEXOS to PyPSA.

    This is the main function for converting PLEXOS transmission lines.
    It combines line creation and flow limit setting in one call.

    Parameters
    ----------
    network : pypsa.Network
        Network to add lines to
    db : PlexosDB
        PLEXOS database
    default_s_nom : float, default 1000.0
        Default line capacity if none found in PLEXOS
    timeslice_csv : str, optional
        Path to timeslice CSV for time-varying limits

    Returns
    -------
    dict
        Summary statistics about ported lines

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB.from_xml("model.xml")
    >>> summary = port_lines(network, db)
    >>> print(f"Added {summary['lines_added']} lines")
    """
    print("Starting PLEXOS line porting process...")

    # Step 1: Add lines with impedance
    print("1. Adding transmission lines from PLEXOS...")
    lines_added = add_lines_from_plexos(network, db, default_s_nom)

    # Step 2: Set flow limits if lines were added
    if lines_added > 0:
        print("2. Setting line flow limits...")
        set_line_flow_limits(network, db, timeslice_csv)
    else:
        print("2. No lines added, skipping flow limits")

    # Summary
    summary = {
        "lines_added": lines_added,
        "total_lines": len(network.lines),
        "default_capacity_mw": default_s_nom,
        "impedance_method": "synthetic_realistic",
    }

    print(f"Line porting complete! Added {lines_added} lines.")
    return summary
