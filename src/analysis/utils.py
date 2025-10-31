"""Utility functions for PyPSA network analysis.

This module provides helper functions for data filtering, aggregation, and formatting,
adapted from PyPSA-Explorer patterns for flexible spatial resolution support.
"""

from typing import Any

import pandas as pd
import pypsa

# =============================================================================
# Slack Generator Identification
# =============================================================================


def identify_slack_generators(network: pypsa.Network) -> list[str]:
    """Identify slack generators (load shedding/spillage) in the network.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    list[str]
        List of slack generator names
    """
    if "carrier" not in network.generators.columns:
        return []

    slack_carriers = ["load spillage", "load shedding"]
    return network.generators[
        network.generators["carrier"].isin(slack_carriers)
    ].index.tolist()


def filter_slack_generators(
    data: pd.Series | pd.DataFrame,
    slack_generators: list[str],
    level: str | None = "carrier",
) -> pd.Series | pd.DataFrame:
    """Filter out slack generators from data.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Data to filter (index should contain carrier information)
    slack_generators : list[str]
        List of slack generator names
    level : str | None, default "carrier"
        Level name for MultiIndex filtering, None for simple Index

    Returns
    -------
    pd.Series | pd.DataFrame
        Filtered data without slack generators
    """
    if data.empty:
        return data

    slack_carriers = [
        "load spillage",
        "load shedding",
        "Load spillage",
        "Load shedding",
    ]

    # Handle MultiIndex
    if isinstance(data.index, pd.MultiIndex):
        if level and level in data.index.names:
            mask = ~data.index.isin(slack_carriers, level=level)
        else:
            # Try last level (typically carrier)
            mask = ~data.index.get_level_values(-1).isin(slack_carriers)
        return data[mask]
    else:
        # Simple Index
        mask = ~data.index.isin(slack_carriers)
        return data[mask]


# =============================================================================
# Bus Filtering and Spatial Analysis
# =============================================================================


def filter_by_buses(
    data: pd.DataFrame | pd.Series,
    buses: list[str],
    network: pypsa.Network,
    component_type: str | None = None,
) -> pd.DataFrame | pd.Series:
    """Filter data by bus list (spatial subset).

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to filter
    buses : list[str]
        List of bus names to include
    network : pypsa.Network
        PyPSA network object
    component_type : str, optional
        Component type to filter (e.g., "Generator", "Load")

    Returns
    -------
    pd.DataFrame | pd.Series
        Filtered data for specified buses
    """
    if data.empty or not buses:
        return data

    # If data has MultiIndex with 'bus' level
    if isinstance(data.index, pd.MultiIndex) and "bus" in data.index.names:
        mask = data.index.isin(buses, level="bus")
        return data[mask]

    # If component_type specified, filter by component bus assignment
    if component_type:
        try:
            component_df = getattr(network, f"{component_type.lower()}s")
            if "bus" in component_df.columns:
                valid_components = component_df[component_df["bus"].isin(buses)].index
                if isinstance(data.index, pd.MultiIndex):
                    # Filter by component name in index
                    mask = data.index.get_level_values(0).isin(valid_components)
                else:
                    mask = data.index.isin(valid_components)
                return data[mask]
        except AttributeError:
            pass

    return data


def get_bus_carriers(network: pypsa.Network) -> list[str]:
    """Get unique bus carriers from network.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    list[str]
        List of unique bus carriers
    """
    if "carrier" in network.buses.columns:
        return sorted(network.buses["carrier"].dropna().unique().tolist())
    return []


def get_buses_by_carrier(network: pypsa.Network, carrier: str) -> list[str]:
    """Get buses of a specific carrier type.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object
    carrier : str
        Bus carrier (e.g., "AC", "DC", "gas")

    Returns
    -------
    list[str]
        List of bus names with specified carrier
    """
    if "carrier" in network.buses.columns:
        return network.buses[network.buses["carrier"] == carrier].index.tolist()
    return network.buses.index.tolist()


def detect_spatial_resolution(network: pypsa.Network) -> str:
    """Detect the spatial resolution of the network.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    str
        Spatial resolution: "single", "zonal", or "nodal"
    """
    n_buses = len(network.buses)

    if n_buses <= 2:
        return "single"
    elif n_buses <= 20:
        return "zonal"
    else:
        return "nodal"


# =============================================================================
# Data Aggregation
# =============================================================================


def aggregate_by_carrier(
    data: pd.DataFrame | pd.Series, level: str = "carrier"
) -> pd.Series:
    """Aggregate MultiIndex data by carrier.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data with MultiIndex
    level : str, default "carrier"
        Level name to group by

    Returns
    -------
    pd.Series
        Aggregated data by carrier
    """
    if isinstance(data, pd.Series):
        if isinstance(data.index, pd.MultiIndex):
            if level in data.index.names:
                return data.groupby(level=level).sum()
            else:
                # Try to aggregate by last level
                return data.groupby(level=-1).sum()
        return data
    elif isinstance(data, pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            if level in data.index.names:
                return data.groupby(level=level).sum().sum(axis=1)
            else:
                return data.groupby(level=-1).sum().sum(axis=1)
        return data.sum(axis=1)
    return data


def aggregate_by_component(
    data: pd.DataFrame | pd.Series, level: str = "component"
) -> pd.Series:
    """Aggregate MultiIndex data by component.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data with MultiIndex
    level : str, default "component"
        Level name to group by

    Returns
    -------
    pd.Series
        Aggregated data by component
    """
    return aggregate_by_carrier(data, level=level)


def aggregate_by_bus(
    data: pd.DataFrame | pd.Series,
    network: pypsa.Network,
    component_type: str | None = None,
) -> pd.Series:
    """Aggregate data by bus for spatial analysis.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        Data to aggregate
    network : pypsa.Network
        PyPSA network object
    component_type : str, optional
        Component type (e.g., "Generator", "Load")

    Returns
    -------
    pd.Series
        Aggregated data by bus
    """
    if data.empty:
        return pd.Series(dtype=float)

    # If data has bus in MultiIndex
    if isinstance(data.index, pd.MultiIndex) and "bus" in data.index.names:
        return data.groupby(level="bus").sum()

    # Otherwise, need to map components to buses
    if component_type:
        try:
            component_df = getattr(network, f"{component_type.lower()}s")
            if "bus" in component_df.columns:
                bus_mapping = component_df["bus"].to_dict()

                if isinstance(data, pd.Series):
                    result = pd.Series(dtype=float)
                    for component, value in data.items():
                        if component in bus_mapping:
                            bus = bus_mapping[component]
                            result[bus] = result.get(bus, 0) + value
                    return result
        except AttributeError:
            pass

    return data


# =============================================================================
# Carrier Name Formatting
# =============================================================================


def format_carrier_name(carrier: str) -> str:
    """Format carrier name for display (nice_names equivalent).

    Parameters
    ----------
    carrier : str
        Carrier name

    Returns
    -------
    str
        Formatted carrier name
    """
    # Capitalize first letter of each word
    words = carrier.split()
    formatted = " ".join(word.capitalize() for word in words)

    # Special cases
    replacements = {
        "Pv": "PV",
        "Ac": "AC",
        "Dc": "DC",
        "Phs": "PHS",
        "Ocgt": "OCGT",
        "Ccgt": "CCGT",
        "H2": "H₂",  # Hydrogen with subscript
        "Co2": "CO₂",  # CO2 with subscript
    }

    for old, new in replacements.items():
        formatted = formatted.replace(old, new)

    return formatted


def extract_carrier_from_index(index_value: Any) -> str:
    """Extract carrier name from index value (handles tuples from MultiIndex).

    Parameters
    ----------
    index_value : Any
        Index value (string or tuple)

    Returns
    -------
    str
        Carrier name
    """
    if isinstance(index_value, str):
        return index_value
    elif isinstance(index_value, tuple):
        # Assume carrier is first element or last element
        # Try first
        if isinstance(index_value[0], str):
            return index_value[0]
        # Try last
        if isinstance(index_value[-1], str):
            return index_value[-1]
        return str(index_value)
    else:
        return str(index_value)


# =============================================================================
# Time Series Utilities
# =============================================================================


def sample_timeseries(data: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
    """Sample large time series for plotting (from PyPSA-Explorer pattern).

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    max_points : int, default 5000
        Maximum number of points to keep

    Returns
    -------
    pd.DataFrame
        Sampled data
    """
    if len(data) <= max_points:
        return data

    # Uniform sampling
    step = len(data) // max_points
    return data.iloc[::step]


def get_snapshot_range(
    network: pypsa.Network,
    start_snapshot: int | str | None = None,
    end_snapshot: int | str | None = None,
) -> pd.DatetimeIndex | pd.Index:
    """Get subset of snapshots from network.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object
    start_snapshot : int | str, optional
        Starting snapshot index or timestamp
    end_snapshot : int | str, optional
        Ending snapshot index or timestamp

    Returns
    -------
    pd.DatetimeIndex | pd.Index
        Subset of snapshots
    """
    snapshots = network.snapshots

    if start_snapshot is None and end_snapshot is None:
        return snapshots

    if isinstance(snapshots, pd.MultiIndex):
        # Multi-period case - just use all for now
        return snapshots

    if start_snapshot is not None:
        if isinstance(start_snapshot, int):
            snapshots = snapshots[start_snapshot:]
        else:
            snapshots = snapshots[snapshots >= start_snapshot]

    if end_snapshot is not None:
        if isinstance(end_snapshot, int):
            snapshots = snapshots[:end_snapshot]
        else:
            snapshots = snapshots[snapshots <= end_snapshot]

    return snapshots


# =============================================================================
# Data Validation and Checking
# =============================================================================


def has_time_series_data(network: pypsa.Network) -> bool:
    """Check if network has time series dispatch data.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    bool
        True if network has time series data
    """
    if not hasattr(network, "generators_t"):
        return False

    if not hasattr(network.generators_t, "p"):
        return False

    return not network.generators_t.p.empty


def has_storage_units(network: pypsa.Network) -> bool:
    """Check if network has storage units.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    bool
        True if network has storage units
    """
    return len(network.storage_units) > 0


def has_stores(network: pypsa.Network) -> bool:
    """Check if network has stores.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    bool
        True if network has stores
    """
    return len(network.stores) > 0


def has_links(network: pypsa.Network) -> bool:
    """Check if network has links.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    bool
        True if network has links
    """
    return len(network.links) > 0


def is_multi_period(network: pypsa.Network) -> bool:
    """Check if network has multiple investment periods.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network object

    Returns
    -------
    bool
        True if network has investment periods
    """
    return (
        hasattr(network, "investment_periods")
        and network.investment_periods is not None
        and len(network.investment_periods) > 0
    )


# =============================================================================
# Query Building (PyPSA-Explorer pattern)
# =============================================================================


def build_query_string(filters: dict[str, Any]) -> str:
    """Build pandas query string from filter dictionary.

    Parameters
    ----------
    filters : dict
        Dictionary of column: value filters

    Returns
    -------
    str
        Query string for DataFrame.query()
    """
    conditions = []
    for col, value in filters.items():
        if isinstance(value, list | tuple):
            # Multiple values - use isin
            values_str = ", ".join(f"'{v}'" for v in value)
            conditions.append(f"{col} in [{values_str}]")
        elif isinstance(value, str):
            conditions.append(f"{col} == '{value}'")
        else:
            conditions.append(f"{col} == {value}")

    return " & ".join(conditions) if conditions else ""


def filter_dataframe_by_query(
    df: pd.DataFrame, query: str | None = None
) -> pd.DataFrame:
    """Filter DataFrame using query string.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to filter
    query : str, optional
        Query string

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if query and query.strip():
        try:
            return df.query(query)
        except Exception:
            # If query fails, return original
            return df
    return df
