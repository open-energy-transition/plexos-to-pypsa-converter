"""Data extraction and metric calculation for PyPSA networks.

This module provides functions to extract metrics from PyPSA networks using the
statistics API, adapted from PyPSA-Explorer patterns for static analysis.
"""

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.analysis.utils import (
    filter_by_buses,
    filter_slack_generators,
    identify_slack_generators,
)

# =============================================================================
# Energy Balance Metrics
# =============================================================================


def calculate_energy_balance(
    network: pypsa.Network,
    bus_carrier: str | None = None,
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    aggregate: bool = False,
    nice_names: bool = True,
) -> pd.DataFrame | pd.Series:
    """Calculate energy balance from network.

    Extracts energy balance using PyPSA's statistics.energy_balance accessor.
    This shows energy flows by component and carrier over time.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    bus_carrier : str, optional
        Filter by bus carrier (e.g., "AC", "DC", "gas")
    buses : list[str], optional
        Filter by specific buses (spatial subset)
    exclude_slack : bool, default True
        Exclude load shedding/spillage slack generators
    aggregate : bool, default False
        If True, aggregate over all snapshots
    nice_names : bool, default True
        Use readable carrier names

    Returns
    -------
    pd.DataFrame | pd.Series
        Energy balance data
        - If aggregate=False: DataFrame with MultiIndex (snapshot, component, carrier)
        - If aggregate=True: Series grouped by carrier
    """
    try:
        # Get energy balance from PyPSA statistics
        result = network.statistics.energy_balance(
            bus_carrier=bus_carrier, nice_names=nice_names
        )

        # Filter by buses if specified
        if buses:
            result = filter_by_buses(result, buses, network)

        # Filter slack generators if requested
        if exclude_slack:
            slack_gens = identify_slack_generators(network)
            if slack_gens:
                result = filter_slack_generators(result, slack_gens, level="carrier")

        # Aggregate if requested
        if aggregate and isinstance(result, pd.DataFrame):
            # Sum over all snapshots and components
            if isinstance(result.index, pd.MultiIndex):
                result = result.groupby(level="carrier").sum()
            else:
                result = result.sum()

    except AttributeError:
        # Fallback if energy_balance not available
        return pd.DataFrame()
    else:
        return result


def calculate_supply(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate energy supply (generation) by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        Energy supply (MWh) grouped by specified dimension
    """
    result = network.statistics.supply(groupby=groupby, nice_names=nice_names)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network, component_type="Generator")

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


def calculate_withdrawal(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate energy withdrawal (consumption) by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        Energy withdrawal (MWh) grouped by specified dimension
    """
    result = network.statistics.withdrawal(groupby=groupby, nice_names=nice_names)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network, component_type="Load")

    return result


# =============================================================================
# Capacity Metrics
# =============================================================================


def calculate_optimal_capacity(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate optimal capacity by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        Optimal capacity (MW) grouped by specified dimension
    """
    result = network.statistics.optimal_capacity(groupby=groupby, nice_names=nice_names)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network, component_type="Generator")

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


def calculate_installed_capacity(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate installed capacity by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        Installed capacity (MW) grouped by specified dimension
    """
    result = network.statistics.installed_capacity(
        groupby=groupby, nice_names=nice_names
    )

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network, component_type="Generator")

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


def calculate_capacity_factor(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
) -> pd.Series:
    """Calculate capacity factors by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators

    Returns
    -------
    pd.Series
        Capacity factor (0-1) grouped by specified dimension
    """
    result = network.statistics.capacity_factor(groupby=groupby)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network, component_type="Generator")

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


# =============================================================================
# Cost Metrics
# =============================================================================


def calculate_costs(
    network: pypsa.Network,
    cost_type: str = "total",
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series | dict[str, pd.Series]:
    """Calculate costs by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    cost_type : str, default "total"
        Cost type: "capex", "opex", or "total"
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series | dict[str, pd.Series]
        If cost_type is "capex" or "opex": Returns Series
        If cost_type is "total": Returns dict with "capex" and "opex" keys
    """
    if cost_type == "total":
        capex = calculate_capex(network, groupby, buses, exclude_slack, nice_names)
        opex = calculate_opex(network, groupby, buses, exclude_slack, nice_names)
        return {"capex": capex, "opex": opex, "total": capex + opex}
    elif cost_type == "capex":
        return calculate_capex(network, groupby, buses, exclude_slack, nice_names)
    elif cost_type == "opex":
        return calculate_opex(network, groupby, buses, exclude_slack, nice_names)
    else:
        msg = f"Invalid cost_type: {cost_type}"
        raise ValueError(msg)


def calculate_capex(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate capital expenditure by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        CAPEX grouped by specified dimension
    """
    result = network.statistics.capex(groupby=groupby, nice_names=nice_names)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network)

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


def calculate_opex(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    nice_names: bool = True,
) -> pd.Series:
    """Calculate operational expenditure by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    nice_names : bool, default True
        Use readable names

    Returns
    -------
    pd.Series
        OPEX grouped by specified dimension
    """
    result = network.statistics.opex(groupby=groupby, nice_names=nice_names)

    # Filter by buses if specified
    if buses:
        result = filter_by_buses(result, buses, network)

    # Filter slack if requested
    if exclude_slack and groupby == "carrier":
        slack_gens = identify_slack_generators(network)
        if slack_gens:
            result = filter_slack_generators(result, slack_gens, level="carrier")

    return result


# =============================================================================
# Storage Metrics
# =============================================================================


def calculate_storage_state(
    network: pypsa.Network,
    storage_units: list[str] | None = None,
    buses: list[str] | None = None,
) -> pd.DataFrame:
    """Extract storage state of charge time series.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    storage_units : list[str], optional
        Specific storage units to include
    buses : list[str], optional
        Filter by buses connected to storage

    Returns
    -------
    pd.DataFrame
        Time series of storage state of charge (MWh)
    """
    if not hasattr(network, "storage_units_t") or not hasattr(
        network.storage_units_t, "state_of_charge"
    ):
        return pd.DataFrame()

    soc = network.storage_units_t.state_of_charge

    # Filter by storage units
    if storage_units:
        soc = soc[[s for s in storage_units if s in soc.columns]]

    # Filter by buses
    if buses:
        storage_buses = network.storage_units[
            network.storage_units["bus"].isin(buses)
        ].index
        soc = soc[[s for s in storage_buses if s in soc.columns]]

    return soc


def calculate_store_state(
    network: pypsa.Network,
    stores: list[str] | None = None,
    buses: list[str] | None = None,
) -> pd.DataFrame:
    """Extract store energy level time series.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    stores : list[str], optional
        Specific stores to include
    buses : list[str], optional
        Filter by buses connected to stores

    Returns
    -------
    pd.DataFrame
        Time series of store energy levels (MWh)
    """
    if not hasattr(network, "stores_t") or not hasattr(network.stores_t, "e"):
        return pd.DataFrame()

    store_e = network.stores_t.e

    # Filter by stores
    if stores:
        store_e = store_e[[s for s in stores if s in store_e.columns]]

    # Filter by buses
    if buses:
        store_buses = network.stores[network.stores["bus"].isin(buses)].index
        store_e = store_e[[s for s in store_buses if s in store_e.columns]]

    return store_e


# =============================================================================
# Transmission/Link Metrics
# =============================================================================


def calculate_transmission_flows(
    network: pypsa.Network, buses: list[str] | None = None, aggregate: bool = True
) -> pd.DataFrame | pd.Series:
    """Extract transmission line and link flows.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    buses : list[str], optional
        Filter by buses (includes lines/links connected to these buses)
    aggregate : bool, default True
        If True, return total flow per line/link; if False, return time series

    Returns
    -------
    pd.DataFrame | pd.Series
        Transmission flows (MW or MWh if aggregated)
    """
    flows = pd.DataFrame()

    # Get line flows
    if hasattr(network, "lines_t") and hasattr(network.lines_t, "p0"):
        line_flows = network.lines_t.p0

        # Filter by buses
        if buses:
            lines = network.lines[
                network.lines["bus0"].isin(buses) | network.lines["bus1"].isin(buses)
            ].index
            line_flows = line_flows[[l for l in lines if l in line_flows.columns]]

        if aggregate:
            flows["Lines"] = line_flows.abs().sum()
        else:
            flows = line_flows

    # Get link flows
    if hasattr(network, "links_t") and hasattr(network.links_t, "p0"):
        link_flows = network.links_t.p0

        # Filter by buses
        if buses:
            links = network.links[
                network.links["bus0"].isin(buses) | network.links["bus1"].isin(buses)
            ].index
            link_flows = link_flows[[l for l in links if l in link_flows.columns]]

        if aggregate:
            if flows.empty:
                flows["Links"] = link_flows.abs().sum()
            else:
                flows = pd.concat(
                    [flows, pd.DataFrame({"Links": link_flows.abs().sum()})]
                )
        elif flows.empty:
            flows = link_flows
        else:
            flows = pd.concat([flows, link_flows], axis=1)

    return flows.sum(axis=1) if aggregate and isinstance(flows, pd.DataFrame) else flows


# =============================================================================
# Curtailment Metrics
# =============================================================================


def calculate_curtailment(
    network: pypsa.Network,
    carriers: list[str] | None = None,
    buses: list[str] | None = None,
) -> pd.Series:
    """Calculate curtailed energy by carrier.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    carriers : list[str], optional
        Specific carriers (default: ["wind", "solar"])
    buses : list[str], optional
        Filter by specific buses

    Returns
    -------
    pd.Series
        Curtailed energy (MWh) by carrier
    """
    if carriers is None:
        carriers = ["wind", "solar"]

    try:
        result = network.statistics.curtailment(carrier=carriers)

        # Filter by buses if specified
        if buses:
            result = filter_by_buses(result, buses, network, component_type="Generator")

    except (AttributeError, KeyError):
        return pd.Series(dtype=float)
    else:
        return result
