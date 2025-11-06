"""Visualization functions for PyPSA network analysis.

This module provides matplotlib-based plotting functions adapted from PyPSA-Explorer
patterns for static analysis of energy systems.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from plexos_to_pypsa_converter.analysis.metrics import (
    calculate_capacity_factor,
    calculate_costs,
    calculate_energy_balance,
    calculate_installed_capacity,
    calculate_optimal_capacity,
    calculate_storage_state,
    calculate_store_state,
    calculate_supply,
    calculate_transmission_flows,
)
from plexos_to_pypsa_converter.analysis.styles import (
    apply_default_style,
    assign_colors_to_carriers,
    format_axis_labels,
    format_legend,
    style_capacity_plot,
    style_cost_plot,
    style_energy_balance_plot,
)
from plexos_to_pypsa_converter.analysis.utils import (
    sample_timeseries,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _save_figure(
    fig: plt.Figure, save_path: str | Path, dpi: int = 150, bbox_inches: str = "tight"
) -> None:
    """Save figure to file.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    save_path : str | Path
        Path to save figure
    dpi : int, default 150
        Resolution in dots per inch
    bbox_inches : str, default "tight"
        Bounding box setting
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved plot: {save_path}")


# =============================================================================
# Energy Balance Visualizations
# =============================================================================


def plot_energy_balance_timeseries(
    network: pypsa.Network,
    bus_carrier: str | None = None,
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    max_points: int = 5000,
    figsize: tuple[float, float] = (14, 7),
    ax: plt.Axes | None = None,
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> plt.Axes:
    """Plot energy balance time series as stacked area chart.

    Adapted from PyPSA-Explorer's energy balance timeseries visualization.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    bus_carrier : str, optional
        Filter by bus carrier (e.g., "AC", "DC")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    max_points : int, default 5000
        Maximum time series points (will sample if exceeded)
    figsize : tuple, default (14, 7)
        Figure size if creating new figure
    ax : plt.Axes, optional
        Existing axes to plot on
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments passed to ax.stackplot

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get energy balance data
    balance = calculate_energy_balance(
        network,
        bus_carrier=bus_carrier,
        buses=buses,
        exclude_slack=exclude_slack,
        aggregate=False,
        nice_names=True,
    )

    if balance.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No energy balance data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Convert to pivot table for plotting (snapshots x carriers)
    if isinstance(balance.index, pd.MultiIndex):
        # Aggregate by snapshot and carrier
        balance_pivot = balance.groupby(level=[0, -1]).sum().unstack(fill_value=0)
    else:
        balance_pivot = balance

    # Sample if too many points
    if len(balance_pivot) > max_points:
        balance_pivot = sample_timeseries(balance_pivot, max_points)

    # Separate positive (supply) and negative (withdrawal) flows
    supply_carriers = balance_pivot.columns[balance_pivot.sum() > 0]
    withdrawal_carriers = balance_pivot.columns[balance_pivot.sum() < 0]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot supply (positive)
    if len(supply_carriers) > 0:
        supply_data = balance_pivot[supply_carriers].clip(lower=0)
        colors_dict = assign_colors_to_carriers(
            supply_carriers.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in supply_carriers]
        ax.stackplot(
            balance_pivot.index,
            supply_data.T,
            labels=supply_carriers,
            colors=colors,
            alpha=0.8,
            **kwargs,
        )

    # Plot withdrawal (negative) - flip to negative
    if len(withdrawal_carriers) > 0:
        withdrawal_data = balance_pivot[withdrawal_carriers].clip(upper=0)
        colors_dict = assign_colors_to_carriers(
            withdrawal_carriers.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in withdrawal_carriers]
        ax.stackplot(
            balance_pivot.index,
            withdrawal_data.T,
            labels=withdrawal_carriers,
            colors=colors,
            alpha=0.8,
            **kwargs,
        )

    # Apply styling
    style_energy_balance_plot(ax)
    format_axis_labels(
        ax, xlabel="Time", ylabel="Power (MW)", title="Energy Balance Over Time"
    )

    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

    return ax


def plot_energy_balance_totals(
    network: pypsa.Network,
    bus_carrier: str | None = None,
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (10, 8),
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> plt.Axes:
    """Plot aggregated energy balance as horizontal bar chart.

    Adapted from PyPSA-Explorer's energy balance totals visualization.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    bus_carrier : str, optional
        Filter by bus carrier
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (10, 8)
        Figure size if creating new figure
    ax : plt.Axes, optional
        Existing axes to plot on
    save_path : str | Path, optional
        Path to save the figure
    dpi : int, default 150
        Resolution for saved figure
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments passed to ax.barh

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get aggregated energy balance
    balance = calculate_energy_balance(
        network,
        bus_carrier=bus_carrier,
        buses=buses,
        exclude_slack=exclude_slack,
        aggregate=True,
        nice_names=True,
    )

    if balance.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No energy balance data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Separate supply and withdrawal
    supply = balance[balance > 0].sort_values(ascending=True)
    withdrawal = balance[balance < 0].sort_values(ascending=False)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Combine for plotting
    all_carriers = pd.concat([withdrawal, supply])
    colors_dict = assign_colors_to_carriers(
        all_carriers.index.tolist(),
        user_overrides=color_overrides,
        palette=color_palette,
    )
    colors = [colors_dict[c] for c in all_carriers.index]

    # Plot horizontal bars
    y_pos = np.arange(len(all_carriers))
    ax.barh(y_pos, all_carriers.values, color=colors, alpha=0.8, **kwargs)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_carriers.index)
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")

    format_axis_labels(
        ax, xlabel="Energy (MWh)", title="Total Energy Balance by Carrier"
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Save if path provided
    if save_path is not None:
        _save_figure(ax.get_figure(), save_path, dpi=dpi)

    return ax


# =============================================================================
# Capacity Visualizations
# =============================================================================


def plot_capacity_overview(
    network: pypsa.Network,
    capacity_type: str = "optimal",
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> plt.Axes:
    """Plot capacity overview by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network
    capacity_type : str, default "optimal"
        Type of capacity ("optimal" or "installed")
    groupby : str, default "carrier"
        Grouping dimension ("carrier", "component", "bus")
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (10, 6)
        Figure size if creating new figure
    ax : plt.Axes, optional
        Existing axes to plot on
    save_path : str | Path, optional
        Path to save the figure
    dpi : int, default 150
        Resolution for saved figure
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments passed to ax.bar

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get capacity data
    if capacity_type == "optimal":
        capacity = calculate_optimal_capacity(
            network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=True,
        )
    else:
        capacity = calculate_installed_capacity(
            network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=True,
        )

    if capacity.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No capacity data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort by value
    capacity = capacity.sort_values(ascending=False)

    # Get colors based on groupby
    if groupby == "carrier":
        colors_dict = assign_colors_to_carriers(
            capacity.index.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in capacity.index]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(capacity)))

    # Plot bars
    x_pos = np.arange(len(capacity))
    ax.bar(x_pos, capacity.values, color=colors, alpha=0.8, **kwargs)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(capacity.index, rotation=45, ha="right")

    style_capacity_plot(ax)
    title = f"{capacity_type.capitalize()} Capacity by {groupby.capitalize()}"
    format_axis_labels(ax, ylabel="Capacity (MW)", title=title)

    # Save if path provided
    if save_path is not None:
        _save_figure(ax.get_figure(), save_path, dpi=dpi)

    return ax


def plot_capacity_factors(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> plt.Axes:
    """Plot capacity factors by carrier/component/bus.

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
    figsize : tuple, default (10, 6)
        Figure size
    ax : plt.Axes, optional
        Existing axes
    save_path : str | Path, optional
        Path to save the figure
    dpi : int, default 150
        Resolution for saved figure
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments passed to ax.bar

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get capacity factor data
    cf = calculate_capacity_factor(
        network, groupby=groupby, buses=buses, exclude_slack=exclude_slack
    )

    if cf.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No capacity factor data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Sort by value
    cf = cf.sort_values(ascending=False)

    # Get colors
    if groupby == "carrier":
        colors_dict = assign_colors_to_carriers(
            cf.index.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in cf.index]
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(cf)))

    # Plot bars
    x_pos = np.arange(len(cf))
    ax.bar(x_pos, cf.values, color=colors, alpha=0.8, **kwargs)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cf.index, rotation=45, ha="right")
    ax.set_ylim(0, 1)

    format_axis_labels(
        ax,
        ylabel="Capacity Factor",
        title=f"Capacity Factors by {groupby.capitalize()}",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Save if path provided
    if save_path is not None:
        _save_figure(ax.get_figure(), save_path, dpi=dpi)

    return ax


# =============================================================================
# Cost Visualizations
# =============================================================================


def plot_cost_breakdown(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    dpi: int = 150,
    **kwargs,
) -> plt.Axes:
    """Plot cost breakdown (CAPEX + OPEX) by carrier/component/bus.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (10, 6)
        Figure size
    ax : plt.Axes, optional
        Existing axes
    save_path : str | Path, optional
        Path to save the figure
    dpi : int, default 150
        Resolution for saved figure
    **kwargs
        Additional arguments passed to ax.bar

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get cost data
    costs = calculate_costs(
        network,
        cost_type="total",
        groupby=groupby,
        buses=buses,
        exclude_slack=exclude_slack,
        nice_names=True,
    )

    capex = costs["capex"]
    opex = costs["opex"]

    if capex.empty and opex.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No cost data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Combine and sort by total
    total = costs["total"].sort_values(ascending=False)
    capex = capex.reindex(total.index, fill_value=0)
    opex = opex.reindex(total.index, fill_value=0)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot stacked bars
    x_pos = np.arange(len(total))
    ax.bar(x_pos, capex.values, label="CAPEX", alpha=0.8, color="#3498db", **kwargs)
    ax.bar(
        x_pos,
        opex.values,
        bottom=capex.values,
        label="OPEX",
        alpha=0.8,
        color="#e74c3c",
        **kwargs,
    )

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(total.index, rotation=45, ha="right")

    style_cost_plot(ax)
    format_axis_labels(
        ax, ylabel="Cost ($)", title=f"Cost Breakdown by {groupby.capitalize()}"
    )
    format_legend(ax, loc="upper right")

    # Save if path provided
    if save_path is not None:
        _save_figure(ax.get_figure(), save_path, dpi=dpi)

    return ax


def plot_cost_comparison(
    network: pypsa.Network,
    groupby: str = "carrier",
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (12, 6),
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot side-by-side comparison of CAPEX and OPEX.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    groupby : str, default "carrier"
        Grouping dimension
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (12, 6)
        Figure size
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes tuple
    """
    # Get cost data
    costs = calculate_costs(
        network,
        cost_type="total",
        groupby=groupby,
        buses=buses,
        exclude_slack=exclude_slack,
        nice_names=True,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot CAPEX
    capex = costs["capex"].sort_values(ascending=False)
    if not capex.empty:
        if groupby == "carrier":
            colors_dict = assign_colors_to_carriers(
                capex.index.tolist(),
                user_overrides=color_overrides,
                palette=color_palette,
            )
            colors = [colors_dict[c] for c in capex.index]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(capex)))

        x_pos = np.arange(len(capex))
        ax1.bar(x_pos, capex.values, color=colors, alpha=0.8, **kwargs)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(capex.index, rotation=45, ha="right")
        format_axis_labels(ax1, ylabel="CAPEX ($)", title="Capital Expenditure")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Plot OPEX
    opex = costs["opex"].sort_values(ascending=False)
    if not opex.empty:
        if groupby == "carrier":
            colors_dict = assign_colors_to_carriers(
                opex.index.tolist(),
                user_overrides=color_overrides,
                palette=color_palette,
            )
            colors = [colors_dict[c] for c in opex.index]
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(opex)))

        x_pos = np.arange(len(opex))
        ax2.bar(x_pos, opex.values, color=colors, alpha=0.8, **kwargs)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(opex.index, rotation=45, ha="right")
        format_axis_labels(ax2, ylabel="OPEX ($)", title="Operational Expenditure")
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# Storage Visualizations
# =============================================================================


def plot_storage_state_of_charge(
    network: pypsa.Network,
    storage_units: list[str] | None = None,
    buses: list[str] | None = None,
    max_points: int = 5000,
    figsize: tuple[float, float] = (14, 6),
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot storage state of charge time series.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    storage_units : list[str], optional
        Specific storage units to plot
    buses : list[str], optional
        Filter by buses
    max_points : int, default 5000
        Maximum time series points
    figsize : tuple, default (14, 6)
        Figure size
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional arguments passed to ax.plot

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get storage state data
    soc = calculate_storage_state(network, storage_units=storage_units, buses=buses)
    store_e = calculate_store_state(network, buses=buses)

    if soc.empty and store_e.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No storage data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Combine storage units and stores
    all_storage = (
        pd.concat([soc, store_e], axis=1)
        if not soc.empty and not store_e.empty
        else (soc if not soc.empty else store_e)
    )

    # Sample if needed
    if len(all_storage) > max_points:
        all_storage = sample_timeseries(all_storage, max_points)

    # Plot each storage unit/store
    for col in all_storage.columns:
        ax.plot(all_storage.index, all_storage[col], label=col, **kwargs)

    # Formatting
    format_axis_labels(
        ax,
        xlabel="Time",
        ylabel="State of Charge (MWh)",
        title="Storage State of Charge",
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=45)

    if len(all_storage.columns) <= 10:
        format_legend(ax, loc="best")

    return ax


# =============================================================================
# Transmission Visualizations
# =============================================================================


def plot_transmission_flows(
    network: pypsa.Network,
    buses: list[str] | None = None,
    top_n: int = 20,
    figsize: tuple[float, float] = (10, 8),
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot transmission flows (aggregated by line/link).

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    buses : list[str], optional
        Filter by buses
    top_n : int, default 20
        Show top N lines/links by flow
    figsize : tuple, default (10, 8)
        Figure size
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional arguments passed to ax.barh

    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    # Get transmission flows
    flows = calculate_transmission_flows(network, buses=buses, aggregate=True)

    if flows.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No transmission flow data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    # Sort and take top N
    flows = flows.sort_values(ascending=True).tail(top_n)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal bars
    y_pos = np.arange(len(flows))
    ax.barh(y_pos, flows.values, alpha=0.8, color="#34495e", **kwargs)

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(flows.index)

    format_axis_labels(
        ax, xlabel="Total Flow (MWh)", title=f"Top {top_n} Transmission Flows"
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    return ax


# =============================================================================
# Combined/Dashboard Plots
# =============================================================================


def plot_generation_mix(
    network: pypsa.Network,
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (12, 5),
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
    **kwargs,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot generation mix: supply totals and capacity.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (12, 5)
        Figure size
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers
    **kwargs
        Additional arguments

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Figure and axes tuple
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot energy supply
    supply = calculate_supply(
        network,
        groupby="carrier",
        buses=buses,
        exclude_slack=exclude_slack,
        nice_names=True,
    ).sort_values(ascending=False)

    if not supply.empty:
        colors_dict = assign_colors_to_carriers(
            supply.index.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in supply.index]
        x_pos = np.arange(len(supply))
        ax1.bar(x_pos, supply.values, color=colors, alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(supply.index, rotation=45, ha="right")
        format_axis_labels(ax1, ylabel="Energy (MWh)", title="Energy Supply by Carrier")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Plot capacity
    capacity = calculate_optimal_capacity(
        network,
        groupby="carrier",
        buses=buses,
        exclude_slack=exclude_slack,
        nice_names=True,
    ).sort_values(ascending=False)

    if not capacity.empty:
        colors_dict = assign_colors_to_carriers(
            capacity.index.tolist(),
            user_overrides=color_overrides,
            palette=color_palette,
        )
        colors = [colors_dict[c] for c in capacity.index]
        x_pos = np.arange(len(capacity))
        ax2.bar(x_pos, capacity.values, color=colors, alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(capacity.index, rotation=45, ha="right")
        format_axis_labels(
            ax2, ylabel="Capacity (MW)", title="Optimal Capacity by Carrier"
        )
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return fig, (ax1, ax2)


def create_summary_dashboard(
    network: pypsa.Network,
    buses: list[str] | None = None,
    exclude_slack: bool = True,
    figsize: tuple[float, float] = (16, 10),
    save_path: str | Path | None = None,
    dpi: int = 150,
    color_overrides: dict[str, str] | None = None,
    color_palette: str = "retro_metro",
) -> plt.Figure:
    """Create comprehensive summary dashboard with multiple plots.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network
    buses : list[str], optional
        Filter by specific buses
    exclude_slack : bool, default True
        Exclude slack generators
    figsize : tuple, default (16, 10)
        Figure size
    save_path : str | Path, optional
        Path to save the figure
    dpi : int, default 150
        Resolution for saved figure
    color_overrides : dict, optional
        Manual color assignments for specific carriers
    color_palette : str, default "retro_metro"
        Fallback palette for unknown carriers

    Returns
    -------
    plt.Figure
        Figure with multiple subplots
    """
    apply_default_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Energy balance totals
    ax1 = fig.add_subplot(gs[0, 0])
    plot_energy_balance_totals(
        network,
        buses=buses,
        exclude_slack=exclude_slack,
        ax=ax1,
        color_overrides=color_overrides,
        color_palette=color_palette,
    )

    # Capacity overview
    ax2 = fig.add_subplot(gs[0, 1])
    plot_capacity_overview(
        network,
        buses=buses,
        exclude_slack=exclude_slack,
        ax=ax2,
        color_overrides=color_overrides,
        color_palette=color_palette,
    )

    # Cost breakdown
    ax3 = fig.add_subplot(gs[1, 0])
    plot_cost_breakdown(network, buses=buses, exclude_slack=exclude_slack, ax=ax3)

    # Capacity factors
    ax4 = fig.add_subplot(gs[1, 1])
    plot_capacity_factors(
        network,
        buses=buses,
        exclude_slack=exclude_slack,
        ax=ax4,
        color_overrides=color_overrides,
        color_palette=color_palette,
    )

    fig.suptitle("Network Analysis Summary", fontsize=16, fontweight="bold", y=0.995)

    # Save if path provided
    if save_path is not None:
        _save_figure(fig, save_path, dpi=dpi)

    return fig
