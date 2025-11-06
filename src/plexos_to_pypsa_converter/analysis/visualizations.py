"""Visualization and plotting functions for PyPSA network analysis.

This module provides comprehensive plotting capabilities for energy system analysis,
including capacity mix, generation dispatch, cost breakdowns, and validation dashboards.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plexos_to_pypsa_converter.analysis.statistics import NetworkStatistics

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10

# Standard color scheme for carriers
CARRIER_COLORS = {
    "wind": "#74c9e6",
    "solar": "#ffdd00",
    "hydro": "#2980b9",
    "gas": "#c0504e",
    "coal": "#8b4513",
    "nuclear": "#9b59b6",
    "oil": "#34495e",
    "biomass": "#27ae60",
    "battery": "#e74c3c",
    "other": "#95a5a6",
}


def _extract_carrier_name(carrier) -> str:
    """Extract carrier name from string or tuple index.

    PyPSA statistics can return MultiIndex with tuples (e.g., (carrier, bus))
    or simple string indices. This helper safely extracts the carrier name.

    Parameters
    ----------
    carrier : str or tuple
        Carrier identifier from pandas index/columns

    Returns
    -------
    str
        Lowercase carrier name for color mapping
    """
    if isinstance(carrier, str):
        return carrier.lower()
    elif isinstance(carrier, tuple):
        # Extract first element (typically the carrier) from tuple
        return carrier[0].lower() if carrier else "other"
    else:
        # Fallback for any other type
        return str(carrier).lower()


def plot_capacity_mix(
    stats: NetworkStatistics, save_path: Path | None = None, exclude_slack: bool = True
) -> tuple:
    """Plot installed capacity by carrier as bar chart.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    exclude_slack : bool, default True
        If True, exclude load shedding/spillage slack generators

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    capacity = stats.installed_capacity(exclude_slack=exclude_slack)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get colors for carriers
    colors = [
        CARRIER_COLORS.get(_extract_carrier_name(c), CARRIER_COLORS["other"])
        for c in capacity.index
    ]

    capacity.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Installed Capacity (MW)")
    ax.set_xlabel("")
    ax.set_title("Installed Capacity by Technology", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_generation_mix(
    stats: NetworkStatistics, save_path: Path | None = None, exclude_slack: bool = True
) -> tuple:
    """Plot energy generation by carrier as bar chart.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    exclude_slack : bool, default True
        If True, exclude load shedding/spillage slack generators

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    generation = stats.energy_supply(exclude_slack=exclude_slack)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        CARRIER_COLORS.get(_extract_carrier_name(c), CARRIER_COLORS["other"])
        for c in generation.index
    ]

    generation.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Energy Generation (MWh)")
    ax.set_xlabel("")
    ax.set_title("Energy Generation by Technology", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_cost_breakdown(
    stats: NetworkStatistics, save_path: Path | None = None, exclude_slack: bool = True
) -> tuple:
    """Plot CAPEX and OPEX by carrier as stacked bar chart.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    exclude_slack : bool, default True
        If True, exclude load shedding/spillage slack generators

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    capex = stats.capex(exclude_slack=exclude_slack)
    opex = stats.opex(exclude_slack=exclude_slack)

    # Align indices
    all_carriers = list(set(capex.index) | set(opex.index))
    capex = capex.reindex(all_carriers, fill_value=0)
    opex = opex.reindex(all_carriers, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_carriers))
    width = 0.35

    ax.bar(
        x - width / 2, capex, width, label="CAPEX", color="steelblue", edgecolor="black"
    )
    ax.bar(x + width / 2, opex, width, label="OPEX", color="coral", edgecolor="black")

    ax.set_ylabel("Cost (Currency Units)")
    ax.set_title("System Cost Breakdown by Technology", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_carriers, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_generation_stack(
    stats: NetworkStatistics, save_path: Path | None = None, exclude_slack: bool = True
) -> tuple:
    """Plot generation dispatch over time as stacked area chart.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    exclude_slack : bool, default True
        If True, exclude load shedding/spillage slack generators

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    # Get hourly generation by carrier
    network = stats.network
    if not hasattr(network, "generators_t") or not hasattr(network.generators_t, "p"):
        msg = "No generation dispatch data found"
        raise ValueError(msg)

    # Filter slack generators if requested
    if exclude_slack:
        slack_gens = stats._get_slack_generators()
        gen_data = network.generators_t.p.drop(columns=slack_gens, errors="ignore")
        gen_carriers = network.generators.drop(slack_gens, errors="ignore")
    else:
        gen_data = network.generators_t.p
        gen_carriers = network.generators

    gen_by_carrier = gen_data.groupby(gen_carriers.carrier, axis=1).sum()

    fig, ax = plt.subplots(figsize=(16, 8))

    # Get colors for each carrier
    colors = [
        CARRIER_COLORS.get(_extract_carrier_name(c), CARRIER_COLORS["other"])
        for c in gen_by_carrier.columns
    ]

    gen_by_carrier.plot.area(ax=ax, color=colors, alpha=0.8, linewidth=0)
    ax.set_ylabel("Generation (MW)", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title("Generation Dispatch Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_lcoe_comparison(
    stats: NetworkStatistics, save_path: Path | None = None, exclude_slack: bool = True
) -> tuple:
    """Plot Levelized Cost of Energy by carrier.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    exclude_slack : bool, default True
        If True, exclude load shedding/spillage slack generators

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    lcoe = stats.lcoe(exclude_slack=exclude_slack)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        CARRIER_COLORS.get(_extract_carrier_name(c), CARRIER_COLORS["other"])
        for c in lcoe.index
    ]

    lcoe.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("LCOE (Currency/MWh)")
    ax.set_xlabel("")
    ax.set_title(
        "Levelized Cost of Energy by Technology", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_capacity_evolution(
    stats: NetworkStatistics, save_path: Path | None = None
) -> tuple:
    """Plot capacity evolution across investment periods (if applicable).

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects or None if no periods
    """
    if not stats.has_periods:
        print("Network has no investment periods")
        return None

    capacity_by_period = stats.capacity_by_period()

    fig, ax = plt.subplots(figsize=(12, 6))

    for carrier in capacity_by_period.columns:
        color = CARRIER_COLORS.get(
            _extract_carrier_name(carrier), CARRIER_COLORS["other"]
        )
        ax.plot(
            capacity_by_period.index,
            capacity_by_period[carrier],
            marker="o",
            label=carrier,
            color=color,
            linewidth=2,
        )

    ax.set_xlabel("Investment Period", fontsize=12)
    ax.set_ylabel("Installed Capacity (MW)", fontsize=12)
    ax.set_title(
        "Capacity Evolution Across Investment Periods",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_curtailment(stats: NetworkStatistics, save_path: Path | None = None) -> tuple:
    """Plot VRE curtailment by technology.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    curtailment = stats.curtailment()

    if curtailment.empty or curtailment.sum() == 0:
        print("No curtailment data available")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        CARRIER_COLORS.get(_extract_carrier_name(c), CARRIER_COLORS["other"])
        for c in curtailment.index
    ]

    curtailment.plot.bar(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Curtailed Energy (MWh)")
    ax.set_xlabel("")
    ax.set_title("VRE Curtailment by Technology", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_validation_dashboard(
    validation_results: dict, save_path: Path | None = None
) -> tuple:
    """Plot validation results as a dashboard grid.

    Parameters
    ----------
    validation_results : dict
        Results from NetworkValidator.run_all_validations()
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    # Extract checks (exclude 'overall')
    checks = [k for k in validation_results.keys() if k != "overall"]

    # Create grid of status indicators
    y_pos = 0.9
    for check in checks:
        result = validation_results[check]
        status = result.get("status", "UNKNOWN")

        # Status icon
        if status == "PASS":
            icon = "✅"
            color = "green"
        elif status == "WARNING":
            icon = "⚠️"
            color = "orange"
        else:
            icon = "❌"
            color = "red"

        # Draw status
        check_name = check.replace("_", " ").title()
        ax.text(0.05, y_pos, f"{icon} {check_name}", fontsize=12, color=color)

        # Add details
        if "violations_count" in result:
            ax.text(
                0.7,
                y_pos,
                f"{result['violations_count']} violations",
                fontsize=10,
                color="gray",
            )
        elif "warnings_count" in result:
            ax.text(
                0.7,
                y_pos,
                f"{result['warnings_count']} warnings",
                fontsize=10,
                color="gray",
            )

        y_pos -= 0.12

    # Overall status
    overall_status = validation_results["overall"]["status"]
    overall_color = "green" if overall_status == "PASS" else "red"
    ax.text(
        0.5,
        0.05,
        f"Overall Status: {overall_status}",
        fontsize=14,
        fontweight="bold",
        ha="center",
        color=overall_color,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_unserved_energy(
    stats: NetworkStatistics, save_path: Path | None = None
) -> tuple | None:
    """Plot load shedding and load spillage bar chart.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    tuple or None
        (fig, ax) matplotlib figure and axes objects, or None if no slack
    """
    slack_data = stats.unserved_energy()

    if slack_data["load_shedding_mwh"] == 0 and slack_data["load_spillage_mwh"] == 0:
        print("No load shedding or spillage detected")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = [
        "Load Shedding\n(Unserved Energy)",
        "Load Spillage\n(Excess Generation)",
    ]
    values = [slack_data["load_shedding_mwh"], slack_data["load_spillage_mwh"]]
    colors = ["#dd2e23", "#df8e23"]  # Red for shedding, orange for spillage

    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.5)

    # Add percentage labels on bars
    for bar, pct in zip(
        bars, [slack_data["unserved_pct"], slack_data["spillage_pct"]], strict=False
    ):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{pct:.2f}%",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_ylabel("Energy (MWh)", fontsize=12)
    ax.set_title("Slack Generator Usage", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_slack_timeseries(
    stats: NetworkStatistics, save_path: Path | None = None
) -> tuple | None:
    """Plot time series of load shedding and spillage over time.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path

    Returns
    -------
    tuple or None
        (fig, ax) matplotlib figure and axes objects, or None if no slack
    """
    slack_gens = stats._get_slack_generators()

    if not slack_gens or not hasattr(stats.network, "generators_t"):
        print("No slack generator time series data available")
        return None

    # Separate shedding and spillage generators
    shedding_gens = [
        g
        for g in slack_gens
        if stats.network.generators.loc[g, "carrier"] == "load shedding"
    ]
    spillage_gens = [
        g
        for g in slack_gens
        if stats.network.generators.loc[g, "carrier"] == "load spillage"
    ]

    # Get time series data
    shedding_ts = (
        stats.network.generators_t.p[shedding_gens].sum(axis=1)
        if shedding_gens
        else pd.Series(0, index=stats.network.snapshots)
    )
    spillage_ts = (
        abs(stats.network.generators_t.p[spillage_gens].sum(axis=1))
        if spillage_gens
        else pd.Series(0, index=stats.network.snapshots)
    )

    if shedding_ts.sum() == 0 and spillage_ts.sum() == 0:
        print("No slack usage detected in time series")
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot load shedding
    ax1.fill_between(shedding_ts.index, shedding_ts, color="#dd2e23", alpha=0.6)
    ax1.set_ylabel("Load Shedding (MW)", fontsize=12)
    ax1.set_title("Load Shedding Over Time", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3)

    # Plot load spillage
    ax2.fill_between(spillage_ts.index, spillage_ts, color="#df8e23", alpha=0.6)
    ax2.set_ylabel("Load Spillage (MW)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)
    ax2.set_title("Load Spillage Over Time", fontsize=14, fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax1, ax2


def plot_slack_by_bus(
    stats: NetworkStatistics, save_path: Path | None = None, top_n: int = 10
) -> tuple | None:
    """Plot slack generation breakdown by bus/zone.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    save_path : Path, optional
        If provided, save figure to this path
    top_n : int, default 10
        Number of top buses to display

    Returns
    -------
    tuple or None
        (fig, ax) matplotlib figure and axes objects, or None if no slack
    """
    slack_data = stats.unserved_energy()

    if not slack_data["slack_by_bus"]:
        print("No slack usage by bus detected")
        return None

    # Convert to DataFrame
    bus_data = pd.DataFrame(slack_data["slack_by_bus"]).T
    bus_data = bus_data.fillna(0)

    # Sort by total slack and take top N
    bus_data["total"] = bus_data["shedding"] + bus_data["spillage"]
    bus_data = bus_data.sort_values("total", ascending=False).head(top_n)
    bus_data = bus_data.drop("total", axis=1)

    if bus_data.empty:
        print("No slack usage by bus detected")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(bus_data))
    width = 0.35

    ax.bar(
        x - width / 2,
        bus_data["shedding"],
        width,
        label="Load Shedding",
        color="#dd2e23",
        edgecolor="black",
    )
    ax.bar(
        x + width / 2,
        bus_data["spillage"],
        width,
        label="Load Spillage",
        color="#df8e23",
        edgecolor="black",
    )

    ax.set_ylabel("Energy (MWh)", fontsize=12)
    ax.set_xlabel("Bus/Zone", fontsize=12)
    ax.set_title(
        f"Slack Usage by Bus (Top {len(bus_data)})", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bus_data.index, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def create_dashboard(
    stats: NetworkStatistics,
    output_dir: str | Path,
    include_validation: bool = False,
    validation_results: dict | None = None,
) -> dict:
    """Generate complete dashboard with all key plots.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    output_dir : str | Path
        Directory to save plots
    include_validation : bool, default False
        Whether to include validation dashboard
    validation_results : dict, optional
        Validation results (required if include_validation=True)

    Returns
    -------
    dict
        Dictionary mapping plot names to saved file paths
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Overview plots
    print("Generating capacity mix plot...")
    plot_capacity_mix(stats, save_path=plots_dir / "capacity_mix.png")
    saved_files["capacity_mix"] = plots_dir / "capacity_mix.png"

    print("Generating generation mix plot...")
    plot_generation_mix(stats, save_path=plots_dir / "generation_mix.png")
    saved_files["generation_mix"] = plots_dir / "generation_mix.png"

    print("Generating cost breakdown plot...")
    plot_cost_breakdown(stats, save_path=plots_dir / "cost_breakdown.png")
    saved_files["cost_breakdown"] = plots_dir / "cost_breakdown.png"

    # Time series plot
    try:
        print("Generating generation stack plot...")
        plot_generation_stack(stats, save_path=plots_dir / "generation_stack.png")
        saved_files["generation_stack"] = plots_dir / "generation_stack.png"
    except Exception as e:
        print(f"Could not generate generation stack: {e}")

    # Financial plot
    print("Generating LCOE comparison plot...")
    plot_lcoe_comparison(stats, save_path=plots_dir / "lcoe_comparison.png")
    saved_files["lcoe_comparison"] = plots_dir / "lcoe_comparison.png"

    # Curtailment plot
    try:
        print("Generating curtailment plot...")
        result = plot_curtailment(stats, save_path=plots_dir / "curtailment.png")
        if result is not None:
            saved_files["curtailment"] = plots_dir / "curtailment.png"
    except Exception as e:
        print(f"Could not generate curtailment plot: {e}")

    # Multi-period plot (if applicable)
    if stats.has_periods:
        try:
            print("Generating capacity evolution plot...")
            plot_capacity_evolution(
                stats, save_path=plots_dir / "capacity_evolution.png"
            )
            saved_files["capacity_evolution"] = plots_dir / "capacity_evolution.png"
        except Exception as e:
            print(f"Could not generate capacity evolution plot: {e}")

    # Validation dashboard (if requested)
    if include_validation and validation_results:
        print("Generating validation dashboard...")
        plot_validation_dashboard(
            validation_results, save_path=plots_dir / "validation_dashboard.png"
        )
        saved_files["validation_dashboard"] = plots_dir / "validation_dashboard.png"

    # Slack analysis plots (if slack generators exist)
    slack_summary = stats.slack_summary()
    if slack_summary["has_slack"]:
        print("\nGenerating slack analysis plots...")
        try:
            result = plot_unserved_energy(
                stats, save_path=plots_dir / "unserved_energy.png"
            )
            if result is not None:
                saved_files["unserved_energy"] = plots_dir / "unserved_energy.png"
        except Exception as e:
            print(f"Could not generate unserved energy plot: {e}")

        try:
            result = plot_slack_timeseries(
                stats, save_path=plots_dir / "slack_timeseries.png"
            )
            if result is not None:
                saved_files["slack_timeseries"] = plots_dir / "slack_timeseries.png"
        except Exception as e:
            print(f"Could not generate slack timeseries plot: {e}")

        try:
            result = plot_slack_by_bus(stats, save_path=plots_dir / "slack_by_bus.png")
            if result is not None:
                saved_files["slack_by_bus"] = plots_dir / "slack_by_bus.png"
        except Exception as e:
            print(f"Could not generate slack by bus plot: {e}")

    print(f"\n✅ Dashboard complete! Plots saved to: {plots_dir}")
    return saved_files
