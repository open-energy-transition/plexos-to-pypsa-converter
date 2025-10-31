"""Matplotlib styling configuration for PyPSA network visualizations.

This module provides color schemes, styling functions, and plot templates
adapted from PyPSA-Explorer for static matplotlib plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# Color Schemes (adapted from PyPSA-Explorer)
# =============================================================================

# Technology/Carrier colors
CARRIER_COLORS = {
    # Renewables
    "wind": "#74c9e6",
    "onwind": "#74c9e6",
    "offwind": "#6fa5d6",
    "offwind-ac": "#6fa5d6",
    "offwind-dc": "#5a8bc5",
    "solar": "#ffdd00",
    "solar pv": "#ffdd00",
    "solar thermal": "#ffb700",
    "hydro": "#2980b9",
    "ror": "#2980b9",  # run-of-river
    # Conventional
    "gas": "#c0504e",
    "OCGT": "#d97370",
    "CCGT": "#c0504e",
    "coal": "#8b4513",
    "lignite": "#7a3a06",
    "oil": "#34495e",
    "nuclear": "#9b59b6",
    # Bioenergy
    "biomass": "#27ae60",
    "biogas": "#52c373",
    # Storage
    "battery": "#e74c3c",
    "battery storage": "#e74c3c",
    "PHS": "#16a085",  # pumped hydro storage
    "hydro storage": "#16a085",
    # Transmission
    "AC": "#333333",
    "DC": "#666666",
    # Other
    "load": "#d35400",
    "load shedding": "#dd2e23",
    "load spillage": "#df8e23",
    "other": "#95a5a6",
}

# Component type colors
COMPONENT_COLORS = {
    "Generator": "#3498db",
    "StorageUnit": "#e74c3c",
    "Store": "#9b59b6",
    "Load": "#d35400",
    "Line": "#2c3e50",
    "Link": "#34495e",
    "Bus": "#95a5a6",
}

# Bus carrier colors
BUS_CARRIER_COLORS = {
    "AC": "#c0392b",
    "DC": "#8e44ad",
    "gas": "#f39c12",
    "H2": "#3498db",
    "heat": "#e67e22",
    "Li ion": "#e74c3c",
}

# =============================================================================
# Matplotlib Style Configuration
# =============================================================================

DEFAULT_STYLE_CONFIG = {
    # Figure
    "figure.figsize": (12, 6),
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # Font
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    # Axes
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    # Colors
    "axes.prop_cycle": plt.cycler(color=list(CARRIER_COLORS.values())[:10]),
}


def apply_default_style() -> None:
    """Apply the default matplotlib style for energy system plots."""
    plt.rcParams.update(DEFAULT_STYLE_CONFIG)
    sns.set_style("whitegrid")


def reset_style() -> None:
    """Reset matplotlib to default settings."""
    plt.rcdefaults()
    sns.reset_defaults()


# =============================================================================
# Color Accessor Functions
# =============================================================================


def get_carrier_color(carrier: str | tuple, default: str = "#95a5a6") -> str:
    """Get color for a carrier/technology.

    Parameters
    ----------
    carrier : str | tuple
        Carrier name (e.g., "wind", "solar", "gas")
        If tuple (from MultiIndex), extracts carrier string
    default : str, default "#95a5a6"
        Default color if carrier not found

    Returns
    -------
    str
        Hex color code
    """
    # Handle tuple from MultiIndex (extract carrier string)
    if isinstance(carrier, tuple):
        # Carrier is typically the last element in multi-period index
        carrier = carrier[-1] if carrier else ""

    # Convert to string and lowercase
    carrier_str = str(carrier).lower()
    return CARRIER_COLORS.get(carrier_str, default)


def get_component_color(component: str, default: str = "#95a5a6") -> str:
    """Get color for a component type.

    Parameters
    ----------
    component : str
        Component name (e.g., "Generator", "StorageUnit")
    default : str, default "#95a5a6"
        Default color if component not found

    Returns
    -------
    str
        Hex color code
    """
    return COMPONENT_COLORS.get(component, default)


def get_bus_carrier_color(carrier: str, default: str = "#c0392b") -> str:
    """Get color for a bus carrier type.

    Parameters
    ----------
    carrier : str
        Bus carrier (e.g., "AC", "DC", "gas")
    default : str, default "#c0392b"
        Default color if carrier not found

    Returns
    -------
    str
        Hex color code
    """
    return BUS_CARRIER_COLORS.get(carrier, default)


def get_colors_for_carriers(carriers: list[str]) -> list[str]:
    """Get list of colors for multiple carriers.

    Parameters
    ----------
    carriers : list[str]
        List of carrier names

    Returns
    -------
    list[str]
        List of hex color codes
    """
    return [get_carrier_color(c) for c in carriers]


def get_colors_for_components(components: list[str]) -> list[str]:
    """Get list of colors for multiple components.

    Parameters
    ----------
    components : list[str]
        List of component names

    Returns
    -------
    list[str]
        List of hex color codes
    """
    return [get_component_color(c) for c in components]


# =============================================================================
# Plot Style Helpers
# =============================================================================


def format_axis_labels(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """Format axis labels and title with consistent styling.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    title : str, optional
        Plot title
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")


def add_grid(ax: plt.Axes, axis: str = "y", alpha: float = 0.3) -> None:
    """Add grid to plot.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    axis : str, default "y"
        Which axis to add grid ("x", "y", or "both")
    alpha : float, default 0.3
        Grid transparency
    """
    ax.grid(axis=axis, alpha=alpha, linestyle="--")


def format_legend(
    ax: plt.Axes, loc: str = "best", frameon: bool = True, ncol: int = 1
) -> None:
    """Format legend with consistent styling.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    loc : str, default "best"
        Legend location
    frameon : bool, default True
        Whether to draw frame around legend
    ncol : int, default 1
        Number of columns in legend
    """
    ax.legend(loc=loc, frameon=frameon, ncol=ncol, fontsize=10)


# =============================================================================
# Specialized Plot Styles
# =============================================================================


def style_energy_balance_plot(ax: plt.Axes) -> None:
    """Apply consistent styling to energy balance plots.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    """
    format_axis_labels(ax, ylabel="Energy (MWh)")
    add_grid(ax, axis="y")
    format_legend(ax, loc="upper left", ncol=2)


def style_capacity_plot(ax: plt.Axes) -> None:
    """Apply consistent styling to capacity plots.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    """
    format_axis_labels(ax, ylabel="Capacity (MW)")
    add_grid(ax, axis="y")
    ax.tick_params(axis="x", rotation=45)


def style_cost_plot(ax: plt.Axes, currency: str = "$") -> None:
    """Apply consistent styling to cost plots.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object
    currency : str, default "$"
        Currency symbol for y-axis label
    """
    format_axis_labels(ax, ylabel=f"Cost ({currency})")
    add_grid(ax, axis="y")
    ax.tick_params(axis="x", rotation=45)
