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
# Additional Color Palettes for Automatic Assignment
# =============================================================================

# Vibrant, aesthetically pleasing palettes for unknown carriers
AVAILABLE_PALETTES = {
    # Retro Metro - Vibrant and eye-catching (9 colors)
    "retro_metro": [
        "#ea5545",  # Vibrant Red
        "#f46a9b",  # Hot Pink
        "#ef9b20",  # Orange
        "#edbf33",  # Gold
        "#ede15b",  # Yellow
        "#bdcf32",  # Lime
        "#87bc45",  # Green
        "#27aeef",  # Blue
        "#b33dc6",  # Purple
    ],
    # Material Design - Modern Google palette (12 colors)
    "material": [
        "#f44336",  # Red
        "#e91e63",  # Pink
        "#9c27b0",  # Purple
        "#3f51b5",  # Indigo
        "#2196f3",  # Blue
        "#00bcd4",  # Cyan
        "#009688",  # Teal
        "#4caf50",  # Green
        "#cddc39",  # Lime
        "#ffc107",  # Amber
        "#ff9800",  # Orange
        "#ff5722",  # Deep Orange
    ],
    # Seaborn Deep - Moderated, aesthetically pleasing (10 colors)
    "seaborn_deep": [
        "#4C72B0",  # Blue
        "#DD8452",  # Orange
        "#55A868",  # Green
        "#C44E52",  # Red
        "#8172B3",  # Purple
        "#937860",  # Brown
        "#DA8BC3",  # Pink
        "#8C8C8C",  # Grey
        "#CCB974",  # Yellow
        "#64B5CD",  # Cyan
    ],
    # Seaborn Bright - Distinct, vibrant hues (10 colors)
    "seaborn_bright": [
        "#023EFF",  # Blue
        "#FF7C00",  # Orange
        "#1AC938",  # Green
        "#E8000B",  # Red
        "#8B2BE2",  # Purple
        "#9F4800",  # Brown
        "#F14CC1",  # Pink
        "#A3A3A3",  # Grey
        "#FFC400",  # Yellow
        "#00D7FF",  # Cyan
    ],
    # Tableau 10 - Industry standard, balanced (10 colors)
    "tableau10": [
        "#4E79A7",  # Blue
        "#F28E2B",  # Orange
        "#E15759",  # Red
        "#76B7B2",  # Teal
        "#59A14F",  # Green
        "#EDC948",  # Yellow
        "#B07AA1",  # Purple
        "#FF9DA7",  # Pink
        "#9C755F",  # Brown
        "#BAB0AC",  # Grey
    ],
}


# =============================================================================
# Smart Color Management
# =============================================================================


class ColorManager:
    """Smart color assignment for carriers with fallback palettes.

    This class handles automatic color assignment for carriers, with priority:
    1. Predefined colors (CARRIER_COLORS)
    2. User overrides
    3. Fallback palette (avoiding duplicates)

    Parameters
    ----------
    predefined_colors : dict, optional
        Predefined carrier->color mappings (defaults to CARRIER_COLORS)
    fallback_palette : str, default "retro_metro"
        Name of palette to use for unknown carriers

    Examples
    --------
    >>> cm = ColorManager(fallback_palette="material")
    >>> colors = cm.get_colors(["wind", "solar", "biomass", "new_tech"])
    >>> # wind/solar use CARRIER_COLORS, biomass/new_tech use material palette
    """

    def __init__(
        self,
        predefined_colors: dict[str, str] | None = None,
        fallback_palette: str = "retro_metro",
    ):
        """Initialize ColorManager."""
        self.predefined = predefined_colors or CARRIER_COLORS
        self.fallback_palette = fallback_palette

        # Validate palette name
        if fallback_palette not in AVAILABLE_PALETTES:
            msg = f"Unknown palette: {fallback_palette}. Available: {list(AVAILABLE_PALETTES.keys())}"
            raise ValueError(msg)

    def get_colors(
        self,
        carriers: list[str],
        user_overrides: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Get colors for list of carriers.

        Parameters
        ----------
        carriers : list[str]
            List of carrier names
        user_overrides : dict, optional
            User-specified carrier->color mappings

        Returns
        -------
        dict
            Mapping of carrier -> hex color
        """
        user_overrides = user_overrides or {}
        colors = {}
        used_colors = set()

        # Get fallback palette
        palette = AVAILABLE_PALETTES[self.fallback_palette].copy()

        for carrier in carriers:
            # Extract carrier name if it's a tuple (from MultiIndex)
            carrier_name = carrier[-1] if isinstance(carrier, tuple) else carrier
            carrier_lower = str(carrier_name).lower()

            # Priority 1: User overrides
            if carrier in user_overrides:
                color = user_overrides[carrier]
            # Priority 2: Predefined colors (check lowercase)
            elif carrier_lower in self.predefined:
                color = self.predefined[carrier_lower]
            # Priority 3: Assign from fallback palette
            else:
                # Filter out already-used colors
                available = [c for c in palette if c not in used_colors]

                if available:
                    color = available[0]
                    # Remove from palette for next carrier
                    palette.remove(color)
                else:
                    # Palette exhausted - cycle through with alpha variation
                    # Just cycle through the original palette
                    idx = len(colors) % len(AVAILABLE_PALETTES[self.fallback_palette])
                    color = AVAILABLE_PALETTES[self.fallback_palette][idx]

            colors[carrier] = color
            used_colors.add(color)

        return colors


def assign_colors_to_carriers(
    carriers: list[str],
    user_overrides: dict[str, str] | None = None,
    palette: str = "retro_metro",
    predefined: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assign colors to carriers with smart fallback.

    Main public API for color assignment. Handles priority:
    1. Predefined colors (CARRIER_COLORS by default)
    2. User overrides
    3. Fallback palette

    Parameters
    ----------
    carriers : list[str]
        List of carrier names
    user_overrides : dict, optional
        User-specified carrier->color mappings
    palette : str, default "retro_metro"
        Fallback palette name for unknown carriers
    predefined : dict, optional
        Predefined colors (defaults to CARRIER_COLORS)

    Returns
    -------
    dict
        Carrier -> color mapping

    Examples
    --------
    >>> colors = assign_colors_to_carriers(
    ...     ["wind", "solar", "biomass", "new_tech"],
    ...     user_overrides={"new_tech": "#FF00FF"},
    ...     palette="material"
    ... )
    """
    cm = ColorManager(predefined_colors=predefined, fallback_palette=palette)
    return cm.get_colors(carriers, user_overrides=user_overrides)


def get_palette_colors(palette_name: str) -> list[str]:
    """Get colors from a named palette.

    Parameters
    ----------
    palette_name : str
        Name of palette ("retro_metro", "material", "seaborn_deep", etc.)

    Returns
    -------
    list[str]
        List of hex color codes

    Raises
    ------
    ValueError
        If palette_name is not recognized
    """
    if palette_name not in AVAILABLE_PALETTES:
        msg = f"Unknown palette: {palette_name}. Available: {list(AVAILABLE_PALETTES.keys())}"
        raise ValueError(msg)
    return AVAILABLE_PALETTES[palette_name].copy()


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
