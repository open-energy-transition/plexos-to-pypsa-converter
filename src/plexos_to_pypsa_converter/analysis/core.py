"""Core NetworkAnalyzer class for PyPSA network analysis.

This module provides the main user-facing interface for analyzing PyPSA networks,
wrapping the functional API from metrics, utils, and plotting modules.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pypsa

from plexos_to_pypsa_converter.analysis.metrics import (
    calculate_capacity_factor,
    calculate_costs,
    calculate_curtailment,
    calculate_energy_balance,
    calculate_installed_capacity,
    calculate_optimal_capacity,
    calculate_storage_state,
    calculate_store_state,
    calculate_supply,
    calculate_transmission_flows,
    calculate_withdrawal,
)
from plexos_to_pypsa_converter.analysis.plotting import (
    create_summary_dashboard,
    plot_capacity_factors,
    plot_capacity_overview,
    plot_cost_breakdown,
    plot_cost_comparison,
    plot_energy_balance_timeseries,
    plot_energy_balance_totals,
    plot_generation_mix,
    plot_storage_state_of_charge,
    plot_transmission_flows,
)
from plexos_to_pypsa_converter.analysis.utils import (
    detect_spatial_resolution,
    get_bus_carriers,
    has_links,
    has_storage_units,
    has_stores,
    has_time_series_data,
    identify_slack_generators,
    is_multi_period,
)


class NetworkAnalyzer:
    """Main interface for PyPSA network analysis and visualization.

    This class provides a convenient object-oriented interface to analyze
    solved PyPSA networks, extract metrics, and create visualizations.

    Parameters
    ----------
    network : pypsa.Network
        Solved PyPSA network to analyze
    exclude_slack_default : bool, default True
        Default setting for excluding slack generators in analysis
    nice_names_default : bool, default True
        Default setting for using readable carrier names

    Attributes
    ----------
    network : pypsa.Network
        The PyPSA network being analyzed
    spatial_resolution : str
        Detected spatial resolution ("single", "zonal", or "nodal")
    slack_generators : list[str]
        List of identified slack generator names
    multi_period : bool
        Whether network has investment periods
    color_overrides : dict | None
        Manual color assignments for specific carriers
    color_palette : str
        Fallback palette for unknown carriers

    Examples
    --------
    >>> import pypsa
    >>> from plexos_to_pypsa_converter.analysis import NetworkAnalyzer
    >>>
    >>> # Load a solved network
    >>> network = pypsa.Network()
    >>> network.import_from_netcdf("solved_network.nc")
    >>>
    >>> # Create analyzer
    >>> analyzer = NetworkAnalyzer(network)
    >>>
    >>> # Get energy supply by carrier
    >>> supply = analyzer.get_supply()
    >>>
    >>> # Plot energy balance
    >>> fig = analyzer.plot_energy_balance_totals()
    >>> plt.show()
    >>>
    >>> # Create summary dashboard
    >>> dashboard = analyzer.create_dashboard()
    >>> plt.savefig("network_summary.png")
    """

    def __init__(
        self,
        network: pypsa.Network,
        exclude_slack_default: bool = True,
        nice_names_default: bool = True,
        color_overrides: dict[str, str] | None = None,
        color_palette: str = "retro_metro",
    ):
        """Initialize NetworkAnalyzer with a PyPSA network.

        Parameters
        ----------
        network : pypsa.Network
            Solved PyPSA network to analyze
        exclude_slack_default : bool, default True
            Default setting for excluding slack generators
        nice_names_default : bool, default True
            Default setting for using readable carrier names
        color_overrides : dict, optional
            Manual color assignments for specific carriers
        color_palette : str, default "retro_metro"
            Fallback palette for unknown carriers
        """
        self.network = network
        self.exclude_slack_default = exclude_slack_default
        self.nice_names_default = nice_names_default
        self.color_overrides = color_overrides
        self.color_palette = color_palette

        # Cache network characteristics
        self.spatial_resolution = detect_spatial_resolution(network)
        self.slack_generators = identify_slack_generators(network)
        self.multi_period = is_multi_period(network)
        self.bus_carriers = get_bus_carriers(network)

        # Cache capabilities
        self._has_time_series = has_time_series_data(network)
        self._has_storage = has_storage_units(network)
        self._has_stores = has_stores(network)
        self._has_links = has_links(network)

    # =========================================================================
    # Network Information Methods
    # =========================================================================

    def info(self) -> dict[str, Any]:
        """Get summary information about the network.

        Returns
        -------
        dict
            Dictionary with network characteristics
        """
        # Get investment periods (0 if not multi-period)
        investment_periods = (
            len(self.network.investment_periods) if self.multi_period else 0
        )

        return {
            "buses": len(self.network.buses),
            "generators": len(self.network.generators),
            "loads": len(self.network.loads),
            "storage_units": len(self.network.storage_units),
            "stores": len(self.network.stores),
            "links": len(self.network.links),
            "lines": len(self.network.lines),
            "snapshots": len(self.network.snapshots),
            "investment_periods": investment_periods,
            "carriers": len(self.network.carriers),
            "spatial_resolution": self.spatial_resolution,
            "multi_period": self.multi_period,
            "bus_carriers": self.bus_carriers,
            "slack_generators": len(self.slack_generators),
            "has_time_series": self._has_time_series,
            "has_storage": self._has_storage,
            "has_stores": self._has_stores,
            "has_links": self._has_links,
        }

    def print_info(self) -> None:
        """Print formatted network information."""
        info = self.info()
        print("=" * 60)
        print("Network Analysis Summary")
        print("=" * 60)
        print(f"Spatial Resolution: {info['spatial_resolution']}")
        print(f"Multi-period: {info['multi_period']}")
        print("\nComponents:")
        print(f"  Buses: {info['buses']}")
        print(f"  Generators: {info['generators']}")
        print(f"  Loads: {info['loads']}")
        print(f"  Storage Units: {info['storage_units']}")
        print(f"  Stores: {info['stores']}")
        print(f"  Links: {info['links']}")
        print(f"  Lines: {info['lines']}")
        print("\nTime Series:")
        print(f"  Snapshots: {info['snapshots']}")
        print(f"  Has Dispatch Data: {info['has_time_series']}")
        print(
            f"\nBus Carriers: {', '.join(info['bus_carriers']) if info['bus_carriers'] else 'None'}"
        )
        print(f"Slack Generators: {info['slack_generators']}")
        print("=" * 60)

    # =========================================================================
    # Energy Balance Methods
    # =========================================================================

    def get_energy_balance(
        self,
        bus_carrier: str | None = None,
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        aggregate: bool = False,
        nice_names: bool | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Get energy balance from network.

        Parameters
        ----------
        bus_carrier : str, optional
            Filter by bus carrier
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators (uses default if None)
        aggregate : bool, default False
            If True, aggregate over snapshots
        nice_names : bool, optional
            Use readable names (uses default if None)

        Returns
        -------
        pd.DataFrame | pd.Series
            Energy balance data
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_energy_balance(
            self.network,
            bus_carrier=bus_carrier,
            buses=buses,
            exclude_slack=exclude_slack,
            aggregate=aggregate,
            nice_names=nice_names,
        )

    def get_supply(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        nice_names: bool | None = None,
    ) -> pd.Series:
        """Get energy supply by carrier/component/bus.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators
        nice_names : bool, optional
            Use readable names

        Returns
        -------
        pd.Series
            Energy supply (MWh)
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_supply(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=nice_names,
        )

    def get_withdrawal(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        nice_names: bool | None = None,
    ) -> pd.Series:
        """Get energy withdrawal by carrier/component/bus.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        nice_names : bool, optional
            Use readable names

        Returns
        -------
        pd.Series
            Energy withdrawal (MWh)
        """
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_withdrawal(
            self.network, groupby=groupby, buses=buses, nice_names=nice_names
        )

    # =========================================================================
    # Capacity Methods
    # =========================================================================

    def get_optimal_capacity(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        nice_names: bool | None = None,
    ) -> pd.Series:
        """Get optimal capacity by carrier/component/bus.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators
        nice_names : bool, optional
            Use readable names

        Returns
        -------
        pd.Series
            Optimal capacity (MW)
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_optimal_capacity(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=nice_names,
        )

    def get_installed_capacity(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        nice_names: bool | None = None,
    ) -> pd.Series:
        """Get installed capacity by carrier/component/bus.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators
        nice_names : bool, optional
            Use readable names

        Returns
        -------
        pd.Series
            Installed capacity (MW)
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_installed_capacity(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=nice_names,
        )

    def get_capacity_factors(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
    ) -> pd.Series:
        """Get capacity factors by carrier/component/bus.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators

        Returns
        -------
        pd.Series
            Capacity factors (0-1)
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return calculate_capacity_factor(
            self.network, groupby=groupby, buses=buses, exclude_slack=exclude_slack
        )

    # =========================================================================
    # Cost Methods
    # =========================================================================

    def get_costs(
        self,
        cost_type: str = "total",
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        nice_names: bool | None = None,
    ) -> pd.Series | dict[str, pd.Series]:
        """Get costs by carrier/component/bus.

        Parameters
        ----------
        cost_type : str, default "total"
            Cost type: "capex", "opex", or "total"
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by specific buses
        exclude_slack : bool, optional
            Exclude slack generators
        nice_names : bool, optional
            Use readable names

        Returns
        -------
        pd.Series | dict[str, pd.Series]
            Cost data
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )
        nice_names = nice_names if nice_names is not None else self.nice_names_default

        return calculate_costs(
            self.network,
            cost_type=cost_type,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            nice_names=nice_names,
        )

    # =========================================================================
    # Storage Methods
    # =========================================================================

    def get_storage_state(
        self, storage_units: list[str] | None = None, buses: list[str] | None = None
    ) -> pd.DataFrame:
        """Get storage state of charge time series.

        Parameters
        ----------
        storage_units : list[str], optional
            Specific storage units
        buses : list[str], optional
            Filter by buses

        Returns
        -------
        pd.DataFrame
            Storage state of charge (MWh)
        """
        return calculate_storage_state(
            self.network, storage_units=storage_units, buses=buses
        )

    def get_store_state(
        self, stores: list[str] | None = None, buses: list[str] | None = None
    ) -> pd.DataFrame:
        """Get store energy level time series.

        Parameters
        ----------
        stores : list[str], optional
            Specific stores
        buses : list[str], optional
            Filter by buses

        Returns
        -------
        pd.DataFrame
            Store energy levels (MWh)
        """
        return calculate_store_state(self.network, stores=stores, buses=buses)

    # =========================================================================
    # Transmission Methods
    # =========================================================================

    def get_transmission_flows(
        self, buses: list[str] | None = None, aggregate: bool = True
    ) -> pd.DataFrame | pd.Series:
        """Get transmission flows.

        Parameters
        ----------
        buses : list[str], optional
            Filter by buses
        aggregate : bool, default True
            If True, return totals; if False, return time series

        Returns
        -------
        pd.DataFrame | pd.Series
            Transmission flows
        """
        return calculate_transmission_flows(
            self.network, buses=buses, aggregate=aggregate
        )

    # =========================================================================
    # Curtailment Methods
    # =========================================================================

    def get_curtailment(
        self, carriers: list[str] | None = None, buses: list[str] | None = None
    ) -> pd.Series:
        """Get curtailment by carrier.

        Parameters
        ----------
        carriers : list[str], optional
            Carriers to check (default: ["wind", "solar"])
        buses : list[str], optional
            Filter by buses

        Returns
        -------
        pd.Series
            Curtailed energy (MWh)
        """
        return calculate_curtailment(self.network, carriers=carriers, buses=buses)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot_energy_balance_timeseries(
        self,
        bus_carrier: str | None = None,
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        max_points: int = 5000,
        figsize: tuple[float, float] = (14, 7),
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot energy balance time series.

        Parameters
        ----------
        bus_carrier : str, optional
            Filter by bus carrier
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        max_points : int, default 5000
            Maximum time series points
        figsize : tuple, default (14, 7)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_energy_balance_timeseries(
            self.network,
            bus_carrier=bus_carrier,
            buses=buses,
            exclude_slack=exclude_slack,
            max_points=max_points,
            figsize=figsize,
            ax=ax,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_energy_balance_totals(
        self,
        bus_carrier: str | None = None,
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (10, 8),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
        dpi: int = 150,
        **kwargs,
    ) -> plt.Axes:
        """Plot aggregated energy balance.

        Parameters
        ----------
        bus_carrier : str, optional
            Filter by bus carrier
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (10, 8)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        save_path : str, optional
            Path to save the figure
        dpi : int, default 150
            Resolution for saved figure
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_energy_balance_totals(
            self.network,
            bus_carrier=bus_carrier,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            ax=ax,
            save_path=save_path,
            dpi=dpi,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_capacity_overview(
        self,
        capacity_type: str = "optimal",
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
        dpi: int = 150,
        **kwargs,
    ) -> plt.Axes:
        """Plot capacity overview.

        Parameters
        ----------
        capacity_type : str, default "optimal"
            "optimal" or "installed"
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (10, 6)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        save_path : str, optional
            Path to save the figure
        dpi : int, default 150
            Resolution for saved figure
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_capacity_overview(
            self.network,
            capacity_type=capacity_type,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            ax=ax,
            save_path=save_path,
            dpi=dpi,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_capacity_factors(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
        dpi: int = 150,
        **kwargs,
    ) -> plt.Axes:
        """Plot capacity factors.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (10, 6)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        save_path : str, optional
            Path to save the figure
        dpi : int, default 150
            Resolution for saved figure
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_capacity_factors(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            ax=ax,
            save_path=save_path,
            dpi=dpi,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_cost_breakdown(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (10, 6),
        ax: plt.Axes | None = None,
        save_path: str | None = None,
        dpi: int = 150,
        **kwargs,
    ) -> plt.Axes:
        """Plot cost breakdown.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (10, 6)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        save_path : str, optional
            Path to save the figure
        dpi : int, default 150
            Resolution for saved figure
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_cost_breakdown(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            ax=ax,
            save_path=save_path,
            dpi=dpi,
            **kwargs,
        )

    def plot_cost_comparison(
        self,
        groupby: str = "carrier",
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (12, 6),
        **kwargs,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot CAPEX vs OPEX comparison.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping dimension
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (12, 6)
            Figure size
        **kwargs
            Additional plotting arguments

        Returns
        -------
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
            Figure and axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_cost_comparison(
            self.network,
            groupby=groupby,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_generation_mix(
        self,
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (12, 5),
        **kwargs,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        """Plot generation mix (supply + capacity).

        Parameters
        ----------
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (12, 5)
            Figure size
        **kwargs
            Additional plotting arguments

        Returns
        -------
        tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
            Figure and axes
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return plot_generation_mix(
            self.network,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
            **kwargs,
        )

    def plot_storage_state_of_charge(
        self,
        storage_units: list[str] | None = None,
        buses: list[str] | None = None,
        max_points: int = 5000,
        figsize: tuple[float, float] = (14, 6),
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot storage state of charge.

        Parameters
        ----------
        storage_units : list[str], optional
            Specific storage units
        buses : list[str], optional
            Filter by buses
        max_points : int, default 5000
            Maximum time series points
        figsize : tuple, default (14, 6)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        return plot_storage_state_of_charge(
            self.network,
            storage_units=storage_units,
            buses=buses,
            max_points=max_points,
            figsize=figsize,
            ax=ax,
            **kwargs,
        )

    def plot_transmission_flows(
        self,
        buses: list[str] | None = None,
        top_n: int = 20,
        figsize: tuple[float, float] = (10, 8),
        ax: plt.Axes | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot transmission flows.

        Parameters
        ----------
        buses : list[str], optional
            Filter by buses
        top_n : int, default 20
            Show top N flows
        figsize : tuple, default (10, 8)
            Figure size
        ax : plt.Axes, optional
            Existing axes
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        return plot_transmission_flows(
            self.network, buses=buses, top_n=top_n, figsize=figsize, ax=ax, **kwargs
        )

    def create_dashboard(
        self,
        buses: list[str] | None = None,
        exclude_slack: bool | None = None,
        figsize: tuple[float, float] = (16, 10),
        save_path: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Create comprehensive summary dashboard.

        Parameters
        ----------
        buses : list[str], optional
            Filter by buses
        exclude_slack : bool, optional
            Exclude slack generators
        figsize : tuple, default (16, 10)
            Figure size
        save_path : str, optional
            Path to save the figure
        dpi : int, default 150
            Resolution for saved figure

        Returns
        -------
        plt.Figure
            Dashboard figure
        """
        exclude_slack = (
            exclude_slack if exclude_slack is not None else self.exclude_slack_default
        )

        return create_summary_dashboard(
            self.network,
            buses=buses,
            exclude_slack=exclude_slack,
            figsize=figsize,
            save_path=save_path,
            dpi=dpi,
            color_overrides=self.color_overrides,
            color_palette=self.color_palette,
        )
