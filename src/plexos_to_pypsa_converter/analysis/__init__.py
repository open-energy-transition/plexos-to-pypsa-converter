"""Analysis, benchmarking, and visualization tools for solved PyPSA networks.

This module provides comprehensive tools for analyzing solved energy system models,
adapted from PyPSA-Explorer patterns for static matplotlib visualizations.

Main Components
---------------
- NetworkAnalyzer: Primary OOP interface for network analysis
- Metrics: Functional API for extracting data from PyPSA networks
- Plotting: Static matplotlib visualization functions
- Utils: Helper functions for filtering and aggregation
- Styles: Color schemes and matplotlib styling

Example Usage
-------------
Using the NetworkAnalyzer (recommended):

    >>> import pypsa
    >>> from plexos_to_pypsa_converter.analysis import NetworkAnalyzer
    >>>
    >>> # Load solved network
    >>> network = pypsa.Network()
    >>> network.import_from_netcdf("solved_network.nc")
    >>>
    >>> # Create analyzer
    >>> analyzer = NetworkAnalyzer(network)
    >>>
    >>> # Get metrics
    >>> supply = analyzer.get_supply()
    >>> capacity = analyzer.get_optimal_capacity()
    >>>
    >>> # Create visualizations
    >>> analyzer.plot_energy_balance_totals()
    >>> dashboard = analyzer.create_dashboard()

Using functional API:

    >>> from plexos_to_pypsa_converter.analysis.metrics import calculate_energy_balance
    >>> from plexos_to_pypsa_converter.analysis.plotting import plot_energy_balance_totals
    >>>
    >>> balance = calculate_energy_balance(network, exclude_slack=True)
    >>> plot_energy_balance_totals(network)

Legacy API (NetworkStatistics) remains available for backward compatibility.
"""

# Primary API
from plexos_to_pypsa_converter.analysis.core import NetworkAnalyzer

# Functional API - Metrics
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

# Functional API - Plotting
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

# Legacy API (backward compatibility)
from plexos_to_pypsa_converter.analysis.statistics import NetworkStatistics

# Styling
from plexos_to_pypsa_converter.analysis.styles import (
    apply_default_style,
    get_carrier_color,
)

# Utilities
from plexos_to_pypsa_converter.analysis.utils import (
    detect_spatial_resolution,
    identify_slack_generators,
)

__all__ = [
    # Primary API
    "NetworkAnalyzer",
    # Metrics
    "calculate_energy_balance",
    "calculate_supply",
    "calculate_withdrawal",
    "calculate_optimal_capacity",
    "calculate_installed_capacity",
    "calculate_capacity_factor",
    "calculate_costs",
    "calculate_storage_state",
    "calculate_store_state",
    "calculate_transmission_flows",
    "calculate_curtailment",
    # Plotting
    "plot_energy_balance_timeseries",
    "plot_energy_balance_totals",
    "plot_capacity_overview",
    "plot_capacity_factors",
    "plot_cost_breakdown",
    "plot_cost_comparison",
    "plot_generation_mix",
    "plot_storage_state_of_charge",
    "plot_transmission_flows",
    "create_summary_dashboard",
    # Utilities
    "detect_spatial_resolution",
    "identify_slack_generators",
    "apply_default_style",
    "get_carrier_color",
    # Legacy
    "NetworkStatistics",
]
