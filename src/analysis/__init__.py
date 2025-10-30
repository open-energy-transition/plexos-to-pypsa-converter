"""Analysis, benchmarking, and visualization tools for solved PyPSA networks.

This module provides comprehensive tools for analyzing solved energy system models,
including:
- Statistics calculation using PyPSA's statistics API
- Validation and sanity checking
- Visualization and plotting
- Multi-model benchmarking
- Report generation

Example Usage:
    from analysis import NetworkStatistics
    from analysis.visualizations import create_dashboard

    stats = NetworkStatistics("results/model/network.nc")
    create_dashboard(stats, output_dir="analysis_output")
"""

from analysis.statistics import NetworkStatistics

__all__ = ["NetworkStatistics"]
