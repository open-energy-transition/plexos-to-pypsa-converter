"""Report generation for PyPSA network analysis results.

This module provides functions to export analysis results in various formats
including Markdown, Excel, and JSON.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from plexos_to_pypsa_converter.analysis.statistics import NetworkStatistics


def generate_markdown_report(stats: NetworkStatistics, output_path: str | Path) -> None:
    """Generate Markdown report with summary statistics and plots.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    output_path : str | Path
        Path to output Markdown file

    Examples
    --------
    >>> stats = NetworkStatistics("results/model/network.nc")
    >>> generate_markdown_report(stats, "results/model/analysis/report.md")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = stats.generate_summary()

    # Build markdown content
    md_content = f"""# Network Analysis Report

## Model Information

- **Buses**: {summary["network_info"]["buses"]}
- **Generators**: {summary["network_info"]["generators"]}
- **Loads**: {summary["network_info"]["loads"]}
- **Links**: {summary["network_info"]["links"]}
- **Storage Units**: {summary["network_info"]["storage_units"]}
- **Snapshots**: {summary["network_info"]["snapshots"]}
- **Has Investment Periods**: {summary["network_info"]["has_periods"]}
- **Objective Value**: {summary["network_info"]["objective"]:,.2f}

## Capacity Summary

*Note: All capacity, generation, and cost metrics exclude slack generators (load shedding/spillage). See Slack Analysis section below for unserved energy metrics.*

### Installed Capacity (MW)

| Carrier | Capacity (MW) |
|---------|--------------|
"""

    # Add capacity table
    for carrier, capacity in summary["capacity"]["installed"].items():
        md_content += f"| {carrier} | {capacity:,.1f} |\n"

    md_content += """
### Capacity Factors

| Carrier | Capacity Factor (%) |
|---------|-------------------|
"""

    # Add capacity factor table
    for carrier, cf in summary["capacity"]["capacity_factor"].items():
        md_content += f"| {carrier} | {cf * 100:.1f}% |\n"

    md_content += """
## Generation Summary

### Energy Supply (MWh)

| Carrier | Generation (MWh) |
|---------|-----------------|
"""

    # Add generation table
    for carrier, gen in summary["generation"]["supply"].items():
        md_content += f"| {carrier} | {gen:,.1f} |\n"

    md_content += """
### VRE Curtailment (MWh)

| Carrier | Curtailed Energy (MWh) |
|---------|----------------------|
"""

    # Add curtailment table
    for carrier, curt in summary["generation"]["curtailment"].items():
        md_content += f"| {carrier} | {curt:,.1f} |\n"

    md_content += f"""
## Cost Summary

- **Total System Cost**: ${summary["costs"]["total"]:,.2f}
- **Total CAPEX**: ${sum(summary["costs"]["capex"].values()):,.2f}
- **Total OPEX**: ${sum(summary["costs"]["opex"].values()):,.2f}

### CAPEX by Carrier

| Carrier | CAPEX |
|---------|-------|
"""

    # Add CAPEX table
    for carrier, capex in summary["costs"]["capex"].items():
        md_content += f"| {carrier} | ${capex:,.2f} |\n"

    md_content += """
### OPEX by Carrier

| Carrier | OPEX |
|---------|------|
"""

    # Add OPEX table
    for carrier, opex in summary["costs"]["opex"].items():
        md_content += f"| {carrier} | ${opex:,.2f} |\n"

    md_content += """
### LCOE by Carrier ($/MWh)

| Carrier | LCOE ($/MWh) |
|---------|-------------|
"""

    # Add LCOE table
    for carrier, lcoe in summary["costs"]["lcoe"].items():
        md_content += f"| {carrier} | ${lcoe:,.2f} |\n"

    # Add slack analysis section if slack generators exist
    if summary["slack"]["has_slack"]:
        unserved = summary["slack"]["unserved_energy"]
        md_content += f"""
## Slack Analysis

### Unserved Energy Summary

- **Load Shedding (Unserved Energy)**: {unserved["load_shedding_mwh"]:,.2f} MWh ({unserved["unserved_pct"]:.2f}% of demand)
- **Load Spillage (Excess Generation)**: {unserved["load_spillage_mwh"]:,.2f} MWh ({unserved["spillage_pct"]:.2f}% of generation)
- **Number of Slack Generators**: {summary["slack"]["num_slack_gens"]}
- **Total Slack Capacity**: {summary["slack"]["slack_capacity_mw"]:,.1f} MW

"""
        # Add slack by bus table if available
        if unserved["slack_by_bus"]:
            md_content += """### Slack Usage by Bus/Zone

| Bus/Zone | Load Shedding (MWh) | Load Spillage (MWh) |
|----------|---------------------|---------------------|
"""
            for bus, values in unserved["slack_by_bus"].items():
                shedding = values.get("shedding", 0)
                spillage = values.get("spillage", 0)
                md_content += f"| {bus} | {shedding:,.2f} | {spillage:,.2f} |\n"

    md_content += """
## Visualizations

![Capacity Mix](plots/capacity_mix.png)

![Generation Mix](plots/generation_mix.png)

![Cost Breakdown](plots/cost_breakdown.png)

![Generation Stack](plots/generation_stack.png)

![LCOE Comparison](plots/lcoe_comparison.png)
"""

    # Add multi-period plot if applicable
    if summary["network_info"]["has_periods"]:
        md_content += """
![Capacity Evolution](plots/capacity_evolution.png)
"""

    # Add slack visualization plots if slack generators exist
    if summary["slack"]["has_slack"]:
        md_content += """
### Slack Analysis Plots

![Unserved Energy](plots/unserved_energy.png)

![Slack Time Series](plots/slack_timeseries.png)

![Slack by Bus](plots/slack_by_bus.png)
"""

    # Write to file
    output_path.write_text(md_content)
    print(f"✅ Markdown report saved to: {output_path}")


def export_to_excel(stats: NetworkStatistics, output_path: str | Path) -> None:
    """Export all statistics to Excel workbook with multiple sheets.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    output_path : str | Path
        Path to output Excel file

    Examples
    --------
    >>> stats = NetworkStatistics("results/model/network.nc")
    >>> export_to_excel(stats, "results/model/analysis/statistics.xlsx")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Installed Capacity
        capacity = stats.installed_capacity()
        capacity.to_frame(name="Capacity (MW)").to_excel(writer, sheet_name="Capacity")

        # Sheet 2: Generation
        generation = stats.energy_supply()
        generation.to_frame(name="Generation (MWh)").to_excel(
            writer, sheet_name="Generation"
        )

        # Sheet 3: CAPEX
        capex = stats.capex()
        capex.to_frame(name="CAPEX").to_excel(writer, sheet_name="CAPEX")

        # Sheet 4: OPEX
        opex = stats.opex()
        opex.to_frame(name="OPEX").to_excel(writer, sheet_name="OPEX")

        # Sheet 5: LCOE
        lcoe = stats.lcoe()
        lcoe.to_frame(name="LCOE ($/MWh)").to_excel(writer, sheet_name="LCOE")

        # Sheet 6: Capacity Factors
        cf = stats.capacity_factor()
        cf.to_frame(name="Capacity Factor").to_excel(
            writer, sheet_name="Capacity Factor"
        )

        # Sheet 7: Summary
        summary = stats.generate_summary()
        summary_data = {
            "Metric": [
                "Total Buses",
                "Total Generators",
                "Total Loads",
                "Total Links",
                "Total Storage Units",
                "Total Snapshots",
                "Has Investment Periods",
                "Objective Value",
                "Total System Cost",
                "Total CAPEX",
                "Total OPEX",
            ],
            "Value": [
                summary["network_info"]["buses"],
                summary["network_info"]["generators"],
                summary["network_info"]["loads"],
                summary["network_info"]["links"],
                summary["network_info"]["storage_units"],
                summary["network_info"]["snapshots"],
                summary["network_info"]["has_periods"],
                summary["network_info"]["objective"],
                summary["costs"]["total"],
                sum(summary["costs"]["capex"].values()),
                sum(summary["costs"]["opex"].values()),
            ],
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 8: Slack Analysis (if applicable)
        if summary["slack"]["has_slack"]:
            unserved = summary["slack"]["unserved_energy"]
            slack_summary_data = {
                "Metric": [
                    "Load Shedding (MWh)",
                    "Load Spillage (MWh)",
                    "Unserved Energy (%)",
                    "Spillage (%)",
                    "Number of Slack Generators",
                    "Total Slack Capacity (MW)",
                ],
                "Value": [
                    unserved["load_shedding_mwh"],
                    unserved["load_spillage_mwh"],
                    unserved["unserved_pct"],
                    unserved["spillage_pct"],
                    summary["slack"]["num_slack_gens"],
                    summary["slack"]["slack_capacity_mw"],
                ],
            }
            pd.DataFrame(slack_summary_data).to_excel(
                writer, sheet_name="Slack Analysis", index=False
            )

            # Add slack by bus breakdown if available
            if unserved["slack_by_bus"]:
                slack_by_bus = pd.DataFrame(unserved["slack_by_bus"]).T
                slack_by_bus.columns = ["Load Shedding (MWh)", "Load Spillage (MWh)"]
                slack_by_bus.to_excel(writer, sheet_name="Slack by Bus")

        # Sheet 9/10: Multi-period data (if applicable)
        if stats.has_periods:
            try:
                capacity_by_period = stats.capacity_by_period()
                capacity_by_period.to_excel(writer, sheet_name="Capacity by Period")
            except Exception:
                logging.exception(
                    "Exception occurred while exporting capacity by period to Excel"
                )

    print(f"✅ Excel report saved to: {output_path}")


def export_to_json(stats: NetworkStatistics, output_path: str | Path) -> None:
    """Export full statistics dictionary to JSON format.

    Parameters
    ----------
    stats : NetworkStatistics
        Network statistics object
    output_path : str | Path
        Path to output JSON file

    Examples
    --------
    >>> stats = NetworkStatistics("results/model/network.nc")
    >>> export_to_json(stats, "results/model/analysis/statistics.json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = stats.generate_summary()

    with output_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"✅ JSON report saved to: {output_path}")
