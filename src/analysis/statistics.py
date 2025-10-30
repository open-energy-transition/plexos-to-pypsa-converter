"""Core statistics calculation for PyPSA networks using the statistics API.

This module provides the NetworkStatistics class which wraps PyPSA's statistics API
to calculate comprehensive metrics for energy system analysis.
"""

import copy
from pathlib import Path

import pandas as pd
import pypsa


class NetworkStatistics:
    """Calculate comprehensive statistics for solved PyPSA networks.

    This class wraps PyPSA's statistics API to provide easy access to:
    - Capacity metrics (installed, optimal, capacity factors)
    - Energy metrics (supply, withdrawal, curtailment)
    - Financial metrics (CAPEX, OPEX, LCOE)
    - Multi-period analysis (if investment periods present)

    Parameters
    ----------
    network_path : str | Path
        Path to solved PyPSA network NetCDF file

    Attributes
    ----------
    network : pypsa.Network
        Loaded PyPSA network
    has_periods : bool
        Whether network has investment periods defined

    Examples
    --------
    >>> stats = NetworkStatistics("results/sem-2024-2032/network.nc")
    >>> capacity = stats.installed_capacity()
    >>> total_cost = stats.total_system_cost()
    """

    def __init__(self, network_path: str | Path) -> None:
        """Load network from NetCDF file and detect investment periods."""
        self.network = pypsa.Network(str(network_path))
        self.has_periods = hasattr(self.network, "investment_periods") and (
            self.network.investment_periods is not None
        )

    # ========================================================================
    # Slack Generator Helpers
    # ========================================================================

    def _get_slack_generators(self) -> list[str]:
        """Get list of slack generator names.

        Returns
        -------
        list[str]
            Names of slack generators (load shedding and load spillage)
        """
        if "carrier" not in self.network.generators.columns:
            return []
        slack_carriers = ["load spillage", "load shedding"]
        return self.network.generators[
            self.network.generators["carrier"].isin(slack_carriers)
        ].index.tolist()

    def _get_network_without_slack(self) -> pypsa.Network:
        """Create a temporary network view with slack generators removed.

        Returns
        -------
        pypsa.Network
            Network with slack generators filtered out
        """
        # Create a deep copy to avoid modifying original
        n = copy.deepcopy(self.network)
        slack_gens = self._get_slack_generators()

        if slack_gens:
            # Filter slack generators from components
            n.generators = n.generators.drop(slack_gens)
            # Filter from time-series data if it exists
            if hasattr(n, "generators_t") and hasattr(n.generators_t, "p"):
                n.generators_t.p = n.generators_t.p.drop(
                    columns=slack_gens, errors="ignore"
                )

        return n

    # ========================================================================
    # Capacity Metrics
    # ========================================================================

    def installed_capacity(
        self,
        groupby: str = "carrier",
        nice_names: bool = True,
        exclude_slack: bool = True,
    ) -> pd.Series:
        """Get installed capacity by carrier or other grouping.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation ("carrier", "bus", "country", etc.)
        nice_names : bool, default True
            Use readable carrier names
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            Installed capacity (MW) grouped by specified dimension
        """
        # Get capacity from original network (avoid deep copy issues)
        result = self.network.statistics.installed_capacity(
            groupby=groupby, nice_names=nice_names
        )

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            slack_carriers = ["load spillage", "load shedding"]
            if nice_names:
                slack_carriers = ["Load spillage", "Load shedding"]
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    def optimal_capacity(
        self,
        groupby: str = "carrier",
        nice_names: bool = True,
        exclude_slack: bool = True,
    ) -> pd.Series:
        """Get optimal capacity (including expansion) by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            Optimal capacity (MW) grouped by specified dimension
        """
        # Get capacity from original network (avoid deep copy issues)
        result = self.network.statistics.optimal_capacity(
            groupby=groupby, nice_names=nice_names
        )

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            slack_carriers = ["load spillage", "load shedding"]
            if nice_names:
                slack_carriers = ["Load spillage", "Load shedding"]
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    def capacity_factor(
        self, groupby: str = "carrier", exclude_slack: bool = True
    ) -> pd.Series:
        """Calculate capacity factors by carrier.

        Capacity factor = actual generation / (capacity * hours)

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            Capacity factor (0-1) grouped by specified dimension
        """
        # Get capacity factor from original network (avoid deep copy issues)
        result = self.network.statistics.capacity_factor(groupby=groupby)

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            # Capacity factor doesn't have nice_names parameter, use lowercase
            slack_carriers = ["load spillage", "load shedding"]
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    # ========================================================================
    # Energy Metrics
    # ========================================================================

    def energy_supply(
        self,
        groupby: str = "carrier",
        nice_names: bool = True,
        exclude_slack: bool = True,
    ) -> pd.Series:
        """Get energy supply (generation) by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            Energy supply (MWh) grouped by specified dimension
        """
        # Get supply from original network (avoid deep copy issues)
        result = self.network.statistics.supply(groupby=groupby, nice_names=nice_names)

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            slack_carriers = ["load spillage", "load shedding"]
            if nice_names:
                slack_carriers = ["Load spillage", "Load shedding"]
            # Filter out slack carriers from result
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    def energy_withdrawal(
        self, groupby: str = "carrier", nice_names: bool = True
    ) -> pd.Series:
        """Get energy withdrawal (consumption) by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            Energy withdrawal (MWh) grouped by specified dimension
        """
        return self.network.statistics.withdrawal(
            groupby=groupby, nice_names=nice_names
        )

    def curtailment(self, carrier: list[str] | None = None) -> pd.Series:
        """Get curtailed energy by VRE technology.

        Parameters
        ----------
        carrier : list[str], optional
            Specific carriers to check (default: ["wind", "solar"])

        Returns
        -------
        pd.Series
            Curtailed energy (MWh) by carrier
        """
        if carrier is None:
            carrier = ["wind", "solar"]
        return self.network.statistics.curtailment(carrier=carrier)

    # ========================================================================
    # Financial Metrics
    # ========================================================================

    def capex(
        self,
        groupby: str = "carrier",
        nice_names: bool = True,
        exclude_slack: bool = True,
    ) -> pd.Series:
        """Get capital expenditure by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            CAPEX ($M or configured currency) grouped by specified dimension
        """
        # Get CAPEX from original network (avoid deep copy issues)
        result = self.network.statistics.capex(groupby=groupby, nice_names=nice_names)

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            slack_carriers = ["load spillage", "load shedding"]
            if nice_names:
                slack_carriers = ["Load spillage", "Load shedding"]
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    def opex(
        self,
        groupby: str = "carrier",
        nice_names: bool = True,
        exclude_slack: bool = True,
    ) -> pd.Series:
        """Get operational expenditure by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series
            OPEX ($M/year or configured currency) grouped by specified dimension
        """
        # Get OPEX from original network (avoid deep copy issues)
        result = self.network.statistics.opex(groupby=groupby, nice_names=nice_names)

        # Filter slack carriers if requested and groupby is "carrier"
        if exclude_slack and groupby == "carrier":
            slack_carriers = ["load spillage", "load shedding"]
            if nice_names:
                slack_carriers = ["Load spillage", "Load shedding"]
            result = result[~result.index.isin(slack_carriers, level="carrier")]

        return result

    def total_system_cost(self, exclude_slack: bool = True) -> float:
        """Calculate total system cost (CAPEX + OPEX).

        Parameters
        ----------
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        float
            Total system cost in configured currency
        """
        return (
            self.capex(exclude_slack=exclude_slack).sum()
            + self.opex(exclude_slack=exclude_slack).sum()
        )

    def lcoe(
        self, carrier: str | None = None, exclude_slack: bool = True
    ) -> pd.Series | float:
        """Calculate Levelized Cost of Energy by carrier.

        LCOE = (CAPEX + OPEX) / Energy Generated

        Parameters
        ----------
        carrier : str, optional
            Specific carrier to calculate LCOE for.
            If None, calculates for all carriers.
        exclude_slack : bool, default True
            If True, exclude load shedding/spillage slack generators

        Returns
        -------
        pd.Series or float
            LCOE ($/MWh or configured currency/MWh) by carrier
        """
        capex = self.capex(groupby="carrier", exclude_slack=exclude_slack)
        opex = self.opex(groupby="carrier", exclude_slack=exclude_slack)
        energy = self.energy_supply(groupby="carrier", exclude_slack=exclude_slack)

        # Avoid division by zero
        lcoe = (capex + opex) / energy.replace(0, float("nan"))

        if carrier is not None:
            return lcoe.get(carrier, float("nan"))
        return lcoe

    def revenue(self, groupby: str = "carrier", nice_names: bool = True) -> pd.Series:
        """Get revenue by carrier (if available in network).

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            Revenue grouped by specified dimension
        """
        return self.network.statistics.revenue(groupby=groupby, nice_names=nice_names)

    # ========================================================================
    # Multi-Period Methods
    # ========================================================================

    def capacity_by_period(self, carrier: str | None = None) -> pd.DataFrame:
        """Get capacity evolution across investment periods.

        Parameters
        ----------
        carrier : str, optional
            Specific carrier to get capacity for.
            If None, returns all carriers.

        Returns
        -------
        pd.DataFrame
            Capacity (MW) with periods as index, carriers as columns
        """
        if not self.has_periods:
            # Return single-period data as DataFrame
            cap = self.installed_capacity()
            if carrier:
                cap = cap[cap.index == carrier]
            return pd.DataFrame(cap).T

        results = []
        for period in self.network.investment_periods:
            try:
                cap = self.network.statistics.installed_capacity(
                    period=period, groupby="carrier"
                )
                results.append(cap)
            except Exception:
                # If period-specific statistics fail, use overall
                cap = self.installed_capacity()
                results.append(cap)

        df = pd.DataFrame(results, index=self.network.investment_periods)

        if carrier:
            df = df[[carrier]] if carrier in df.columns else pd.DataFrame()

        return df

    def generation_by_period(self, carrier: str | None = None) -> pd.DataFrame:
        """Get generation evolution across investment periods.

        Parameters
        ----------
        carrier : str, optional
            Specific carrier to get generation for.
            If None, returns all carriers.

        Returns
        -------
        pd.DataFrame
            Generation (MWh) with periods as index, carriers as columns
        """
        if not self.has_periods:
            gen = self.energy_supply()
            if carrier:
                gen = gen[gen.index == carrier]
            return pd.DataFrame(gen).T

        results = []
        for period in self.network.investment_periods:
            try:
                gen = self.network.statistics.supply(period=period, groupby="carrier")
                results.append(gen)
            except Exception:
                gen = self.energy_supply()
                results.append(gen)

        df = pd.DataFrame(results, index=self.network.investment_periods)

        if carrier:
            df = df[[carrier]] if carrier in df.columns else pd.DataFrame()

        return df

    # ========================================================================
    # Slack Analysis Metrics
    # ========================================================================

    def unserved_energy(self) -> dict:
        """Calculate load shedding (unserved energy) and load spillage metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - load_shedding_mwh: Total unserved energy (MWh)
            - load_spillage_mwh: Total excess generation absorbed (MWh)
            - unserved_pct: Percentage of total demand unserved
            - spillage_pct: Percentage of total generation spilled
            - slack_by_bus: Dict of slack usage by bus
        """
        slack_gens = self._get_slack_generators()

        result = {
            "load_shedding_mwh": 0.0,
            "load_spillage_mwh": 0.0,
            "unserved_pct": 0.0,
            "spillage_pct": 0.0,
            "slack_by_bus": {},
        }

        if not slack_gens or not hasattr(self.network, "generators_t"):
            return result

        # Calculate load shedding (positive power from "load shedding" generators)
        shedding_gens = [
            g
            for g in slack_gens
            if self.network.generators.loc[g, "carrier"] == "load shedding"
        ]
        for gen in shedding_gens:
            if gen in self.network.generators_t.p.columns:
                gen_output = self.network.generators_t.p[gen]
                shedding = gen_output[gen_output > 0].sum()
                result["load_shedding_mwh"] += shedding
                if shedding > 0:
                    bus = self.network.generators.loc[gen, "bus"]
                    if bus not in result["slack_by_bus"]:
                        result["slack_by_bus"][bus] = {"shedding": 0.0, "spillage": 0.0}
                    result["slack_by_bus"][bus]["shedding"] += shedding

        # Calculate load spillage (negative power from "load spillage" generators)
        spillage_gens = [
            g
            for g in slack_gens
            if self.network.generators.loc[g, "carrier"] == "load spillage"
        ]
        for gen in spillage_gens:
            if gen in self.network.generators_t.p.columns:
                gen_output = self.network.generators_t.p[gen]
                spillage = abs(gen_output[gen_output < 0].sum())
                result["load_spillage_mwh"] += spillage
                if spillage > 0:
                    bus = self.network.generators.loc[gen, "bus"]
                    if bus not in result["slack_by_bus"]:
                        result["slack_by_bus"][bus] = {"shedding": 0.0, "spillage": 0.0}
                    result["slack_by_bus"][bus]["spillage"] += spillage

        # Calculate percentages
        try:
            total_demand = self.network.loads_t.p_set.sum().sum()
            if total_demand > 0:
                result["unserved_pct"] = (
                    result["load_shedding_mwh"] / total_demand
                ) * 100

            total_generation = self.energy_supply(exclude_slack=True).sum()
            if total_generation > 0:
                result["spillage_pct"] = (
                    result["load_spillage_mwh"] / total_generation
                ) * 100
        except (AttributeError, KeyError):
            pass

        return result

    def slack_summary(self) -> dict:
        """Generate comprehensive slack generator summary.

        Returns
        -------
        dict
            Summary including:
            - has_slack: Whether network has slack generators
            - num_slack_gens: Number of slack generators
            - unserved_energy: Dict from unserved_energy() method
            - slack_capacity_mw: Total slack capacity
        """
        slack_gens = self._get_slack_generators()

        summary = {
            "has_slack": len(slack_gens) > 0,
            "num_slack_gens": len(slack_gens),
            "unserved_energy": self.unserved_energy(),
            "slack_capacity_mw": 0.0,
        }

        if slack_gens:
            summary["slack_capacity_mw"] = self.network.generators.loc[
                slack_gens, "p_nom"
            ].sum()

        return summary

    # ========================================================================
    # Summary Methods
    # ========================================================================

    def generate_summary(self) -> dict:
        """Generate comprehensive summary dictionary with all metrics.

        Note: All capacity, generation, and cost metrics exclude slack generators
        (load shedding/spillage) by default. Slack metrics are reported separately.

        Returns
        -------
        dict
            Nested dictionary containing:
            - capacity: installed, optimal, capacity_factor (excludes slack)
            - generation: supply, curtailment (excludes slack)
            - costs: capex, opex, total, lcoe (excludes slack)
            - slack: unserved energy and slack generator metrics
            - network_info: basic network information
        """
        return {
            "capacity": {
                "installed": self.installed_capacity().to_dict(),
                "capacity_factor": self.capacity_factor().to_dict(),
            },
            "generation": {
                "supply": self.energy_supply().to_dict(),
                "curtailment": self.curtailment().to_dict(),
            },
            "costs": {
                "capex": self.capex().to_dict(),
                "opex": self.opex().to_dict(),
                "total": self.total_system_cost(),
                "lcoe": self.lcoe().to_dict(),
            },
            "slack": self.slack_summary(),
            "network_info": {
                "buses": len(self.network.buses),
                "generators": len(self.network.generators),
                "loads": len(self.network.loads),
                "links": len(self.network.links),
                "storage_units": len(self.network.storage_units),
                "snapshots": len(self.network.snapshots),
                "has_periods": self.has_periods,
                "objective": float(self.network.objective),
            },
        }
