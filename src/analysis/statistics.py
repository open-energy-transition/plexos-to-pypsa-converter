"""Core statistics calculation for PyPSA networks using the statistics API.

This module provides the NetworkStatistics class which wraps PyPSA's statistics API
to calculate comprehensive metrics for energy system analysis.
"""

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
    # Capacity Metrics
    # ========================================================================

    def installed_capacity(
        self, groupby: str = "carrier", nice_names: bool = True
    ) -> pd.Series:
        """Get installed capacity by carrier or other grouping.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation ("carrier", "bus", "country", etc.)
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            Installed capacity (MW) grouped by specified dimension
        """
        return self.network.statistics.installed_capacity(
            groupby=groupby, nice_names=nice_names
        )

    def optimal_capacity(
        self, groupby: str = "carrier", nice_names: bool = True
    ) -> pd.Series:
        """Get optimal capacity (including expansion) by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            Optimal capacity (MW) grouped by specified dimension
        """
        return self.network.statistics.optimal_capacity(
            groupby=groupby, nice_names=nice_names
        )

    def capacity_factor(self, groupby: str = "carrier") -> pd.Series:
        """Calculate capacity factors by carrier.

        Capacity factor = actual generation / (capacity * hours)

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation

        Returns
        -------
        pd.Series
            Capacity factor (0-1) grouped by specified dimension
        """
        return self.network.statistics.capacity_factor(groupby=groupby)

    # ========================================================================
    # Energy Metrics
    # ========================================================================

    def energy_supply(
        self, groupby: str = "carrier", nice_names: bool = True
    ) -> pd.Series:
        """Get energy supply (generation) by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            Energy supply (MWh) grouped by specified dimension
        """
        return self.network.statistics.supply(groupby=groupby, nice_names=nice_names)

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

    def capex(self, groupby: str = "carrier", nice_names: bool = True) -> pd.Series:
        """Get capital expenditure by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            CAPEX ($M or configured currency) grouped by specified dimension
        """
        return self.network.statistics.capex(groupby=groupby, nice_names=nice_names)

    def opex(self, groupby: str = "carrier", nice_names: bool = True) -> pd.Series:
        """Get operational expenditure by carrier.

        Parameters
        ----------
        groupby : str, default "carrier"
            Grouping for aggregation
        nice_names : bool, default True
            Use readable carrier names

        Returns
        -------
        pd.Series
            OPEX ($M/year or configured currency) grouped by specified dimension
        """
        return self.network.statistics.opex(groupby=groupby, nice_names=nice_names)

    def total_system_cost(self) -> float:
        """Calculate total system cost (CAPEX + OPEX).

        Returns
        -------
        float
            Total system cost in configured currency
        """
        return self.capex().sum() + self.opex().sum()

    def lcoe(self, carrier: str | None = None) -> pd.Series | float:
        """Calculate Levelized Cost of Energy by carrier.

        LCOE = (CAPEX + OPEX) / Energy Generated

        Parameters
        ----------
        carrier : str, optional
            Specific carrier to calculate LCOE for.
            If None, calculates for all carriers.

        Returns
        -------
        pd.Series or float
            LCOE ($/MWh or configured currency/MWh) by carrier
        """
        capex = self.capex(groupby="carrier")
        opex = self.opex(groupby="carrier")
        energy = self.energy_supply(groupby="carrier")

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
    # Summary Methods
    # ========================================================================

    def generate_summary(self) -> dict:
        """Generate comprehensive summary dictionary with all metrics.

        Returns
        -------
        dict
            Nested dictionary containing:
            - capacity: installed, optimal, capacity_factor
            - generation: supply, curtailment
            - costs: capex, opex, total, lcoe
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
