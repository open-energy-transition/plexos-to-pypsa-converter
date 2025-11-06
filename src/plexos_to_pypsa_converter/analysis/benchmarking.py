"""Benchmarking and comparison tools for multiple PyPSA models.

This module provides tools for comparing multiple solved models and benchmarking
against PLEXOS baseline results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plexos_to_pypsa_converter.analysis.statistics import NetworkStatistics


class ModelBenchmark:
    """Compare metrics across multiple solved PyPSA models.

    This class enables cross-model comparison of capacity, generation, costs,
    and identification of outlier values.

    Parameters
    ----------
    model_paths : dict[str, Path]
        Mapping of model_id to network file path.
        Example: {"caiso-irp23": Path("results/caiso-irp23/network.nc")}

    Attributes
    ----------
    models : dict[str, NetworkStatistics]
        Dictionary of model statistics objects

    Examples
    --------
    >>> benchmark = ModelBenchmark({
    ...     "caiso-irp23": Path("results/caiso-irp23/network.nc"),
    ...     "sem-2024-2032": Path("results/sem-2024-2032/network.nc")
    ... })
    >>> capacity_comp = benchmark.compare_capacity_mix()
    """

    def __init__(self, model_paths: dict[str, Path]) -> None:
        """Initialize benchmark with multiple models."""
        self.models = {}
        for model_id, path in model_paths.items():
            try:
                self.models[model_id] = NetworkStatistics(path)
                print(f"Loaded: {model_id}")
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")

    def compare_capacity_mix(self) -> pd.DataFrame:
        """Compare installed capacity across models.

        Returns
        -------
        pd.DataFrame
            Capacity (MW) with carriers as index, models as columns
        """
        results = {}
        for model_id, stats in self.models.items():
            results[model_id] = stats.installed_capacity()

        return pd.DataFrame(results).fillna(0)

    def compare_generation_mix(self) -> pd.DataFrame:
        """Compare energy generation across models.

        Returns
        -------
        pd.DataFrame
            Generation (MWh) with carriers as index, models as columns
        """
        results = {}
        for model_id, stats in self.models.items():
            results[model_id] = stats.energy_supply()

        return pd.DataFrame(results).fillna(0)

    def compare_costs(self) -> pd.DataFrame:
        """Compare total system costs across models.

        Returns
        -------
        pd.DataFrame
            Costs with models as index, cost types as columns
            (CAPEX, OPEX, Total)
        """
        data = []
        for model_id, stats in self.models.items():
            data.append(
                {
                    "Model": model_id,
                    "CAPEX": stats.capex().sum(),
                    "OPEX": stats.opex().sum(),
                    "Total": stats.total_system_cost(),
                }
            )

        return pd.DataFrame(data).set_index("Model")

    def compare_with_ranges(self) -> pd.DataFrame:
        """Compare models with statistical ranges for context.

        Returns
        -------
        pd.DataFrame
            Capacity comparison with added statistical columns
            (Mean, Min, Max, Std Dev, CV%)
        """
        capacity = self.compare_capacity_mix()

        # Add statistical columns
        stats_df = capacity.copy()
        stats_df["Mean"] = capacity.mean(axis=1)
        stats_df["Min"] = capacity.min(axis=1)
        stats_df["Max"] = capacity.max(axis=1)
        stats_df["Std Dev"] = capacity.std(axis=1)
        stats_df["CV (%)"] = (stats_df["Std Dev"] / stats_df["Mean"]) * 100

        return stats_df

    def identify_outliers(self, threshold: float = 2.0) -> dict:
        """Identify models with outlier values (>N std dev from mean).

        Parameters
        ----------
        threshold : float, default 2.0
            Number of standard deviations to consider as outlier

        Returns
        -------
        dict
            Dictionary mapping model_metric to list of outlier carriers
        """
        outliers = {}

        for metric_name, compare_func in [
            ("capacity", self.compare_capacity_mix),
            ("generation", self.compare_generation_mix),
        ]:
            data = compare_func()
            mean = data.mean(axis=1)
            std = data.std(axis=1)

            for model_id in data.columns:
                z_score = (data[model_id] - mean).abs() / std
                outlier_carriers = data.index[z_score > threshold].tolist()

                if outlier_carriers:
                    outliers[f"{model_id}_{metric_name}"] = outlier_carriers

        return outliers

    def plot_comparison_dashboard(self, output_dir: str | Path) -> None:
        """Generate comparison plots for all models.

        Parameters
        ----------
        output_dir : str | Path
            Directory to save comparison plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Capacity comparison
        capacity = self.compare_capacity_mix()
        fig, ax = plt.subplots(figsize=(14, 7))
        capacity.T.plot.bar(ax=ax, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Installed Capacity (MW)")
        ax.set_title("Capacity Mix Comparison Across Models", fontweight="bold")
        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(
            output_dir / "capacity_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Generation comparison
        generation = self.compare_generation_mix()
        fig, ax = plt.subplots(figsize=(14, 7))
        generation.T.plot.bar(ax=ax, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Energy Generation (MWh)")
        ax.set_title("Generation Mix Comparison Across Models", fontweight="bold")
        ax.legend(title="Carrier", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(
            output_dir / "generation_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 3. Cost comparison
        costs = self.compare_costs()
        fig, ax = plt.subplots(figsize=(10, 6))
        costs.plot.bar(ax=ax, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Cost (Currency Units)")
        ax.set_title("System Cost Comparison Across Models", fontweight="bold")
        ax.legend(title="Cost Type")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(output_dir / "cost_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Comparison plots saved to: {output_dir}")


class PLEXOSBenchmark:
    """Compare PyPSA results against original PLEXOS baseline.

    This class enables validation of PyPSA conversion by comparing results
    against PLEXOS solution files.

    Parameters
    ----------
    pypsa_network_path : Path
        Path to solved PyPSA network
    plexos_results_path : Path
        Path to PLEXOS solution CSV/Excel files

    Examples
    --------
    >>> benchmark = PLEXOSBenchmark(
    ...     pypsa_network_path=Path("results/sem/network.nc"),
    ...     plexos_results_path=Path("data/sem/plexos_solution.csv")
    ... )
    >>> accuracy = benchmark.assess_accuracy()
    """

    def __init__(self, pypsa_network_path: Path, plexos_results_path: Path) -> None:
        """Initialize PLEXOS benchmark."""
        self.pypsa_stats = NetworkStatistics(pypsa_network_path)
        self.plexos_results_path = plexos_results_path
        self.plexos_results = self._load_plexos_results()

    def _load_plexos_results(self) -> dict:
        """Load PLEXOS results from file.

        This is a placeholder - actual implementation depends on PLEXOS
        output format (CSV, Excel, database, etc.)

        Returns
        -------
        dict
            Dictionary with PLEXOS results (generation, costs, etc.)
        """
        # TODO: Implement based on actual PLEXOS output format
        # For now, return empty dict
        print(
            "Warning: PLEXOS results loading not yet implemented. "
            "Override _load_plexos_results() method for your specific format."
        )
        return {"generation": pd.Series(), "total_cost": 0}

    def compare_generation_mix(self) -> pd.DataFrame:
        """Compare total generation by carrier.

        Returns
        -------
        pd.DataFrame
            Comparison table with PyPSA, PLEXOS, Difference columns
        """
        pypsa_gen = self.pypsa_stats.energy_supply()
        plexos_gen = self.plexos_results.get("generation", pd.Series())

        comparison = pd.DataFrame(
            {
                "PyPSA (MWh)": pypsa_gen,
                "PLEXOS (MWh)": plexos_gen,
            }
        )

        comparison["Difference (MWh)"] = (
            comparison["PyPSA (MWh)"] - comparison["PLEXOS (MWh)"]
        )
        comparison["Difference (%)"] = (
            comparison["Difference (MWh)"] / comparison["PLEXOS (MWh)"]
        ) * 100

        return comparison

    def compare_costs(self) -> pd.DataFrame:
        """Compare total system costs.

        Returns
        -------
        pd.DataFrame
            Cost comparison table
        """
        pypsa_cost = self.pypsa_stats.total_system_cost()
        plexos_cost = self.plexos_results.get("total_cost", 0)

        diff_pct = (
            ((pypsa_cost - plexos_cost) / plexos_cost) * 100 if plexos_cost > 0 else 0
        )

        return pd.DataFrame(
            {
                "Metric": ["Total System Cost"],
                "PyPSA": [pypsa_cost],
                "PLEXOS": [plexos_cost],
                "Difference (%)": [diff_pct],
            }
        )

    def assess_accuracy(self) -> dict:
        """Assess overall accuracy with tolerance thresholds.

        Returns
        -------
        dict
            Accuracy assessment with keys:
            - generation_match: bool (within 5%)
            - cost_match: bool (within 10%)
            - max_generation_diff_pct: float
            - cost_diff_pct: float
            - overall_accuracy: str (Excellent/Good/Fair/Poor)
        """
        gen_comparison = self.compare_generation_mix()
        cost_comparison = self.compare_costs()

        # Check if PLEXOS data is available
        if gen_comparison["PLEXOS (MWh)"].sum() == 0:
            return {
                "generation_match": False,
                "cost_match": False,
                "max_generation_diff_pct": float("nan"),
                "cost_diff_pct": float("nan"),
                "overall_accuracy": "No PLEXOS Data",
            }

        max_gen_diff = gen_comparison["Difference (%)"].abs().max()
        cost_diff = cost_comparison["Difference (%)"].iloc[0]

        gen_match = max_gen_diff < 5
        cost_match = abs(cost_diff) < 10

        # Determine overall accuracy
        if gen_match and cost_match:
            accuracy = "Excellent"
        elif max_gen_diff < 15 and abs(cost_diff) < 20:
            accuracy = "Good"
        elif max_gen_diff < 30 and abs(cost_diff) < 40:
            accuracy = "Fair"
        else:
            accuracy = "Poor"

        return {
            "generation_match": gen_match,
            "cost_match": cost_match,
            "max_generation_diff_pct": float(max_gen_diff),
            "cost_diff_pct": float(cost_diff),
            "overall_accuracy": accuracy,
        }
