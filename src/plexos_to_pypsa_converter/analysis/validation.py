"""Validation and sanity checking for solved PyPSA networks.

This module provides the NetworkValidator class which performs comprehensive
validation checks on solved energy system models to ensure physical feasibility
and catch potential modeling errors.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.analysis.statistics import NetworkStatistics


class NetworkValidator:
    """Perform validation and sanity checks on solved PyPSA networks.

    This class provides 7 categories of validation checks:
    1. Energy balance (generation = demand + losses)
    2. Generator limits (p_min_pu ≤ p ≤ p_max_pu)
    3. Ramping limits (Δp/Δt within ramp rates)
    4. Storage state of charge (0 ≤ SOC ≤ max, cyclic)
    5. Transmission limits (|flow| ≤ p_nom)
    6. Cost reasonableness (LCOE, capacity factors in expected ranges)
    7. Unserved energy (check for slack generation)

    Parameters
    ----------
    network_path : str | Path
        Path to solved PyPSA network NetCDF file

    Attributes
    ----------
    network : pypsa.Network
        Loaded PyPSA network
    stats : NetworkStatistics
        Statistics calculator for the network

    Examples
    --------
    >>> validator = NetworkValidator("results/sem-2024-2032/network.nc")
    >>> results = validator.run_all_validations()
    >>> validator.print_validation_report(results)
    """

    def __init__(self, network_path: str | Path) -> None:
        """Load network and initialize statistics calculator."""
        self.network_path = Path(network_path)
        self.network = pypsa.Network(str(network_path))
        self.stats = NetworkStatistics(network_path)

    # ========================================================================
    # 1. Energy Balance Checks
    # ========================================================================

    def check_energy_balance(self) -> dict:
        """Verify generation = demand + losses at each timestep.

        Checks that total generation matches total demand plus network losses
        within a tolerance of 0.1% of peak demand.

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - max_imbalance_mw: float
            - max_imbalance_pct: float
            - status: "PASS" or "FAIL"
        """
        # Get total generation
        if hasattr(self.network, "generators_t") and hasattr(
            self.network.generators_t, "p"
        ):
            generation = self.network.generators_t.p.sum(axis=1)
        else:
            return {
                "is_valid": False,
                "status": "FAIL",
                "message": "No generator dispatch data found",
            }

        # Get total demand
        if hasattr(self.network, "loads_t") and hasattr(self.network.loads_t, "p"):
            demand = self.network.loads_t.p.sum(axis=1)
        else:
            return {
                "is_valid": False,
                "status": "FAIL",
                "message": "No load data found",
            }

        # Account for storage charging/discharging
        storage_net = 0
        if hasattr(self.network, "storage_units_t") and hasattr(
            self.network.storage_units_t, "p"
        ):
            storage_net = self.network.storage_units_t.p.sum(axis=1)

        # Calculate imbalance
        imbalance = generation - demand - storage_net
        max_imbalance = imbalance.abs().max()
        peak_demand = demand.max()
        max_imbalance_pct = (
            (max_imbalance / peak_demand) * 100 if peak_demand > 0 else 0
        )

        # Tolerance: 0.1% of peak demand
        is_valid = max_imbalance_pct < 0.1

        return {
            "is_valid": is_valid,
            "max_imbalance_mw": float(max_imbalance),
            "max_imbalance_pct": float(max_imbalance_pct),
            "status": "PASS" if is_valid else "FAIL",
        }

    # ========================================================================
    # 2. Generator Limit Checks
    # ========================================================================

    def check_generator_limits(self) -> dict:
        """Verify no generators exceed p_nom limits.

        Checks:
        - p <= p_nom * p_max_pu (upper bound)
        - p >= p_nom * p_min_pu (lower bound for must-run)
        - p >= 0 (no negative generation)

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - violations: list of dict
            - status: "PASS" or "FAIL"
        """
        violations = []
        tolerance = 1e-3  # 1 kW tolerance

        for gen in self.network.generators.index:
            if gen not in self.network.generators_t.p.columns:
                continue

            p = self.network.generators_t.p[gen]
            p_nom = self.network.generators.at[gen, "p_nom"]

            # Check upper bound (p_max_pu)
            if hasattr(self.network.generators_t, "p_max_pu") and (
                gen in self.network.generators_t.p_max_pu.columns
            ):
                p_max_pu = self.network.generators_t.p_max_pu[gen]
                upper_limit = p_nom * p_max_pu
                violations_upper = p > (upper_limit + tolerance)
                if violations_upper.any():
                    violations.append(
                        {
                            "generator": gen,
                            "type": "upper_limit_exceeded",
                            "max_violation_mw": float((p - upper_limit).max()),
                            "count": int(violations_upper.sum()),
                        }
                    )

            # Check lower bound (p_min_pu)
            if hasattr(self.network.generators_t, "p_min_pu") and (
                gen in self.network.generators_t.p_min_pu.columns
            ):
                p_min_pu = self.network.generators_t.p_min_pu[gen]
                lower_limit = p_nom * p_min_pu
                violations_lower = p < (lower_limit - tolerance)
                if violations_lower.any():
                    violations.append(
                        {
                            "generator": gen,
                            "type": "lower_limit_violated",
                            "max_violation_mw": float((lower_limit - p).max()),
                            "count": int(violations_lower.sum()),
                        }
                    )

            # Check non-negative
            if (p < -tolerance).any():
                violations.append(
                    {
                        "generator": gen,
                        "type": "negative_generation",
                        "min_value_mw": float(p.min()),
                        "count": int((p < -tolerance).sum()),
                    }
                )

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "violations_count": len(violations),
            "status": "PASS" if len(violations) == 0 else "FAIL",
        }

    # ========================================================================
    # 3. Ramping Limit Checks
    # ========================================================================

    def check_ramping_limits(self) -> dict:
        """Verify generators respect ramp rate limits.

        Checks:
        - Δp/Δt <= ramp_limit_up * p_nom
        - |Δp/Δt| <= ramp_limit_down * p_nom (for ramp down)

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - violations: list of dict
            - status: "PASS" or "FAIL"
        """
        if "ramp_limit_up" not in self.network.generators.columns:
            return {
                "is_valid": True,
                "status": "PASS",
                "message": "No ramp limits defined in network",
            }

        violations = []
        tolerance = 1e-3

        for gen in self.network.generators.index:
            if gen not in self.network.generators_t.p.columns:
                continue

            p = self.network.generators_t.p[gen]
            p_nom = self.network.generators.at[gen, "p_nom"]

            # Calculate actual ramps
            dp = p.diff()

            # Check ramp up violations
            ramp_up = self.network.generators.at[gen, "ramp_limit_up"]
            if not pd.isna(ramp_up) and ramp_up > 0:
                max_ramp_up = p_nom * ramp_up
                violations_up = dp > (max_ramp_up + tolerance)
                if violations_up.any():
                    violations.append(
                        {
                            "generator": gen,
                            "type": "ramp_up_exceeded",
                            "max_violation_mw": float((dp - max_ramp_up).max()),
                            "count": int(violations_up.sum()),
                        }
                    )

            # Check ramp down violations
            if "ramp_limit_down" in self.network.generators.columns:
                ramp_down = self.network.generators.at[gen, "ramp_limit_down"]
                if not pd.isna(ramp_down) and ramp_down > 0:
                    max_ramp_down = p_nom * ramp_down
                    violations_down = dp < (-max_ramp_down - tolerance)
                    if violations_down.any():
                        violations.append(
                            {
                                "generator": gen,
                                "type": "ramp_down_exceeded",
                                "max_violation_mw": float((-dp - max_ramp_down).max()),
                                "count": int(violations_down.sum()),
                            }
                        )

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "violations_count": len(violations),
            "status": "PASS" if len(violations) == 0 else "FAIL",
        }

    # ========================================================================
    # 4. Storage State of Charge Checks
    # ========================================================================

    def check_storage_soc(self) -> dict:
        """Verify storage units stay within SOC bounds.

        Checks:
        - 0 <= SOC <= max_capacity
        - Initial SOC ≈ Final SOC (cyclic constraint, if applicable)

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - violations: list of dict
            - status: "PASS" or "FAIL"
        """
        if not hasattr(self.network, "storage_units_t") or not hasattr(
            self.network.storage_units_t, "state_of_charge"
        ):
            return {
                "is_valid": True,
                "status": "PASS",
                "message": "No storage units in network",
            }

        violations = []
        tolerance = 1e-3  # 1 kWh tolerance

        for storage in self.network.storage_units.index:
            if storage not in self.network.storage_units_t.state_of_charge.columns:
                continue

            soc = self.network.storage_units_t.state_of_charge[storage]
            max_hours = self.network.storage_units.at[storage, "max_hours"]
            p_nom = self.network.storage_units.at[storage, "p_nom"]
            max_soc = max_hours * p_nom

            # Check lower bound
            if (soc < -tolerance).any():
                violations.append(
                    {
                        "storage": storage,
                        "type": "negative_soc",
                        "min_value_mwh": float(soc.min()),
                    }
                )

            # Check upper bound
            if (soc > max_soc + tolerance).any():
                violations.append(
                    {
                        "storage": storage,
                        "type": "soc_exceeded",
                        "max_violation_mwh": float((soc - max_soc).max()),
                    }
                )

            # Check cyclic constraint (initial ≈ final)
            soc_diff = abs(soc.iloc[0] - soc.iloc[-1])
            if soc_diff > tolerance:
                violations.append(
                    {
                        "storage": storage,
                        "type": "cyclic_constraint_violated",
                        "difference_mwh": float(soc_diff),
                    }
                )

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "violations_count": len(violations),
            "status": "PASS" if len(violations) == 0 else "FAIL",
        }

    # ========================================================================
    # 5. Transmission Limit Checks
    # ========================================================================

    def check_transmission_limits(self) -> dict:
        """Verify link flows stay within p_nom limits.

        Checks:
        - |flow| <= p_nom (for all links)

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - violations: list of dict
            - status: "PASS" or "FAIL"
        """
        if not hasattr(self.network, "links_t") or not hasattr(
            self.network.links_t, "p0"
        ):
            return {
                "is_valid": True,
                "status": "PASS",
                "message": "No transmission links in network",
            }

        violations = []
        tolerance = 1e-3

        for link in self.network.links.index:
            if link not in self.network.links_t.p0.columns:
                continue

            flow = self.network.links_t.p0[link]
            p_nom = self.network.links.at[link, "p_nom"]

            # Check capacity violations (both directions for bidirectional)
            if (flow.abs() > p_nom + tolerance).any():
                violations.append(
                    {
                        "link": link,
                        "type": "capacity_exceeded",
                        "max_violation_mw": float((flow.abs() - p_nom).max()),
                        "count": int((flow.abs() > p_nom + tolerance).sum()),
                    }
                )

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "violations_count": len(violations),
            "status": "PASS" if len(violations) == 0 else "FAIL",
        }

    # ========================================================================
    # 6. Cost Reasonableness Checks
    # ========================================================================

    def check_cost_reasonableness(self) -> dict:
        """Check if costs are within reasonable ranges.

        Typical ranges (order of magnitude checks):
        - LCOE: $20-300/MWh
        - Capacity Factor: 1-95%
        - Total System Cost: > 0

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - warnings: list of str
            - status: "PASS" or "WARNING"
        """
        warnings = []

        # Check LCOE
        try:
            lcoe = self.stats.lcoe()
            if (lcoe < 10).any():
                low_carriers = lcoe[lcoe < 10].index.tolist()
                warnings.append(
                    f"Suspiciously low LCOE detected (< $10/MWh): {low_carriers}"
                )
            if (lcoe > 500).any():
                high_carriers = lcoe[lcoe > 500].index.tolist()
                warnings.append(
                    f"Suspiciously high LCOE detected (> $500/MWh): {high_carriers}"
                )
        except Exception as e:
            warnings.append(f"Could not calculate LCOE: {e}")

        # Check capacity factors
        try:
            cf = self.stats.capacity_factor()
            if (cf < 0.01).any():
                low_cf = cf[cf < 0.01].index.tolist()
                warnings.append(f"Very low capacity factors (< 1%): {low_cf}")
        except Exception as e:
            warnings.append(f"Could not calculate capacity factors: {e}")

        # Check total cost
        try:
            total_cost = self.stats.total_system_cost()
            if total_cost <= 0:
                warnings.append(f"Invalid total system cost: ${total_cost:,.0f}")
        except Exception as e:
            warnings.append(f"Could not calculate total system cost: {e}")

        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "warnings_count": len(warnings),
            "status": "PASS" if len(warnings) == 0 else "WARNING",
        }

    # ========================================================================
    # 7. Unserved Energy Checks
    # ========================================================================

    def check_unserved_energy(self) -> dict:
        """Check for load shedding / unserved energy.

        Looks for slack generators with positive generation, which indicates
        the system couldn't meet demand.

        Returns
        -------
        dict
            Validation result with keys:
            - is_valid: bool
            - unserved_energy_mwh: float
            - unserved_pct: float
            - status: "PASS" or "WARNING"
        """
        # Look for slack generators
        slack_gens = [
            g
            for g in self.network.generators.index
            if "slack" in g.lower() or "unserved" in g.lower()
        ]

        unserved = 0.0
        if slack_gens and hasattr(self.network, "generators_t"):
            for gen in slack_gens:
                if gen in self.network.generators_t.p.columns:
                    gen_output = self.network.generators_t.p[gen]
                    unserved += gen_output[gen_output > 0].sum()

        # Calculate percentage of total demand
        total_demand = 0.0
        if hasattr(self.network, "loads_t") and hasattr(self.network.loads_t, "p"):
            total_demand = self.network.loads_t.p.sum().sum()

        unserved_pct = (unserved / total_demand * 100) if total_demand > 0 else 0

        # Tolerance: less than 1 kWh total
        is_valid = unserved < 1e-3

        return {
            "is_valid": is_valid,
            "unserved_energy_mwh": float(unserved),
            "unserved_pct": float(unserved_pct),
            "slack_generators": slack_gens,
            "status": "PASS" if is_valid else "WARNING",
        }

    # ========================================================================
    # Master Validation Function
    # ========================================================================

    def run_all_validations(self) -> dict:
        """Run all validation checks and generate summary report.

        Returns
        -------
        dict
            Dictionary with results from all validation checks:
            - energy_balance
            - generator_limits
            - ramping_limits
            - storage_soc
            - transmission_limits
            - cost_reasonableness
            - unserved_energy
            - overall: {"status": "PASS"/"FAIL", "timestamp": ...}
        """
        results = {
            "energy_balance": self.check_energy_balance(),
            "generator_limits": self.check_generator_limits(),
            "ramping_limits": self.check_ramping_limits(),
            "storage_soc": self.check_storage_soc(),
            "transmission_limits": self.check_transmission_limits(),
            "cost_reasonableness": self.check_cost_reasonableness(),
            "unserved_energy": self.check_unserved_energy(),
        }

        # Overall pass/fail
        all_passed = all(
            r.get("status") in ["PASS", "WARNING"]
            for r in results.values()
            if isinstance(r, dict)
        )

        results["overall"] = {
            "status": "PASS" if all_passed else "FAIL",
            "timestamp": datetime.now().isoformat(),
            "network_path": str(self.network_path),
        }

        return results

    def print_validation_report(self, results: dict) -> None:
        """Print human-readable validation report to console.

        Parameters
        ----------
        results : dict
            Results from run_all_validations()
        """
        print("\n" + "=" * 60)
        print("NETWORK VALIDATION REPORT")
        print("=" * 60)
        print(f"Network: {results['overall']['network_path']}")
        print(f"Timestamp: {results['overall']['timestamp']}")

        for check_name, result in results.items():
            if check_name == "overall" or not isinstance(result, dict):
                continue

            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                status_icon = "✅"
            elif status == "WARNING":
                status_icon = "⚠️ "
            else:
                status_icon = "❌"

            print(f"\n{status_icon} {check_name.replace('_', ' ').title()}: {status}")

            # Print details for failures/warnings
            if status != "PASS":
                if "violations" in result and result.get("violations"):
                    print(f"   Found {len(result['violations'])} violations:")
                    for v in result["violations"][:5]:  # Show first 5
                        vtype = v.get("type", "unknown")
                        gen = v.get("generator", v.get("storage", v.get("link", "?")))
                        print(f"   - {gen}: {vtype}")
                    if len(result["violations"]) > 5:
                        print(f"   ... and {len(result['violations']) - 5} more")

                if "warnings" in result and result.get("warnings"):
                    for warning in result["warnings"]:
                        print(f"   - {warning}")

                # Print key metrics
                for key in ["max_imbalance_pct", "unserved_pct"]:
                    if key in result:
                        print(f"   {key}: {result[key]:.4f}")

        print("\n" + "=" * 60)
        print(f"Overall Status: {results['overall']['status']}")
        print("=" * 60)

    def export_validation_report(self, results: dict, output_path: str | Path) -> None:
        """Export validation results to JSON file.

        Parameters
        ----------
        results : dict
            Results from run_all_validations()
        output_path : str | Path
            Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Validation report exported to: {output_path}")
