"""AEMO 2024 ISP Progressive Change Model - Workflow execution.

This script demonstrates using the registry-driven workflow system to execute
the standard processing pipeline for the AEMO Progressive Change model.

For manual/custom workflow execution, see legacy/aemo_2024_prog_manual.py

To run the default workflow:
    network, summary = run_model_workflow("aemo-2024-isp-progressive-change")
To override specific parameters:
    network, summary = run_model_workflow(
        "aemo-2024-isp-progressive-change",
        optimize__year=2026,  # Different year
    )
"""

from workflow import run_model_workflow

if __name__ == "__main__":
    # Execute standard workflow from registry
    network, summary = run_model_workflow(
        "aemo-2024-isp-progressive-change",
        use_investment_periods=True,
        optimization_period=2030,
    )

    print("\n" + "=" * 60)
    print("AEMO 2024 ISP Progressive Change Workflow Complete")
    print("=" * 60)
    print(
        f"\nNetwork: {len(network.buses)} buses, {len(network.generators)} generators"
    )
    print(f"Snapshots: {len(network.snapshots)}")
    opt_summary = summary.get("optimize", {})
    period_filter = opt_summary.get("period_filter")
    label_map = getattr(network, "investment_period_label_map", {})
    if period_filter is not None:
        label = label_map.get(int(period_filter), str(period_filter))
        print(f"Optimisation period: {label} (idx {period_filter})")
    print(f"\nOptimization status: {network.results.status}")
    print(f"Objective value: {network.objective:,.0f}")
