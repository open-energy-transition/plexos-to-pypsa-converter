"""NREL Extended IEEE 118-bus Model - Workflow execution.

This script demonstrates using the registry-driven workflow system to execute
the standard processing pipeline for the NREL 118-bus model.

For manual/custom workflow execution, see legacy/nrel-118_manual.py
"""

from plexos_to_pypsa_converter.workflow import run_model_workflow

if __name__ == "__main__":
    # Execute standard workflow from registry
    network, summary = run_model_workflow("nrel-118", solve=True)

    print("\n" + "=" * 60)
    print("NREL Extended IEEE 118-bus Workflow Complete")
    print("=" * 60)
    print(
        f"\nNetwork: {len(network.buses)} buses, {len(network.generators)} generators"
    )
    print(f"Snapshots: {len(network.snapshots)}")
    print(f"\nOptimization status: {network.results.status}")
    print(f"Objective value: {network.objective:,.0f}")
