"""SEM 2024-2032 Validation Model - Workflow execution.

This script demonstrates using the registry-driven workflow system to execute
the standard processing pipeline for the SEM model.

For manual/custom workflow execution, see legacy/sem_2024_manual.py

To run the default workflow:
    network, summary = run_model_workflow("sem-2024-2032", solve=True)

To override specific parameters:
    network, summary = run_model_workflow(
        "sem-2024-2032",
        solve=True,
        optimize__year=2024,  # Override optimization year
        scale_p_min_pu__scaling_factor=0.5,  # Different scaling
    )
"""

from plexos_to_pypsa_converter.workflow import run_model_workflow

if __name__ == "__main__":
    # Execute standard workflow from registry
    network, summary = run_model_workflow("sem-2024-2032", solve=True)

    print("\n" + "=" * 60)
    print("SEM 2024-2032 Workflow Complete")
    print("=" * 60)
    print(
        f"\nNetwork: {len(network.buses)} buses, {len(network.generators)} generators"
    )
