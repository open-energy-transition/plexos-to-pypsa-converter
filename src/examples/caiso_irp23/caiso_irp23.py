"""CAISO IRP 2023 Stochastic Model - Workflow execution.

This script demonstrates using the registry-driven workflow system to execute
the standard processing pipeline for the CAISO IRP23 model.

For manual/custom workflow execution, see legacy/caiso_irp23_manual.py

To run the default workflow:
    network, summary = run_model_workflow("caiso-irp23")
To override specific parameters:
    network, summary = run_model_workflow(
        "caiso-irp23",
        optimize__year=2026,  # Override optimization year
    )
"""

from workflow import run_model_workflow

if __name__ == "__main__":
    # Execute standard workflow from registry
    network, summary = run_model_workflow("caiso-irp23")

    print("\n" + "=" * 60)
    print("CAISO IRP 2023 Stochastic Workflow Complete")
    print("=" * 60)
