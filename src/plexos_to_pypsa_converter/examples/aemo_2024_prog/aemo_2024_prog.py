"""AEMO 2024 ISP Progressive Change Model - Workflow execution.

This script demonstrates using the registry-driven workflow system to execute
the standard processing pipeline for the AEMO Progressive Change model.

For manual/custom workflow execution, see legacy/aemo_2024_prog_manual.py

To run the default workflow (including optimization):
    network, summary = run_model_workflow("aemo-2024-isp-progressive-change", solve=True)
To override specific parameters:
    network, summary = run_model_workflow(
        "aemo-2024-isp-progressive-change",
        solve=True,
        optimize__year=2026,  # Different year
    )
"""

from plexos_to_pypsa_converter.workflow import run_model_workflow

if __name__ == "__main__":
    # Execute standard workflow from registry
    network, summary = run_model_workflow(
        "aemo-2024-isp-progressive-change", solve=True
    )

    print("\n" + "=" * 60)
    print("AEMO 2024 ISP Progressive Change Workflow Complete")
    print("=" * 60)
