#!/usr/bin/env python3
"""Integration test for model conversion.

This script tests that models can be successfully converted from PLEXOS
to PyPSA format using the CSV workflow system. It's designed to work
with the CI system (model-tests.yaml).

Usage:
    python tests/integration/test_model_conversion.py \
        --model-id sem-2024-2032 \
        --no-consistency-check \
        --run-solve \
        --solver-name highs \
        --snapshot-limit 50 \
        --output-file sem_stats.txt
"""

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plexos_to_pypsa_converter.db.registry import MODEL_REGISTRY
from plexos_to_pypsa_converter.workflow.executor import run_model_workflow

EXCLUDED_STEPS: set[str] = {"save_network"}


def _build_conversion_workflow(workflow: dict, excluded_steps: Iterable[str]) -> dict:
    """Return a shallow copy of the workflow without excluded steps."""
    workflow_copy = dict(workflow)
    workflow_copy["steps"] = [
        step for step in workflow["steps"] if step["name"] not in excluded_steps
    ]
    return workflow_copy


def _run_optional_consistency_check(network, skip_check: bool) -> None:
    if skip_check:
        return
    print("\nRunning consistency check...")
    network.consistency_check()
    print("  - Consistency check passed")


def _run_optional_solve(
    model_id: str,
    network,
    snapshot_limit: int,
    solver_name: str,
) -> tuple[float, str, int]:
    """Run an optional solve on the converted network."""
    print(f"\n{'-' * 60}")
    print(f"Running solve for {model_id}")
    print(f"Solver: {solver_name}")
    print(f"Snapshot limit: {snapshot_limit}")
    print(f"{'-' * 60}\n")

    snapshots = network.snapshots[:snapshot_limit]
    print(f"  - Original snapshots: {len(network.snapshots)}")
    print(f"  - Limited snapshots: {len(snapshots)}")

    result = network.optimize(snapshots=snapshots, solver_name=solver_name)
    objective = result[0]
    status = result[1]

    print("\nSolve results:")
    print(f"  - Objective: {objective}")
    print(f"  - Status: {status}")

    assert status == "optimal", f"Solve failed with status: {status}"
    assert objective is not None, "No objective value returned"

    return objective, status, len(snapshots)


def test_sem_conversion(args):
    """Test SEM 2024-2032 model converts successfully.

    Tests full workflow EXCEPT optimization:
    - Network creation
    - VRE profile loading
    - Storage inflows
    - Generator units (retirements/builds)
    - Outage parsing and application
    - Ramp conflict fixes
    - Slack generators
    - Generator count (should be >= 50)
    - Bus creation
    - Optional consistency check

    Note: The workflow skips the ``save_network`` step for speed. Optimization
    is always disabled here; optional solves are run separately on a limited
    snapshot set.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: sem-2024-2032")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out save_network
        model_config = MODEL_REGISTRY["sem-2024-2032"]
        workflow = model_config["processing_workflow"]
        workflow_no_save = _build_conversion_workflow(workflow, EXCLUDED_STEPS)

        print("Running workflow (save_network step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "sem-2024-2032",
            workflow_overrides=workflow_no_save,
            solve=False,
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed:")
        for step_name in summary.keys():
            print(f"  - {step_name}")

        # Validate network structure
        print("\nNetwork structure validated:")
        print(f"  - Buses: {len(network.buses)}")
        print(f"  - Generators: {len(network.generators)}")
        print(f"  - Links: {len(network.links)}")
        print(f"  - Storage units: {len(network.storage_units)}")
        print(f"  - Snapshots: {len(network.snapshots)}")

        # Assertions
        assert len(network.buses) > 0, "No buses created"
        assert len(network.generators) >= 50, (
            f"Expected at least 50 generators, got {len(network.generators)}"
        )
        assert len(network.snapshots) > 0, "No snapshots created"

        # Optional consistency check
        _run_optional_consistency_check(network, args.no_consistency_check)

        if args.run_solve:
            objective, status, used_snapshots = _run_optional_solve(
                "sem-2024-2032",
                network,
                snapshot_limit=args.snapshot_limit,
                solver_name=args.solver_name,
            )

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
                if args.run_solve:
                    f.write(f"solve_status={status}\n")
                    f.write(f"solve_objective={objective}\n")
                    f.write(f"solve_snapshots={used_snapshots}\n")
            print(f"\nStats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("SEM conversion test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("SEM conversion test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_aemo_conversion(args):
    """Test AEMO 2024 ISP Progressive Change model converts successfully.

    Tests full workflow EXCEPT optimization:
    - Network creation
    - VRE profile loading
    - Storage inflows
    - Generator units (retirements/builds)
    - Outage parsing and application
    - Generator count (should be >= 100 for AEMO)
    - Bus creation
    - Optional consistency check

    Note: Optimization and save_network steps are removed unless --run-solve
    is supplied.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: aemo-2024-isp-progressive-change")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out save_network
        model_config = MODEL_REGISTRY["aemo-2024-isp-progressive-change"]
        workflow = model_config["processing_workflow"]
        workflow_no_save = _build_conversion_workflow(workflow, EXCLUDED_STEPS)

        print("Running workflow (save_network step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "aemo-2024-isp-progressive-change",
            workflow_overrides=workflow_no_save,
            solve=False,
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed:")
        for step_name in summary.keys():
            print(f"  - {step_name}")

        # Validate network structure
        print("\nNetwork structure validated:")
        print(f"  - Buses: {len(network.buses)}")
        print(f"  - Generators: {len(network.generators)}")
        print(f"  - Links: {len(network.links)}")
        print(f"  - Storage units: {len(network.storage_units)}")
        print(f"  - Snapshots: {len(network.snapshots)}")

        # Assertions
        assert len(network.buses) > 0, "No buses created"
        assert len(network.generators) >= 100, (
            f"Expected at least 100 generators, got {len(network.generators)}"
        )
        assert len(network.snapshots) > 0, "No snapshots created"

        # Optional consistency check
        _run_optional_consistency_check(network, args.no_consistency_check)

        if args.run_solve:
            objective, status, used_snapshots = _run_optional_solve(
                "aemo-2024-isp-progressive-change",
                network,
                snapshot_limit=args.snapshot_limit,
                solver_name=args.solver_name,
            )

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
                if args.run_solve:
                    f.write(f"solve_status={status}\n")
                    f.write(f"solve_objective={objective}\n")
                    f.write(f"solve_snapshots={used_snapshots}\n")
            print(f"\nStats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("AEMO conversion test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("AEMO conversion test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_caiso_conversion(args):
    """Test CAISO IRP23 model converts successfully.

    Tests full workflow EXCEPT optimization:
    - Network creation
    - VRE profile loading
    - Storage inflows
    - Generator units (retirements/builds)
    - Outage parsing and application
    - Ramp conflict fixes
    - Slack generators
    - Generator count (should be >= 100 for CAISO)
    - Bus creation
    - Optional consistency check

    Note: Optimization and save_network steps are removed unless --run-solve
    is supplied.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: caiso-irp23")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out save_network
        model_config = MODEL_REGISTRY["caiso-irp23"]
        workflow = model_config["processing_workflow"]
        workflow_no_save = _build_conversion_workflow(workflow, EXCLUDED_STEPS)

        print("Running workflow (save_network step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "caiso-irp23",
            workflow_overrides=workflow_no_save,
            solve=False,
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed:")
        for step_name in summary.keys():
            print(f"  - {step_name}")

        # Validate network structure
        print("\nNetwork structure validated:")
        print(f"  - Buses: {len(network.buses)}")
        print(f"  - Generators: {len(network.generators)}")
        print(f"  - Links: {len(network.links)}")
        print(f"  - Storage units: {len(network.storage_units)}")
        print(f"  - Snapshots: {len(network.snapshots)}")

        # Assertions
        assert len(network.buses) > 0, "No buses created"
        assert len(network.generators) >= 500, (
            f"Expected at least 500 generators, got {len(network.generators)}"
        )
        assert len(network.snapshots) > 0, "No snapshots created"

        # Optional consistency check
        _run_optional_consistency_check(network, args.no_consistency_check)

        if args.run_solve:
            objective, status, used_snapshots = _run_optional_solve(
                "caiso-irp23",
                network,
                snapshot_limit=args.snapshot_limit,
                solver_name=args.solver_name,
            )

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
                if args.run_solve:
                    f.write(f"solve_status={status}\n")
                    f.write(f"solve_objective={objective}\n")
                    f.write(f"solve_snapshots={used_snapshots}\n")
            print(f"\nStats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("CAISO IRP23 conversion test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("CAISO IRP23 conversion test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Test PLEXOS to PyPSA model conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test SEM conversion
    python tests/integration/test_model_conversion.py --model-id sem-2024-2032 --output-file sem_stats.txt

    # Test AEMO conversion without consistency check
    python tests/integration/test_model_conversion.py --model-id aemo-2024-isp-progressive-change --no-consistency-check --output-file aemo_stats.txt

    # Test CAISO conversion
    python tests/integration/test_model_conversion.py --model-id caiso-irp23 --output-file caiso_stats.txt

    # Test SEM conversion plus solve
    python tests/integration/test_model_conversion.py --model-id sem-2024-2032 --run-solve --snapshot-limit 50 --output-file sem_stats.txt
        """,
    )
    parser.add_argument(
        "--model-id",
        required=True,
        choices=["sem-2024-2032", "aemo-2024-isp-progressive-change", "caiso-irp23"],
        help="Model ID to test",
    )
    parser.add_argument(
        "--no-consistency-check",
        action="store_true",
        help="Skip PyPSA consistency check (faster)",
    )
    parser.add_argument(
        "--run-solve",
        action="store_true",
        help="Run a limited optimization after conversion",
    )
    parser.add_argument(
        "--solver-name",
        default="highs",
        help="Solver to use when running the optional optimization (default: highs)",
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=50,
        help="Number of snapshots to include in the optional solve (default: 50)",
    )
    parser.add_argument(
        "--output-file",
        help="File to write combined conversion and optional solve statistics",
    )

    args = parser.parse_args()

    # Route to appropriate test function
    if args.model_id == "sem-2024-2032":
        success = test_sem_conversion(args)
    elif args.model_id == "aemo-2024-isp-progressive-change":
        success = test_aemo_conversion(args)
    elif args.model_id == "caiso-irp23":
        success = test_caiso_conversion(args)
    else:
        print(f"Unknown model: {args.model_id}")
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
