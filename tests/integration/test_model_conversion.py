#!/usr/bin/env python3
"""Integration test for model conversion.

This script tests that models can be successfully converted from PLEXOS
to PyPSA format using the CSV workflow system. It's designed to work
with the CI system (model-tests.yaml).

Usage:
    python tests/integration/test_model_conversion.py \
        --model-id sem-2024-2032 \
        --no-consistency-check \
        --output-file model_stats.txt
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from db.registry import MODEL_REGISTRY
from workflow.executor import run_model_workflow


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

    Note: Optimization is skipped for faster CI testing.
          Use test_model_solve.py to test solving.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: sem-2024-2032")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step
        model_config = MODEL_REGISTRY["sem-2024-2032"]
        workflow = model_config["processing_workflow"]

        # Create modified workflow without optimize step
        workflow_no_optimize = {
            "csv_dir_pattern": workflow.get("csv_dir_pattern"),
            "solver_config": workflow.get("solver_config"),
            "steps": [
                step
                for step in workflow["steps"]
                if step["name"] not in ["optimize", "save_network"]
            ],
        }

        print("Running workflow (optimize step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "sem-2024-2032", workflow_overrides=workflow_no_optimize
        )

        # Verify optimize step did NOT run
        assert "optimize" not in summary, (
            "Optimization should not run in conversion test"
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed (optimize excluded):")
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
        if not args.no_consistency_check:
            print("\nRunning consistency check...")
            network.consistency_check()
            print("  - Consistency check passed")

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
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

    Note: Optimization is skipped for faster CI testing.
          Use test_model_solve.py to test solving.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: aemo-2024-isp-progressive-change")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step
        model_config = MODEL_REGISTRY["aemo-2024-isp-progressive-change"]
        workflow = model_config["processing_workflow"]

        # Create modified workflow without optimize step
        workflow_no_optimize = {
            "csv_dir_pattern": workflow.get("csv_dir_pattern"),
            "solver_config": workflow.get("solver_config"),
            "steps": [
                step
                for step in workflow["steps"]
                if step["name"] not in ["optimize", "save_network"]
            ],
        }

        print("Running workflow (optimize step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "aemo-2024-isp-progressive-change", workflow_overrides=workflow_no_optimize
        )

        # Verify optimize step did NOT run
        assert "optimize" not in summary, (
            "Optimization should not run in conversion test"
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed (optimize excluded):")
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
        if not args.no_consistency_check:
            print("\nRunning consistency check...")
            network.consistency_check()
            print("  - Consistency check passed")

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
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

    Note: Optimization is skipped for faster CI testing.
          Use test_model_solve.py to test solving.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if conversion succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing conversion: caiso-irp23")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step only
        model_config = MODEL_REGISTRY["caiso-irp23"]
        workflow = model_config["processing_workflow"]
        workflow_no_optimize = workflow.copy()
        workflow_no_optimize["steps"] = [
            step
            for step in workflow["steps"]
            if step["name"] not in ["optimize", "save_network"]
        ]

        print("Running workflow (optimize step excluded for faster testing)...")
        network, summary = run_model_workflow(
            "caiso-irp23", workflow_overrides=workflow_no_optimize
        )

        # Verify optimize step did NOT run
        assert "optimize" not in summary, (
            "Optimization should not run in conversion test"
        )

        # Show workflow steps that were completed
        print("\nWorkflow steps completed (optimize excluded):")
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
        if not args.no_consistency_check:
            print("\nRunning consistency check...")
            network.consistency_check()
            print("  - Consistency check passed")

        # Write stats to output file
        if args.output_file:
            with Path(args.output_file).open("w") as f:
                f.write(f"buses={len(network.buses)}\n")
                f.write(f"generators={len(network.generators)}\n")
                f.write(f"links={len(network.links)}\n")
                f.write(f"storage_units={len(network.storage_units)}\n")
                f.write(f"snapshots={len(network.snapshots)}\n")
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
        "--output-file", required=True, help="File to write conversion statistics"
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
