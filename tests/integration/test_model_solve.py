#!/usr/bin/env python3
"""Integration test for model solving with limited snapshots.

This script tests that models can be successfully optimized using PyPSA
after conversion. To keep tests fast, it limits the number of snapshots
to 50 by default.

Usage:
    python tests/integration/test_model_solve.py \
        --model-id sem-2024-2032 \
        --snapshot-limit 50 \
        --output-file solve_stats.txt
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from workflow.executor import run_model_workflow


def test_sem_solve_limited_snapshots(args):
    """Test SEM 2024-2032 model solves successfully with limited snapshots.

    Note: This runs the full workflow including optimization, but only
    solves for the first N snapshots to keep tests fast.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if solve succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing solve: sem-2024-2032")
    print(f"Snapshot limit: {args.snapshot_limit}")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step
        from db.registry import MODEL_REGISTRY

        model_config = MODEL_REGISTRY["sem-2024-2032"]
        workflow = model_config["processing_workflow"]
        workflow_no_optimize = workflow.copy()
        workflow_no_optimize["steps"] = [
            step for step in workflow["steps"] if step["name"] != "optimize"
        ]

        print("Running workflow (optimize step excluded for solve test)...")
        network, summary = run_model_workflow(
            "sem-2024-2032", workflow_overrides=workflow_no_optimize
        )

        # Limit snapshots for fast testing
        snapshots = network.snapshots[: args.snapshot_limit]
        print(f"  - Original snapshots: {len(network.snapshots)}")
        print(f"  - Limited snapshots: {len(snapshots)}")

        # Run optimization with limited snapshots using default solver
        result = network.optimize(snapshots=snapshots, solver_name="highs")

        solve = result[0]
        status = result[1]

        # Validate solve results
        print("\n✓ Solve results:")
        print(f"  - Objective: {solve}")
        print(f"  - Status: {status}")

        assert status == "optimal", f"Solve failed with status: {status}"
        assert solve is not None, "No objective value returned"

        # Write stats to output file
        if args.output_file:
            from pathlib import Path

            with Path(args.output_file).open("w") as f:
                f.write(f"status={status}\n")
                f.write(f"objective={solve}\n")
                f.write(
                    f"snapshots={args.snapshot_limit if 'optimize' not in summary else summary['optimize'].get('snapshots_count')}\n"
                )
            print(f"\nSolve stats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("SEM solve test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("SEM solve test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_aemo_solve_limited_snapshots(args):
    """Test AEMO 2024 ISP Progressive Change model solves successfully.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if solve succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing solve: aemo-2024-isp-progressive-change")
    print(f"Snapshot limit: {args.snapshot_limit}")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step
        from db.registry import MODEL_REGISTRY

        model_config = MODEL_REGISTRY["aemo-2024-isp-progressive-change"]
        workflow = model_config["processing_workflow"]
        workflow_no_optimize = workflow.copy()
        workflow_no_optimize["steps"] = [
            step for step in workflow["steps"] if step["name"] != "optimize"
        ]

        print("Running workflow (optimize step excluded for solve test)...")
        network, summary = run_model_workflow(
            "aemo-2024-isp-progressive-change", workflow_overrides=workflow_no_optimize
        )

        # Limit snapshots for fast testing
        snapshots = network.snapshots[: args.snapshot_limit]
        print(f"  - Original snapshots: {len(network.snapshots)}")
        print(f"  - Limited snapshots: {len(snapshots)}")

        # Run optimization with limited snapshots using default solver
        result = network.optimize(snapshots=snapshots, solver_name="highs")

        solve = result[0]
        status = result[1]

        # Validate solve results
        print("\n✓ Solve results:")
        print(f"  - Objective: {solve}")
        print(f"  - Status: {status}")

        assert status == "optimal", f"Solve failed with status: {status}"
        assert solve is not None, "No objective value returned"

        # Write stats to output file
        if args.output_file:
            from pathlib import Path

            with Path(args.output_file).open("w") as f:
                f.write(f"status={status}\n")
                f.write(f"objective={solve}\n")
                f.write(
                    f"snapshots={args.snapshot_limit if 'optimize' not in summary else summary['optimize'].get('snapshots_count')}\n"
                )
            print(f"\nSolve stats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("AEMO solve test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("AEMO solve test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_caiso_solve_limited_snapshots(args):
    """Test CAISO IRP23 model solves successfully with limited snapshots.

    Args:
        args: Command-line arguments

    Returns:
        bool: True if solve succeeds
    """
    print(f"\n{'=' * 60}")
    print("Testing solve: caiso-irp23")
    print(f"Snapshot limit: {args.snapshot_limit}")
    print(f"{'=' * 60}\n")

    try:
        # Get workflow from registry and filter out optimize step
        from db.registry import MODEL_REGISTRY

        model_config = MODEL_REGISTRY["caiso-irp23"]
        workflow = model_config["processing_workflow"]
        workflow_no_optimize = workflow.copy()
        workflow_no_optimize["steps"] = [
            step for step in workflow["steps"] if step["name"] != "optimize"
        ]

        print("Running workflow (optimize step excluded for solve test)...")
        network, summary = run_model_workflow(
            "caiso-irp23", workflow_overrides=workflow_no_optimize
        )

        # Limit snapshots for fast testing
        snapshots = network.snapshots[: args.snapshot_limit]
        print(f"  - Original snapshots: {len(network.snapshots)}")
        print(f"  - Limited snapshots: {len(snapshots)}")

        # Run optimization with limited snapshots using default solver
        result = network.optimize(snapshots=snapshots, solver_name="highs")

        solve = result[0]
        status = result[1]

        # Validate solve results
        print("\n✓ Solve results:")
        print(f"  - Objective: {solve}")
        print(f"  - Status: {status}")

        assert status == "optimal", f"Solve failed with status: {status}"
        assert solve is not None, "No objective value returned"

        # Write stats to output file
        if args.output_file:
            from pathlib import Path

            with Path(args.output_file).open("w") as f:
                f.write(f"status={status}\n")
                f.write(f"objective={solve}\n")
                f.write(
                    f"snapshots={args.snapshot_limit if 'optimize' not in summary else summary['optimize'].get('snapshots_count')}\n"
                )
            print(f"\nSolve stats written to {args.output_file}")

        print(f"\n{'=' * 60}")
        print("CAISO IRP23 solve test PASSED")
        print(f"{'=' * 60}\n")

        return True

    except Exception as e:
        print(f"\n{'=' * 60}")
        print("CAISO IRP23 solve test FAILED")
        print(f"{'=' * 60}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Test PLEXOS to PyPSA model optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test SEM solve with 50 snapshots
    python tests/integration/test_model_solve.py --model-id sem-2024-2032 --snapshot-limit 50 --output-file sem_solve.txt

    # Test AEMO solve with 100 snapshots
    python tests/integration/test_model_solve.py --model-id aemo-2024-isp-progressive-change --snapshot-limit 100 --output-file aemo_solve.txt

    # Test CAISO solve with default snapshot limit
    python tests/integration/test_model_solve.py --model-id caiso-irp23 --output-file caiso_solve.txt
    """,
    )
    parser.add_argument(
        "--model-id",
        required=True,
        choices=["sem-2024-2032", "aemo-2024-isp-progressive-change", "caiso-irp23"],
        help="Model ID to test",
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=50,
        help="Number of snapshots to solve (default: 50)",
    )
    parser.add_argument(
        "--output-file", required=True, help="File to write solve statistics"
    )

    args = parser.parse_args()

    # Route to appropriate test function
    if args.model_id == "sem-2024-2032":
        success = test_sem_solve_limited_snapshots(args)
    elif args.model_id == "aemo-2024-isp-progressive-change":
        success = test_aemo_solve_limited_snapshots(args)
    elif args.model_id == "caiso-irp23":
        success = test_caiso_solve_limited_snapshots(args)
    else:
        print(f"Unknown model: {args.model_id}")
        sys.exit(1)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
