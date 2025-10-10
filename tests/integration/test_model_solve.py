#!/usr/bin/env python3
"""Integration tests for PLEXOS model solving.

Usage as CLI:
    python tests/integration/test_model_solve.py --model-id sem-2024-2032 [--snapshot-limit 50] [--solver highs]

Usage with pytest:
    pytest tests/integration/test_model_solve.py -v
"""

from pathlib import Path

import pytest


def run_model_solve_test(
    model_id: str,
    snapshot_limit: int = 50,
    solver: str = "highs",
    output_file: str | None = None,
) -> dict:
    """Test model solving with a subset of snapshots."""
    from src.network.conversion import create_model

    network, _ = create_model(model_id)
    snapshots_subset = network.snapshots[:snapshot_limit]

    print(
        f"\nSolving {model_id} with {len(snapshots_subset)} snapshots using {solver} solver..."
    )
    network.optimize(solver_name=solver, snapshots=snapshots_subset)

    result = {
        "objective": float(network.objective),
        "status": str(network.status),
        "snapshots_solved": len(snapshots_subset),
    }

    print(f"Model {model_id} solved successfully!")
    print(f"   Objective value: {result['objective']:.2f}")
    print(f"   Status: {result['status']}")

    if output_file:
        Path(output_file).write_text(
            "\n".join(f"{k}={v}" for k, v in result.items()) + "\n"
        )

    return result


# Pytest tests
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["sem-2024-2032"])
def test_model_solve(model_id):
    """Test model solving (slow test, marked with @pytest.mark.slow)."""
    result = run_model_solve_test(model_id, snapshot_limit=10, solver="highs")
    assert result["status"] == "ok"
    assert result["objective"] is not None
    assert result["snapshots_solved"] == 10


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["marei-eu"])
def test_model_solve_marei(model_id):
    """Test solving multi-sector model (very slow)."""
    result = run_model_solve_test(model_id, snapshot_limit=5, solver="highs")
    assert result["status"] == "ok"
    assert result["objective"] is not None


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Test PLEXOS model solving with PyPSA optimization"
    )
    parser.add_argument(
        "--model-id", required=True, help="Model ID from MODEL_REGISTRY"
    )
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        default=50,
        help="Number of snapshots to solve (default: 50)",
    )
    parser.add_argument(
        "--solver",
        default="highs",
        help="Solver to use: highs, glpk, gurobi, etc. (default: highs)",
    )
    parser.add_argument("--output-file", help="File to save solve results")

    args = parser.parse_args()

    try:
        run_model_solve_test(
            args.model_id, args.snapshot_limit, args.solver, args.output_file
        )
        sys.exit(0)
    except Exception as e:
        print(f"\nSolve failed for {args.model_id}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
