#!/usr/bin/env python3
"""Integration tests for PLEXOS model conversion.

This module can be used both as a pytest test suite and as a standalone CLI script.

Usage as CLI:
    # With consistency checks (default)
    python tests/integration/test_model_conversion.py --model-id sem-2024-2032

    # Skip consistency checks for speed
    python tests/integration/test_model_conversion.py --model-id sem-2024-2032 --no-consistency-check

    # Save stats to file
    python tests/integration/test_model_conversion.py --model-id marei-eu --output-file stats.txt

Usage with pytest:
    # All conversion tests
    pytest tests/integration/test_model_conversion.py -v

    # Specific model
    pytest tests/integration/test_model_conversion.py::test_model_conversion[sem-2024-2032] -v

    # Skip consistency checks for faster testing
    pytest tests/integration/test_model_conversion.py::test_model_conversion_no_checks -v
"""

from pathlib import Path

import pytest


def run_model_conversion_test(
    model_id: str,
    run_consistency_check: bool = True,
    output_file: str | None = None,
) -> dict:
    """Test model conversion and optionally run consistency checks.

    Parameters
    ----------
    model_id : str
        Model identifier from MODEL_REGISTRY
    run_consistency_check : bool, default True
        Whether to run PyPSA consistency checks
    output_file : str, optional
        Path to file where statistics will be saved

    Returns
    -------
    dict
        Dictionary with model statistics including buses, generators, storage, snapshots

    Raises
    ------
    Exception
        If model conversion or consistency check fails
    """
    from src.network.conversion import create_model

    # Create model (will auto-download if not cached)
    network, setup_summary = create_model(model_id)

    # Print stats to stdout
    print(f"\nModel {model_id} converted successfully!")
    print(f"   Buses: {len(network.buses)}")
    print(f"   Generators: {len(network.generators)}")
    print(f"   Storage units: {len(network.storage_units)}")
    print(f"   Snapshots: {len(network.snapshots)}")

    # Run consistency check (default behavior)
    if run_consistency_check:
        print("\nRunning consistency check...")
        network.consistency_check()
        print("Consistency check passed!")
    else:
        print("\nConsistency check skipped (--no-consistency-check)")

    # Collect statistics
    stats = {
        "buses": len(network.buses),
        "generators": len(network.generators),
        "storage": len(network.storage_units),
        "snapshots": len(network.snapshots),
    }
    # Save stats to file if requested
    if output_file:
        with Path(output_file).open("w") as f:
            for key, value in stats.items():
                f.write(f"{key}={value}\n")

    return stats


# Pytest tests


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["sem-2024-2032", "marei-eu"])
def test_model_conversion(model_id):
    """Test model conversion with consistency checks.

    This test verifies that models can be successfully converted from PLEXOS
    to PyPSA format and pass PyPSA's built-in consistency checks.
    """
    stats = run_model_conversion_test(model_id, run_consistency_check=True)

    # Basic sanity checks
    assert stats["buses"] > 0, f"Model {model_id} should have at least one bus"
    assert stats["snapshots"] > 0, f"Model {model_id} should have at least one snapshot"


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["sem-2024-2032", "marei-eu"])
def test_model_conversion_no_checks(model_id):
    """Test model conversion without consistency checks (faster).

    This test is useful for quick verification during development when
    you want to check basic conversion functionality without the overhead
    of consistency checks.
    """
    stats = run_model_conversion_test(model_id, run_consistency_check=False)

    # Basic sanity checks
    assert stats["buses"] > 0
    assert stats["snapshots"] > 0


# CLI interface

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Test PLEXOS model conversion to PyPSA format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with consistency checks (default)
  python tests/integration/test_model_conversion.py --model-id sem-2024-2032

  # Skip consistency checks for faster testing
  python tests/integration/test_model_conversion.py --model-id marei-eu --no-consistency-check

  # Save statistics to file
  python tests/integration/test_model_conversion.py --model-id sem-2024-2032 --output-file stats.txt
        """,
    )
    parser.add_argument(
        "--model-id", required=True, help="Model ID from MODEL_REGISTRY"
    )
    parser.add_argument(
        "--no-consistency-check",
        action="store_true",
        help="Skip PyPSA consistency checks (runs by default)",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="File to save statistics (for CI integration)",
    )

    args = parser.parse_args()

    try:
        stats = run_model_conversion_test(
            model_id=args.model_id,
            run_consistency_check=not args.no_consistency_check,
            output_file=args.output_file,
        )
        sys.exit(0)
    except Exception as e:
        print(f"\nModel {args.model_id} conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
