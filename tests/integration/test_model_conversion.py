#!/usr/bin/env python3
"""Integration tests for PLEXOS model conversion.

Usage as CLI:
    python tests/integration/test_model_conversion.py --model-id sem-2024-2032 [--no-consistency-check] [--output-file stats.txt]

Usage with pytest:
    pytest tests/integration/test_model_conversion.py -v
"""

from pathlib import Path

import pytest

from network.conversion import create_model


def run_model_conversion_test(
    model_id: str, run_consistency_check: bool = True, output_file: str | None = None
) -> dict:
    """Test model conversion and optionally run consistency checks."""

    network, _ = create_model(model_id)

    # Collect statistics
    stats = {
        "buses": len(network.buses),
        "generators": len(network.generators),
        "storage": len(network.storage_units),
        "snapshots": len(network.snapshots),
    }

    print(f"\nModel {model_id} converted successfully!")
    for key, value in stats.items():
        print(f"   {key.title()}: {value}")

    if run_consistency_check:
        print("\nRunning consistency check...")
        network.consistency_check()
        print("Consistency check passed!")
    else:
        print("\nConsistency check skipped")

    if output_file:
        Path(output_file).write_text(
            "\n".join(f"{k}={v}" for k, v in stats.items()) + "\n"
        )

    return stats


# Pytest tests
@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["sem-2024-2032", "marei-eu"])
def test_model_conversion(model_id):
    """Test model conversion with consistency checks."""
    stats = run_model_conversion_test(model_id, run_consistency_check=True)
    assert stats["buses"] > 0
    assert stats["snapshots"] > 0


@pytest.mark.integration
@pytest.mark.parametrize("model_id", ["sem-2024-2032", "marei-eu"])
def test_model_conversion_no_checks(model_id):
    """Test model conversion without consistency checks (faster)."""
    stats = run_model_conversion_test(model_id, run_consistency_check=False)
    assert stats["buses"] > 0
    assert stats["snapshots"] > 0


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Test PLEXOS model conversion")
    parser.add_argument(
        "--model-id", required=True, help="Model ID from MODEL_REGISTRY"
    )
    parser.add_argument(
        "--no-consistency-check",
        action="store_true",
        help="Skip PyPSA consistency checks",
    )
    parser.add_argument("--output-file", help="File to save statistics")

    args = parser.parse_args()

    try:
        run_model_conversion_test(
            args.model_id, not args.no_consistency_check, args.output_file
        )
        sys.exit(0)
    except Exception as e:
        print(f"\nModel {args.model_id} conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
