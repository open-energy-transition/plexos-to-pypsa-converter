"""Test script for CSV-based SEM model creation.

This script demonstrates the new CSV-based conversion approach:
1. Checks if CSVs already exist (in csvs_from_xml/)
2. If not, automatically generates them from the XML file using COAD
3. Uses the CSV files to create the PyPSA network

This is much faster than the PlexosDB approach for subsequent runs.
"""

from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60

network, setup_summary = create_model(MODEL_ID, use_csv=True)

if __name__ == "__main__":
    print("=" * 70)
    print("SEM 2024-2032 Model - CSV-Based Conversion Test")
    print("=" * 70)
    print()
    print("This script will:")
    print("1. Check if CSVs exist (first run: auto-generates from XML)")
    print("2. Create PyPSA network from CSVs")
    print("3. Run a test optimization")
    print()

    # Create network using CSV-based conversion
    # On first run: auto-generates CSVs from XML using COAD
    # On subsequent runs: reuses existing CSVs (much faster!)
    network, setup_summary = create_model(MODEL_ID, use_csv=True)

    print("\n" + "=" * 70)
    print("Setup Summary:")
    print("=" * 70)
    if "target_node" in setup_summary:
        print(f"  Target node: {setup_summary['target_node']}")
    if "peak_demand" in setup_summary:
        print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
    print(f"  Total buses: {len(network.buses)}")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total storage units: {len(network.storage_units)}")
    print(f"  Total links: {len(network.links)}")
    print(f"  Total loads: {len(network.loads)}")
    print(f"  Total snapshots: {len(network.snapshots)}")

    # Consistency check
    print("\nRunning consistency check...")
    network.consistency_check()
    print("✓ Network is consistent!")

    # Select subset of snapshots for optimization
    print(f"\nSelecting {SNAPSHOTS_PER_YEAR} snapshots per year for optimization...")
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]
    print(f"Selected {len(subset)} snapshots for optimization")

    # Optimize network
    print("\nOptimizing network with HiGHS solver...")
    result = network.optimize(solver_name="highs", snapshots=subset)

    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print(f"Status: {result}")
    print(f"Objective value: {network.objective:.2f}")
    print()
    print("CSV-based conversion test successful! ✓")
