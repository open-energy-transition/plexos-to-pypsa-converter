from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60

if __name__ == "__main__":
    # Create network using unified factory (uses default target_node strategy)
    network, setup_summary = create_model(MODEL_ID)

    print("\nSetup Summary:")
    if "target_node" in setup_summary:
        print(f"  Target node: {setup_summary['target_node']}")
    if "peak_demand" in setup_summary:
        print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
    print(f"  Total buses: {len(network.buses)}")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total storage units: {len(network.storage_units)}")

    network.consistency_check()

    # Select subset of snapshots for optimization
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

    # Optimize network
    network.optimize(solver_name="highs", snapshots=subset)
