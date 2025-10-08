from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "caiso-irp23"
SNAPSHOTS_PER_YEAR = 50

if __name__ == "__main__":
    # Create network using unified factory (uses default aggregate_node strategy)
    network, setup_summary = create_model(MODEL_ID)

    print("\nSetup Summary:")
    if "aggregate_node_name" in setup_summary:
        print(f"  Aggregate node: {setup_summary['aggregate_node_name']}")
    if "peak_demand" in setup_summary:
        print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
    if "generator_summary" in setup_summary:
        print(
            f"  Generators reassigned: {setup_summary['generator_summary']['reassigned_count']}"
        )
    if "link_summary" in setup_summary:
        print(
            f"  Links reassigned: {setup_summary['link_summary']['reassigned_count']}"
        )
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
    network.optimize(solver_name="highs", snapshots=subset)  # type: ignore
