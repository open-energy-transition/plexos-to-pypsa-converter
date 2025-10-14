from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "aemo-2024-isp-progressive-change"
SNAPSHOTS_PER_YEAR = 50

if __name__ == "__main__":
    # Create network using unified factory
    network, setup_summary = create_model(MODEL_ID)

    network.consistency_check()

    # Select subset of snapshots for optimization
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

    # Optimization configuration
    SOLVER_CONFIG = {
        "solver_name": "gurobi",
        "solver_options": {
            "Threads": 6,
            "Method": 2,  # barrier
            "Crossover": 0,
            "BarConvTol": 1.0e-5,
            "Seed": 123,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
        },
    }

    # Optimize network
    use_subset = True  # Set to False for full network optimization
    snapshots = subset if use_subset else network.snapshots

    print(f"\nOptimizing network with {len(snapshots)} snapshots...")
    network.optimize(snapshots=snapshots, **SOLVER_CONFIG)  # type: ignore
    print("  Optimization complete!")

    # Save results
    output_file = "aemo_2024_results.nc"
    print(f"Saving results to {output_file}...")
    network.export_to_netcdf(output_file)
    print("  Results saved!")
