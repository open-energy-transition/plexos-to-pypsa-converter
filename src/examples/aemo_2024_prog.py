from collections import defaultdict

import pandas as pd

from src.network.electricity_sector import create_model_from_xml
from src.utils.model_paths import find_model_xml, get_model_directory

# Constants
MODEL_ID = "aemo-2024-isp-progressive"
SNAPSHOTS_PER_YEAR = 50

# Find and validate model data
xml_file = find_model_xml(MODEL_ID)
main_dir = get_model_directory(MODEL_ID)

if xml_file is None or main_dir is None:
    msg = (
        f"Model '{MODEL_ID}' not found. "
        f"Please download and extract the AEMO 2024 ISP model data."
    )
    raise FileNotFoundError(msg)

# Convert to strings for compatibility
xml_file = str(xml_file)
main_dir = str(main_dir)

# Create network
network = create_model_from_xml(
    xml_file_path=xml_file,
    main_directory=main_dir,
    demand_assignment_strategy="per_node",
)

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
