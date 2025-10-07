from collections import defaultdict

import pandas as pd

from src.network.electricity_sector import create_model_from_xml
from src.utils.model_paths import find_model_xml, get_model_directory

# Find model data in src/examples/data/
model_id = "aemo-2024-isp-progressive"
xml_file = find_model_xml(model_id)
main_dir = get_model_directory(model_id)

if xml_file is None or main_dir is None:
    msg = (
        f"Model '{model_id}' not found in src/examples/data/. "
        f"Please download and extract the AEMO 2024 ISP model data to:\n"
        f"  src/examples/data/AEMO/2024 ISP/2024 ISP Progressive Change/"
    )
    raise FileNotFoundError(msg)

# Convert Path objects to strings for compatibility
xml_file = str(xml_file)
main_dir = str(main_dir)

network = create_model_from_xml(
    xml_file_path=xml_file,
    main_directory=main_dir,
    demand_assignment_strategy="per_node",  # AEMO uses traditional per-node assignment of load
)

network.consistency_check()

# select a subset of snapshots for optimization
x = 50  # number of snapshots to select per year
snapshots_by_year: defaultdict[int, list] = defaultdict(list)
for snap in network.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < x:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# Configuration
use_subset = True  # Set to True to optimize on subset, False for full network

# solve the network
if use_subset:
    print(f"\nOptimizing network with {len(subset)} snapshots...")
    network.optimize(
        solver_name="gurobi",
        snapshots=subset,
        solver_options={
            "Threads": 6,
            "Method": 2,  # barrier
            "Crossover": 0,
            "BarConvTol": 1.0e-5,
            "Seed": 123,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
        },
    )  # type: ignore
else:
    print(f"\nOptimizing network with {len(network.snapshots)} snapshots...")
    network.optimize(
        solver_name="gurobi",
        solver_options={
            "Threads": 6,
            "Method": 2,  # barrier
            "Crossover": 0,
            "BarConvTol": 1.0e-5,
            "Seed": 123,
            "AggFill": 0,
            "PreDual": 0,
            "GURO_PAR_BARDENSETHRESH": 200,
        },
    )  # type: ignore

print("  Optimization complete!")

# Save results
output_file = "aemo_2024_results.nc"
print(f"Saving results to {output_file}...")
network.export_to_netcdf(output_file)
print("  Results saved!")
