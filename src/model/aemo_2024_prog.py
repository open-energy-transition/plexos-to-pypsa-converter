from collections import defaultdict
from typing import DefaultDict

import pandas as pd

from src.network.electricity_sector import create_model_from_xml

path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/2_Modeling/Plexos Converter/Input Models"
main_dir = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
xml_file = f"{main_dir}/2024 ISP Progressive Change Model.xml"

network = create_model_from_xml(
    xml_file_path=xml_file,
    main_directory=main_dir,
    demand_assignment_strategy="per_node",  # AEMO uses traditional per-node assignment of load
)

network.consistency_check()

# select a subset of snapshots for optimization
x = 50  # number of snapshots to select per year
snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
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
