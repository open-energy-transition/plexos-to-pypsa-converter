from collections import defaultdict
from typing import DefaultDict

import pandas as pd

from plexos_pypsa.model.data_driven import create_model_from_xml

path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
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

# solve the network
print(f"\nOptimizing network with {len(subset)} snapshots...")
network.optimize(solver_name="highs", snapshots=subset)  # type: ignore
print("  Optimization complete!")
