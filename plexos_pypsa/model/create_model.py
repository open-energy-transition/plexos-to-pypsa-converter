from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.core import add_buses, add_carriers, add_loads, add_snapshots
from plexos_pypsa.network.generators import (
    add_generators,
    set_capacity_ratings,
    set_capital_costs,
    set_generator_efficiencies,
    set_vre_profiles,
)
from plexos_pypsa.network.links import add_links, set_link_flows

# list XML file
path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
file_xml = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
file_timeslice = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/timeslice/timeslice_RefYear4006.csv"
# file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/sem/2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"

# specify renewables profiles and demand paths
path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
path_demand = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/demand"

# load PlexosDB from XML file
plexos_db = PlexosDB.from_xml(file_xml)

# initialize PyPSA network
n = pypsa.Network()

# add buses
add_buses(n, plexos_db)

# add snapshots
add_snapshots(n, path_demand)

# add carriers
add_carriers(n, plexos_db)

# add generators
add_generators(n, plexos_db)
set_capacity_ratings(n, plexos_db, timeslice_csv=file_timeslice)
set_generator_efficiencies(n, plexos_db, use_incr=True)
set_capital_costs(n, plexos_db)
set_vre_profiles(n, plexos_db, path_ren)

# add links
add_links(n, plexos_db)
set_link_flows(n, plexos_db)

# add demand/loads
add_loads(n, path_demand)

# add storage (TODO: fix)
# add_storage(n, plexos_db)
# add_hydro_inflows(n, plexos_db, path_ren)

# run consistency check on network
n.consistency_check()

# select a subset of snapshots
# subset = n.snapshots[:50]  # the first 50 snapshots

# in each year in the snapshots, select the first x snapshots
x = 60  # number of snapshots to select per year
snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
for snap in n.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < x:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# solve the network
n.optimize(solver_name="highs", snapshots=subset)  # type: ignore

# save to file
# n.export_to_netcdf("converted_network.nc")
# print("Network exported to converted_network.nc")
