import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.constraints import add_constraints
from plexos_pypsa.network.core import add_buses, add_loads, add_snapshots
from plexos_pypsa.network.generators import (
    add_generators,
    set_capacity_ratings,
    set_capital_costs,
    set_generator_efficiencies,
    set_vre_profiles,
)
from plexos_pypsa.network.links import add_links, set_link_flows
from plexos_pypsa.network.storage import add_hydro_inflows, add_storage
from plexos_pypsa.network.summarize import check_constraints

# list XML file
path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
file_xml = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
# file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/sem/2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"

# specify renewables profiles and demand paths
path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
path_demand = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/demand"

# load PlexosDB from XML file
plexos_db = PlexosDB.from_xml(file_xml)

# initialize PyPSA network
network = pypsa.Network()

# add buses
add_buses(network, plexos_db)

# add snapshots
add_snapshots(network, path_demand)

# add generators
add_generators(network, plexos_db)
set_generator_efficiencies(network, plexos_db, use_incr=True)
set_capital_costs(network, plexos_db)
set_capacity_ratings(network, plexos_db)
set_vre_profiles(network, plexos_db, path_ren)

# add links
add_links(network, plexos_db)
set_link_flows(network, plexos_db)

# add demand/loads
add_loads(network, path_demand)

# add storage (TODO: fix)
add_storage(network, plexos_db)
add_hydro_inflows(network, plexos_db, path_ren)

# add constraints (TODO: fix)
add_constraints(network, plexos_db)
check_constraints(network)

# save to file
network.export_to_netcdf("converted_network.nc")
print("Network exported to converted_network.nc")

# solve network
# network.optimize(solver_name="highs")
