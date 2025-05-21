import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.add import (
    add_buses,
    add_generators,
    add_snapshot,
    set_generator_efficiencies,
)

# list XML file
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
# file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/sem/2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"

# specify demand data path
path_aemo = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change"
path_demand = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/Traces/demand"

# load PlexosDB from XML file
plexos_db = PlexosDB.from_xml(file_xml)

# initialize PyPSA network
network = pypsa.Network()

# add buses
add_buses(network, plexos_db)

# add snapshots
add_snapshot(network, path_demand)

# add generators
add_generators(network, plexos_db)
set_generator_efficiencies(network, plexos_db, use_incr=False)
eff1 = network.generators["efficiency"].copy()

set_generator_efficiencies(network, plexos_db, use_incr=True)
eff2 = network.generators["efficiency"].copy()

# call column in eff1 "eff_base_only" and in eff2 "eff_with_incr" and merge them
eff1.name = "eff_base_only"
eff2.name = "eff_with_incr"
eff = (
    eff1.to_frame()
    .merge(eff2.to_frame(), left_index=True, right_index=True)
    .reset_index()
)

# save eff to file
eff.to_csv("plexos_pypsa/data/scratch/efficiencies.csv", index=False)
