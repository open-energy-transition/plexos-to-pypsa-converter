import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.add import (
    add_buses,
    add_constraints,
    add_generators,
    add_lines,
    add_storage,
)
from plexos_pypsa.network.summarize import check_constraints

# load PLEXOS input XML
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
plexos_db = PlexosDB.from_xml(file_xml)

# initialize PyPSA network
network = pypsa.Network()

# add to PyPSA network
add_buses(network, plexos_db)
add_generators(network, plexos_db)
add_storage(network, plexos_db)
add_lines(network, plexos_db)
add_constraints(network, plexos_db)

check_constraints(network)

# save to file
network.export_to_netcdf("converted_network.nc")
print("Network exported to converted_network.nc")
