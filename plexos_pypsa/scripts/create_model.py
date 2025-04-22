import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.add import add_buses, add_generators, add_lines

# Load PLEXOS input XML
file_xml = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/My Drive/open-tyndp/aemo/2024/2024 ISP Model/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
plexos_db = PlexosDB.from_xml(file_xml)

# Initialize PyPSA network
network = pypsa.Network()

# --- Execute all steps ---
add_buses(network, plexos_db)
add_generators(network, plexos_db)
add_lines(network, plexos_db)

# list generators in network
# Optional: Save to file
network.export_to_netcdf("converted_network.nc")
print("Network exported to converted_network.nc")
