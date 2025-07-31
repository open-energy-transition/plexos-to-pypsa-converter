from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.core import setup_network
from plexos_pypsa.model.data_driven import create_sem_model_data_driven

def create_sem_model(use_data_driven: bool = False):
    """
    Create SEM PyPSA model using traditional or data-driven approach.
    
    Parameters
    ---------- 
    use_data_driven : bool, default False
        If True, uses automatic path discovery from database.
        If False, uses hardcoded paths (legacy behavior).
        
    Returns
    -------
    pypsa.Network
        Configured SEM PyPSA network
    """
    # Define XML file path
    path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
    file_xml = f"{path_root}/SEM/SEM 2024-2032/SEM PLEXOS Forecast Model 2024-2032( Public Version)/PUBLIC Validation 2024-2032 Model 2025-03-14.xml"
    
    if use_data_driven:
        print("Creating SEM PyPSA Model using data-driven approach...")
        return create_sem_model_data_driven(
            xml_file_path=file_xml,
            main_directory=f"{path_root}/SEM/SEM 2024-2032"
        )
    
    # Legacy approach with hardcoded paths
    file_timeslice = None

    # specify renewables profiles and demand paths
    path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
    path_demand = f"{path_root}/SEM/SEM 2024-2032/demand"
    path_hydro_inflows = f"{path_root}/SEM/SEM 2024-2032/hydro"

    print("Creating SEM PyPSA Model using traditional approach...")
    print(f"XML file: {file_xml}")
    print(f"Demand path: {path_demand}")
    print(f"VRE profiles path: {path_ren}")
    print(f"Hydro inflows path: {path_hydro_inflows}")

    # load PlexosDB from XML file
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(file_xml)

    # initialize PyPSA network
    n = pypsa.Network()

    # Set up complete network with demand assigned to SEM node
    # SEM 2024: Assign all demand to the "SEM" node specifically
    print("\nSetting up complete network...")
    setup_summary = setup_network(
        network=n,
        db=plexos_db,
        snapshots_source=path_demand,
        demand_source=path_demand,
        target_node="SEM",
        model_name="Opt A 24-32 (Avail, Uplift, Wheeling)--MIP 25/26",
        timeslice_csv=file_timeslice,
        vre_profiles_path=path_ren,
        inflow_path=path_hydro_inflows,
    )
    
    return n, setup_summary


# Create model using legacy approach by default
n, setup_summary = create_sem_model(use_data_driven=False)

print("\nSetup Summary:")
print(f"  Mode: {setup_summary['mode']}")
print(f"  Target node: {setup_summary['target_node']}")
print(f"  Format type: {setup_summary['format_type']}")
if setup_summary["format_type"] == "iteration":
    print(
        f"  Iterations processed: {setup_summary['iterations_processed']}"
    )
    print(
        f"  Loads created: {setup_summary['loads_added']} (Load1_{setup_summary['target_node']} to Load{setup_summary['iterations_processed']}_{setup_summary['target_node']})"
    )
else:
    print(f"  Zones aggregated: {setup_summary['zones_aggregated']}")
print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
print(f"  Total buses: {len(n.buses)}")
print(f"  Total generators: {len(n.generators)}")
print(f"  Total storage units: {len(n.storage_units)}")
if hasattr(n.storage_units_t, 'inflow') and len(n.storage_units_t.inflow.columns) > 0:
    print(f"  Storage with inflows: {len(n.storage_units_t.inflow.columns)}")

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
