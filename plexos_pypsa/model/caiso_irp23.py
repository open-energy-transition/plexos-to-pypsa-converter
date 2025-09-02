from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.core import setup_network
from plexos_pypsa.network.electricity_sector import create_caiso_model_data_driven


def create_caiso_model(use_data_driven: bool = False):
    """
    Create CAISO PyPSA model using traditional or data-driven approach.

    Parameters
    ----------
    use_data_driven : bool, default False
        If True, uses automatic path discovery from database.
        If False, uses hardcoded paths (legacy behavior).

    Returns
    -------
    pypsa.Network
        Configured CAISO PyPSA network
    """
    # Define XML file path
    path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/2_Modeling/Plexos Converter/Input Models"
    file_xml = f"{path_root}/CAISO/IRP/IRP23 - 25MMT Stochastic models with CEC 2023 IEPR Load Forecast/caiso-irp23-stochastic-2024-0517/CAISOIRP23Stochastic 20240517.xml"

    if use_data_driven:
        print("Creating CAISO PyPSA Model using data-driven approach...")
        return create_caiso_model_data_driven(
            xml_file_path=file_xml,
            main_directory=f"{path_root}/CAISO/IRP/IRP23 - 25MMT Stochastic models with CEC 2023 IEPR Load Forecast/caiso-irp23-stochastic-2024-0517",
        )

    # Legacy approach with hardcoded paths
    file_timeslice = None
    # specify renewables profiles and demand paths
    path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
    path_demand = f"{path_root}/CAISO/IRP/IRP23 - 25MMT Stochastic models with CEC 2023 IEPR Load Forecast/caiso-irp23-stochastic-2024-0517/LoadProfile"
    path_hydro_inflows = f"{path_root}/CAISO/IRP/IRP23 - 25MMT Stochastic models with CEC 2023 IEPR Load Forecast/caiso-irp23-stochastic-2024-0517/hydro"

    print("Creating CAISO PyPSA Model using traditional approach...")
    print(f"XML file: {file_xml}")
    print(f"Demand path: {path_demand}")
    print(f"VRE profiles path: {path_ren}")
    print(f"Hydro inflows path: {path_hydro_inflows}")

    # load PlexosDB from XML file
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(file_xml)

    # initialize PyPSA network
    n = pypsa.Network()

    # Set up complete network with demand aggregation
    # CAISO IRP23: Aggregate all demand to a single node and assign all generators/links to it
    print("\nSetting up complete network...")
    setup_summary = setup_network(
        network=n,
        db=plexos_db,
        snapshots_source=path_demand,
        demand_source=path_demand,
        aggregate_node_name="CAISO_Load_Aggregate",
        model_name="M01Y2024 PSP23_25MMT",
        timeslice_csv=file_timeslice,
        vre_profiles_path=path_ren,
        inflow_path=path_hydro_inflows,
    )

    return n, setup_summary


# Create model using legacy approach by default
n, setup_summary = create_caiso_model(use_data_driven=False)

print("\nSetup Summary:")
print(f"  Mode: {setup_summary['mode']}")
print(f"  Aggregate node: {setup_summary['aggregate_node_name']}")
print(f"  Format type: {setup_summary['format_type']}")
if setup_summary["format_type"] == "iteration":
    print(f"  Iterations processed: {setup_summary['iterations_processed']}")
    print(
        f"  Loads created: {setup_summary['loads_added']} (Load1_{setup_summary['aggregate_node_name']} to Load{setup_summary['iterations_processed']}_{setup_summary['aggregate_node_name']})"
    )
else:
    print(f"  Zones aggregated: {setup_summary['zones_aggregated']}")
print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
print(
    f"  Generators reassigned: {setup_summary['generator_summary']['reassigned_count']}"
)
print(f"  Links reassigned: {setup_summary['link_summary']['reassigned_count']}")
print(f"  Total storage units: {len(n.storage_units)}")
if hasattr(n.storage_units_t, "inflow") and len(n.storage_units_t.inflow.columns) > 0:
    print(f"  Storage with inflows: {len(n.storage_units_t.inflow.columns)}")

# run consistency check on network
n.consistency_check()

# select a subset of snapshots
# subset = n.snapshots[:50]  # the first 50 snapshots

# in each year in the snapshots, select the first x snapshots
x = 50  # number of snapshots to select per year
snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
for snap in n.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < x:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# solve the network
n.optimize(solver_name="highs", snapshots=subset)  # type: ignore
