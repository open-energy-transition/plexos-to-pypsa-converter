from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.network.core import setup_network
from src.utils.model_paths import find_model_xml, get_model_directory


def create_sem_model(use_data_driven: bool = False) -> tuple[pypsa.Network, dict]:
    """Create SEM PyPSA model using traditional or data-driven approach.

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
    # Find model data in src/examples/data/
    model_id = "sem-2024-2032"
    file_xml = find_model_xml(model_id)
    model_dir = get_model_directory(model_id)

    if file_xml is None or model_dir is None:
        msg = (
            f"Model '{model_id}' not found in src/examples/data/. "
            f"Please download and extract the SEM 2024-2032 model data to:\n"
            f"  src/examples/data/sem-2024-2032/"
        )
        raise FileNotFoundError(msg)

    # Convert to strings
    file_xml = str(file_xml)
    model_dir = str(model_dir)

    # Set up paths
    file_timeslice = None

    # specify renewables profiles and demand paths
    # Note: Uses AEMO VRE profiles - ensure AEMO model is also downloaded
    aemo_dir = get_model_directory("aemo-2024-isp-progressive")
    path_ren = str(aemo_dir) if aemo_dir else model_dir
    path_demand = f"{model_dir}/demand"
    path_hydro_inflows = f"{model_dir}/hydro"  # May not exist in actual data

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
    print(f"  Iterations processed: {setup_summary['iterations_processed']}")
    print(
        f"  Loads created: {setup_summary['loads_added']} (Load1_{setup_summary['target_node']} to Load{setup_summary['iterations_processed']}_{setup_summary['target_node']})"
    )
else:
    print(f"  Zones aggregated: {setup_summary['zones_aggregated']}")
print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
print(f"  Total buses: {len(n.buses)}")
print(f"  Total generators: {len(n.generators)}")
print(f"  Total storage units: {len(n.storage_units)}")
if hasattr(n.storage_units_t, "inflow") and len(n.storage_units_t.inflow.columns) > 0:
    print(f"  Storage with inflows: {len(n.storage_units_t.inflow.columns)}")

# run consistency check on network
n.consistency_check()

# in each year in the snapshots, select the first x snapshots
x = 60  # number of snapshots to select per year
snapshots_by_year: defaultdict[int, list] = defaultdict(list)
for snap in n.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < x:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# solve the network
n.optimize(solver_name="highs", snapshots=subset)  # type: ignore
