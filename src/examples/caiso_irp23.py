from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.network.core import setup_network
from src.utils.model_paths import find_model_xml, get_model_directory


def create_caiso_model(
    use_data_driven: bool = False,
) -> tuple[pypsa.Network, dict]:
    """Create CAISO PyPSA model using traditional or data-driven approach.

    Parameters
    ----------
    use_data_driven : bool, default False
        If True, uses automatic path discovery from database.
        If False, uses hardcoded paths (legacy behavior).

    Returns
    -------
    tuple[pypsa.Network, dict]
        Tuple containing configured CAISO PyPSA network and setup summary
    """
    # Find model data in src/examples/data/
    model_id = "caiso-irp23"
    file_xml = find_model_xml(model_id)
    model_dir = get_model_directory(model_id)

    if file_xml is None or model_dir is None:
        msg = (
            f"Model '{model_id}' not found in src/examples/data/. "
            f"Please download and extract the CAISO IRP23 model data to:\n"
            f"  src/examples/data/CAISO/IRP/..."
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
    path_demand = f"{model_dir}/LoadProfile"
    path_hydro_inflows = f"{model_dir}/hydro"

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

# in each year in the snapshots, select the first x snapshots
x = 50  # number of snapshots to select per year
snapshots_by_year: defaultdict[int, list] = defaultdict(list)
for snap in n.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < x:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# solve the network
n.optimize(solver_name="highs", snapshots=subset)  # type: ignore
