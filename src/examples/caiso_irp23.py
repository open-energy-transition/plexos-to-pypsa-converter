from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.network.core import setup_network
from src.utils.model_paths import find_model_xml, get_model_directory

# Constants
MODEL_ID = "caiso-irp23"
AEMO_MODEL_ID = "aemo-2024-isp-progressive"
SNAPSHOTS_PER_YEAR = 50
MODEL_NAME = "M01Y2024 PSP23_25MMT"
AGGREGATE_NODE = "CAISO_Load_Aggregate"


def create_caiso_model() -> tuple[pypsa.Network, dict]:
    """Create CAISO PyPSA model.

    Returns
    -------
    tuple[pypsa.Network, dict]
        Configured CAISO PyPSA network and setup summary
    """
    # Find and validate model data
    file_xml = find_model_xml(MODEL_ID)
    model_dir = get_model_directory(MODEL_ID)

    if file_xml is None or model_dir is None:
        msg = f"Model '{MODEL_ID}' not found. Please download and extract the CAISO IRP23 model data."
        raise FileNotFoundError(msg)

    # Set up paths - use AEMO for VRE profiles
    aemo_dir = get_model_directory(AEMO_MODEL_ID)
    path_ren = str(aemo_dir) if aemo_dir else str(model_dir)
    path_demand = f"{model_dir}/LoadProfile"
    path_hydro_inflows = f"{model_dir}/hydro"

    print("Creating CAISO PyPSA Model...")
    print(f"XML file: {file_xml}")
    print(f"VRE profiles path: {path_ren}")

    # Load database and initialize network
    plexos_db = PlexosDB.from_xml(str(file_xml))
    network = pypsa.Network()

    # Set up network with demand aggregation
    setup_summary = setup_network(
        network=network,
        db=plexos_db,
        snapshots_source=path_demand,
        demand_source=path_demand,
        aggregate_node_name=AGGREGATE_NODE,
        model_name=MODEL_NAME,
        timeslice_csv=None,
        vre_profiles_path=path_ren,
        inflow_path=path_hydro_inflows,
    )

    return network, setup_summary


# Create and validate model
network, setup_summary = create_caiso_model()

print("\nSetup Summary:")
print(f"  Aggregate node: {setup_summary['aggregate_node_name']}")
print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
print(
    f"  Generators reassigned: {setup_summary['generator_summary']['reassigned_count']}"
)
print(f"  Links reassigned: {setup_summary['link_summary']['reassigned_count']}")
print(f"  Total storage units: {len(network.storage_units)}")

network.consistency_check()

# Select subset of snapshots for optimization
snapshots_by_year: defaultdict[int, list] = defaultdict(list)
for snap in network.snapshots:
    year = pd.Timestamp(snap).year
    if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
        snapshots_by_year[year].append(snap)

subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

# Optimize network
network.optimize(solver_name="highs", snapshots=subset)  # type: ignore
