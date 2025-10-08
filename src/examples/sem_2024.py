from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.network.core import setup_network
from src.utils.model_paths import find_model_xml, get_model_directory

# Constants
MODEL_ID = "sem-2024-2032"
AEMO_MODEL_ID = "aemo-2024-isp-progressive"
SNAPSHOTS_PER_YEAR = 60
MODEL_NAME = "Opt A 24-32 (Avail, Uplift, Wheeling)--MIP 25/26"
TARGET_NODE = "SEM"


def create_sem_model() -> tuple[pypsa.Network, dict]:
    """Create SEM PyPSA model.

    Returns
    -------
    tuple[pypsa.Network, dict]
        Configured SEM PyPSA network and setup summary
    """
    # Find and validate model data
    file_xml = find_model_xml(MODEL_ID)
    model_dir = get_model_directory(MODEL_ID)

    if file_xml is None or model_dir is None:
        msg = f"Model '{MODEL_ID}' not found. Please download and extract the SEM 2024-2032 model data."
        raise FileNotFoundError(msg)

    # Set up paths - use AEMO for VRE profiles
    aemo_dir = get_model_directory(AEMO_MODEL_ID)
    path_ren = str(aemo_dir) if aemo_dir else str(model_dir)
    path_demand = f"{model_dir}/demand"
    path_hydro_inflows = f"{model_dir}/hydro"

    print("Creating SEM PyPSA Model...")
    print(f"XML file: {file_xml}")
    print(f"VRE profiles path: {path_ren}")

    # Load database and initialize network
    plexos_db = PlexosDB.from_xml(str(file_xml))
    network = pypsa.Network()

    # Set up network with demand assigned to SEM node
    setup_summary = setup_network(
        network=network,
        db=plexos_db,
        snapshots_source=path_demand,
        demand_source=path_demand,
        target_node=TARGET_NODE,
        model_name=MODEL_NAME,
        timeslice_csv=None,
        vre_profiles_path=path_ren,
        inflow_path=path_hydro_inflows,
    )

    return network, setup_summary


# Create and validate model
network, setup_summary = create_sem_model()

print("\nSetup Summary:")
print(f"  Target node: {setup_summary['target_node']}")
print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
print(f"  Total buses: {len(network.buses)}")
print(f"  Total generators: {len(network.generators)}")
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
