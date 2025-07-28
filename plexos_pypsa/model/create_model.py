from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexos_pypsa.network.core import setup_network


def create_aemo_model():
    """
    Examples
    --------
    >>> network = create_aemo_model()
    >>> print(f"Network has {len(network.buses)} buses and {len(network.loads)} loads")
    >>> network.optimize(solver_name="highs")
    """
    # list XML file
    path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
    file_xml = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
    file_timeslice = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/timeslice/timeslice_RefYear4006.csv"

    # specify renewables profiles and demand paths
    path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
    path_demand = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/demand"

    print("Creating AEMO PyPSA Model...")
    print(f"XML file: {file_xml}")
    print(f"Demand path: {path_demand}")
    print(f"VRE profiles path: {path_ren}")

    # load PlexosDB from XML file
    print("\nLoading Plexos database...")
    plexos_db = PlexosDB.from_xml(file_xml)

    # initialize PyPSA network
    n = pypsa.Network()

    # set up complete network using unified function
    # AEMO model: Uses traditional per-node load assignment (each CSV file maps to a bus)
    print("\nSetting up complete network...")
    setup_summary = setup_network(
        n, 
        plexos_db, 
        snapshots_source=path_demand, 
        demand_source=path_demand,
        timeslice_csv=file_timeslice,
        vre_profiles_path=path_ren
    )

    print("\nNetwork Setup Complete:")
    print(f"  Mode: {setup_summary['mode']}")
    print(f"  Format type: {setup_summary['format_type']}")
    if setup_summary["format_type"] == "iteration":
        print(
            f"  Iterations processed: {setup_summary.get('iterations_processed', 'N/A')}"
        )
        print(f"  Loads created: {setup_summary['loads_added']}")
    else:  # zone format
        print(f"  Loads mapped to buses: {setup_summary['loads_added']}")
        if setup_summary.get("loads_skipped", 0) > 0:
            print(f"  Loads skipped (no matching bus): {setup_summary['loads_skipped']}")

    print(f"  Total buses: {len(n.buses)}")
    print(f"  Total generators: {len(n.generators)}")
    print(f"  Total links: {len(n.links)}")
    print(f"  Total snapshots: {len(n.snapshots)}")

    # add storage (TODO: fix)
    # add_storage(n, plexos_db)
    # add_hydro_inflows(n, plexos_db, path_ren)

    # run consistency check on network
    print("\nRunning network consistency check...")
    n.consistency_check()
    print("  Network consistency check passed!")

    return n


if __name__ == "__main__":
    # create the network
    network = create_aemo_model()

    # select a subset of snapshots for optimization
    print("\nPreparing optimization subset...")
    x = 50  # number of snapshots to select per year
    snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < x:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]
    print(f"  Selected {len(subset)} snapshots from {len(snapshots_by_year)} years")

    # solve the network
    print(f"\nOptimizing network with {len(subset)} snapshots...")
    network.optimize(solver_name="highs", snapshots=subset)  # type: ignore
    print("  Optimization complete!")

    # save to file
    # network.export_to_netcdf("aemo_network.nc")
    # print("Network exported to aemo_network.nc")
