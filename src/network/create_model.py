from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from network.core import setup_network
from network.electricity_sector import create_aemo_model_data_driven
from utils.model_paths import find_model_xml, get_model_directory


def create_aemo_model(use_data_driven: bool = False) -> pypsa.Network:
    """Create AEMO PyPSA model using traditional or data-driven approach.

    Parameters
    ----------
    use_data_driven : bool, default False
        If True, uses automatic path discovery from database.
        If False, uses hardcoded paths (legacy behavior).

    Examples
    --------
    Traditional approach:
    >>> network = create_aemo_model()

    Data-driven approach:
    >>> network = create_aemo_model(use_data_driven=True)

    >>> print(f"Network has {len(network.buses)} buses and {len(network.loads)} loads")
    >>> network.optimize(solver_name="highs")
    """
    # Find model data in src/examples/data/
    model_id = "aemo-2024-isp-progressive-change"
    file_xml = find_model_xml(model_id)
    main_dir = get_model_directory(model_id)

    if file_xml is None or main_dir is None:
        msg = (
            f"Model '{model_id}' not found in src/examples/data/. "
            f"Please download and extract the AEMO 2024 ISP model data to:\n"
            f"  src/examples/data/aemo-2024-isp-progressive-change/"
        )
        raise FileNotFoundError(msg)

    # Convert Path objects to strings for compatibility
    file_xml = str(file_xml)
    main_dir = str(main_dir)

    if use_data_driven:
        print("Creating AEMO PyPSA Model using data-driven approach...")
        return create_aemo_model_data_driven(
            xml_file_path=file_xml,
            main_directory=main_dir,
        )

    # Legacy approach with model-relative paths
    file_timeslice = f"{main_dir}/Traces/timeslice/timeslice_RefYear4006.csv"

    # specify renewables profiles and demand paths
    path_ren = main_dir
    path_demand = f"{main_dir}/Traces/demand"
    path_hydro_inflows = f"{main_dir}/Traces/hydro"

    print("Creating AEMO PyPSA Model using traditional approach...")
    print(f"XML file: {file_xml}")
    print(f"Demand path: {path_demand}")
    print(f"VRE profiles path: {path_ren}")
    print(f"Hydro inflows path: {path_hydro_inflows}")

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
        vre_profiles_path=path_ren,
        inflow_path=path_hydro_inflows,
    )

    print("\nNetwork Setup Complete:")
    print(f"  Mode: {setup_summary['mode']}")
    print(f"  Format type: {setup_summary['format_type']}")
    if setup_summary["format_type"] == "iteration":
        print(
            f"  Iterations processed: {setup_summary.get('iterations_processed', 'N/A')}"
        )
        print(f"  Loads created: {len(n.loads)}")
    else:  # zone format
        print(f"  Loads mapped to buses: {len(n.loads)}")
        if setup_summary.get("loads_skipped", 0) > 0:
            print(
                f"  Loads skipped (no matching bus): {setup_summary['loads_skipped']}"
            )

    print(f"  Total buses: {len(n.buses)}")
    print(f"  Total generators: {len(n.generators)}")
    print(f"  Total links: {len(n.links)}")
    print(f"  Total storage units: {len(n.storage_units)}")
    if (
        hasattr(n.storage_units_t, "inflow")
        and len(n.storage_units_t.inflow.columns) > 0
    ):
        print(f"  Storage with inflows: {len(n.storage_units_t.inflow.columns)}")
    print(f"  Total snapshots: {len(n.snapshots)}")

    return n


if __name__ == "__main__":
    # create the network
    network = create_aemo_model()

    # select a subset of snapshots for optimization
    print("\nPreparing optimization subset...")
    x = 50  # number of snapshots to select per year
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
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

network = create_aemo_model()
network.consistency_check()
