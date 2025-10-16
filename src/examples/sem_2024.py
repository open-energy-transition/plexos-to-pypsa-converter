from collections import defaultdict

import pandas as pd

from network.conversion import create_model
from network.generators_csv import load_data_file_profiles_csv

# Constants
MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60

# SEM-specific manual mappings (CSV export lacks "Rating.Data File" column)
SEM_VRE_MAPPINGS = {
    "Wind NI -- All": "StochasticWindNI",
    "Wind ROI": "StochasticWindROI",
    "Wind Offshore": "StochasticWindOffshore",
    "Wind Offshore -- Arklow Phase 1": "StochasticWindROI",  # Uses ROI profile per report
    "Solar NI -- All": "StochasticSolarNI",
    "Solar ROI": "StochasticSolarROI",
}


if __name__ == "__main__":
    # Create network using unified factory (uses default target_node strategy)
    network, setup_summary = create_model(MODEL_ID, use_csv=True)

    print("\nSetup Summary:")
    if "target_node" in setup_summary:
        print(f"  Target node: {setup_summary['target_node']}")
    if "peak_demand" in setup_summary:
        print(f"  Peak demand: {setup_summary['peak_demand']:.2f} MW")
    print(f"  Total buses: {len(network.buses)}")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total storage units: {len(network.storage_units)}")

    # Load VRE profiles using generic loader with manual mappings
    print("\nLoading VRE profiles...")
    summary = load_data_file_profiles_csv(
        network=network,
        csv_dir="src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model",
        profiles_path="src/examples/data/sem-2024-2032/CSV Files",
        property_name="Rating",
        target_property="p_max_pu",
        target_type="generators_t",
        apply_mode="replace",
        scenario="1",
        generator_filter=lambda gen: "Wind" in gen or "Solar" in gen,
        carrier_mapping={"Wind": "Wind", "Solar": "Solar"},
        value_scaling=0.01,  # Convert percentage to fraction
        manual_mappings=SEM_VRE_MAPPINGS,  # Fallback for incomplete CSV export
    )

    # Also set p_min_pu for VRE generators (must-run at capacity factor)
    if summary["processed_generators"] > 0:
        for gen in network.generators.index:
            if (
                "Wind" in gen or "Solar" in gen
            ) and gen in network.generators_t.p_max_pu.columns:
                network.generators_t.p_min_pu[gen] = network.generators_t.p_max_pu[gen]

    print(
        f"✓ Successfully loaded VRE profiles for {summary['processed_generators']} generators"
    )

    network.consistency_check()

    # Select subset of snapshots for optimization
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

    # Optimize network
    network.optimize(solver_name="gurobi")
