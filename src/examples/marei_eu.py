from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "marei-eu"
SNAPSHOTS_PER_YEAR = 100

if __name__ == "__main__":
    # Configuration - can override defaults from MODEL_REGISTRY
    use_csv_integration = True
    infrastructure_scenario = "PCI"
    pricing_scheme = "Production"

    # Create model using unified factory
    network, setup_summary = create_model(
        MODEL_ID,
        use_csv_integration=use_csv_integration,
        infrastructure_scenario=infrastructure_scenario,
        pricing_scheme=pricing_scheme,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("MAREI-EU MODEL SETUP SUMMARY")
    print("=" * 60)
    print(f"Network type: {setup_summary['network_type']}")
    print(f"Sectors: {', '.join(setup_summary['sectors'])}")

    # Print key summaries
    elec_summary = setup_summary["electricity"]
    gas_summary = setup_summary["gas"]

    print(
        f"\nElectricity: {elec_summary['buses']} buses, {elec_summary['generators']} generators"
    )
    print(f"Gas: {gas_summary['buses']} buses, {gas_summary['pipelines']} pipelines")
    print(f"Total Components: {len(network.buses)} buses, {len(network.links)} links")

    # Run consistency check and optimize
    print("\nRunning consistency check...")
    network.consistency_check()
    print("Network consistency check passed!")

    # Select subset of snapshots for optimization
    snapshots_by_year: defaultdict[int, list] = defaultdict(list)
    for snap in network.snapshots:
        year = pd.Timestamp(snap).year
        if len(snapshots_by_year[year]) < SNAPSHOTS_PER_YEAR:
            snapshots_by_year[year].append(snap)

    subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

    # Optimize network
    try:
        print(f"\nOptimizing network with {len(subset)} snapshots...")
        network.optimize(solver_name="highs", snapshots=subset)  # type: ignore
        print("Optimization complete!")

        # Save results
        output_file = "marei_eu_results.nc"
        print(f"\nSaving results to {output_file}...")
        network.export_to_netcdf(output_file)
        print("  Results saved!")

    except Exception as e:
        print(f"Optimization failed: {e}")
        print("Network created successfully but optimization encountered issues.")
