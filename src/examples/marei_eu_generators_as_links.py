"""MaREI-EU Multi-Sector Model with Generators-as-Links Option.

This script creates a PyPSA network from the MaREI-EU PLEXOS model where conventional
generators (coal, gas, nuclear, etc.) can be represented as Links instead of Generators.
This enables multi-sector coupling by connecting fuel buses to electric buses with
efficiency conversions.
"""

from collections import defaultdict

import pandas as pd

from network.conversion import create_model

# Constants
MODEL_ID = "marei-eu"

if __name__ == "__main__":
    # Configuration - can override defaults from MODEL_REGISTRY
    # Set generators_as_links=True to use Link representation for conventional generators
    # Set testing_mode=True for faster development, False for complete model
    # Set use_csv=True to enable MaREI CSV data integration
    generators_as_links = True  # Enable generators-as-links functionality
    testing_mode = False  # Full model to investigate actual PLEXOS data
    use_csv = True  # Enable CSV integration for enhanced model
    infrastructure_scenario = "PCI"  # Infrastructure scenario: 'PCI', 'High', 'Low'
    pricing_scheme = (
        "Production"  # Gas pricing: 'Production', 'Postage', 'Trickle', 'Uniform'
    )

    # Create model using unified factory
    network, setup_summary = create_model(
        MODEL_ID,
        generators_as_links=generators_as_links,
        testing_mode=testing_mode,
        use_csv=use_csv,
        infrastructure_scenario=infrastructure_scenario,
        pricing_scheme=pricing_scheme,
    )

    # Print setup summary
    print("\n" + "=" * 70)
    print("MAREI-EU MODEL SETUP SUMMARY (Generators-as-Links Version)")
    print("=" * 70)
    print(f"Network type: {setup_summary['network_type']}")
    print(f"Sectors: {', '.join(setup_summary['sectors'])}")
    print(
        f"Generators represented as: {'Links' if generators_as_links else 'Generators'}"
    )
    print(f"CSV integration: {'Enabled' if use_csv else 'Disabled'}")

    # CSV integration summary (if enabled)
    if use_csv and setup_summary.get("csv_data_loaded", False):
        csv_summary = setup_summary.get("csv_integration", {})
        print("\nCSV Data Integration:")
        print(
            f"  Infrastructure scenario: {setup_summary.get('infrastructure_scenario', 'N/A')}"
        )
        print(f"  Gas pricing scheme: {setup_summary.get('pricing_scheme', 'N/A')}")
        print(f"  Data categories loaded: {csv_summary.get('data_categories', 0)}")
        print(
            f"  Available datasets: {', '.join(csv_summary.get('available_datasets', []))}"
        )

        # EU countries summary
        if setup_summary.get("eu_countries"):
            print(
                f"  EU countries: {len(setup_summary['eu_countries'])} ({', '.join(setup_summary['eu_countries'][:5])}...)"
            )

    # Electricity sector summary
    elec_summary = setup_summary["electricity"]
    print("\nElectricity Sector:")
    print(f"  Buses: {elec_summary['buses']}")
    print(f"  Generators: {elec_summary['generators']}")
    print(f"  Loads: {elec_summary['loads']}")
    print(f"  Lines: {elec_summary['lines']}")
    print(f"  Storage: {elec_summary['storage']}")

    # Enhanced gas sector summary (with CSV integration)
    gas_summary = setup_summary["gas"]
    if use_csv:
        print("\nGas Sector (Enhanced with CSV Integration):")
    else:
        print("\nGas Sector (Enhanced with PyPSA Patterns):")
    print(f"  Gas buses: {gas_summary['buses']} (enhanced carrier typing)")
    print(
        f"  Gas fields: {gas_summary.get('fields', 0)} (Store components - finite reserves)"
    )
    print(f"  Gas pipelines: {gas_summary['pipelines']} (Link components with losses)")
    print(f"  Gas storage: {gas_summary['storage']} (Store components with cycling)")
    print(
        f"  Gas plants: {gas_summary.get('plants', 0)} (gas→electricity conversion Links)"
    )
    print(f"  Gas demand: {gas_summary['demand']} (Load components)")
    if use_csv:
        print(
            f"  LNG terminals: {gas_summary.get('lng', 0)} (Store components from CSV)"
        )

    # Enhanced sector coupling summary
    coupling_summary = setup_summary["sector_coupling"]
    print("\nEnhanced Sector Coupling:")
    if use_csv:
        # CSV integration mode
        print(
            f"  Gas-to-electric links: {coupling_summary.get('gas_to_elec_links', 0)} (CSV enhanced)"
        )
        print(
            f"  Gas plants (from PlexosDB): {coupling_summary.get('gas_plants_added', 0)}"
        )
        if generators_as_links:
            print(
                f"  Generator links: {coupling_summary.get('gas_generators', 0)} (generators-as-links mode)"
            )
        print(
            f"  Conversion efficiency range: {coupling_summary.get('efficiency_range', 'N/A')}"
        )
        if coupling_summary.get("fuel_types"):
            print(f"  Fuel types: {', '.join(coupling_summary['fuel_types'])}")
    else:
        # Traditional PlexosDB mode
        if generators_as_links:
            print(
                f"  Conventional generator-links: {coupling_summary.get('generator_links', 0)} (fuel→electricity)"
            )
            print(
                f"  Renewable generators: {coupling_summary.get('renewable_generators', 0)} (standard generators)"
            )
            print(
                f"  Gas plants: {coupling_summary.get('gas_plants_added', 0)} (gas→electricity from gas.py)"
            )
            print(
                f"  Gas-fired generators: {coupling_summary.get('gas_generators', 0)} (generators-as-links mode)"
            )
            print(
                f"  Multi-sector links: {coupling_summary.get('sector_coupling_links', 0)} (electrolysis/fuel_cell)"
            )
            print(f"  Fuel types represented: {coupling_summary.get('fuel_types', [])}")
        else:
            print(
                f"  Gas-to-electric generators: {coupling_summary.get('gas_generators', 0)}"
            )
            print(
                f"  Gas plants: {coupling_summary.get('gas_plants_added', 0)} (from gas sector)"
            )
        print(
            f"  Conversion efficiency range: {coupling_summary.get('efficiency_range', 'N/A')}"
        )

    # Enhanced network totals with multi-sector breakdown
    print("\nTotal Network Components (Multi-Sector):")
    print(f"  Total buses: {len(network.buses)} (electricity + gas + other carriers)")
    print(
        f"  Total generators: {len(network.generators)} (renewables + remaining conventional)"
    )
    print(
        f"  Total links: {len(network.links)} (transmission + pipelines + conversions + generator-links)"
    )
    print(f"  Total storage units: {len(network.storage_units)} (electricity storage)")
    print(f"  Total stores: {len(network.stores)} (gas fields + gas storage)")
    print(f"  Total loads: {len(network.loads)} (electricity + gas demand)")

    # Enhanced carrier breakdown
    if len(network.buses) > 0:
        carriers = network.buses.carrier.value_counts()
        print("\nBus Carrier Distribution (Multi-Sector):")
        for carrier, count in carriers.items():
            print(f"  {carrier}: {count} buses")

    # Link breakdown by carrier
    if len(network.links) > 0 and "carrier" in network.links.columns:
        link_carriers = network.links.carrier.value_counts()
        print("\nLink Carrier Distribution:")
        for carrier, count in link_carriers.items():
            print(f"  {carrier}: {count} links")

    # Run consistency check
    print("\nRunning network consistency check...")
    try:
        network.consistency_check()
        print("Network consistency check passed!")
    except Exception as e:
        print(f"Network consistency check failed: {e}")
        print("Proceeding with caution...")

    # Select subset of snapshots for optimization
    if len(network.snapshots) > 0:
        x = 8760  # number of snapshots to select per year
        snapshots_by_year: defaultdict[int, list] = defaultdict(list)
        for snap in network.snapshots:
            year = pd.Timestamp(snap).year
            if len(snapshots_by_year[year]) < x:
                snapshots_by_year[year].append(snap)

        subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

        # Configuration
        use_subset = False  # Set to True to optimize on subset, False for full network

        # solve the network
        try:
            if use_subset:
                print(f"\nOptimizing network with {len(subset)} snapshots...")
                network.optimize(
                    solver_name="gurobi",
                    snapshots=subset,
                    solver_options={
                        "Threads": 6,
                        "Method": 2,  # barrier
                        "Crossover": 0,
                        "BarConvTol": 1.0e-5,
                        "Seed": 123,
                        "AggFill": 0,
                        "PreDual": 0,
                        "GURO_PAR_BARDENSETHRESH": 200,
                    },
                )  # type: ignore
            else:
                print(
                    f"\nOptimizing network with {len(network.snapshots)} snapshots..."
                )
                network.optimize(
                    solver_name="gurobi",
                    solver_options={
                        "Threads": 6,
                        "Method": 2,  # barrier
                        "Crossover": 0,
                        "BarConvTol": 1.0e-5,
                        "Seed": 123,
                        "AggFill": 0,
                        "PreDual": 0,
                        "GURO_PAR_BARDENSETHRESH": 200,
                    },
                )  # type: ignore

            print(" Optimization complete!")

            # Print optimization results summary
            print("\nOptimization Results:")
            print(f"  Objective value: {network.objective:.2f}")

            if hasattr(network, "generators_t") and "p" in network.generators_t:
                total_gen = network.generators_t.p.sum().sum()
                print(f"  Total generation: {total_gen:.2f} MWh")

            if hasattr(network, "links_t") and "p0" in network.links_t:
                if generators_as_links:
                    # Show fuel-to-electric conversion with more detailed debugging
                    gen_link_cols = [
                        col for col in network.links_t.p0.columns if "gen_link" in col
                    ]
                    print(
                        f"   DEBUG - Generator-link columns found: {len(gen_link_cols)}"
                    )
                    if len(gen_link_cols) > 0:
                        print(
                            f"   DEBUG - Sample gen-link columns: {gen_link_cols[:3]}"
                        )
                        fuel_to_elec = network.links_t.p0[gen_link_cols].sum().sum()
                        print(f"  Fuel-to-electric conversion: {fuel_to_elec:.2f} MWh")

                        # Check if any generator-links are actually operating
                        active_gen_links = (
                            network.links_t.p0[gen_link_cols] > 0.01
                        ).any()
                        active_count = active_gen_links.sum()
                        print(
                            f"  DEBUG - Active generator-links: {active_count} out of {len(gen_link_cols)}"
                        )
                    else:
                        print("  DEBUG - No generator-link columns found in links_t.p0")
                        print(
                            f"  DEBUG - Available link columns: {list(network.links_t.p0.columns[:5])}"
                        )
                else:
                    # Show gas-to-electric conversion
                    total_gas_elec = (
                        network.links_t.p0[
                            network.links_t.p0.columns[
                                network.links_t.p0.columns.str.contains("gas_to_elec")
                            ]
                        ]
                        .sum()
                        .sum()
                    )
                    print(f"  Gas-to-electric conversion: {total_gas_elec:.2f} MWh")

            # Save results
            output_file = "marei_eu_generators_as_links_results.nc"
            print(f"\nSaving results to {output_file}...")
            network.export_to_netcdf(output_file)
            print("  Results saved!")

        except Exception as e:
            print(f"Optimization failed: {e}")
            print("Network created successfully but optimization encountered issues.")
    else:
        print("No snapshots found in network - skipping optimization")

    if testing_mode:
        print("\nTesting mode was enabled - model may be incomplete")
