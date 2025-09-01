"""
MaREI-EU Multi-Sector Model with Generators-as-Links Option

This script creates a PyPSA network from the MaREI-EU PLEXOS model where conventional
generators (coal, gas, nuclear, etc.) can be represented as Links instead of Generators.
This enables multi-sector coupling by connecting fuel buses to electric buses with
efficiency conversions.
"""

from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.db.models import INPUT_XMLS
from plexos_pypsa.network.multi_sector_db import setup_gas_electric_network_db


def create_marei_eu_generators_as_links_model(
    generators_as_links: bool = True, testing_mode: bool = False
):
    """
    Create MaREI-EU PyPSA model with optional generators-as-links conversion.

    The MaREI-EU model includes:
    - Electricity network (Node, Generator, Line, Storage)
    - Gas network represented as PyPSA Links and Nodes
    - Conventional generators optionally as Links for multi-sector coupling
    - Sector coupling through fuel-to-electric conversion Links

    Parameters
    ----------
    generators_as_links : bool, optional
        If True, represent conventional generators (coal, gas, nuclear, etc.) as Links
        connecting fuel buses to electric buses. If False, use standard Generators.
        Default True.
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.
        Default False creates complete model.

    Returns
    -------
    pypsa.Network
        Multi-sector PyPSA network with gas and electricity
    dict
        Setup summary with model statistics
    """
    # Get XML file path from models registry
    xml_file = INPUT_XMLS["marei-eu"]

    print("Creating MaREI-EU Multi-Sector PyPSA Model (Generators-as-Links Version)...")
    print(f"XML file: {xml_file}")
    print(f"Generators as links: {generators_as_links}")

    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")

    # Load PLEXOS database
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(xml_file)

    # Initialize PyPSA network
    network = pypsa.Network()

    # Set up multi-sector network with gas and electricity using database queries
    print("\nSetting up multi-sector network (Gas + Electricity)...")
    print("   Using direct database queries to discover gas and electricity components")

    if generators_as_links:
        print("   Representing conventional generators as fuel-to-electric Links")
        print("   Enabling multi-sector coupling through fuel buses")
    else:
        print("   Using standard Generator representation")

    setup_summary = setup_gas_electric_network_db(
        network=network,
        db=plexos_db,
        generators_as_links=generators_as_links,
        testing_mode=testing_mode,
    )

    return network, setup_summary


if __name__ == "__main__":
    # Create the multi-sector model
    # Set generators_as_links=True to use Link representation for conventional generators
    # Set testing_mode=True for faster development, False for complete model
    generators_as_links = True  # NEW: Enable generators-as-links functionality
    testing_mode = False  # Full model to investigate actual PLEXOS data

    network, setup_summary = create_marei_eu_generators_as_links_model(
        generators_as_links=generators_as_links, testing_mode=testing_mode
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

    # Electricity sector summary
    elec_summary = setup_summary["electricity"]
    print("\nElectricity Sector:")
    print(f"  Buses: {elec_summary['buses']}")
    print(f"  Generators: {elec_summary['generators']}")
    print(f"  Loads: {elec_summary['loads']}")
    print(f"  Lines: {elec_summary['lines']}")
    print(f"  Storage: {elec_summary['storage']}")

    # Gas sector summary
    gas_summary = setup_summary["gas"]
    print("\nGas Sector:")
    print(f"  Gas buses: {gas_summary['buses']}")
    print(f"  Gas pipelines: {gas_summary['pipelines']}")
    print(f"  Gas storage: {gas_summary['storage']}")
    print(f"  Gas demand: {gas_summary['demand']}")
    print(f"  Gas fields: {gas_summary['fields']}")

    # Sector coupling summary
    coupling_summary = setup_summary["sector_coupling"]
    print("\nSector Coupling:")
    if generators_as_links:
        print(
            f"  Conventional generator-links: {coupling_summary.get('generator_links', 0)}"
        )
        print(
            f"  Renewable generators: {coupling_summary.get('renewable_generators', 0)}"
        )
        print(f"  Fuel types represented: {coupling_summary.get('fuel_types', [])}")
    else:
        print(f"  Gas-to-electric generators: {coupling_summary['gas_generators']}")
    print(f"  Conversion efficiency range: {coupling_summary['efficiency_range']}")

    # Network totals
    print("\nTotal Network Components:")
    print(f"  Total buses: {len(network.buses)}")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total links: {len(network.links)}")
    print(f"  Total storage units: {len(network.storage_units)}")
    print(f"  Total loads: {len(network.loads)}")

    # Carrier breakdown
    if len(network.buses) > 0:
        carriers = network.buses.carrier.value_counts()
        print("\nCarrier Distribution:")
        for carrier, count in carriers.items():
            print(f"  {carrier}: {count} buses")

    # Run consistency check
    print("\nRunning network consistency check...")
    try:
        network.consistency_check()
        print("✓ Network consistency check passed!")
    except Exception as e:
        print(f"⚠ Network consistency check failed: {e}")
        print("Proceeding with caution...")

    # Select subset of snapshots for optimization
    if len(network.snapshots) > 0:
        x = 8760  # number of snapshots to select per year
        snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
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
                print(f"\nOptimizing network with {len(network.snapshots)} snapshots...")
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

            print("✓ Optimization complete!")

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
                        f"  🔍 DEBUG - Generator-link columns found: {len(gen_link_cols)}"
                    )
                    if len(gen_link_cols) > 0:
                        print(
                            f"  🔍 DEBUG - Sample gen-link columns: {gen_link_cols[:3]}"
                        )
                        fuel_to_elec = network.links_t.p0[gen_link_cols].sum().sum()
                        print(f"  Fuel-to-electric conversion: {fuel_to_elec:.2f} MWh")

                        # Check if any generator-links are actually operating
                        active_gen_links = (
                            network.links_t.p0[gen_link_cols] > 0.01
                        ).any()
                        active_count = active_gen_links.sum()
                        print(
                            f"  🔍 DEBUG - Active generator-links: {active_count} out of {len(gen_link_cols)}"
                        )
                    else:
                        print(
                            "  🔍 DEBUG - No generator-link columns found in links_t.p0"
                        )
                        print(
                            f"  🔍 DEBUG - Available link columns: {list(network.links_t.p0.columns[:5])}"
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
            print(f"⚠ Optimization failed: {e}")
            print("Network created successfully but optimization encountered issues.")
    else:
        print("⚠ No snapshots found in network - skipping optimization")

    if testing_mode:
        print("\n⚠️  Testing mode was enabled - model may be incomplete")
