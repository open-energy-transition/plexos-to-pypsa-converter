"""
PLEXOS-MESSAGE Multi-Sector Model (Electricity + Hydrogen + Ammonia)

This script creates a PyPSA network from the PLEXOS-MESSAGE model which includes
electricity, hydrogen, and ammonia sectors with process-based sector coupling.
"""

from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.db.models import INPUT_XMLS
from plexos_pypsa.network.multi_sector_db import setup_flow_network_db


def create_plexos_message_model(testing_mode: bool = False):
    """
    Create PLEXOS-MESSAGE PyPSA model with electricity, hydrogen, and ammonia sectors.

    The PLEXOS-MESSAGE model includes:
    - Flow Network represented as PyPSA Nodes and Links
    - Process-based conversions as PyPSA Links for sector coupling
    - Multi-sector representation (Electricity, Hydrogen, Ammonia)

    Parameters
    ----------
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.
        Default False creates complete model.

    Returns
    -------
    pypsa.Network
        Multi-sector PyPSA network
    dict
        Setup summary with model statistics
    """
    # Get XML file path from models registry
    xml_file = INPUT_XMLS["plexos-message"]

    print("Creating PLEXOS-MESSAGE Multi-Sector PyPSA Model...")
    print(f"XML file: {xml_file}")

    # Load PLEXOS database
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(xml_file)

    # Initialize PyPSA network
    network = pypsa.Network()

    # Set up multi-sector flow network using database queries
    print(
        "\nSetting up multi-sector flow network (Electricity + Hydrogen + Ammonia)..."
    )
    print("   Using direct database queries to discover Flow Network components")
    print("   Representing all sectors through PyPSA Links and Nodes")

    setup_summary = setup_flow_network_db(
        network=network, db=plexos_db, testing_mode=testing_mode
    )

    return network, setup_summary


# n, s = create_plexos_message_model()  # Disabled to avoid double execution

if __name__ == "__main__":
    # Create the multi-sector model
    # Set testing_mode=True for faster development, False for complete model
    testing_mode = False  # Change to True for testing
    network, setup_summary = create_plexos_message_model(testing_mode=testing_mode)

    # Print setup summary
    print("\n" + "=" * 60)
    print("PLEXOS-MESSAGE MODEL SETUP SUMMARY")
    print("=" * 60)
    print(f"Network type: {setup_summary['network_type']}")
    print(f"Sectors: {', '.join(setup_summary['sectors'])}")

    # Flow network summary by carrier
    for sector in setup_summary["sectors"]:
        sector_summary = setup_summary[sector.lower()]
        print(f"\n{sector.title()} Sector:")
        print(f"  Flow nodes: {sector_summary['nodes']}")
        print(f"  Flow paths: {sector_summary['paths']}")
        print(f"  Flow storage: {sector_summary['storage']}")
        if "demand" in sector_summary:
            print(f"  Demand: {sector_summary['demand']}")

    # Process-based coupling summary
    process_summary = setup_summary["processes"]
    print("\nProcess-Based Sector Coupling:")
    for process_type, count in process_summary.items():
        print(f"  {process_type}: {count}")

    # Facilities summary
    if "facilities" in setup_summary:
        facility_summary = setup_summary["facilities"]
        print("\nFacilities:")
        for facility_type, count in facility_summary.items():
            print(f"  {facility_type}: {count}")

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
        x = 24  # number of snapshots to select per year (fewer for multi-sector complexity)
        snapshots_by_year: DefaultDict[int, list] = defaultdict(list)
        for snap in network.snapshots:
            year = pd.Timestamp(snap).year
            if len(snapshots_by_year[year]) < x:
                snapshots_by_year[year].append(snap)

        subset = [snap for snaps in snapshots_by_year.values() for snap in snaps]

        # Configuration
        use_subset = True  # Set to True to optimize on subset, False for full network

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

            # Sector-specific generation/conversion summaries
            if hasattr(network, "generators_t") and "p" in network.generators_t:
                total_gen = network.generators_t.p.sum().sum()
                print(f"  Total generation: {total_gen:.2f} MWh")

            if hasattr(network, "links_t") and "p0" in network.links_t:
                # Electrolysis (electricity to hydrogen)
                electrolysis_links = [
                    col
                    for col in network.links_t.p0.columns
                    if "electrolysis" in col.lower()
                ]
                if electrolysis_links:
                    electrolysis_total = (
                        network.links_t.p0[electrolysis_links].sum().sum()
                    )
                    print(f"  Electrolysis (Elec→H2): {electrolysis_total:.2f} MWh")

                # Hydrogen to electricity
                h2_elec_links = [
                    col
                    for col in network.links_t.p0.columns
                    if "h2power" in col.lower() or "fuel_cell" in col.lower()
                ]
                if h2_elec_links:
                    h2_elec_total = network.links_t.p0[h2_elec_links].sum().sum()
                    print(f"  H2 Power (H2→Elec): {h2_elec_total:.2f} MWh")

                # Ammonia synthesis
                ammonia_links = [
                    col
                    for col in network.links_t.p0.columns
                    if "ammonia" in col.lower()
                ]
                if ammonia_links:
                    ammonia_total = network.links_t.p0[ammonia_links].sum().sum()
                    print(f"  Ammonia synthesis (H2→NH3): {ammonia_total:.2f} MWh")

            # Save results
            output_file = "plexos_message_results.nc"
            print(f"\nSaving results to {output_file}...")
            network.export_to_netcdf(output_file)
            print("  Results saved!")

        except Exception as e:
            print(f"⚠ Optimization failed: {e}")
            print("Network created successfully but optimization encountered issues.")
    else:
        print("⚠ No snapshots found in network - skipping optimization")
