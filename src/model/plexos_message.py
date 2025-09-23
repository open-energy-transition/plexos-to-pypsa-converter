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

from src.db.models import INPUT_XMLS
from src.network.multi_sector_db import (
    setup_enhanced_flow_network_with_csv,
    setup_flow_network_db,
)


def create_plexos_message_model(
    testing_mode: bool = False, use_csv_integration: bool = True
):
    """
    Create PLEXOS-MESSAGE PyPSA model with electricity, hydrogen, and ammonia sectors.

    The PLEXOS-MESSAGE model includes:
    - Flow Network represented as PyPSA Nodes and Links
    - Process-based conversions as PyPSA Links for sector coupling
    - Multi-sector representation (Electricity, Hydrogen, Ammonia)
    - CSV data integration for costs, demand profiles, and infrastructure

    Parameters
    ----------
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.
        Default False creates complete model.
    use_csv_integration : bool, optional
        If True, use enhanced setup with CSV data integration following PyPSA best practices.
        If False, use traditional flow network setup. Default True.

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
    print(f"CSV Integration: {'Enabled' if use_csv_integration else 'Disabled'}")

    # Load PLEXOS database
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(xml_file)

    # Initialize PyPSA network
    network = pypsa.Network()

    if use_csv_integration:
        # Use enhanced setup with CSV data integration
        print(
            "\nSetting up enhanced multi-sector flow network with CSV data integration..."
        )
        print("   Following PyPSA best practices for sector coupling")
        print("   Integrating BuildCosts.csv, Load.csv, H2_Demand_With_Blending.csv")
        print(
            "   Creating sector coupling links (Electrolysis, H2Power, Haber-Bosch, Ammonia Cracking)"
        )

        # Define inputs folder path
        inputs_folder = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/2_Modeling/Plexos Converter/Input Models/University College Cork/MaREI/MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link/MESSAGEix-GLOBIOM-EN-NPi2020-500-Soft-Link-main/Inputs"

        setup_summary = setup_enhanced_flow_network_with_csv(
            network=network,
            db=plexos_db,
            inputs_folder=inputs_folder,
            testing_mode=testing_mode,
        )
    else:
        # Use traditional flow network setup
        print(
            "\nSetting up multi-sector flow network (Electricity + Hydrogen + Ammonia)..."
        )
        print("   Using direct database queries to discover Flow Network components")
        print("   Representing all sectors through PyPSA Links and Nodes")

        setup_summary = setup_flow_network_db(
            network=network, db=plexos_db, testing_mode=testing_mode
        )

    return network, setup_summary


if __name__ == "__main__":
    # Create the multi-sector model
    # Set testing_mode=True for faster development, False for complete model
    testing_mode = True  # Change to True for testing
    use_csv_integration = True  # Enable CSV data integration with PyPSA best practices

    network, setup_summary = create_plexos_message_model(
        testing_mode=testing_mode, use_csv_integration=use_csv_integration
    )

    # Print setup summary
    print("\n" + "=" * 60)
    print("PLEXOS-MESSAGE MODEL SETUP SUMMARY")
    print("=" * 60)
    print(f"Network type: {setup_summary['network_type']}")
    print(f"Sectors: {', '.join(setup_summary['sectors'])}")

    # CSV integration summary (if enabled)
    if use_csv_integration and setup_summary.get("csv_data_loaded", False):
        csv_summary = setup_summary.get("csv_integration", {})
        print("\nCSV Data Integration:")
        print(f"  Cost files loaded: {csv_summary.get('cost_files', 0)}")
        print(f"  Time series files loaded: {csv_summary.get('time_series_files', 0)}")
        print(
            f"  Infrastructure files loaded: {csv_summary.get('infrastructure_files', 0)}"
        )
        print(f"  Snapshots source: {setup_summary.get('snapshots_source', 'N/A')}")
        print(f"  Snapshots count: {setup_summary.get('snapshots_count', 0)}")

    # Multi-sector buses (if CSV integration enabled)
    if "multi_sector_buses" in setup_summary:
        buses_summary = setup_summary["multi_sector_buses"]
        print("\nMulti-Sector Buses:")
        for sector, count in buses_summary.items():
            print(f"  {sector.title()}: {count}")
    elif "buses_by_sector" in setup_summary:
        # Traditional buses summary
        buses_summary = setup_summary["buses_by_sector"]
        print("\nBuses by sector:")
        for sector, count in buses_summary.items():
            print(f"  {sector.title()}: {count}")

    # Sector coupling links (if CSV integration enabled)
    if "sector_coupling_links" in setup_summary:
        coupling_summary = setup_summary["sector_coupling_links"]
        print("\nSector Coupling Links (PyPSA Best Practices):")
        for link_type, count in coupling_summary.items():
            print(f"  {link_type.replace('_', ' ').title()}: {count}")

    # Multi-sector loads (if CSV integration enabled)
    if "multi_sector_loads" in setup_summary:
        loads_summary = setup_summary["multi_sector_loads"]
        print("\nMulti-Sector Loads:")
        for sector, count in loads_summary.items():
            print(f"  {sector.title()}: {count}")

    # Multi-sector storage (if CSV integration enabled)
    if "multi_sector_storage" in setup_summary:
        storage_summary = setup_summary["multi_sector_storage"]
        print("\nMulti-Sector Storage:")
        for sector, count in storage_summary.items():
            print(f"  {sector.title()}: {count}")

    # H2 Pipelines (if CSV integration enabled)
    if "h2_pipelines" in setup_summary:
        print(f"\nH2 Pipeline Infrastructure: {setup_summary['h2_pipelines']} links")

    # Traditional summaries (if not using CSV integration)
    if not use_csv_integration or not setup_summary.get("csv_data_loaded", False):
        # Paths
        if "paths" in setup_summary:
            paths_summary = setup_summary["paths"]
            print("\nPaths:")
            for process_type, count in paths_summary.items():
                print(f"  {process_type}: {count}")

        # Process-based coupling summary
        if "processes" in setup_summary:
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
    print(f"  Total stores: {len(network.stores)}")
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
