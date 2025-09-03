"""
MaREI-EU Multi-Sector Model (Electricity + Gas)

This script creates a PyPSA network from the MaREI-EU PLEXOS model which includes
both electricity and gas sectors with sector coupling through gas-fired generators.
"""

from collections import defaultdict
from typing import DefaultDict

import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.db.models import INPUT_XMLS
from plexos_pypsa.network.multi_sector_db import setup_gas_electric_network_db


def create_marei_eu_model():
    """
    Create MaREI-EU PyPSA model with gas and electricity sectors.

    The MaREI-EU model includes:
    - Electricity network (Node, Generator, Line, Storage)
    - Gas network represented as PyPSA Links and Nodes
    - Sector coupling through gas-to-electric conversion Links

    Returns
    -------
    pypsa.Network
        Multi-sector PyPSA network with gas and electricity
    dict
        Setup summary with model statistics
    """
    # Get XML file path from models registry
    xml_file = INPUT_XMLS["marei-eu"]

    print("Creating MaREI-EU Multi-Sector PyPSA Model...")
    print(f"XML file: {xml_file}")

    # Load PLEXOS database
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(xml_file)

    # Initialize PyPSA network
    network = pypsa.Network()

    # Set up multi-sector network with gas and electricity using database queries
    print("\nSetting up multi-sector network (Gas + Electricity)...")
    print("   Using direct database queries to discover gas and electricity components")
    print("   Representing gas sector through Links and Nodes for sector coupling")

    setup_summary = setup_gas_electric_network_db(network=network, db=plexos_db)

    return network, setup_summary


if __name__ == "__main__":
    # Create the multi-sector model
    network, setup_summary = create_marei_eu_model()

    # Print setup summary
    print("\n" + "=" * 60)
    print("MAREI-EU MODEL SETUP SUMMARY")
    print("=" * 60)
    print(f"Network type: {setup_summary['network_type']}")
    print(f"Sectors: {', '.join(setup_summary['sectors'])}")

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
    print("\nGas Sector (Enhanced):")
    print(f"  Gas buses: {gas_summary['buses']}")
    print(f"  Gas fields: {gas_summary.get('fields', 0)} (Store components)")
    print(f"  Gas pipelines: {gas_summary['pipelines']} (Link components)")
    print(f"  Gas storage: {gas_summary['storage']} (Store components)")
    print(f"  Gas plants: {gas_summary.get('plants', 0)} (gas→electricity Links)")
    print(f"  Gas demand: {gas_summary['demand']}")

    # Enhanced sector coupling summary
    coupling_summary = setup_summary["sector_coupling"]
    print("\nEnhanced Sector Coupling:")
    print(f"  Gas plants: {coupling_summary.get('gas_plants_added', 0)} (from gas sector)")
    print(f"  Gas-to-electric generators: {coupling_summary['gas_generators']}")
    print(f"  Multi-sector links: {coupling_summary.get('sector_coupling_links', 0)} (electrolysis/fuel_cell)")
    print(f"  Conversion efficiency range: {coupling_summary['efficiency_range']}")
    if coupling_summary.get('fuel_types'):
        print(f"  Fuel types: {', '.join(coupling_summary['fuel_types'])}")

    # Network totals with enhanced multi-sector breakdown
    print("\nTotal Network Components:")
    print(f"  Total buses: {len(network.buses)} (electricity + gas)")
    print(f"  Total generators: {len(network.generators)}")
    print(f"  Total links: {len(network.links)} (transmission + pipelines + conversions)")
    print(f"  Total storage units: {len(network.storage_units)}")
    print(f"  Total stores: {len(network.stores)} (gas fields + gas storage)")
    print(f"  Total loads: {len(network.loads)} (electricity + gas demand)")
    
    # Component breakdown by carrier
    if len(network.buses) > 0:
        carriers = network.buses.carrier.value_counts()
        print("\nBus Carrier Distribution:")
        for carrier, count in carriers.items():
            print(f"  {carrier}: {count} buses")

    # Run consistency check
    print("\nRunning network consistency check...")
    network.consistency_check()
    print("✓ Network consistency check passed!")

    # Select subset of snapshots for optimization
    x = 100  # number of snapshots to select per year
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
        if hasattr(network, "generators_t") and "p" in network.generators_t:
            total_gen = network.generators_t.p.sum().sum()
            print(f"  Total generation: {total_gen:.2f} MWh")
        if hasattr(network, "links_t") and "p0" in network.links_t:
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
        output_file = "marei_eu_results.nc"
        print(f"\nSaving results to {output_file}...")
        network.export_to_netcdf(output_file)
        print("  Results saved!")

    except Exception as e:
        print(f"⚠ Optimization failed: {e}")
        print("Network created successfully but optimization encountered issues.")
