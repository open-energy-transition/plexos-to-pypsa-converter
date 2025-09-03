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
    generators_as_links: bool = True,
    testing_mode: bool = False,
    use_csv_integration: bool = False,
    csv_data_path: str = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/2_Modeling/Plexos Converter/Input Models/University College Cork/MaREI/EU Power & Gas Model/CSV Files",
    infrastructure_scenario: str = "PCI",
    pricing_scheme: str = "Production"
):
    """
    Create MaREI-EU PyPSA model with optional generators-as-links conversion.

    The MaREI-EU model includes:
    - Electricity network (Node, Generator, Line, Storage)
    - Gas network represented as PyPSA Links and Nodes
    - Conventional generators optionally as Links for multi-sector coupling
    - Sector coupling through fuel-to-electric conversion Links
    - Optional CSV data integration for enhanced demand profiles and infrastructure

    Parameters
    ----------
    generators_as_links : bool, optional
        If True, represent conventional generators (coal, gas, nuclear, etc.) as Links
        connecting fuel buses to electric buses. If False, use standard Generators.
        Default True.
    testing_mode : bool, optional
        If True, process only limited subsets of components for faster testing.
        Default False creates complete model.
    use_csv_integration : bool, optional
        If True, integrates MaREI CSV data for detailed demand profiles and infrastructure.
        If False, uses PlexosDB-only approach (legacy behavior). Default False.
    csv_data_path : str, optional
        Path to MaREI CSV Files directory containing demand, infrastructure, and pricing data
    infrastructure_scenario : str, optional
        Infrastructure scenario for gas network ('PCI', 'High', 'Low'). Default 'PCI'.
    pricing_scheme : str, optional
        Gas pricing mechanism ('Production', 'Postage', 'Trickle', 'Uniform'). Default 'Production'.

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
    print(f"CSV integration: {'Enabled' if use_csv_integration else 'Disabled'}")

    if testing_mode:
        print("⚠️  TESTING MODE: Processing limited subsets for faster development")
        
    if use_csv_integration:
        print(f"CSV data path: {csv_data_path}")
        print(f"Infrastructure scenario: {infrastructure_scenario}")
        print(f"Pricing scheme: {pricing_scheme}")

    # Load PLEXOS database
    print("\nLoading PLEXOS database...")
    plexos_db = PlexosDB.from_xml(xml_file)

    # Initialize PyPSA network
    network = pypsa.Network()

    if use_csv_integration:
        # Enhanced setup with CSV data integration
        print("\nSetting up enhanced multi-sector network with CSV data integration...")
        print("   Combining PLEXOS database topology with MaREI CSV demand and infrastructure data")
        print("   Following PyPSA multi-sector patterns for gas/electricity coupling")

        if generators_as_links:
            print("   Representing conventional generators as fuel-to-electric Links")
            print("   Enabling multi-sector coupling through fuel buses")
        else:
            print("   Using standard Generator representation")
        
        from plexos_pypsa.network.multi_sector_db import setup_marei_csv_network
        
        setup_summary = setup_marei_csv_network(
            network=network,
            db=plexos_db,
            csv_data_path=csv_data_path,
            infrastructure_scenario=infrastructure_scenario,
            pricing_scheme=pricing_scheme,
            generators_as_links=generators_as_links
        )
    else:
        # Traditional PlexosDB-only setup
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
    # Set use_csv_integration=True to enable MaREI CSV data integration
    generators_as_links = True  # Enable generators-as-links functionality
    testing_mode = False  # Full model to investigate actual PLEXOS data
    use_csv_integration = True  # Enable CSV integration for enhanced model
    infrastructure_scenario = "PCI"  # Infrastructure scenario: 'PCI', 'High', 'Low'
    pricing_scheme = "Production"  # Gas pricing: 'Production', 'Postage', 'Trickle', 'Uniform'

    network, setup_summary = create_marei_eu_generators_as_links_model(
        generators_as_links=generators_as_links,
        testing_mode=testing_mode,
        use_csv_integration=use_csv_integration,
        infrastructure_scenario=infrastructure_scenario,
        pricing_scheme=pricing_scheme
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
    print(f"CSV integration: {'Enabled' if use_csv_integration else 'Disabled'}")
    
    # CSV integration summary (if enabled)
    if use_csv_integration and setup_summary.get('csv_data_loaded', False):
        csv_summary = setup_summary.get('csv_integration', {})
        print(f"\nCSV Data Integration:")
        print(f"  Infrastructure scenario: {setup_summary.get('infrastructure_scenario', 'N/A')}")
        print(f"  Gas pricing scheme: {setup_summary.get('pricing_scheme', 'N/A')}")
        print(f"  Data categories loaded: {csv_summary.get('data_categories', 0)}")
        print(f"  Available datasets: {', '.join(csv_summary.get('available_datasets', []))}")
        
        # EU countries summary
        if setup_summary.get('eu_countries'):
            print(f"  EU countries: {len(setup_summary['eu_countries'])} ({', '.join(setup_summary['eu_countries'][:5])}...)")

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
    if use_csv_integration:
        print("\nGas Sector (Enhanced with CSV Integration):")
    else:
        print("\nGas Sector (Enhanced with PyPSA Patterns):")
    print(f"  Gas buses: {gas_summary['buses']} (enhanced carrier typing)")
    print(f"  Gas fields: {gas_summary.get('fields', 0)} (Store components - finite reserves)")
    print(f"  Gas pipelines: {gas_summary['pipelines']} (Link components with losses)")
    print(f"  Gas storage: {gas_summary['storage']} (Store components with cycling)")
    print(f"  Gas plants: {gas_summary.get('plants', 0)} (gas→electricity conversion Links)")
    print(f"  Gas demand: {gas_summary['demand']} (Load components)")
    if use_csv_integration:
        print(f"  LNG terminals: {gas_summary.get('lng', 0)} (Store components from CSV)")

    # Enhanced sector coupling summary
    coupling_summary = setup_summary["sector_coupling"]
    print("\nEnhanced Sector Coupling:")
    if use_csv_integration:
        # CSV integration mode
        print(f"  Gas-to-electric links: {coupling_summary.get('gas_to_elec_links', 0)} (CSV enhanced)")
        print(f"  Gas plants (from PlexosDB): {coupling_summary.get('gas_plants_added', 0)}")
        if generators_as_links:
            print(f"  Generator links: {coupling_summary.get('gas_generators', 0)} (generators-as-links mode)")
        print(f"  Conversion efficiency range: {coupling_summary.get('efficiency_range', 'N/A')}")
        if coupling_summary.get('fuel_types'):
            print(f"  Fuel types: {', '.join(coupling_summary['fuel_types'])}")
    else:
        # Traditional PlexosDB mode
        if generators_as_links:
            print(f"  Conventional generator-links: {coupling_summary.get('generator_links', 0)} (fuel→electricity)")
            print(f"  Renewable generators: {coupling_summary.get('renewable_generators', 0)} (standard generators)")
            print(f"  Gas plants: {coupling_summary.get('gas_plants_added', 0)} (gas→electricity from gas.py)")
            print(f"  Gas-fired generators: {coupling_summary.get('gas_generators', 0)} (generators-as-links mode)")
            print(f"  Multi-sector links: {coupling_summary.get('sector_coupling_links', 0)} (electrolysis/fuel_cell)")
            print(f"  Fuel types represented: {coupling_summary.get('fuel_types', [])}")
        else:
            print(f"  Gas-to-electric generators: {coupling_summary.get('gas_generators', 0)}")
            print(f"  Gas plants: {coupling_summary.get('gas_plants_added', 0)} (from gas sector)")
        print(f"  Conversion efficiency range: {coupling_summary.get('efficiency_range', 'N/A')}")

    # Enhanced network totals with multi-sector breakdown
    print("\nTotal Network Components (Multi-Sector):")
    print(f"  Total buses: {len(network.buses)} (electricity + gas + other carriers)")
    print(f"  Total generators: {len(network.generators)} (renewables + remaining conventional)")
    print(f"  Total links: {len(network.links)} (transmission + pipelines + conversions + generator-links)")
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
    if len(network.links) > 0 and 'carrier' in network.links.columns:
        link_carriers = network.links.carrier.value_counts()
        print("\nLink Carrier Distribution:")
        for carrier, count in link_carriers.items():
            print(f"  {carrier}: {count} links")

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
