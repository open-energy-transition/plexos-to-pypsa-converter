from collections import defaultdict

import pandas as pd
import pypsa
from plexosdb import PlexosDB

from src.db.models import get_model_xml_path
from src.network.multi_sector_db import (
    setup_gas_electric_network_db,
    setup_marei_csv_network,
)
from src.utils.model_paths import get_model_directory

# Constants
MODEL_ID = "marei-eu"
SNAPSHOTS_PER_YEAR = 100


def create_marei_eu_model(
    use_csv_integration: bool = False,
    csv_data_path: str | None = None,
    infrastructure_scenario: str = "PCI",
    pricing_scheme: str = "Production",
) -> tuple[pypsa.Network, dict]:
    """Create MaREI-EU PyPSA model with gas and electricity sectors.

    Parameters
    ----------
    use_csv_integration : bool, default False
        If True, integrates MaREI CSV data for enhanced model.
    csv_data_path : str, optional
        Path to MaREI CSV Files directory. Auto-determined if None.
    infrastructure_scenario : str, default "PCI"
        Infrastructure scenario ('PCI', 'High', 'Low')
    pricing_scheme : str, default "Production"
        Gas pricing mechanism ('Production', 'Postage', 'Trickle', 'Uniform')

    Returns
    -------
    tuple[pypsa.Network, dict]
        Multi-sector PyPSA network and setup summary
    """
    # Find and validate model data
    xml_file = get_model_xml_path(MODEL_ID)
    if xml_file is None:
        msg = f"Model '{MODEL_ID}' not found. Please download and extract the MaREI EU model data."
        raise FileNotFoundError(msg)

    # Auto-determine CSV path if needed
    if csv_data_path is None and use_csv_integration:
        model_dir = get_model_directory(MODEL_ID)
        if model_dir:
            csv_data_path = str(model_dir / "CSV Files")
        else:
            msg = "Could not locate MaREI model directory"
            raise FileNotFoundError(msg)

    print("Creating MaREI-EU Multi-Sector PyPSA Model...")
    print(f"CSV integration: {'Enabled' if use_csv_integration else 'Disabled'}")

    # Load database and initialize network
    plexos_db = PlexosDB.from_xml(str(xml_file))
    network = pypsa.Network()

    # Set up network based on integration type
    if use_csv_integration:
        setup_summary = setup_marei_csv_network(
            network=network,
            db=plexos_db,
            csv_data_path=csv_data_path,
            infrastructure_scenario=infrastructure_scenario,
            pricing_scheme=pricing_scheme,
            generators_as_links=False,
        )
    else:
        setup_summary = setup_gas_electric_network_db(network=network, db=plexos_db)

    return network, setup_summary


if __name__ == "__main__":
    # Configuration
    use_csv_integration = True
    infrastructure_scenario = "PCI"
    pricing_scheme = "Production"

    # Create model
    network, setup_summary = create_marei_eu_model(
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
    print("✓ Network consistency check passed!")

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
        print("✓ Optimization complete!")

        # Save results
        output_file = "marei_eu_results.nc"
        print(f"\nSaving results to {output_file}...")
        network.export_to_netcdf(output_file)
        print("  Results saved!")

    except Exception as e:
        print(f"⚠ Optimization failed: {e}")
        print("Network created successfully but optimization encountered issues.")
