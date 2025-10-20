from collections import defaultdict

import pandas as pd

from network.conversion import create_model
from network.generators_csv import load_data_file_profiles_csv
from network.outages import (
    apply_outage_schedule,
    build_outage_schedule,
    generate_stochastic_outages_csv,
    parse_explicit_outages_from_properties,
)

# Constants
MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60

# Create network
print("Creating SEM network...")
network, summary = create_model(MODEL_ID, use_csv=True)

# Load VRE profiles (if needed)
SEM_VRE_MAPPINGS = {
    "Wind NI -- All": "StochasticWindNI",
    "Wind ROI": "StochasticWindROI",
    "Wind Offshore": "StochasticWindOffshore",
    "Wind Offshore -- Arklow Phase 1": "StochasticWindROI",
    "Solar NI -- All": "StochasticSolarNI",
    "Solar ROI": "StochasticSolarROI",
}

load_data_file_profiles_csv(
    network=network,
    csv_dir="src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model",
    profiles_path="src/examples/data/sem-2024-2032/CSV Files",
    property_name="Rating",
    target_property="p_max_pu",
    target_type="generators_t",
    apply_mode="set_both_min_max",
    scenario="1",
    generator_filter=lambda gen: "Wind" in gen or "Solar" in gen,
    carrier_mapping={"Wind": "Wind", "Solar": "Solar"},
    value_scaling=0.01,
    manual_mappings=SEM_VRE_MAPPINGS,
)

# Get demand profile for intelligent maintenance scheduling
try:
    demand = network.loads_t.p_set.sum(axis=1)
    print(f"\nDemand profile: peak={demand.max():.1f} MW, min={demand.min():.1f} MW")
    has_demand = True
except Exception:
    print("\nNo demand profile available, maintenance will be scheduled uniformly")
    demand = None
    has_demand = False

# Generate outages (explicit + stochastic with proper accounting)
csv_dir = "src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model"

# Parse explicit outages from "Units Out" property
explicit_events = parse_explicit_outages_from_properties(
    csv_dir=csv_dir,
    network=network,
    property_name="Units Out",
)

# Generate stochastic outages accounting for explicit outages
stochastic_events = generate_stochastic_outages_csv(
    csv_dir=csv_dir,
    network=network,
    include_forced=True,  # Monte Carlo forced outages
    include_maintenance=True,  # PASA-like maintenance scheduling
    demand_profile=demand if has_demand else None,
    random_seed=42,  # For reproducibility
    existing_outage_events=explicit_events,  # Deducts explicit hours from FOR/MR budgets
    generator_filter=lambda gen: "Wind" not in gen
    and "Solar" not in gen,  # Exclude VRE (variability already in capacity factors)
)

# Combine all events
events = explicit_events + stochastic_events

# Build outage schedule
schedule = build_outage_schedule(events, network.snapshots)

# Apply to network using generalized function
summary2 = apply_outage_schedule(network, schedule)

# Consistency check
network.consistency_check()

# select 2023 snapshots as subset
network_subset = network.snapshots[network.snapshots.year == 2023]
network.optimize(solver_name="gurobi", snapshots=network_subset)
print(f"✓ Optimization complete: objective = {network.objective:.2f}")


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

    # Test outages module functionality
    print("\n" + "=" * 70)
    print("TESTING OUTAGES MODULE")
    print("=" * 70)

    # Part 1: Explicit Outages (from CSV properties)
    print("\n1. EXPLICIT OUTAGES (from Units Out property)")
    print("-" * 70)
    from network.outages import (
        build_outage_schedule,
        parse_explicit_outages_from_properties,
    )

    explicit_events = parse_explicit_outages_from_properties(
        csv_dir="src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model",
        network=network,
        property_name="Units Out",
    )

    if explicit_events:
        print(f"✓ Found {len(explicit_events)} explicit outage events")
        print("  Sample events:")
        for event in explicit_events[:3]:
            print(
                f"    - {event.generator}: {event.start} to {event.end} (CF={event.capacity_factor})"
            )

        # Build and apply explicit outage schedule
        explicit_schedule = build_outage_schedule(
            outage_events=explicit_events,
            snapshots=network.snapshots,
        )
        print(f"✓ Built explicit outage schedule: {explicit_schedule.shape}")

        # Apply to network using generalized function
        summary = apply_outage_schedule(network, explicit_schedule)
        print(
            f"✓ Applied explicit outages to {summary['affected_generators']} generators"
        )
        if summary["initialized_p_min_pu"] > 0:
            print(
                f"  Initialized p_min_pu for {summary['initialized_p_min_pu']} generators"
            )
    else:
        print("  No explicit outages found in SEM data")

    # Part 2: Stochastic Outages (Monte Carlo + Maintenance Scheduling)
    print("\n2. STOCHASTIC OUTAGES (Monte Carlo + PASA-like)")
    print("-" * 70)
    from network.outages import generate_stochastic_outages_csv

    # Get demand profile for maintenance scheduling
    try:
        demand = network.loads_t.p_set.sum(axis=1)
        has_demand = True
        print(
            f"  Using demand profile for maintenance scheduling (peak={demand.max():.1f} MW)"
        )
    except Exception:
        demand = None
        has_demand = False
        print("  No demand profile available, using uniform maintenance scheduling")

    # Generate stochastic outages accounting for explicit outages (exclude VRE, limit to AA* for testing)
    stochastic_events = generate_stochastic_outages_csv(
        csv_dir="src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model",
        network=network,
        include_forced=True,
        include_maintenance=True,
        demand_profile=demand if has_demand else None,
        random_seed=42,  # For reproducibility
        existing_outage_events=explicit_events,  # Deducts explicit hours from FOR/MR budgets
        generator_filter=lambda gen: gen.startswith("AA")
        and "Wind" not in gen
        and "Solar" not in gen,  # Limit to AA* thermal generators for testing
    )

    if stochastic_events:
        # Build and apply stochastic outage schedule
        stochastic_schedule = build_outage_schedule(
            outage_events=stochastic_events,
            snapshots=network.snapshots,
        )

        # Apply to network using generalized function
        summary = apply_outage_schedule(network, stochastic_schedule)
        print(
            f"✓ Applied stochastic outages to {summary['affected_generators']} generators"
        )
        if summary["initialized_p_min_pu"] > 0:
            print(
                f"  Initialized p_min_pu for {summary['initialized_p_min_pu']} generators"
            )

        # Show statistics by outage type
        forced_count = sum(1 for e in stochastic_events if e.outage_type == "forced")
        maint_count = sum(
            1 for e in stochastic_events if e.outage_type == "maintenance"
        )
        forced_hours = sum(
            (e.end - e.start).total_seconds() / 3600
            for e in stochastic_events
            if e.outage_type == "forced"
        )
        maint_hours = sum(
            (e.end - e.start).total_seconds() / 3600
            for e in stochastic_events
            if e.outage_type == "maintenance"
        )

        print("\nStochastic Outage Statistics:")
        print(
            f"  Forced outages: {forced_count} events, {forced_hours:.1f} hours total"
        )
        print(f"  Maintenance: {maint_count} events, {maint_hours:.1f} hours total")
    else:
        print("  No stochastic outages generated")

    print("\n" + "=" * 70 + "\n")

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
