from network.conversion import create_model
from network.generators_csv import (
    apply_generator_units_timeseries_csv,
    load_data_file_profiles_csv,
)
from network.outages import (
    apply_outage_schedule,
    build_outage_schedule,
    generate_stochastic_outages_csv,
)
from network.storage_csv import add_storage_inflows_csv

MODEL_ID = "aemo-2024-isp-progressive-change"
SOLVER_CONFIG = {
    "solver_name": "gurobi",
    "solver_options": {
        "Threads": 6,
        "Method": 2,  # barrier
        "Crossover": 0,
        "BarConvTol": 1.0e-5,
        "Seed": 123,
        "AggFill": 0,
        "PreDual": 0,
        "GURO_PAR_BARDENSETHRESH": 200,
    },
}
xml_csv_dir = "src/examples/data/aemo-2024-isp-progressive-change/csvs_from_xml/NEM"
model_path = "src/examples/data/aemo-2024-isp-progressive-change"

network, summary = create_model(MODEL_ID, use_csv=True)

# Load VRE profiles from CSVs, allowing curtailment (p_min_pu = 0)
vre_summary = load_data_file_profiles_csv(
    network=network,
    csv_dir=xml_csv_dir,
    profiles_path=model_path,
    property_name="Rating",
    target_property="p_max_pu",
    target_type="generators_t",
    apply_mode="replace",  # Allow curtailment: p_min_pu = 0, p_max_pu = profile
    scenario=1,  # No scenario filtering needed
    generator_filter=None,  # Load all generators that have Rating.Data File references
    carrier_mapping={"Wind": "wind", "Solar": "solar"},  # Map categories to carriers
    value_scaling=1.0,  # Trace files already have 0-1 capacity factors
)

# Add hydrological inflows to storage units
inflows_summary = add_storage_inflows_csv(
    network=network,
    csv_dir=xml_csv_dir,
    inflow_path=model_path,
)

# Apply generator Units time series (retirements, new builds, capacity scaling)
# IMPORTANT: Must be called AFTER VRE profiles are loaded
units_summary = apply_generator_units_timeseries_csv(network, xml_csv_dir)

# Get demand profile for intelligent maintenance scheduling
try:
    demand = network.loads_t.p_set.sum(axis=1)
    print(f"\nDemand profile: peak={demand.max():.1f} MW, min={demand.min():.1f} MW")
    has_demand = True
except Exception:
    print("\nNo demand profile available, maintenance will be scheduled uniformly")
    demand = None
    has_demand = False

# AEMO model doesn't have explicit "Units Out" property - skip explicit outages parsing
# Generate stochastic outages (uses Forced Outage Rate from Time varying properties)
# For AEMO: Filter out VRE generators by carrier (empty carrier = VRE)
stochastic_events = generate_stochastic_outages_csv(
    csv_dir=xml_csv_dir,
    network=network,
    include_forced=True,  # Monte Carlo forced outages
    include_maintenance=True,  # PASA-like maintenance scheduling
    demand_profile=demand if has_demand else None,
    random_seed=42,  # For reproducibility
    existing_outage_events=None,  # AEMO has no explicit outages
    generator_filter=lambda gen: network.generators.at[gen, "carrier"]
    != "",  # Exclude VRE (empty carrier)
)

# Build outage schedule (only stochastic events for AEMO)
schedule = build_outage_schedule(stochastic_events, network.snapshots)
summary2 = apply_outage_schedule(network, schedule)

# Consistency check
network.consistency_check()

# Optimize
network_subset = network.snapshots[network.snapshots.year == 2025]
network.optimize(snapshots=network_subset, **SOLVER_CONFIG)
