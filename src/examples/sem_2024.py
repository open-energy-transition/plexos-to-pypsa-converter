from network.conversion import create_model
from network.generators_csv import (
    apply_generator_units_timeseries_csv,
    load_data_file_profiles_csv,
)
from network.outages import (
    apply_outage_schedule,
    build_outage_schedule,
    generate_stochastic_outages_csv,
    parse_explicit_outages_from_properties,
)
from network.storage_csv import add_storage_inflows_csv

# Constants
MODEL_ID = "sem-2024-2032"
SNAPSHOTS_PER_YEAR = 60
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
SEM_VRE_MAPPINGS = {
    "Wind NI -- All": "StochasticWindNI",
    "Wind ROI": "StochasticWindROI",
    "Wind Offshore": "StochasticWindOffshore",
    "Wind Offshore -- Arklow Phase 1": "StochasticWindROI",
    "Solar NI -- All": "StochasticSolarNI",
    "Solar ROI": "StochasticSolarROI",
}

xml_csv_dir = "src/examples/data/sem-2024-2032/csvs_from_xml/SEM Forecast model"
model_path = "src/examples/data/sem-2024-2032"

# Create the model
network, summary = create_model(MODEL_ID, use_csv=True)

# scale all p_min_pu down to avoid infeasibility due to must-run constraints
for gen in network.generators.index:
    if gen in network.generators_t.p_min_pu.columns:
        network.generators_t.p_min_pu[gen] *= 0.7

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
    generator_filter=lambda gen: "Wind" in gen or "Solar" in gen,
    carrier_mapping={"Wind": "Wind", "Solar": "Solar"},
    value_scaling=0.01,
    manual_mappings=SEM_VRE_MAPPINGS,
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

# Get explicit outages from CSV
# Parse explicit outages from "Units Out" property
explicit_events = parse_explicit_outages_from_properties(
    csv_dir=xml_csv_dir,
    network=network,
    property_name="Units Out",
)

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
outage_summary = apply_outage_schedule(network, schedule)

network.consistency_check()
network_subset = network.snapshots[network.snapshots.year == 2023]
network.optimize(**SOLVER_CONFIG)

# # Add curtailment/slack link to handle excess must-run generation
# if "curtailment" not in network.carriers.index:
#     network.add("Carrier", "curtailment")
#     print("Added 'curtailment' carrier")

# # Add dummy dump bus
# network.add("Bus", "curtailment_dump")

# # Add Link from SEM to dump (consumes excess power)
# network.add(
#     "Link",
#     "Curtailment_SEM",
#     bus0="SEM",  # From SEM
#     bus1="curtailment_dump",  # To nowhere (dump)
#     p_nom=5000,  # Max curtailment capacity (MW)
#     marginal_cost=1000,
# )


# # Helper function to report curtailment usage
# def report_curtailment(network):
#     """Report curtailment generator usage after optimization."""
#     if "Curtailment_SEM" in network.links.index:
#         if (
#             not network.links_t.p0.empty
#             and "Curtailment_SEM" in network.links_t.p0.columns
#         ):
#             curtailment = network.links_t.p0["Curtailment_SEM"]
#             total_curtailed = curtailment.sum()
#             max_curtailed = curtailment.max()
#             hours_used = (curtailment > 0.1).sum()  # Hours with >0.1 MW curtailment

#             print("\n=== Curtailment Report ===")
#             print(f"Total energy curtailed: {total_curtailed:.1f} MWh")
#             print(f"Max curtailment: {max_curtailed:.1f} MW")
#             print(f"Hours with curtailment: {hours_used} / {len(curtailment)}")

#             if total_curtailed > 0:
#                 print("✓ Curtailment generator absorbed excess must-run generation")
#             else:
#                 print("✓ No curtailment needed (demand always > must-run)")

# # Report curtailment usage
# report_curtailment(network)

# # DIAGNOSTIC: Check for infeasibility issues
# diagnostic_results = diagnose_infeasibility(network)
# suggest_fixes(diagnostic_results)

# # Calculate minimum required generation from must-run constraints
# min_required_gen = pd.Series(0.0, index=network.snapshots)
# for gen in network.generators.index:
#     if gen in network.generators_t.p_min_pu.columns:
#         p_nom = network.generators.at[gen, "p_nom"]
#         p_min_pu = network.generators_t.p_min_pu[gen]
#         min_required_gen += p_nom * p_min_pu

# # Get total demand
# if not network.loads_t.p_set.empty:
#     total_demand = network.loads_t.p_set.sum(axis=1)

#     # Check if must-run exceeds demand
#     exceeds = min_required_gen > total_demand
#     if exceeds.any():
#         violation_count = exceeds.sum()
#         print(
#             f"Must-run generation exceeds demand at {violation_count} timesteps! This makes the problem INFEASIBLE (can't reduce generation below minimum)"
#         )

#         # Show worst violations
#         excess = min_required_gen - total_demand
#         worst_times = excess.nlargest(5).index
#         print("\n    Worst violations:")
#         for t in worst_times:
#             print(
#                 f"      {t}: must-run={min_required_gen.loc[t]:.2f} MW, "
#                 f"demand={total_demand.loc[t]:.2f} MW, "
#                 f"excess={excess.loc[t]:.2f} MW"
#             )

#         print("\n💡 SOLUTION: Reduce p_min_pu constraints or add flexibility")
#         print("   Option 1: Scale down p_min_pu proportionally")
#         print("   Option 2: Allow some generators to curtail (set p_min_pu=0)")
#         print("   Option 3: Add storage or flexible demand")
#     else:
#         print("✓ Must-run constraints are feasible (never exceed demand)")
#         min_excess = (total_demand - min_required_gen).min()
#         print(f"  Minimum flexibility: {min_excess:.2f} MW")
# else:
#     print("⚠️  No demand found - cannot check must-run feasibility")
