import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from plexos_pypsa.network.core import setup_network

path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
file_xml = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
file_timeslice = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/timeslice/timeslice_RefYear4006.csv"


db = PlexosDB.from_xml(file_xml)


def create_aemo_model():
    """
    Examples
    --------
    >>> network = create_aemo_model()
    >>> print(f"Network has {len(network.buses)} buses and {len(network.loads)} loads")
    >>> network.optimize(solver_name="highs")
    """
    # list XML file
    path_root = "/Users/meas/Library/CloudStorage/GoogleDrive-measrainsey.meng@openenergytransition.org/Shared drives/OET Shared Drive/Projects/[008] ENTSOE - Open TYNDP I/2 - interim deliverables (working files)/Plexos Converter/Input Models"
    file_xml = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/2024 ISP Progressive Change Model.xml"
    file_timeslice = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/timeslice/timeslice_RefYear4006.csv"

    # specify renewables profiles and demand paths
    path_ren = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change"
    path_demand = f"{path_root}/AEMO/2024 ISP/2024 ISP Progressive Change/Traces/demand"

    print("Creating AEMO PyPSA Model...")
    print(f"XML file: {file_xml}")
    print(f"Demand path: {path_demand}")
    print(f"VRE profiles path: {path_ren}")

    # load PlexosDB from XML file
    print("\nLoading Plexos database...")
    plexos_db = PlexosDB.from_xml(file_xml)

    # initialize PyPSA network
    n = pypsa.Network()

    # set up complete network using unified function
    # AEMO model: Uses traditional per-node load assignment (each CSV file maps to a bus)
    print("\nSetting up complete network...")
    setup_summary = setup_network(
        n,
        plexos_db,
        snapshots_source=path_demand,
        demand_source=path_demand,
        timeslice_csv=file_timeslice,
        vre_profiles_path=path_ren,
    )

    print("\nNetwork Setup Complete:")
    print(f"  Mode: {setup_summary['mode']}")
    print(f"  Format type: {setup_summary['format_type']}")
    if setup_summary["format_type"] == "iteration":
        print(
            f"  Iterations processed: {setup_summary.get('iterations_processed', 'N/A')}"
        )
        print(f"  Loads created: {setup_summary['loads_added']}")
    else:  # zone format
        print(f"  Loads mapped to buses: {setup_summary['loads_added']}")
        if setup_summary.get("loads_skipped", 0) > 0:
            print(
                f"  Loads skipped (no matching bus): {setup_summary['loads_skipped']}"
            )

    print(f"  Total buses: {len(n.buses)}")
    print(f"  Total generators: {len(n.generators)}")
    print(f"  Total links: {len(n.links)}")
    print(f"  Total batteries: {len(n.storage_units)}")
    print(f"  Total snapshots: {len(n.snapshots)}")

    # # run consistency check on network
    # print("\nRunning network consistency check...")
    # n.consistency_check()
    # print("  Network consistency check passed!")

    return n


def parse_generator_production_rates(db, snapshots, generator_list=None):
    """
    Parse generator production rates from the PlexosDB using SQL and return a DataFrame
    with index=snapshots, columns=generator names, values=production rates.

    This function is similar to parse_generator_ratings() but simplified to only handle
    "Production Rate" properties and ignore timeslice logic, using only t_date_from and
    t_date_to for time-dependent behavior.

    Parameters
    ----------
    db : PlexosDB
        The Plexos database containing generator data.
    snapshots : pd.DatetimeIndex
        The time snapshots for the time series.
    generator_list : list, optional
        List of generator names to include. If None, includes all generators.

    Returns
    -------
    pd.DataFrame
        DataFrame with index=snapshots, columns=generator names, values=production rates.
        Missing values are filled with 0.0.

    Examples
    --------
    >>> db = PlexosDB.from_xml("model.xml")
    >>> snapshots = pd.date_range('2024-01-01', periods=24, freq='H')
    >>> production_rates = parse_generator_production_rates(db, snapshots)
    >>> print(production_rates.head())
    """
    print(f"Parsing production rates for {len(snapshots)} snapshots...")

    # 1. Get generator object_id and name
    gen_query = """
        SELECT o.object_id, o.name
        FROM t_object o
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.name = 'Generator'
    """
    gen_rows = db.query(gen_query)
    gen_df = pd.DataFrame(gen_rows, columns=["object_id", "generator"])

    # Filter to specific generators if provided
    if generator_list is not None:
        gen_df = gen_df[gen_df["generator"].isin(generator_list)]
        print(f"  Filtered to {len(gen_df)} specified generators")
    else:
        print(f"  Found {len(gen_df)} generators in database")

    if gen_df.empty:
        print("  No generators found, returning empty DataFrame")
        return pd.DataFrame(index=snapshots)

    # 2. Get data_id for each generator (child_object_id) from t_membership
    membership_query = """
        SELECT m.parent_object_id AS parent_object_id, 
               m.child_object_id AS child_object_id, 
               m.membership_id AS membership_id
        FROM t_membership m
        JOIN t_object p ON m.parent_object_id = p.object_id
        JOIN t_class pc ON p.class_id = pc.class_id
        JOIN t_object c ON m.child_object_id = c.object_id
        JOIN t_class cc ON c.class_id = cc.class_id
    """
    membership_rows = db.query(membership_query)
    membership_df = pd.DataFrame(
        membership_rows,
        columns=["parent_object_id", "child_object_id", "membership_id"],
    )

    # 3. Get Production Rate property values with t_date_from/to (ignoring timeslices)
    prop_query = """
        SELECT
            d.data_id,
            d.membership_id,
            p.name as property,
            d.value,
            df.date as t_date_from,
            dt.date as t_date_to
        FROM t_data d
        JOIN t_property p ON d.property_id = p.property_id
        LEFT JOIN t_date_from df ON d.data_id = df.data_id
        LEFT JOIN t_date_to dt ON d.data_id = dt.data_id
        WHERE p.name = 'Production Rate'
    """
    prop_rows = db.query(prop_query)

    if not prop_rows:
        print("  No Production Rate properties found, returning zeros")
        result = pd.DataFrame(0.0, index=snapshots, columns=gen_df["generator"])
        return result

    prop_df = pd.DataFrame(
        prop_rows,
        columns=[
            "data_id",
            "membership_id",
            "property",
            "value",
            "t_date_from",
            "t_date_to",
        ],
    )

    print(f"  Found {len(prop_df)} Production Rate property entries")

    # 4. Merge: generator object_id -> data_id -> property info
    merged = pd.merge(membership_df, prop_df, on="membership_id", how="inner")
    merged = pd.merge(
        merged, gen_df, left_on="child_object_id", right_on="object_id", how="inner"
    )

    print(f"  Merged data contains {len(merged)} generator-property combinations")

    def build_generator_production_timeseries(merged, gen_df, snapshots):
        """
        For each generator, create a time series for production rates using date logic only.
        Returns a DataFrame: index=snapshots, columns=generator names, values=production rates.
        """
        gen_series = {}

        for gen in gen_df["generator"]:
            props = merged[merged["generator"] == gen]

            # Start with all zeros
            ts = pd.Series(0.0, index=snapshots)

            if props.empty:
                # No production rate data for this generator
                gen_series[gen] = ts
                continue

            # Build property entries with date conversion
            property_entries = []
            for _, p in props.iterrows():
                entry = {
                    "property": p["property"],
                    "value": float(p["value"]) if p["value"] is not None else 0.0,
                    "from": pd.to_datetime(p["t_date_from"])
                    if pd.notnull(p["t_date_from"])
                    else None,
                    "to": pd.to_datetime(p["t_date_to"])
                    if pd.notnull(p["t_date_to"])
                    else None,
                    "data_id": p["data_id"],
                }
                property_entries.append(entry)

            # Sort by from date to handle overlapping periods correctly
            property_entries.sort(
                key=lambda x: x["from"] if x["from"] is not None else pd.Timestamp.min
            )

            # Track which snapshots have been set
            already_set = pd.Series(False, index=snapshots)

            for entry in property_entries:
                # Create date mask
                mask = pd.Series(True, index=snapshots)

                # Apply date from constraint
                if entry["from"] is not None:
                    mask &= snapshots >= entry["from"]

                # Apply date to constraint
                if entry["to"] is not None:
                    mask &= snapshots <= entry["to"]

                # For date-specific entries, only set where not already set
                # (later entries override earlier ones in overlapping periods)
                if entry["from"] is not None or entry["to"] is not None:
                    to_set = mask & ~already_set
                    ts.loc[to_set] = entry["value"]
                    already_set |= mask
                else:
                    # Non-date-specific entries fill remaining unset values
                    ts.loc[ts == 0.0] = entry["value"]

            gen_series[gen] = ts

        # Concatenate all generator Series into a single DataFrame
        if gen_series:
            result = pd.concat(gen_series, axis=1)
            result.index = snapshots
            return result
        else:
            return pd.DataFrame(index=snapshots)

    # Build the production rate timeseries for each generator
    production_timeseries = build_generator_production_timeseries(
        merged, gen_df, snapshots
    )

    print(
        f"  Generated time series for {len(production_timeseries.columns)} generators"
    )
    return production_timeseries


network = create_aemo_model()

snapshots = network.snapshots
generators = network.generators.index.tolist()
production_rates = parse_generator_production_rates(db, snapshots, generators)


# 1. Get generator object_id and name
gen_query = """
    SELECT o.object_id, o.name
    FROM t_object o
    JOIN t_class c ON o.class_id = c.class_id
    WHERE c.name = 'Generator'
"""
gen_rows = db.query(gen_query)
gen_df = pd.DataFrame(gen_rows, columns=["object_id", "generator"])


# 2. Get data_id for each generator (child_object_id) from t_membership
membership_query = """
    SELECT m.parent_object_id AS parent_object_id, m.child_object_id AS child_object_id, m.membership_id AS membership_id
    FROM t_membership m
    JOIN t_object p ON m.parent_object_id = p.object_id
    JOIN t_class pc ON p.class_id = pc.class_id
    JOIN t_object c ON m.child_object_id = c.object_id
    JOIN t_class cc ON c.class_id = cc.class_id
"""
membership_rows = db.query(membership_query)
membership_df = pd.DataFrame(
    membership_rows,
    columns=["parent_object_id", "child_object_id", "membership_id"],
)

# 3. Get property values for each data_id (Rating, Rating Factor, or Max Capacity), with t_date_from/to,
# including both entries with and without a tag/timeslice (class_id=76).
prop_query = """
    SELECT
        d.data_id,
        d.membership_id,
        p.name as property,
        d.value,
        df.date as t_date_from,
        dt.date as t_date_to,
        o.name as timeslice
    FROM t_data d
    JOIN t_property p ON d.property_id = p.property_id
    LEFT JOIN t_date_from df ON d.data_id = df.data_id
    LEFT JOIN t_date_to dt ON d.data_id = dt.data_id
    LEFT JOIN t_tag t ON d.data_id = t.data_id
    LEFT JOIN t_object o ON t.object_id = o.object_id AND o.class_id = 76
    WHERE (p.name = 'Production Rate')
"""
prop_rows = db.query(prop_query)

prop_df = pd.DataFrame(
    prop_rows,
    columns=[
        "data_id",
        "membership_id",
        "property",
        "value",
        "t_date_from",
        "t_date_to",
        "timeslice",
    ],
)

# 4. Merge: generator object_id -> data_id -> property info
merged = pd.merge(membership_df, prop_df, on="membership_id", how="inner")
merged = pd.merge(
    merged, gen_df, left_on="child_object_id", right_on="object_id", how="inner"
)

merged[merged.t_date_from.notnull()]


# Example usage of the new function
if __name__ == "__main__":
    # First, let's check the actual date ranges in the data using the existing merged DataFrame
    print("Checking existing production rate data date ranges...")
    date_ranges = merged[merged.t_date_from.notnull() | merged.t_date_to.notnull()][
        ["generator", "t_date_from", "t_date_to", "value"]
    ].drop_duplicates()

    if not date_ranges.empty:
        print("Sample date ranges found in production rate data:")
        print(date_ranges.head(10))

        # Use a date range that might have data based on what we found
        min_date = (
            pd.to_datetime(date_ranges["t_date_from"].dropna()).min()
            if not date_ranges["t_date_from"].dropna().empty
            else pd.Timestamp("2024-01-01")
        )
        max_date = (
            pd.to_datetime(date_ranges["t_date_to"].dropna()).max()
            if not date_ranges["t_date_to"].dropna().empty
            else pd.Timestamp("2024-12-31")
        )

        print(f"Date range in data: {min_date} to {max_date}")

        # Create snapshots that overlap with actual data
        if min_date != pd.Timestamp("2024-01-01"):
            snapshots = pd.date_range(min_date, periods=24, freq="h")
        else:
            snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
    else:
        print("No date-specific production rate data found, using default date range")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")

    # Test with a few specific generators
    test_generators = ["AGLHAL02", "BARRON-1", "BBTHREE1"]

    print(
        f"\nTesting parse_generator_production_rates function with snapshots: {snapshots[0]} to {snapshots[-1]}"
    )
    production_rates = parse_generator_production_rates(db, snapshots, test_generators)

    print(f"\nResult shape: {production_rates.shape}")
    print(f"Generators with production rate data: {list(production_rates.columns)}")
    print("Sample data (first 5 hours):")
    print(production_rates.head())

    # Show any non-zero production rates
    non_zero_data = production_rates.loc[:, (production_rates != 0).any()]
    if not non_zero_data.empty:
        print("\nGenerators with non-zero production rates:")
        print(non_zero_data.describe())
    else:
        print(
            "\nNo non-zero production rates found for test period. This may be normal if:"
        )
        print("  - Production rates are only defined for specific date ranges")
        print("  - The test generators don't have time-dependent production rates")
        print("  - The data uses different date formats than expected")
