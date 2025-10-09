import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pypsa  # type: ignore
from plexosdb import PlexosDB  # type: ignore

from src.network.core import setup_network
from src.utils.model_paths import find_model_xml, get_model_directory

# Find model data in src/examples/data/
model_id = "aemo-2024-isp-progressive"
xml_path = find_model_xml(model_id)
model_dir = get_model_directory(model_id)

if xml_path is None or model_dir is None:
    raise FileNotFoundError(
        f"Model '{model_id}' not found in src/examples/data/. "
        f"Please download and extract the AEMO 2024 ISP model data."
    )

file_xml = str(xml_path)
file_timeslice = str(model_dir / "Traces" / "timeslice" / "timeslice_RefYear4006.csv")


def create_aemo_model():
    """Examples
    --------
    >>> network = create_aemo_model()
    >>> print(f"Network has {len(network.buses)} buses and {len(network.loads)} loads")
    >>> network.optimize(solver_name="highs")
    """
    # Find model data in src/examples/data/
    model_id = "aemo-2024-isp-progressive"
    xml_path = find_model_xml(model_id)
    model_dir = get_model_directory(model_id)

    if xml_path is None or model_dir is None:
        raise FileNotFoundError(
            f"Model '{model_id}' not found in src/examples/data/. "
            f"Please download and extract the AEMO 2024 ISP model data."
        )

    file_xml = str(xml_path)
    file_timeslice = str(
        model_dir / "Traces" / "timeslice" / "timeslice_RefYear4006.csv"
    )

    # specify renewables profiles and demand paths
    path_ren = str(model_dir)
    path_demand = str(model_dir / "Traces" / "demand")

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
    """Parse generator production rates from the PlexosDB using SQL and return a DataFrame
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
        """For each generator, create a time series for production rates using simplified date logic.

        Logic:
        - Single value: use across all snapshots
        - Multiple values: sort by t_date_from, create sequential periods
          - First value: start → first t_date_from
          - Each subsequent: t_date_from → next t_date_from
          - Last value: last t_date_from → end
        - Ignore t_date_to completely

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

            # Extract unique production rate values with their t_date_from
            unique_entries = []
            for _, p in props.iterrows():
                value = float(p["value"]) if p["value"] is not None else 0.0
                date_from = (
                    pd.to_datetime(p["t_date_from"])
                    if pd.notnull(p["t_date_from"])
                    else None
                )

                # Add to unique entries if not already present
                entry = {"value": value, "date_from": date_from}
                if entry not in unique_entries:
                    unique_entries.append(entry)

            # Case 1: Single value - use across all snapshots
            if len(unique_entries) == 1:
                ts[:] = unique_entries[0]["value"]
                gen_series[gen] = ts
                continue

            # Case 2: Multiple values - create sequential periods
            # Sort by date_from (None/undated entries first, then by date)
            unique_entries.sort(
                key=lambda x: x["date_from"]
                if x["date_from"] is not None
                else pd.Timestamp.min
            )

            # Create transition points from the dated entries
            transition_dates = []
            for entry in unique_entries:
                if entry["date_from"] is not None:
                    transition_dates.append(entry["date_from"])

            transition_dates = sorted(
                set(transition_dates)
            )  # Remove duplicates and sort

            # Apply values in sequential periods
            for i, entry in enumerate(unique_entries):
                if entry["date_from"] is None:
                    # Undated entry: applies from start until first transition date (if any)
                    if transition_dates:
                        mask = snapshots < transition_dates[0]
                        ts.loc[mask] = entry["value"]
                    else:
                        # No transition dates, applies to all snapshots
                        ts[:] = entry["value"]
                else:
                    # Dated entry: find its position in transition dates
                    current_date = entry["date_from"]
                    date_idx = transition_dates.index(current_date)

                    if date_idx == len(transition_dates) - 1:
                        # Last transition date: applies from this date to end
                        mask = snapshots >= current_date
                    else:
                        # Not the last: applies from this date until next transition
                        next_date = transition_dates[date_idx + 1]
                        mask = (snapshots >= current_date) & (snapshots < next_date)

                    ts.loc[mask] = entry["value"]

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


def plot_production_rates(production_rates, output_dir="src/figures/production-rates"):
    """Create individual plots for each generator showing production rate vs time.

    Parameters
    ----------
    production_rates : pd.DataFrame
        DataFrame with index=snapshots, columns=generator names, values=production rates.
    output_dir : str
        Directory to save the plot files (relative to current working directory).

    Returns
    -------
    dict
        Summary of plotting results including counts of plots created and skipped.
    """
    print("Creating production rate plots...")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # Track plotting results
    plots_created = 0
    plots_skipped = 0
    skipped_generators = []

    def clean_filename(name):
        """Convert generator name to valid filename."""
        # Replace special characters with underscores
        cleaned = re.sub(r'[<>:"/\\|?*\s]', "_", name)
        # Remove multiple consecutive underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip("_")
        return cleaned

    # Plot each generator
    for generator in production_rates.columns:
        try:
            # Get the time series for this generator
            ts = production_rates[generator]

            # Skip generators with all-zero production rates
            if (ts == 0.0).all():
                plots_skipped += 1
                skipped_generators.append(generator)
                continue

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot the time series
            ax.plot(ts.index, ts.values, linewidth=1.5, color="blue")

            # Formatting
            ax.set_title(
                f"Production Rate - {generator}", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Production Rate (kg/MWh)", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            if len(ts.index) > 1:
                # Determine appropriate date formatting based on time range
                time_range = ts.index[-1] - ts.index[0]
                if time_range.days <= 1:
                    # For daily or shorter ranges, show hours
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.xaxis.set_major_locator(
                        mdates.HourLocator(interval=max(1, len(ts) // 10))
                    )
                elif time_range.days <= 7:
                    # For weekly ranges, show days
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                else:
                    # For longer ranges, show months/days
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                    ax.xaxis.set_major_locator(mdates.MonthLocator())

                # Rotate x-axis labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

            # Tight layout to prevent label cutoff
            plt.tight_layout()

            # Save the plot
            filename = clean_filename(generator) + ".png"
            filepath = str(Path(output_dir) / filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()  # Close the figure to free memory

            plots_created += 1

            # Print progress every 50 plots
            if plots_created % 50 == 0:
                print(f"  Created {plots_created} plots so far...")

        except Exception as e:
            print(f"  Error plotting {generator}: {e}")
            plots_skipped += 1
            skipped_generators.append(generator)

    # Summary
    total_generators = len(production_rates.columns)
    print("\nPlotting complete:")
    print(f"  Total generators: {total_generators}")
    print(f"  Plots created: {plots_created}")
    print(f"  Plots skipped: {plots_skipped}")

    if skipped_generators:
        print(f"  Skipped generators (first 10): {skipped_generators[:10]}")
        if len(skipped_generators) > 10:
            print(f"  ... and {len(skipped_generators) - 10} more")

    return {
        "total_generators": total_generators,
        "plots_created": plots_created,
        "plots_skipped": plots_skipped,
        "skipped_generators": skipped_generators,
    }


# Load the PlexosDB from XML file
db = PlexosDB.from_xml(file_xml)

# Run full AEMO model
n = create_aemo_model()
snapshots = n.snapshots
generators = n.generators.index.tolist()
production_rates = parse_generator_production_rates(db, snapshots, generators)
plot_summary = plot_production_rates(production_rates, "src/figures/production-rates")

print("\nPlot summary:")
print(f"  Total generators processed: {plot_summary['total_generators']}")
print(f"  Plots successfully created: {plot_summary['plots_created']}")
print(f"  Plots skipped (no data): {plot_summary['plots_skipped']}")
