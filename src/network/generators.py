import logging

import numpy as np
import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from db.parse import (
    find_bus_for_object,
    find_fuel_for_generator,
    get_dataid_timeslice_map,
    get_property_active_mask,
    read_timeslice_activity,
)
from network.carriers import parse_fuel_prices
from network.costs import set_capital_costs_generic
from utils.paths import contains_path_pattern, extract_filename, safe_join

logger = logging.getLogger(__name__)


def add_generators(
    network: Network, db: PlexosDB, generators_as_links=False, fuel_bus_prefix="fuel_"
):
    """Add generators from a Plexos database to a PyPSA network.

    This function retrieves generator objects from the provided Plexos database,
    extracts their properties, and adds them to the PyPSA network. If a generator
    lacks properties or an associated bus, it is skipped and reported at the end.

    Parameters
    ----------
    network : Network
        The PyPSA network to which the generators will be added.
    db : PlexosDB
        The Plexos database containing generator data.
    generators_as_links : bool, optional
        If True, represent conventional generators (coal, gas, nuclear, etc.) as Links
        connecting fuel buses to electric buses. Default False.
    fuel_bus_prefix : str, optional
        Prefix for fuel bus names when generators_as_links=True. Default "fuel_".

    Raises
    ------
    Exception
        If an error occurs while retrieving generator properties.

    Notes
    -----
    - Generators without a "Max Capacity" property will not have `p_nom` specified.
    - Generators without an associated bus will be skipped and reported.
    - A summary of skipped generators is printed at the end.
    - When generators_as_links=True, conventional generators become fuel→electric links.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_generators(network, db)

    # With generators-as-links
    >>> add_generators(network, db, generators_as_links=True, fuel_bus_prefix="fuel_")
    """
    empty_generators = []
    skipped_generators = []
    generators = db.list_objects_by_class(ClassEnum.Generator)

    for gen in generators:
        try:
            props = db.get_object_properties(ClassEnum.Generator, gen)
        except Exception:
            empty_generators.append(gen)
            continue

        # Extract Max Capacity (p_nom)
        p_max = next(
            (float(p["value"]) for p in props if p["property"] == "Max Capacity"), None
        )
        if p_max is None:
            print(f"Warning: 'Max Capacity' not found for {gen}. No p_nom set.")

        # Extract generator properties
        prop_map = {
            "Min Capacity": "p_nom_min",
            "Start Cost": "start_up_cost",
            "Shutdown Cost": "shut_down_cost",
            "Min Up Time": "min_up_time",
            "Min Down Time": "min_down_time",
            "Max Ramp Up": "ramp_limit_up",
            "Max Ramp Down": "ramp_limit_down",
            "Ramp Up Rate": "ramp_limit_start_up",
            "Ramp Down Rate": "ramp_limit_start_down",
            "Technical Life": "lifetime",
        }
        gen_attrs = {}
        for prop, attr in prop_map.items():
            val = next(
                (float(p["value"]) for p in props if p["property"] == prop), None
            )
            if val is not None:
                gen_attrs[attr] = val

        # Find associated bus/node
        bus = find_bus_for_object(db, gen, ClassEnum.Generator)
        if bus is None:
            print(f"Warning: No associated bus found for generator {gen}")
            skipped_generators.append(gen)
            continue

        # Find associated fuel/carrier
        carrier = find_fuel_for_generator(db, gen)

        # Determine if this is a conventional (fuel-based) generator for generators-as-links
        conventional_fuels = [
            "Natural Gas",
            "Natural Gas CCGT",
            "Natural Gas OCGT",
            "Coal",
            "Hard Coal",
            "Lignite",
            "Nuclear",
            "Oil",
            "Biomass",
            "Biomass Waste",
            "Solids Fired",
            "Gas",
        ]
        is_conventional = carrier in conventional_fuels if carrier else False

        # Add generator to the network
        if generators_as_links and is_conventional and carrier:
            # Create fuel bus if it doesn't exist
            fuel_carrier = (
                carrier.replace(" CCGT", "").replace(" OCGT", "").replace(" Waste", "")
            )
            fuel_bus_name = (
                f"{fuel_bus_prefix}{fuel_carrier.replace(' ', '_')}"
                if fuel_bus_prefix
                else f"{fuel_carrier.replace(' ', '_')}"
            )

            if fuel_bus_name not in network.buses.index:
                # Add fuel carrier if needed
                if fuel_carrier not in network.carriers.index:
                    network.add("Carrier", fuel_carrier)
                network.add("Bus", fuel_bus_name, carrier=fuel_carrier)

            # Create generator-link (fuel bus → electric bus)
            link_name = f"gen_link_{gen}"
            if link_name not in network.links.index:
                # Set efficiency based on technology
                if "CCGT" in carrier:
                    efficiency = 0.55  # Combined cycle
                elif "OCGT" in carrier:
                    efficiency = 0.35  # Open cycle
                elif "Nuclear" in carrier:
                    efficiency = 0.33  # Nuclear thermal
                elif "Coal" in carrier or "Lignite" in carrier:
                    efficiency = 0.40  # Coal thermal
                elif "Biomass" in carrier:
                    efficiency = 0.30  # Biomass thermal
                else:
                    efficiency = 0.40  # Default thermal

                # Create link with p_nom
                if p_max is not None:
                    network.add(
                        "Link",
                        link_name,
                        bus0=fuel_bus_name,
                        bus1=bus,
                        p_nom=p_max,
                        efficiency=efficiency,
                        carrier="conversion",
                    )
                else:
                    network.add(
                        "Link",
                        link_name,
                        bus0=fuel_bus_name,
                        bus1=bus,
                        efficiency=efficiency,
                        carrier="conversion",
                    )

                # Set ramp limits and other properties on the link
                for attr, val in gen_attrs.items():
                    if attr in ["ramp_limit_up", "ramp_limit_down"]:
                        network.links.loc[link_name, attr] = val

                # Add infinite fuel supply generator (simplified)
                fuel_supply_name = f"fuel_supply_{fuel_carrier.replace(' ', '_')}"
                if fuel_supply_name not in network.generators.index:
                    network.add(
                        "Generator",
                        fuel_supply_name,
                        bus=fuel_bus_name,
                        p_nom=99999,
                        carrier=fuel_carrier,
                        marginal_cost=0.1,
                    )
        # Add as standard generator (renewables or when generators_as_links=False)
        elif p_max is not None:
            if carrier is not None:
                network.add("Generator", gen, bus=bus, p_nom=p_max, carrier=carrier)
            else:
                network.add("Generator", gen, bus=bus, p_nom=p_max)
            for attr, val in gen_attrs.items():
                network.generators.loc[gen, attr] = val
        elif carrier is not None:
            network.add("Generator", gen, bus=bus, carrier=carrier)
        else:
            network.add("Generator", gen, bus=bus)
    # Report skipped generators
    if empty_generators:
        print(f"\nSkipped {len(empty_generators)} generators with no properties:")
        for g in empty_generators:
            print(f"  - {g}")

    if skipped_generators:
        print(f"\nSkipped {len(skipped_generators)} generators with no associated bus:")
        for g in skipped_generators:
            print(f"  - {g}")


def parse_generator_ratings(db: PlexosDB, network, timeslice_csv=None):
    """Parse generator ratings from the PlexosDB using SQL and return a DataFrame with index=network.snapshots, columns=generator names.
    Applies rating values according to these rules:
    - If a property is linked to a timeslice, use the timeslice activity to set the property for the relevant snapshots (takes precedence over t_date_from/t_date_to).
    - If "Rating" is present (possibly time-dependent), use it as p_max_pu (normalized by Max Capacity).
    - Otherwise, if "Rating Factor" is present (possibly time-dependent), use it as p_max_pu (Rating Factor is a percentage of Max Capacity).
    - If neither, fallback to "Max Capacity".

    This is how the time-dependent ratings are handled:
    - "t_date_from" Only: Value is effective from this date onward, until superseded.
    - "t_date_to" Only: Value applies up to and including this date, starting from simulation start or last defined "Date From".
    - Both: Value applies within the defined date range.
    All snapshots with no rating are filled with Max Capacity.

    for each generator, create a time series for Rating, Rating Factor, or Max Capacity using merged
    use the following logic:
    - If Rating Factor is available, use it (as p_max_pu = Rating Factor / 100)
    - If Rating is available, use it (as p_max_pu = Rating / p_nom)
    - If neither is available, use Max Capacity (p_max_pu = Max Capacity / p_nom)
    - If no Max Capacity is available, fallback to p_max_pu = 1.0
    - If no timeslice is defined, the property is always active, UNLESS it is overridden by a date-specific entry:
    - If timeslice exists, use the timeslice timeseries file to determine and set when the property is active (if the timeslice doesn't exist in the timeslice file but has names like M1, M2, M3, .. M12, assume it is active for the full month).
    - "t_date_from" Only: Value is effective from this date onward, until superseded.
    - "t_date_to" Only: Value applies up to and including this date, starting from simulation start or last defined "Date From" or timeslice.
    - "t_date_from" and "t_date_to": Value applies within the defined date range.

    """
    snapshots = network.snapshots
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)
    dataid_to_timeslice = (
        get_dataid_timeslice_map(db) if timeslice_csv is not None else {}
    )

    # 1. Get generator object_id and name
    gen_query = """
        SELECT o.object_id, o.name
        FROM t_object o
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.name = 'Generator'
    """
    gen_rows = db.query(gen_query)
    gen_df = pd.DataFrame(gen_rows, columns=["object_id", "generator"])

    # only keep generators that are in the network
    gen_df = gen_df[gen_df["generator"].isin(network.generators.index)]

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
        WHERE (p.name = 'Rating' OR p.name = 'Max Capacity' OR p.name = 'Rating Factor')
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

    def build_generator_p_max_pu_timeseries(
        merged,
        gen_df,
        network,
        snapshots,
        timeslice_activity=None,
        dataid_to_timeslice=None,
    ):
        """For each generator, create a time series for p_max_pu using Rating, Rating Factor, or Max Capacity, with timeslice and date logic.
        Returns a DataFrame: index=snapshots, columns=generator names, values=p_max_pu.
        """
        gen_series = {}
        for gen in gen_df["generator"]:
            props = merged[merged["generator"] == gen]
            # Build property entries
            property_entries = []
            for _, p in props.iterrows():
                entry = {
                    "property": p["property"],
                    "value": float(p["value"]),
                    "from": pd.to_datetime(p["t_date_from"])
                    if pd.notnull(p["t_date_from"])
                    else None,
                    "to": pd.to_datetime(p["t_date_to"])
                    if pd.notnull(p["t_date_to"])
                    else None,
                    "data_id": p["data_id"],
                }
                if dataid_to_timeslice and p["data_id"] in dataid_to_timeslice:
                    entry["timeslices"] = dataid_to_timeslice[p["data_id"]]
                property_entries.append(entry)
            if property_entries:
                prop_df_entries = pd.DataFrame(property_entries)
            else:
                # Create empty DataFrame with expected columns
                prop_df_entries = pd.DataFrame(
                    columns=["property", "value", "from", "to", "data_id", "timeslices"]
                )

            # Helper to build a time series for a property
            def build_ts(prop_name, entries, fallback=None):
                ts = pd.Series(index=snapshots, dtype=float)
                prop_rows = entries[entries["property"] == prop_name].copy()
                prop_rows["from_sort"] = pd.to_datetime(prop_rows["from"])
                prop_rows = prop_rows.sort_values("from_sort", na_position="first")
                # Track which snapshots have been set, to handle overrides
                already_set = pd.Series(False, index=snapshots)
                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if is_time_specific:
                        mask = get_property_active_mask(
                            row, snapshots, timeslice_activity, dataid_to_timeslice
                        )
                        # Only override where not already set, or where overlapping (newer overrides older)
                        to_set = mask & (~already_set | mask)
                        ts.loc[to_set] = row["value"]
                        already_set |= mask
                # Non-time-specific entries fill remaining unset values
                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if not is_time_specific:
                        ts.loc[ts.isnull()] = row["value"]
                if fallback is not None:
                    ts = ts.fillna(fallback)
                return ts

            # Get Max Capacity (for fallback and for scaling)
            maxcap = None
            maxcap_entries = prop_df_entries[
                prop_df_entries["property"] == "Max Capacity"
            ]
            if not maxcap_entries.empty:
                maxcap = maxcap_entries.iloc[0]["value"]

            # Build time series for Rating and Rating Factor
            rating_ts = build_ts("Rating", prop_df_entries)
            rating_factor_ts = build_ts("Rating Factor", prop_df_entries)
            # Get p_nom for scaling Rating
            p_nom = (
                network.generators.loc[gen, "p_nom"]
                if gen in network.generators.index
                and "p_nom" in network.generators.columns
                else maxcap
            )
            # Start with all NaN
            ts = pd.Series(index=snapshots, dtype=float)
            # At each snapshot: use Rating Factor if present, else Rating if present, else 1.0
            if p_nom:
                # Use Rating Factor if present
                mask_rf = rating_factor_ts.notnull()
                ts[mask_rf] = rating_factor_ts[mask_rf] / 100.0
                # Where Rating Factor is not present, use Rating if present
                mask_rating = ts.isnull() & rating_ts.notnull()
                ts[mask_rating] = rating_ts[mask_rating] / p_nom
            # Where neither is present, fallback to 1.0
            ts = ts.fillna(1.0)
            gen_series[gen] = ts
        # Concatenate all generator Series into a single DataFrame
        result = pd.concat(gen_series, axis=1)
        result.index = snapshots
        return result

    # Build the p_max_pu timeseries for each generator
    p_max_pu_timeseries = build_generator_p_max_pu_timeseries(
        merged, gen_df, network, snapshots, timeslice_activity, dataid_to_timeslice
    )

    return p_max_pu_timeseries


def parse_generator_min_stable_levels(db: PlexosDB, network, timeslice_csv=None):
    """Parse generator minimum stable levels from the PlexosDB and return DataFrame with index=network.snapshots, columns=generator names.

    Applies minimum generation constraints according to these rules:
    - If "Min Stable Factor" is present (percentage), use it as p_min_pu (Min Stable Factor / 100)
    - If "Min Stable Level" is present (MW), use it as p_min_pu (Min Stable Level / p_nom)
    - If "Min Pump Load" is present (for storage), use it as p_min_pu (Min Pump Load / p_nom)
    - If none present, fallback to p_min_pu = 0.0 (no minimum constraint)

    Time-dependent handling (same logic as parse_generator_ratings):
    - If timeslice is defined, use the timeslice timeseries to determine when property is active
    - "t_date_from" Only: Value effective from this date onward
    - "t_date_to" Only: Value applies up to and including this date
    - Both: Value applies within the defined date range
    - No date/timeslice: Property is always active unless overridden

    Parameters
    ----------
    db : PlexosDB
        The Plexos database containing generator data
    network : Network
        PyPSA network with generators and snapshots
    timeslice_csv : str, optional
        Path to timeslice CSV file for time-dependent properties

    Returns
    -------
    pd.DataFrame
        Time series with index=snapshots, columns=generator names, values=p_min_pu (0-1)

    Examples
    --------
    >>> p_min_pu_timeseries = parse_generator_min_stable_levels(db, network, timeslice_csv)
    >>> network.generators_t.p_min_pu = p_min_pu_timeseries
    """
    snapshots = network.snapshots
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)
    dataid_to_timeslice = (
        get_dataid_timeslice_map(db) if timeslice_csv is not None else {}
    )

    # 1. Get generator object_id and name
    gen_query = """
        SELECT o.object_id, o.name
        FROM t_object o
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.name = 'Generator'
    """
    gen_rows = db.query(gen_query)
    gen_df = pd.DataFrame(gen_rows, columns=["object_id", "generator"])

    # Only keep generators that are in the network
    gen_df = gen_df[gen_df["generator"].isin(network.generators.index)]

    # 2. Get data_id for each generator from t_membership
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

    # 3. Get property values for Min Stable Level, Min Stable Factor, Min Pump Load
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
        WHERE (p.name = 'Min Stable Level' OR p.name = 'Min Stable Factor' OR p.name = 'Min Pump Load')
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

    def build_generator_p_min_pu_timeseries(
        merged,
        gen_df,
        network,
        snapshots,
        timeslice_activity=None,
        dataid_to_timeslice=None,
    ):
        """For each generator, create a time series for p_min_pu using Min Stable Factor, Min Stable Level, or Min Pump Load.
        Returns a DataFrame: index=snapshots, columns=generator names, values=p_min_pu.
        """
        gen_series = {}
        for gen in gen_df["generator"]:
            props = merged[merged["generator"] == gen]

            # Build property entries
            property_entries = []
            for _, p in props.iterrows():
                entry = {
                    "property": p["property"],
                    "value": float(p["value"]),
                    "from": pd.to_datetime(p["t_date_from"])
                    if pd.notnull(p["t_date_from"])
                    else None,
                    "to": pd.to_datetime(p["t_date_to"])
                    if pd.notnull(p["t_date_to"])
                    else None,
                    "data_id": p["data_id"],
                }
                if dataid_to_timeslice and p["data_id"] in dataid_to_timeslice:
                    entry["timeslices"] = dataid_to_timeslice[p["data_id"]]
                property_entries.append(entry)

            if property_entries:
                prop_df_entries = pd.DataFrame(property_entries)
            else:
                # Create empty DataFrame with expected columns
                prop_df_entries = pd.DataFrame(
                    columns=["property", "value", "from", "to", "data_id", "timeslices"]
                )

            # Helper to build a time series for a property
            def build_ts(prop_name, entries, fallback=None):
                ts = pd.Series(index=snapshots, dtype=float)
                prop_rows = entries[entries["property"] == prop_name].copy()
                prop_rows["from_sort"] = pd.to_datetime(prop_rows["from"])
                prop_rows = prop_rows.sort_values("from_sort", na_position="first")

                # Track which snapshots have been set
                already_set = pd.Series(False, index=snapshots)

                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if is_time_specific:
                        mask = get_property_active_mask(
                            row, snapshots, timeslice_activity, dataid_to_timeslice
                        )
                        to_set = mask & (~already_set | mask)
                        ts.loc[to_set] = row["value"]
                        already_set |= mask

                # Non-time-specific entries fill remaining unset values
                for _, row in prop_rows.iterrows():
                    is_time_specific = (
                        pd.notnull(row.get("from"))
                        or pd.notnull(row.get("to"))
                        or (
                            dataid_to_timeslice
                            and row["data_id"] in dataid_to_timeslice
                        )
                    )
                    if not is_time_specific:
                        ts.loc[ts.isnull()] = row["value"]

                if fallback is not None:
                    ts = ts.fillna(fallback)
                return ts

            # Build time series for each property
            min_stable_level_ts = build_ts("Min Stable Level", prop_df_entries)
            min_stable_factor_ts = build_ts("Min Stable Factor", prop_df_entries)
            min_pump_load_ts = build_ts("Min Pump Load", prop_df_entries)

            # Get p_nom for scaling
            p_nom = (
                network.generators.loc[gen, "p_nom"]
                if gen in network.generators.index
                and "p_nom" in network.generators.columns
                else None
            )

            # Start with all NaN
            ts = pd.Series(index=snapshots, dtype=float)

            # Priority: Min Stable Factor > Min Stable Level > Min Pump Load > 0.0
            if p_nom:
                # Use Min Stable Factor if present (percentage)
                mask_factor = min_stable_factor_ts.notnull()
                ts[mask_factor] = min_stable_factor_ts[mask_factor] / 100.0

                # Where Factor not present, use Min Stable Level if present (MW)
                mask_level = ts.isnull() & min_stable_level_ts.notnull()
                ts[mask_level] = min_stable_level_ts[mask_level] / p_nom

                # Where neither present, use Min Pump Load if present (MW)
                mask_pump = ts.isnull() & min_pump_load_ts.notnull()
                ts[mask_pump] = min_pump_load_ts[mask_pump] / p_nom

            # Fallback to 0.0 (no minimum constraint)
            ts = ts.fillna(0.0)
            gen_series[gen] = ts

        # Concatenate all generator Series into a single DataFrame
        result = pd.concat(gen_series, axis=1)
        result.index = snapshots
        return result

    # Build the p_min_pu timeseries for each generator
    p_min_pu_timeseries = build_generator_p_min_pu_timeseries(
        merged, gen_df, network, snapshots, timeslice_activity, dataid_to_timeslice
    )

    return p_min_pu_timeseries


def set_capacity_ratings(network: Network, db: PlexosDB, timeslice_csv=None):
    """Set the capacity ratings for generators in the PyPSA network based on the Plexos database.

    This function retrieves generator ratings from the Plexos database and sets the
    `p_max_pu` attribute for relevant generators in the PyPSA network.

    Parameters
    ----------
    network : Network
        The PyPSA network to which the capacity ratings will be applied.
    db : PlexosDB
        The Plexos database containing generator data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent ratings.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_capacity_ratings(network, db, "path/to/timeslice.csv")
    """
    # Get the generator ratings from the database (already p_max_pu)
    generator_ratings = parse_generator_ratings(db, network, timeslice_csv)

    # Only keep generators present in both network and generator_ratings
    valid_gens = [
        gen for gen in network.generators.index if gen in generator_ratings.columns
    ]
    missing_gens = [
        gen for gen in network.generators.index if gen not in generator_ratings.columns
    ]

    # Assign all columns at once to avoid fragmentation
    network.generators_t.p_max_pu.loc[:, valid_gens] = generator_ratings[
        valid_gens
    ].copy()

    # Warn about missing generators
    for gen in missing_gens:
        print(f"Warning: Generator {gen} not found in ratings DataFrame.")


def set_min_stable_levels(
    network: Network, db: PlexosDB, timeslice_csv: str | None = None
) -> None:
    """Set the minimum stable levels (p_min_pu) for generators in the PyPSA network.

    This function retrieves minimum generation constraints from the Plexos database
    (Min Stable Level, Min Stable Factor, Min Pump Load) and sets the `p_min_pu`
    time series for generators.

    This ensures thermal generators with minimum load constraints (e.g., 60% min load)
    have those constraints properly represented in PyPSA, which is critical for:
    - Realistic unit commitment modeling
    - Proper handling of generator outages (outages scale both p_max_pu and p_min_pu)
    - Avoiding infeasibility when generators cannot be fully turned off

    Parameters
    ----------
    network : Network
        The PyPSA network to which the minimum stable levels will be applied.
    db : PlexosDB
        The Plexos database containing generator data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent minimum levels.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_min_stable_levels(network, db, "path/to/timeslice.csv")

    Notes
    -----
    Property precedence:
    1. Min Stable Factor (percentage) → p_min_pu = value / 100
    2. Min Stable Level (MW) → p_min_pu = value / p_nom
    3. Min Pump Load (MW, for storage) → p_min_pu = value / p_nom
    4. Fallback: p_min_pu = 0.0 (no minimum constraint)
    """
    # Get the generator minimum stable levels from the database
    generator_min_levels = parse_generator_min_stable_levels(db, network, timeslice_csv)

    if generator_min_levels.empty:
        print(
            "No minimum stable levels found. All generators can operate down to 0 MW."
        )
        return

    # Only keep generators present in both network and generator_min_levels
    valid_gens = [
        gen for gen in network.generators.index if gen in generator_min_levels.columns
    ]
    missing_gens = [
        gen
        for gen in network.generators.index
        if gen not in generator_min_levels.columns
    ]

    # Assign all columns at once to avoid fragmentation
    network.generators_t.p_min_pu.loc[:, valid_gens] = generator_min_levels[
        valid_gens
    ].copy()

    # Report statistics
    nonzero_gens = [gen for gen in valid_gens if (generator_min_levels[gen] > 0).any()]

    print(
        f"Set p_min_pu for {len(valid_gens)} generators ({len(nonzero_gens)} with nonzero minimum)"
    )

    # Debug: Show some examples
    if nonzero_gens:
        sample_gens = nonzero_gens[:3]
        for gen in sample_gens:
            min_val = generator_min_levels[gen].min()
            max_val = generator_min_levels[gen].max()
            avg_val = generator_min_levels[gen].mean()
            print(
                f"  {gen}: p_min_pu range=[{min_val:.3f}, {max_val:.3f}], avg={avg_val:.3f}"
            )

    # Warn about missing generators (expected - not all generators have minimum constraints)
    if missing_gens:
        print(
            f"Note: {len(missing_gens)} generators have no minimum stable level (can operate down to 0 MW)"
        )


def set_generator_efficiencies(network: Network, db: PlexosDB, use_incr: bool = True):
    """Set the efficiency for each generator in the PyPSA network based on
    'Heat Rate Base' and 'Heat Rate Incr*' properties from the PlexosDB.

    The efficiency is calculated as:
        efficiency = (p_nom / fuel) * 3.6
    where:
        fuel = hr_base + (hr_inc * p_nom)
    If multiple hr_inc are present, p_nom is divided into equal segments.

    If use_incr is False, only Heat Rate Base is used. If Heat Rate Incr is found,
    a warning is printed.

    The result is stored in network.generators['efficiency'].
    """
    efficiencies = []
    for gen in network.generators.index:
        props = db.get_object_properties(ClassEnum.Generator, gen)
        p_nom = (
            network.generators.at[gen, "p_nom"]
            if "p_nom" in network.generators.columns
            else None
        )
        if p_nom is None or np.isnan(p_nom):
            efficiencies.append(np.nan)
            print(f"Warning: 'p_nom' not found for {gen}. No efficiency set.")
            continue
        hr_base = next(
            (float(p["value"]) for p in props if p["property"] == "Heat Rate Base"),
            None,
        )
        hr_incs = [
            float(p["value"])
            for p in props
            if p["property"].lower().startswith("heat rate incr")
        ]
        if hr_base is None:
            efficiencies.append(1)
            continue
        if not use_incr or not hr_incs:
            if not use_incr and hr_incs:
                print(
                    f"Heat Rate Incr found for {gen}. Only Heat Rate Base will be used."
                )
            fuel = hr_base
        elif len(hr_incs) == 1:
            fuel = hr_base + hr_incs[0] * p_nom
        else:
            n = len(hr_incs)
            p_seg = p_nom / n
            fuel = hr_base + sum(hr * p_seg for hr in hr_incs)
        efficiency = (p_nom / fuel) * 3.6 if fuel else 1
        if efficiency > 1:
            print(
                f"   - Warning: Calculated efficiency > 1 for {gen} (efficiency={efficiency:.3f})"
            )
        efficiencies.append(efficiency)
    network.generators["efficiency"] = efficiencies


def set_vre_profiles(network: Network, db: PlexosDB, path: str):
    """Add time series profiles for solar and wind generators to the PyPSA network.
    Also set the generator's carrier to "Solar" or "Wind" if a profile is found.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        The Plexos database containing generator data.
    path : str
        Path to the folder containing generation profile files.
    """

    def _raise_unsupported_resolution(filename: str):
        """Raise unsupported resolution error."""
        msg = f"Unsupported resolution in file: {filename}"
        raise ValueError(msg)

    dispatch_dict = {}
    for gen in network.generators.index:
        # Skip Adelaide_Desal_FFP
        if gen == "Adelaide_Desal_FFP":
            print(f"Skipping generator {gen}")
            continue

        # Retrieve generator properties from the database
        props = db.get_object_properties(ClassEnum.Generator, gen)

        filename = next(
            (
                p["texts"]
                for p in props
                if contains_path_pattern(p["texts"], "Traces/solar/")
                or contains_path_pattern(p["texts"], "Traces/wind/")
            ),
            None,
        )

        if filename:
            profile_type = (
                "solar" if contains_path_pattern(filename, "Traces/solar/") else "wind"
            )
            # Extract just the filename to avoid path concatenation issues
            clean_filename = extract_filename(filename.strip())
            file_path = safe_join(path, "Traces", profile_type, clean_filename)

            # Set carrier to "Solar" or "Wind"
            carrier_value = "Solar" if profile_type == "solar" else "Wind"
            network.generators.at[gen, "carrier"] = carrier_value

            try:
                df = pd.read_csv(file_path)
                df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])
                df.columns = pd.Index(
                    [
                        str(int(col))
                        if col.strip().isdigit()
                        and col not in {"Year", "Month", "Day", "datetime"}
                        else col
                        for col in df.columns
                    ]
                )
                non_date_columns = [
                    col
                    for col in df.columns
                    if col not in {"Year", "Month", "Day", "datetime"}
                ]
                if len(non_date_columns) == 24:
                    resolution = 60
                elif len(non_date_columns) == 48:
                    resolution = 30
                else:
                    _raise_unsupported_resolution(filename)

                df_long = df.melt(
                    id_vars=["datetime"],
                    value_vars=non_date_columns,
                    var_name="time",
                    value_name="cf",
                )
                if resolution == 60:
                    df_long["time"] = pd.to_timedelta(
                        (df_long["time"].astype(int) - 1) * 60, unit="m"
                    )
                elif resolution == 30:
                    df_long["time"] = pd.to_timedelta(
                        (df_long["time"].astype(int) - 1) * 30, unit="m"
                    )
                df_long["series"] = df_long["datetime"].dt.floor("D") + df_long["time"]
                df_long.set_index("series", inplace=True)
                df_long.drop(columns=["datetime", "time"], inplace=True)

                # Get original p_max_pu for the generator
                p_max_pu = network.generators_t.p_max_pu[gen].copy()

                # Align index
                dispatch = df_long["cf"].reindex(p_max_pu.index).fillna(0) * p_max_pu

                # Collect dispatch series for batch assignment
                dispatch_dict[gen] = dispatch

                print(
                    f" - Added {profile_type} profile for generator {gen} from {filename}"
                )

            except Exception as e:
                print(f"Failed to process profile for generator {gen}: {e}")

    # Assign all dispatch columns at once to avoid fragmentation
    if dispatch_dict:
        dispatch_df = pd.DataFrame(
            dispatch_dict, index=network.generators_t.p_max_pu.index
        )
        network.generators_t.p_max_pu.loc[:, dispatch_df.columns] = dispatch_df
        network.generators_t.p_min_pu.loc[:, dispatch_df.columns] = dispatch_df


def set_capital_costs(network: Network, db: PlexosDB):
    """Set the capital_cost for each generator in the PyPSA network.

    This is a wrapper function that calls the generic capital cost function.
    For detailed documentation, see set_capital_costs_generic in costs.py.

    Parameters
    ----------
    network : Network
        The PyPSA network containing generators.
    db : PlexosDB
        The Plexos database containing generator data.
    """
    set_capital_costs_generic(network, db, "Generator", ClassEnum.Generator)


def set_marginal_costs(network: Network, db: PlexosDB, timeslice_csv=None):
    """Set the marginal costs for generators in the PyPSA network based on fuel prices,
    heat rates, and VO&M charges from the Plexos database.

    The marginal cost is calculated as:
    marginal_cost = (fuel_price * heat_rate_inc) + vo_m_charge

    If there are multiple heat rate inc values, the average is used.
    If any required properties are missing, a warning is printed and the generator is skipped.

    Parameters
    ----------
    network : Network
        The PyPSA network containing generators.
    db : PlexosDB
        The Plexos database containing generator and fuel data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent fuel prices.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_marginal_costs(network, db, "path/to/timeslice.csv")
    """
    # Get fuel prices for all carriers
    fuel_prices = parse_fuel_prices(db, network, timeslice_csv)

    if fuel_prices.empty:
        logger.warning("No fuel prices found. Cannot set marginal costs.")
        return

    snapshots = network.snapshots
    marginal_costs_dict = {}
    skipped_generators = []

    for gen in network.generators.index:
        try:
            # Get generator properties from database
            gen_props = db.get_object_properties(ClassEnum.Generator, gen)
        except Exception as e:
            logger.warning(f"Error retrieving properties for generator {gen}: {e}")
            skipped_generators.append(gen)
            continue

        # Find the generator's carrier/fuel
        carrier = None
        if "carrier" in network.generators.columns and pd.notna(
            network.generators.loc[gen, "carrier"]
        ):
            carrier = network.generators.loc[gen, "carrier"]
        else:
            # Try to find carrier from database
            carrier = find_fuel_for_generator(db, gen)

        if carrier is None or carrier not in fuel_prices.columns:
            skipped_generators.append(gen)
            continue

        # Get fuel price time series
        fuel_price_ts = fuel_prices[carrier]

        if fuel_price_ts.isna().all() or (fuel_price_ts == 0).all():
            skipped_generators.append(gen)
            continue

        # Get Heat Rate Inc values
        heat_rate_inc_values = []
        for prop in gen_props:
            prop_name = prop.get("property", "")
            if prop_name.startswith("Heat Rate Incr") or prop_name == "Heat Rate Inc":
                try:
                    heat_rate_inc_values.append(float(prop["value"]))
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid Heat Rate Inc value for generator {gen}: {prop['value']}"
                    )

        # Calculate average heat rate inc
        if heat_rate_inc_values:
            heat_rate_inc = sum(heat_rate_inc_values) / len(heat_rate_inc_values)
        else:
            logger.warning(f"No Heat Rate Inc found for generator {gen}")
            skipped_generators.append(gen)
            continue

        # Get VO&M Charge
        vo_m_charge = None
        for prop in gen_props:
            if prop.get("property") == "VO&M Charge":
                try:
                    vo_m_charge = float(prop["value"])
                    break
                except (ValueError, TypeError):
                    logger.warning(
                        f"Invalid VO&M Charge value for generator {gen}: {prop['value']}"
                    )

        if vo_m_charge is None:
            logger.warning(f"No VO&M Charge found for generator {gen}")
            skipped_generators.append(gen)
            continue

        # Calculate marginal cost time series
        marginal_cost_ts = (fuel_price_ts * heat_rate_inc) + vo_m_charge
        marginal_costs_dict[gen] = marginal_cost_ts

        logger.debug(
            f"Generator {gen}: heat_rate_inc={heat_rate_inc}, vo_m_charge={vo_m_charge}"
        )

    # Create DataFrame from marginal costs dictionary
    if marginal_costs_dict:
        marginal_costs_df = pd.DataFrame(marginal_costs_dict, index=snapshots)

        # Only keep generators present in both network and marginal_costs_df
        valid_gens = [
            gen for gen in network.generators.index if gen in marginal_costs_df.columns
        ]

        # Assign time series to network
        # Check if marginal_cost time series exists, if not initialize it
        if not hasattr(network.generators_t, "marginal_cost"):
            network.generators_t["marginal_cost"] = pd.DataFrame(
                index=snapshots, columns=network.generators.index, dtype=float
            )

        network.generators_t.marginal_cost.loc[:, valid_gens] = marginal_costs_df[
            valid_gens
        ].copy()

        # Report success
        successful_gens = len(valid_gens)
        print(f"Successfully set marginal costs for {successful_gens} generators")
    else:
        print("No generators had complete data for marginal cost calculation")

    # Report skipped generators
    if skipped_generators:
        print(
            f"Skipped adding marginal costs for {len(skipped_generators)} generators due to missing cost properties:"
        )
        for gen in skipped_generators:
            print(f"  - {gen}")


def reassign_generators_to_node(network: Network, target_node: str):
    """Reassign all generators to a specific node.

    This is useful when demand is aggregated to a single node and all generators
    need to be connected to the same node for a meaningful optimization.

    Parameters
    ----------
    network : Network
        The PyPSA network containing generators.
    target_node : str
        Name of the node to assign all generators to.

    Returns
    -------
    dict
        Summary information about the reassignment.
    """
    if target_node not in network.buses.index:
        msg = f"Target node '{target_node}' not found in network buses"
        raise ValueError(msg)

    original_assignments = network.generators["bus"].copy()
    unique_original_buses = original_assignments.unique()

    # Reassign all generators to the target node
    network.generators["bus"] = target_node

    reassigned_count = len(network.generators)
    print(f"Reassigned {reassigned_count} generators to node '{target_node}'")
    print(
        f"  - Originally spread across {len(unique_original_buses)} buses: {list(unique_original_buses)[:5]}{'...' if len(unique_original_buses) > 5 else ''}"
    )

    return {
        "reassigned_count": reassigned_count,
        "target_node": target_node,
        "original_buses": list(unique_original_buses),
        "original_assignments": original_assignments,
    }


def port_generators(
    network: Network,
    db: PlexosDB,
    timeslice_csv=None,
    vre_profiles_path=None,
    target_node=None,
    generators_as_links=False,
    fuel_bus_prefix="fuel_",
):
    """Comprehensive function to add generators and set all their properties in the PyPSA network.

    This function combines all generator-related operations:
    - Adds generators from the Plexos database
    - Sets capacity ratings (p_max_pu)
    - Sets generator efficiencies
    - Sets capital costs
    - Sets marginal costs (time-dependent)
    - Sets VRE profiles for solar and wind generators
    - Optionally converts conventional generators to fuel→electric links
    - Optionally reassigns all generators to a specific node

    Parameters
    ----------
    network : Network
        The PyPSA network to which generators will be added.
    db : PlexosDB
        The Plexos database containing generator data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    vre_profiles_path : str, optional
        Path to the folder containing VRE generation profile files.
    target_node : str, optional
        If specified, all generators will be reassigned to this node after setup.
        This is useful when demand is aggregated to a single node.
    generators_as_links : bool, optional
        If True, represent conventional generators (coal, gas, nuclear, etc.) as Links
        connecting fuel buses to electric buses. If False, use standard Generators.
        Default False.
    fuel_bus_prefix : str, optional
        Prefix for fuel bus names when generators_as_links=True.
        Default "fuel_" creates buses like "fuel_Natural_Gas_CCGT".
        Use "" for no prefix.

    Returns
    -------
    dict or None
        If target_node is specified, returns summary information about reassignment.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> port_generators(network, db,
    ...                 timeslice_csv="path/to/timeslice.csv",
    ...                 vre_profiles_path="path/to/profiles")

    # With node reassignment for aggregated demand
    >>> reassignment_info = port_generators(network, db,
    ...                                   timeslice_csv="path/to/timeslice.csv",
    ...                                   vre_profiles_path="path/to/profiles",
    ...                                   target_node="Load_Aggregate")
    """
    print("Starting generator porting process...")

    # Step 1: Add generators
    if generators_as_links:
        print("1. Adding generators (with generators-as-links conversion)...")
    else:
        print("1. Adding generators...")
    add_generators(
        network,
        db,
        generators_as_links=generators_as_links,
        fuel_bus_prefix=fuel_bus_prefix,
    )

    # Step 2: Set capacity ratings (p_max_pu)
    print("2. Setting capacity ratings...")
    set_capacity_ratings(network, db, timeslice_csv=timeslice_csv)

    # Step 2b: Set minimum stable levels (p_min_pu)
    print("2b. Setting minimum stable levels (p_min_pu)...")
    set_min_stable_levels(network, db, timeslice_csv=timeslice_csv)

    # Step 3: Set generator efficiencies
    print("3. Setting generator efficiencies...")
    set_generator_efficiencies(network, db, use_incr=True)

    # Step 4: Set capital costs
    print("4. Setting capital costs...")
    set_capital_costs(network, db)

    # Step 5: Set marginal costs (time-dependent)
    print("5. Setting marginal costs...")
    set_marginal_costs(network, db, timeslice_csv=timeslice_csv)

    # Step 6: Set VRE profiles (if path provided)
    if vre_profiles_path:
        print("6. Setting VRE profiles...")
        set_vre_profiles(network, db, vre_profiles_path)
    else:
        print("6. Skipping VRE profiles (no path provided)")

    # Step 7: Reassign generators to target node if specified
    if target_node:
        print(f"7. Reassigning generators to node '{target_node}'...")
        return reassign_generators_to_node(network, target_node)
    else:
        print("7. Skipping generator reassignment (no target node specified)")

    print(f"Generator porting complete! Added {len(network.generators)} generators.")
