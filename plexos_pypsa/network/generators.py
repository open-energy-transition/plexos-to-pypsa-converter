import logging
import os

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import (
    find_bus_for_object,
    find_fuel_for_generator,
    get_dataid_timeslice_map,
    read_timeslice_activity,
)

logger = logging.getLogger(__name__)


def add_generators(network: Network, db: PlexosDB):
    """
    Adds generators from a Plexos database to a PyPSA network.

    This function retrieves generator objects from the provided Plexos database,
    extracts their properties, and adds them to the PyPSA network. If a generator
    lacks properties or an associated bus, it is skipped and reported at the end.

    Parameters
    ----------
    network : Network
        The PyPSA network to which the generators will be added.
    db : PlexosDB
        The Plexos database containing generator data.

    Raises
    ------
    Exception
        If an error occurs while retrieving generator properties.

    Notes
    -----
    - Generators without a "Max Capacity" property will not have `p_nom` specified.
    - Generators without an associated bus will be skipped and reported.
    - A summary of skipped generators is printed at the end.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_generators(network, db)
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
            "VO&M Charge": "marginal_cost",
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

        # Add generator to the network
        if p_max is not None:
            if carrier is not None:
                network.add("Generator", gen, bus=bus, p_nom=p_max, carrier=carrier)
            else:
                network.add("Generator", gen, bus=bus, p_nom=p_max)
            for attr, val in gen_attrs.items():
                network.generators.loc[gen, attr] = val
        else:
            if carrier is not None:
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
    """
    Parse generator ratings from the PlexosDB using SQL and return a DataFrame with index=network.snapshots, columns=generator names.
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

    # 3. Get property values for each data_id (Rating, Rating Factor, or Max Capacity), with t_date_from/to
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
        WHERE p.name = 'Rating' OR p.name = 'Max Capacity' OR p.name = 'Rating Factor'
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
        ],
    )

    # 4. Merge: generator object_id -> data_id -> property info
    merged = pd.merge(membership_df, prop_df, on="membership_id", how="inner")
    merged = pd.merge(
        merged, gen_df, left_on="child_object_id", right_on="object_id", how="inner"
    )

    # 5. Collect generator Series in a dictionary for efficient concatenation
    gen_series = {}

    for gen in gen_df["generator"]:
        props = merged[merged["generator"] == gen]
        # Get all property entries (time-dependent, timeslice-dependent, or static)
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
            # Attach timeslice info if available
            if p["data_id"] in dataid_to_timeslice:
                # Ensure timeslices are always a list of full names
                entry["timeslices"] = dataid_to_timeslice[p["data_id"]]
            property_entries.append(entry)

        # Build a DataFrame for all property entries
        prop_df_entries = pd.DataFrame(property_entries)

        # For each property type (Rating, Rating Factor, Max Capacity), build a time series
        def build_ts(prop_name, fallback=None):
            ts = pd.Series(index=snapshots, dtype=float)
            # 1. Timeslice-based entries (take precedence)
            if timeslice_activity is not None:
                for _, row in prop_df_entries[
                    prop_df_entries["property"] == prop_name
                ].iterrows():
                    for tslice in row["timeslices"]:
                        # Use the full timeslice name as key, warn if not found
                        if tslice in timeslice_activity.columns:
                            mask = timeslice_activity[tslice]
                            ts.loc[mask] = row["value"]
                        else:
                            print(
                                f"Warning: Timeslice '{tslice}' not found in timeslice_activity columns: {list(timeslice_activity.columns)}"
                            )
                        ts.loc[mask] = row["value"]
            # 2. t_date_from/t_date_to entries (only if not already set by timeslice)
            for _, row in prop_df_entries[
                prop_df_entries["property"] == prop_name
            ].iterrows():
                if ("timeslices" in row and row["timeslices"]) or (
                    row["from"] is None and row["to"] is None
                ):
                    continue  # skip if timeslice already handled or static
                if row["from"] is not None and row["to"] is not None:
                    mask = (snapshots >= row["from"]) & (snapshots <= row["to"])
                    ts.loc[mask & ts.isnull()] = row["value"]
                elif row["from"] is not None:
                    mask = snapshots >= row["from"]
                    ts.loc[mask & ts.isnull()] = row["value"]
                elif row["to"] is not None:
                    mask = snapshots <= row["to"]
                    ts.loc[mask & ts.isnull()] = row["value"]
            # 3. Static entries (only if not already set)
            for _, row in prop_df_entries[
                prop_df_entries["property"] == prop_name
            ].iterrows():
                if ("timeslices" in row and row["timeslices"]) or (
                    row["from"] is not None or row["to"] is not None
                ):
                    continue
                ts.loc[ts.isnull()] = row["value"]
            # 4. Fallback
            if fallback is not None:
                ts = ts.fillna(fallback)
            return ts

        # Get Max Capacity (for fallback and for scaling)
        maxcap = None
        maxcap_entries = prop_df_entries[prop_df_entries["property"] == "Max Capacity"]
        if not maxcap_entries.empty:
            maxcap = maxcap_entries.iloc[0]["value"]

        # Build time series for Rating and Rating Factor
        rating_ts = build_ts("Rating")
        rating_factor_ts = build_ts("Rating Factor")
        # If there are any Rating entries, use them (as p_max_pu = Rating / p_nom)
        if rating_ts.notnull().any():
            p_nom = (
                network.generators.loc[gen, "p_nom"]
                if gen in network.generators.index
                and "p_nom" in network.generators.columns
                else maxcap
            )
            ts = rating_ts.fillna(maxcap)
            ts = ts / p_nom if p_nom else ts
            gen_series[gen] = ts
            continue
        # If no Rating, use Rating Factor (as p_max_pu = Rating Factor / 100)
        if rating_factor_ts.notnull().any():
            ts = rating_factor_ts.fillna(1.0)
            gen_series[gen] = ts
            continue
        # If neither, fallback to Max Capacity (p_max_pu = 1.0)
        gen_series[gen] = pd.Series(1.0, index=snapshots)

    # Concatenate all generator Series into a single DataFrame
    result = pd.concat(gen_series, axis=1)
    result.index = snapshots

    return result


def set_capacity_ratings(network: Network, db: PlexosDB, timeslice_csv=None):
    """
    Sets the capacity ratings for generators in the PyPSA network based on the Plexos database.

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


def set_generator_efficiencies(network: Network, db: PlexosDB, use_incr: bool = True):
    """
    Sets the efficiency for each generator in the PyPSA network based on
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
    import numpy as np

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
    """
    Adds time series profiles for solar and wind generators to the PyPSA network.
    Also sets the generator's carrier to "Solar" or "Wind" if a profile is found.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        The Plexos database containing generator data.
    path : str
        Path to the folder containing generation profile files.
    """
    dispatch_dict = {}
    for gen in network.generators.index:
        # Skip Adelaide_Desal_FFP
        if gen == "Adelaide_Desal_FFP":
            print(f"Skipping generator {gen}")
            continue

        # Retrieve generator properties from the database
        props = db.get_object_properties(ClassEnum.Generator, gen)

        # Check if the generator has a solar or wind profile
        filename = next(
            (
                p["texts"]
                for p in props
                if "Traces\\solar\\" in p["texts"] or "Traces\\wind\\" in p["texts"]
            ),
            None,
        )

        if filename:
            # print(f"Found profile for generator {gen}: {filename}")
            profile_type = "solar" if "Traces\\solar\\" in filename else "wind"
            file_path = os.path.join(path, filename.replace("\\", os.sep))

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
                    raise ValueError(f"Unsupported resolution in file: {filename}")

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
        # else:
        #     # If the generator does not have a solar or wind profile, skip it
        #     print(f"Generator {gen} does not have a solar or wind profile. Skipping.")

    # Assign all dispatch columns at once to avoid fragmentation
    if dispatch_dict:
        dispatch_df = pd.DataFrame(
            dispatch_dict, index=network.generators_t.p_max_pu.index
        )
        network.generators_t.p_max_pu.loc[:, dispatch_df.columns] = dispatch_df
        network.generators_t.p_min_pu.loc[:, dispatch_df.columns] = dispatch_df


def set_capital_costs(network: Network, db: PlexosDB):
    """
    Sets the capital_cost for each generator in the PyPSA network based on
    'Build Cost', 'WACC', 'Economic Life' (preferred), 'Technical Life', and 'FO&M Charge' properties from the PlexosDB.

    The capital_cost is calculated as:
        annuity_factor = wacc / (1 - (1 + wacc) ** -lifetime)
        annualized_capex = build_cost * annuity_factor
        capital_cost = annualized_capex + fo_m_charge

    All costs are converted to $/MW (PyPSA convention).

    - Build Cost is given in $/kW, so multiply by 1000 to get $/MW.
    - FO&M Charge is given in $/kW/yr, so multiply by 1000 to get $/MW/yr.
    - capital_cost is stored in $/MW/yr.

    If build_cost, wacc, or lifetime are missing, but FO&M Charge is present,
    capital_cost is set to FO&M Charge only (converted to $/MW/yr).

    If no cost can be calculated or found, capital_cost is set to 0.

    The result is stored in network.generators['capital_cost'].
    """

    capital_costs = []
    for gen in network.generators.index:
        props = db.get_object_properties(ClassEnum.Generator, gen)
        # Extract properties
        build_costs = [
            float(p["value"]) for p in props if p["property"] == "Build Cost"
        ]
        build_cost = sum(build_costs) / len(build_costs) if build_costs else None
        wacc = next((float(p["value"]) for p in props if p["property"] == "WACC"), None)
        # Prefer Economic Life, fallback to Technical Life
        economic_life = next(
            (float(p["value"]) for p in props if p["property"] == "Economic Life"), None
        )
        technical_life = next(
            (float(p["value"]) for p in props if p["property"] == "Technical Life"),
            None,
        )
        lifetime = economic_life if economic_life is not None else technical_life
        fo_m_charge = next(
            (float(p["value"]) for p in props if p["property"] == "FO&M Charge"), None
        )

        # Convert units: $/kW -> $/MW
        build_cost_MW = build_cost * 1000 if build_cost is not None else None
        fo_m_charge_MW = fo_m_charge * 1000 if fo_m_charge is not None else None

        if build_cost_MW is None or wacc is None or lifetime is None or lifetime <= 0:
            if fo_m_charge_MW is not None:
                capital_costs.append(fo_m_charge_MW)
            else:
                capital_costs.append(0.0)
                print(
                    f"Warning: Missing or invalid capital cost data for {gen}. Setting capital_cost to 0."
                )
            continue

        # Calculate annuity factor
        try:
            annuity_factor = wacc / (1 - (1 + wacc) ** (-lifetime))
        except ZeroDivisionError:
            annuity_factor = 1.0

        annualized_capex = build_cost_MW * annuity_factor
        capital_cost = annualized_capex + (
            fo_m_charge_MW if fo_m_charge_MW is not None else 0.0
        )
        capital_costs.append(capital_cost)

    network.generators["capital_cost"] = capital_costs
