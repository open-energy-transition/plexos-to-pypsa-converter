import logging
import os

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object
from plexos_pypsa.db.rating import parse_generator_ratings

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

        # Add generator to the network
        if p_max is not None:
            network.add("Generator", gen, bus=bus, p_nom=p_max)
            for attr, val in gen_attrs.items():
                network.generators.loc[gen, attr] = val
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


def set_capacity_ratings(network: Network, db: PlexosDB):
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

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_capacity_ratings(network, db)
    """
    # Get the generator ratings from the database
    generator_ratings = parse_generator_ratings(db, network)

    # For each generator in the network:
    # - Get the p_nom (max capacity)
    # - Normalize the generating_ratings for the generator by p_nom
    # - Set the p_max_pu time series for the generator

    # Only keep generators present in both network and generator_ratings
    valid_gens = [
        gen for gen in network.generators.index if gen in generator_ratings.columns
    ]
    missing_gens = [
        gen for gen in network.generators.index if gen not in generator_ratings.columns
    ]

    # Normalize ratings by p_nom for valid generators
    for gen in valid_gens:
        p_nom = network.generators.loc[gen, "p_nom"]
        generator_ratings[gen] = generator_ratings[gen] / p_nom

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
            print(f"Found profile for generator {gen}: {filename}")
            profile_type = "solar" if "Traces\\solar\\" in filename else "wind"
            file_path = os.path.join(path, filename.replace("\\", os.sep))

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
