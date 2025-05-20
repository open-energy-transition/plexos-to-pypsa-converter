import logging
import os
from statistics import mean

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object
from plexos_pypsa.db.rating import parse_generator_ratings

logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


def add_buses(network: Network, db: PlexosDB):
    """
    Adds buses to the given network based on the nodes retrieved from the database.

    Parameters
    ----------
    network : pypsa.Network
        The network object to which buses will be added.
    db : PlexosDB
        A PlexosDB object containing the database connection.

    Notes
    -----
    - The function retrieves all nodes from the database and their properties.
    - Each node is added as a bus to the network with its nominal voltage (`v_nom`).
    - If a node does not have a specified voltage property, a default value of 110 kV is used.
    - The function prints the total number of buses added to the network.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_buses(network, db)
    Added 10 buses
    """
    nodes = db.list_objects_by_class(ClassEnum.Node)
    for node in nodes:
        props = db.get_object_properties(ClassEnum.Node, node)
        voltage = next(
            (float(p["value"]) for p in props if p["property"] == "Voltage"), 110
        )  # default to 110 kV if voltage not found

        # add bus to network
        # NOTE: skipping adding v_nom because none of the AEMO nodes have voltage values
        network.add("Bus", name=node)
    print(f"Added {len(nodes)} buses")


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
    for gen in network.generators.index:
        # Check if the generator is in the generator_ratings DataFrame
        if gen in generator_ratings.columns:
            # Get the p_nom (max capacity) for the generator
            p_nom = network.generators.loc[gen, "p_nom"]

            # Normalize the generating_ratings for the generator by p_nom
            generator_ratings[gen] = generator_ratings[gen] / p_nom

            # Set the p_max_pu time series for the generator
            network.generators_t.p_max_pu.loc[:, gen] = generator_ratings[gen]

        else:
            print(f"Warning: Generator {gen} not found in ratings DataFrame.")
            continue


def add_storage(network: Network, db: PlexosDB) -> None:
    """Adds storage units from PLEXOS input db to a PyPSA network."""

    storage_units = db.list_objects_by_class(ClassEnum.Storage)
    logger.info(f"Found {len(storage_units)} storage units in db")

    skipped_units = []

    for su in storage_units:
        try:
            props = db.get_object_properties(ClassEnum.Storage, su)

            # Helper to look up a property by name
            def get_prop(name):
                for p in props:
                    if p["property"] == name:
                        return float(p["value"])
                return None

            # Infer the bus from memberships
            memberships = db.get_memberships_system(su, object_class=ClassEnum.Storage)
            bus = next(
                (
                    m["name"]
                    for m in memberships
                    if m["collection_name"]
                    in {"Storage From", "Head Storage", "Tail Storage", "Storage To"}
                ),
                None,
            )
            if bus is None:
                logger.warning(f"Skipping {su}: no connected bus found via memberships")
                skipped_units.append(su)
                continue

            # Add storage unit to the PyPSA network
            network.storage_units.loc[su] = {
                "bus": bus,
                "carrier": "hydro",  # NOTE: can't find non-hydro storage in AEMO?
                "name": su,
            }

        except Exception as e:
            logger.error(f"Failed to add storage unit {su}: {e}")
            skipped_units.append(su)

    logger.info(
        f"Added {len(storage_units) - len(skipped_units)} storage units to network"
    )
    if skipped_units:
        logger.info(f"Skipped {len(skipped_units)} storage units: {skipped_units}")


def add_links(network: Network, db: PlexosDB):
    """
    Adds transmission links to the given network based on data from the database.

    This function retrieves line objects from the database, extracts their properties,
    and adds them to the network as links. It ensures that the links connect valid buses
    in the network and sets specific attributes like `p_nom`, `p_max_pu`, and `p_min_pu`.

    Parameters
    ----------
    network : pypsa.Network
        The network object to which the links will be added.
    db : Database
        The database object containing line data and their properties.

    Notes
    -----
    - The function retrieves the "From Nodes" and "To Nodes" for each line to determine
      the buses the link connects.
    - The largest positive "Max Flow" property is used to set `p_nom`.
    - The largest negative "Min Flow" property is normalized to `p_nom` and used to set `p_min_pu`.
    - `p_max_pu` is set to 1.0.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_links(network, db)
    Added 10 transmission links
    """
    lines = db.list_objects_by_class(ClassEnum.Line)
    for line in lines:
        props = db.get_object_properties(ClassEnum.Line, line)
        # Find connected nodes
        memberships = db.get_memberships_system(line, object_class=ClassEnum.Line)
        node_from = next(
            (m["name"] for m in memberships if m["collection_name"] == "Node From"),
            None,
        )
        node_to = next(
            (m["name"] for m in memberships if m["collection_name"] == "Node To"), None
        )

        # Helper to extract the greatest property value
        def get_prop(prop, cond=lambda v: True):
            vals = [
                float(p["value"])
                for p in props
                if p["property"] == prop and cond(float(p["value"]))
            ]
            return max(vals) if vals else None

        # Get flows and ratings
        max_flow = get_prop("Max Flow", lambda v: v > 0)
        min_flow = get_prop("Min Flow", lambda v: v < 0)
        max_rating = get_prop("Max Rating", lambda v: v > 0)
        min_rating = get_prop("Min Rating", lambda v: v < 0)

        # Set p_nom as max_flow
        p_nom = max_flow if max_flow is not None else None

        # Prefer ratings if available
        max_val = (
            max_rating
            if max_rating is not None
            else (max_flow if max_flow is not None else 0)
        )
        min_val = (
            min_rating
            if min_rating is not None
            else (min_flow if min_flow is not None else 0)
        )

        # Calculate pu values
        p_min_pu = min_val / p_nom if p_nom else 0
        p_max_pu = max_val / p_nom if p_nom else 1

        # Ramp limits
        ramp_limit_up = get_prop("Max Ramp Up", lambda v: v > 0)
        ramp_limit_down = get_prop("Max Ramp Down", lambda v: v > 0)

        # Add link
        if p_nom is not None:
            network.add(
                "Link",
                name=line,
                bus0=node_from,
                bus1=node_to,
                p_nom=p_nom,
                p_min_pu=p_min_pu,
                p_max_pu=p_max_pu,
            )
            print(
                f"- Added link {line} with p_nom={p_nom} to buses {node_from} and {node_to}"
            )
        else:
            network.add(
                "Link",
                name=line,
                bus0=node_from,
                bus1=node_to,
            )
            print(
                f"- Added link {line} without p_nom to buses {node_from} and {node_to}"
            )

        # Set ramp limits if available
        if ramp_limit_up is not None:
            network.links.loc[line, "ramp_limit_up"] = ramp_limit_up
        if ramp_limit_down is not None:
            network.links.loc[line, "ramp_limit_down"] = ramp_limit_down
    print(f"Added {len(lines)} links")


def add_constraints(network: Network, db: PlexosDB) -> None:
    """
    Adds constraints from the PLEXOS database to the PyPSA network.

    The function infers the possible kinds of constraints, based on the constraint name, properties, and memberships:

    1. Emissions Budgets:
        - Identified by the presence of Production Coefficient and RHS Custom.
        - These constraints limit emissions and can be mapped to pypsa.global_constraints.

    2. Line Flow Limits:
        - Identified by the presence of Flow Forward Coefficient or Flow Back Coefficient.
        - These constraints limit the flow on specific transmission lines.

    3. Generation Limits:
        - Identified by the presence of Generation Coefficient or Installed Capacity Coefficient.
        - These constraints limit the generation capacity of specific generators.

    4. Storage Constraints:
        - Identified by properties like Units Built in Year Coefficient or RHS Year.
        - These constraints limit storage capacity or usage.

    5. Other Constraints:
        - Constraints that don't fit into the above categories can be logged for manual review.

    The constraints are mapped to the PyPSA network as follows:
    - Emissions Budgets: Added as global constraints (network.global_constraints).
    - Line Flow Limits: Applied to specific lines (network.lines).
    - Generation Limits: Applied to specific generators (network.generators).
    - Storage Constraints: Applied to specific storage units (network.storage_units).

    For each constraint, the function checks for all RHS-related properties (RHS Custom, RHS Year, RHS Day, etc.).
    If multiple RHS values exist (e.g., for different time periods),
    the function aggregates or processes them appropriately (currently taking the average).

    Parameters
    ----------
    network : pysa.Network
        The PyPSA network to which constraints will be added.
    db : PlexosDB
        The PLEXOS database containing constraint data.

    Notes
    -----
    - The function retrieves all constraints from the database and their properties.
    - It infers the type of constraint based on the properties and memberships.
    - The function applies the constraints to the network based on their inferred type.
    - The function handles various types of constraints, including emissions budgets,
        line flow limits, generation limits, storage constraints, and capacity constraints.
    - The function logs the application of each constraint and any errors encountered.
    - The function uses the mean of RHS values when multiple are found.
    - The function handles exceptions and logs errors for constraints that fail to apply.
    - The function uses the logger to provide detailed information about the process.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_constraints(network, db)
    Added 5 constraints
    >>> # Check the applied constraints in the network
    >>> print(network.global_constraints)

    """
    constraints = db.list_objects_by_class(ClassEnum.Constraint)
    logger.info(f"Found {len(constraints)} constraints in db")

    for constraint in constraints:
        try:
            # Get constraint properties and memberships
            props = db.get_object_properties(ClassEnum.Constraint, constraint)
            memberships = db.get_memberships_system(
                constraint, object_class=ClassEnum.Constraint
            )

            # Extract relevant properties
            sense = next(
                (float(p["value"]) for p in props if p["property"] == "Sense"), None
            )

            # Collect all RHS values (e.g., RHS Custom, RHS Year, RHS Day)
            rhs_values = [
                float(p["value"])
                for p in props
                if p["property"] in {"RHS Custom", "RHS Year", "RHS Day", "RHS"}
            ]

            # Aggregate RHS values using the mean
            rhs = mean(rhs_values) if rhs_values else None

            # Extract coefficients
            production_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Production Coefficient"
            ]
            generation_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Generation Coefficient"
            ]
            flow_coefficients = [
                float(p["value"])
                for p in props
                if p["property"]
                in {
                    "Flow Forward Coefficient",
                    "Flow Back Coefficient",
                    "Flow Coefficient",
                }
            ]
            storage_coefficients = [
                float(p["value"])
                for p in props
                if p["property"] == "Units Built in Year Coefficient"
            ]
            capacity_coefficients = [
                float(p["value"])
                for p in props
                if p["property"]
                in {"Installed Capacity Coefficient", "Capacity Built Coefficient"}
            ]

            # Infer constraint type
            if production_coefficients and rhs is not None:
                inferred_type = "Emissions Budget"
            elif flow_coefficients:
                inferred_type = "Line Flow Limit"
            elif generation_coefficients:
                inferred_type = "Generation Limit"
            elif storage_coefficients:
                inferred_type = "Storage Constraint"
            elif capacity_coefficients:
                inferred_type = "Capacity Constraint"
            else:
                inferred_type = "Unknown"

            logger.debug(f"Constraint: {constraint}, Type: {inferred_type}, RHS: {rhs}")

            # Apply constraints based on inferred type
            if inferred_type == "Emissions Budget":
                # Apply emissions budget as a global constraint
                network.global_constraints.loc[constraint] = {
                    "type": "primary_energy",
                    "sense": "<=" if sense == -1 else ">=",
                    "constant": rhs,
                    "carrier_attribute": "emissions",
                }
                logger.info(
                    f"Applied emissions budget of {rhs} to constraint {constraint}"
                )

            elif inferred_type == "Line Flow Limit":
                # Apply line flow limits to specific lines
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Line"
                        and membership["name"] in network.lines.index
                    ):
                        network.lines.loc[membership["name"], "s_nom"] = rhs
                logger.info(
                    f"Applied line flow limit of {rhs} to line {membership['name']}"
                )

            elif inferred_type == "Generation Limit":
                # Apply generation limits to specific generators
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Generator"
                        and membership["name"] in network.generators.index
                    ):
                        # TODO: should this be something else?
                        network.generators.loc[membership["name"], "p_nom"] = rhs
                logger.info(
                    f"Applied generation limit of {rhs} to generator {membership['name']}"
                )

            elif inferred_type == "Storage Constraint":
                # Apply storage constraints to specific storage units
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Storage"
                        and membership["name"] in network.storage_units.index
                    ):
                        network.storage_units.loc[membership["name"], "max_hours"] = rhs
                logger.info(
                    f"Applied storage constraint of {rhs} hours to storage {membership['name']}"
                )

            elif inferred_type == "Capacity Constraint":
                # Apply capacity constraints to specific components
                for membership in memberships:
                    logger.debug(f"Membership: {membership}")
                    if (
                        membership["class"] == "Generator"
                        and membership["name"] in network.generators.index
                    ):
                        network.generators.loc[membership["name"], "p_nom_max"] = rhs
                logger.info(
                    f"Applied capacity constraint of {rhs} to generator {membership['name']}"
                )

            else:
                logger.warning(f"Unknown constraint type for {constraint}, skipping.")

        except Exception as e:
            logger.error(f"Failed to apply constraint {constraint}: {e}")


def add_snapshot(network: Network, path: str):
    """
    Reads all {bus}...csv files in the specified path, determines the resolution,
    and creates a unified time series to set as the network snapshots.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    path : str
        Path to the folder containing raw demand data files.
    """
    all_times = []

    # Read all {bus}...csv files in the folder
    for file in os.listdir(path):
        if file.endswith(".csv"):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])
            df.set_index("datetime", inplace=True)

            # Normalize column names to handle both cases (e.g., 1, 2, ...48 or 01, 02, ...48)
            df.columns = pd.Index(
                [
                    str(int(col))
                    if col.strip().isdigit() and col not in {"Year", "Month", "Day"}
                    else col
                    for col in df.columns
                ]
            )

            # Determine the resolution based on the number of columns
            non_date_columns = [
                col for col in df.columns if col not in {"Year", "Month", "Day"}
            ]
            if len(non_date_columns) == 24:
                resolution = 60  # hourly
            elif len(non_date_columns) == 48:
                resolution = 30  # 30 minutes
            else:
                raise ValueError(f"Unsupported resolution in file: {file}")

            # Create a time series for this file
            times = pd.date_range(
                start=df.index.min(),
                end=df.index.max()
                + pd.Timedelta(days=1)
                - pd.Timedelta(minutes=resolution),
                freq=f"{resolution}min",
            )
            all_times.append(times)

    # Combine all time series into a unified time series
    unified_times = (
        pd.concat([pd.Series(times) for times in all_times])
        .drop_duplicates()
        .sort_values()
    )

    # Set the unified time series as the network snapshots
    network.set_snapshots(unified_times)


def add_loads(network: Network, path: str):
    """
    Adds loads to the PyPSA network for each bus based on the corresponding {bus}...csv file.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    path : str
        Path to the folder containing raw demand data files.
    """
    for bus in network.buses.index:
        # Find the corresponding load file for the bus
        file_name = next(
            (
                f
                for f in os.listdir(path)
                if f.startswith(f"{bus}_") and f.endswith(".csv")
            ),
            None,
        )
        if file_name is None:
            print(f"Warning: No load file found for bus {bus}")
            continue
        file_path = os.path.join(path, file_name)

        # Read the load file
        df = pd.read_csv(file_path, index_col=["Year", "Month", "Day"])
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

        # Normalize column names to handle both cases (e.g., 1, 2, ...48 or 01, 02, ...48)
        df.columns = pd.Index(
            [
                str(int(col))
                if col.strip().isdigit()
                and col not in {"Year", "Month", "Day", "datetime"}
                else col
                for col in df.columns
            ]
        )

        # Determine the resolution based on the number of columns
        non_date_columns = [
            col for col in df.columns if col not in {"Year", "Month", "Day", "datetime"}
        ]
        if len(non_date_columns) == 24:
            resolution = 60  # hourly
        elif len(non_date_columns) == 48:
            resolution = 30  # 30 minutes
        else:
            raise ValueError("Unsupported resolution.")

        # Change df to long format, with datetime as index
        df_long = df.melt(
            id_vars=["datetime"],
            value_vars=non_date_columns,
            var_name="time",
            value_name="load",
        )

        # create column with time, depending on the resolution
        if resolution == 60:
            df_long["time"] = pd.to_timedelta(
                (df_long["time"].astype(int) - 1) * 60, unit="m"
            )
        elif resolution == 30:
            df_long["time"] = pd.to_timedelta(
                (df_long["time"].astype(int) - 1) * 30, unit="m"
            )

        # combine datetime and time columns
        # but make sure "0 days" is not added to the datetime
        df_long["series"] = df_long["datetime"].dt.floor("D") + df_long["time"]
        df_long.set_index("series", inplace=True)

        # drop datetime and time columns
        df_long.drop(columns=["datetime", "time"], inplace=True)

        # Add the load to the network
        load_name = f"Load_{bus}"
        network.add("Load", name=load_name, bus=bus)

        # Add the load time series
        network.loads_t.p_set.loc[:, load_name] = df_long
        print(f"- Added load time series for {load_name}")


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

                # Set both p_min_pu and p_max_pu to the dispatch timeseries
                network.generators_t.p_max_pu.loc[:, gen] = dispatch
                network.generators_t.p_min_pu.loc[:, gen] = dispatch

                print(
                    f" - Added {profile_type} profile for generator {gen} from {filename}"
                )

            except Exception as e:
                print(f"Failed to process profile for generator {gen}: {e}")
        # else:
        #     # If the generator does not have a solar or wind profile, skip it
        #     print(f"Generator {gen} does not have a solar or wind profile. Skipping.")


def add_hydro_inflows(network: Network, db: PlexosDB, path: str):
    """
    Adds inflow time series for hydro storage units to the PyPSA network.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        The Plexos database containing storage unit data.
    path : str
        Path to the folder containing inflow profile files.
    """
    for storage_unit in network.storage_units.index:
        # Retrieve storage unit properties from the database
        props = db.get_object_properties(ClassEnum.Storage, storage_unit)

        # Check if the storage unit has a hydro inflow profile
        filename = next(
            (
                p["texts"]
                for p in props
                if "Traces\\hydro\\MonthlyNaturalInflow" in p["texts"]
            ),
            None,
        )

        if filename:
            file_path = os.path.join(path, filename.replace("\\", os.sep))

            try:
                # Read the inflow profile file
                df = pd.read_csv(file_path)

                # Create a date column using Year, Month, and Day
                df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

                # Set the date column as the index and drop unnecessary columns
                daily_inflows = df.set_index("date")["Inflows"]

                # Get the network's snapshots
                snapshots = network.snapshots

                # Resample daily inflows to match the network's snapshots
                inflows_resampled = daily_inflows.reindex(snapshots, method="ffill")

                # Detect the number of time instances per day
                time_instances_per_day = (
                    snapshots.to_series()
                    .groupby(snapshots.to_series().dt.date)
                    .size()
                    .iloc[0]
                )

                # Evenly divide daily inflows across the time instances per day
                inflows_scaled = inflows_resampled / time_instances_per_day

                # Add the inflows as a time series to the storage unit
                network.storage_units_t.inflow[storage_unit] = inflows_scaled

                print(
                    f"Added hydro inflow profile for storage unit {storage_unit} from {filename}"
                )

            except Exception as e:
                print(
                    f"Failed to process inflow profile for storage unit {storage_unit}: {e}"
                )
        else:
            # If the storage unit does not have a hydro inflow profile,      it
            print(
                f"Storage unit {storage_unit} does not have a hydro inflow profile. Skipping."
            )


def set_generator_efficiencies(network: Network, db: PlexosDB):
    """
    Sets the efficiency for each generator in the PyPSA network based on
    'Heat Rate Base' and 'Heat Rate Incr*' properties from the PlexosDB.

    The efficiency is calculated as:
        efficiency = (p_nom / fuel) * (1 / 0.293)
    where:
        fuel = hr_base + (hr_inc * p_nom)
    If multiple hr_inc are present, p_nom is divided into equal segments.

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

        # Extract Heat Rate Base and all Heat Rate Incr*
        hr_base = next(
            (float(p["value"]) for p in props if p["property"] == "Heat Rate Base"),
            None,
        )
        hr_incs = [
            float(p["value"])
            for p in props
            if p["property"].lower().startswith("heat rate incr")
        ]

        if hr_base is None or not hr_incs:
            # NOTE: setting efficiency to 1 if no hr_base or hr_incs are found
            efficiencies.append(1)
            continue

        if len(hr_incs) == 1:
            fuel = hr_base + (hr_incs[0] * p_nom)
        else:
            n = len(hr_incs)
            p_seg = p_nom / n
            fuel = hr_base + sum(hr_incs[i] * p_seg for i in range(n))

        # NOTE: setting efficiency to 1 if fuel is 0
        efficiency = (p_nom / fuel) * (1 / 0.293) if fuel != 0 else 1
        efficiencies.append(efficiency)

    network.generators["efficiency"] = efficiencies
