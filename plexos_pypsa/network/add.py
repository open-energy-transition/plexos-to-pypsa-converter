import logging
import os
from statistics import mean

import pandas as pd
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object

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

        # Find associated bus/node
        bus = find_bus_for_object(db, gen, ClassEnum.Generator)
        if bus is None:
            print(f"Warning: No associated bus found for generator {gen}")
            skipped_generators.append(gen)
            continue

        # Add generator to the network
        if p_max is not None:
            network.add("Generator", gen, bus=bus, p_nom=p_max)
            # print(f"- Added generator {gen} with p_nom={p_max} to bus {bus}")
        else:
            network.add("Generator", gen, bus=bus)
            # print(f"- Added generator {gen} to bus {bus} without p_nom")

    # Report skipped generators
    if empty_generators:
        print(f"\nSkipped {len(empty_generators)} generators with no properties:")
        for g in empty_generators:
            print(f"  - {g}")

    if skipped_generators:
        print(f"\nSkipped {len(skipped_generators)} generators with no associated bus:")
        for g in skipped_generators:
            print(f"  - {g}")


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
        node_from = next(
            (
                m["name"]
                for m in db.get_memberships_system(line, object_class=ClassEnum.Line)
                if m["collection_name"] == "Node From"
            ),
            None,
        )
        node_to = next(
            (
                m["name"]
                for m in db.get_memberships_system(line, object_class=ClassEnum.Line)
                if m["collection_name"] == "Node To"
            ),
            None,
        )

        # Determine p_nom (largest positive "Max Flow")
        p_nom = max(
            (
                float(p["value"])
                for p in props
                if p["property"] == "Max Flow" and float(p["value"]) > 0
            ),
            default=None,
        )

        if p_nom is None:
            continue  # Skip if no valid "Max Flow" is found

        # Determine p_min_pu (largest negative "Min Flow" normalized to p_nom)
        min_flow = max(
            (
                float(p["value"])
                for p in props
                if p["property"] == "Min Flow" and float(p["value"]) < 0
            ),
            default=0,
        )
        p_min_pu = min_flow / p_nom if p_nom != 0 else None

        # Add link to the network
        # If p_nom is None, only add bus0 and bus1
        if p_nom is not None:
            network.add(
                "Link",
                name=line,
                bus0=node_from,
                bus1=node_to,
                p_nom=p_nom,
                p_max_pu=1.0,
                p_min_pu=p_min_pu,
            )
            print(
                f"- Added link {line} with p_nom={p_nom} to buses {node_from} and {node_to}"
            )
        else:
            # If p_nom is None, add the link without p_nom
            network.add(
                "Link",
                name=line,
                bus0=node_from,
                bus1=node_to,
            )
            print(f"- Added link {line} to buses {node_from} and {node_to}")
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
    network : pypsa.Network
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
        print(f"File name: {file_name}")
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
        print(f"- Added load {load_name} to bus {bus}")

        # Add the load time series
        network.loads_t.p_set.loc[:, load_name] = df_long
        print(f"-- Added load time series for {load_name}")


# def add_emissions(network: Network, db: PlexosDB) -> None:
#     """
#     Maps emissions data from the PLEXOS database to the PyPSA network.

#     Parameters
#     ----------
#     network : pypsa.Network
#         The PyPSA network to which emissions data will be added.
#     db : PlexosDB
#         The PLEXOS database containing emissions data.

#     Notes
#     -----
#     - Emission coefficients are applied to generators in the PyPSA network.
#     - Emission budgets are added as global constraints in the PyPSA network.
#     """
#     logger.info("Mapping emissions data from PLEXOS to PyPSA network.")

#     # Retrieve all emissions from the PLEXOS database
#     emissions = db.list_objects_by_class(ClassEnum.Emission)
#     logger.info(f"Found {len(emissions)} emissions in db")

#     for emission in emissions:
#         try:
#             # Get emission properties
#             props = db.get_object_properties(ClassEnum.Emission, emission)
#             logger.debug(f"Emission: {emission}, Properties: {props}")

#             # Extract relevant properties
#             emission_coefficient = next(
#                 (float(p["value"]) for p in props if p["property"] == "Emission Rate"),
#                 None,
#             )
#             if emission_coefficient is None:
#                 logger.warning(f"No 'Emission Rate' found for emission {emission}")

#             emission_budget = next(
#                 (
#                     float(p["value"])
#                     for p in props
#                     if p["property"] == "Emission Budget"
#                 ),
#                 None,
#             )

#             # Map emission coefficients to generators
#             memberships = db.get_memberships_system(
#                 emission, object_class=ClassEnum.Emission
#             )
#             logger.debug(f"Emission: {emission}, Memberships: {memberships}")

#             for membership in memberships:
#                 if (
#                     membership["class"] == "Generator"
#                     and membership["name"] in network.generators.index
#                 ):
#                     network.generators.loc[membership["name"], "emission_factor"] = (
#                         emission_coefficient
#                     )
#                     logger.info(
#                         f"Applied emission factor of {emission_coefficient} to generator {membership['name']}"
#                     )

#             # Add emission budgets as global constraints
#             if emission_budget is not None:
#                 network.global_constraints.loc[emission] = {
#                     "type": "primary_energy",
#                     "sense": "<=",
#                     "constant": emission_budget,
#                     "carrier_attribute": "emissions",
#                 }
#                 logger.info(
#                     f"Added emission budget of {emission_budget} for emission {emission}"
#                 )

#         except Exception as e:
#             logger.error(f"Failed to map emission {emission}: {e}")


# def add_emissions(network: Network, db: PlexosDB) -> None:
#     """
#     Maps emissions data from the PLEXOS database to the PyPSA network.

#     Parameters
#     ----------
#     network : pypsa.Network
#         The PyPSA network to which emissions data will be added.
#     db : PlexosDB
#         The PLEXOS database containing emissions data.

#     Notes
#     -----
#     - Emission coefficients are applied to generators in the PyPSA network.
#     - Emission budgets are added as global constraints in the PyPSA network.
#     """
#     logger.info("Mapping emissions data from PLEXOS to PyPSA network.")

#     # Retrieve all emissions from the PLEXOS database
#     emissions = db.list_objects_by_class(ClassEnum.Emission)
#     logger.info(f"Found {len(emissions)} emissions in db")

#     for emission in emissions:
#         try:
#             # Get emission properties
#             props = db.get_object_properties(ClassEnum.Emission, emission)
#             logger.debug(f"Emission: {emission}, Properties: {props}")

#             # Extract relevant properties
#             emission_factor = next(
#                 (float(p["value"]) for p in props if p["property"] == "Emission Rate"),
#                 None,
#             )
#             if emission_factor is None:
#                 logger.warning(f"No 'Emission Rate' found for emission {emission}")

#             emission_budget = next(
#                 (
#                     float(p["value"])
#                     for p in props
#                     if p["property"] == "Emission Budget"
#                 ),
#                 None,
#             )

#             # Map emission factors to generators
#             memberships = db.get_memberships_system(
#                 emission, object_class=ClassEnum.Emission
#             )
#             logger.debug(f"Emission: {emission}, Memberships: {memberships}")

#             for membership in memberships:
#                 if (
#                     membership["class"] == "Generator"
#                     and membership["name"] in network.generators.index
#                 ):
#                     network.generators.loc[membership["name"], "emission_factor"] = (
#                         emission_factor
#                     )
#                     logger.info(
#                         f"Applied emission factor of {emission_factor} to generator {membership['name']}"
#                     )

#             # Add emission budgets as global constraints
#             if emission_budget is not None:
#                 network.global_constraints.loc[emission] = {
#                     "type": "primary_energy",
#                     "sense": "<=",
#                     "constant": emission_budget,
#                     "carrier_attribute": "emissions",
#                 }
#                 logger.info(
#                     f"Added emission budget of {emission_budget} for emission {emission}"
#                 )

#         except Exception as e:
#             logger.error(f"Failed to map emission {emission}: {e}")


# def add_emissions(network: Network, db: PlexosDB):
#     """
#     Adds emission data from PlexosDB to the PyPSA network.

#     Parameters:
#         db: PlexosDB instance
#         network: PyPSA Network instance
#     """
#     emissions: dict[str, list[dict[str, str | float | None]]] = {}

#     # Step 1: Get all Emission objects
#     emission_objects = db.list_objects_by_class(ClassEnum.Emission)

#     for emission in emission_objects:
#         emission_name = emission["name"]
#         print(f"Processing emission: {emission_name}")

#         # Step 2: Get memberships for this emission (which generators have this emission)
#         memberships = db.get_memberships_system(
#             emission_name, object_class=ClassEnum.Emission
#         )

#         # Get the emission properties: Price and Shadow Price
#         try:
#             emission_properties = db.get_object_properties(
#                 ClassEnum.Emission, emission_name
#             )
#         except Exception as e:
#             print(f"Error retrieving properties for emission {emission_name}: {e}")
#             continue

#         # Extract Price and Shadow Price if they exist
#         price = None
#         shadow_price = None
#         for prop in emission_properties:
#             if prop["property"] == "Price":
#                 price = prop["value"]
#             if prop["property"] == "Shadow Price":
#                 shadow_price = prop["value"]

#         # Step 3: For each membership, link emission to a generator in PyPSA
#         for membership in memberships:
#             if membership["class"] == "Generator":
#                 generator_name = membership["name"]

#                 if generator_name in network.generators.index:
#                     # If the generator exists in the PyPSA network, add the emission information
#                     if generator_name not in emissions:
#                         emissions[generator_name] = []

#                     emissions[generator_name].append(
#                         {
#                             "emission_name": emission_name,
#                             "price": price,
#                             "shadow_price": shadow_price,
#                         }
#                     )
#                 else:
#                     print(f"Generator {generator_name} not found in PyPSA network.")

#     # Step 4: Add emissions to PyPSA network's generator attributes
#     for generator_name, emission_data in emissions.items():
#         # Ensure the network has a column for emissions if it doesn't exist yet
#         if "emissions" not in network.generators.columns:
#             network.generators["emissions"] = None

#         network.generators.loc[generator_name, "emissions"] = emission_data
#         print(f"Added emissions for {generator_name}: {emission_data}")
