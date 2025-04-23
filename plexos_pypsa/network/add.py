import logging
from statistics import mean

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
        )  # default: 110kV
        network.add("Bus", name=node, v_nom=voltage)
    print(f"Added {len(nodes)} buses")


def add_generators(network: Network, db: PlexosDB):
    """
    Adds generators from a Plexos database to a PyPSA network.

    This function retrieves generator objects from the provided Plexos database,
    extracts their properties, and adds them to the PyPSA network. If a generator
    lacks properties or an associated bus, it is either skipped or assigned default
    values. If a generator has no properties at all, it is skipped and reported at
    the end.

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
    - Generators without a "Max Capacity" property will have their `p_nom` set to 0.0.
    - Generators without an associated bus will be assigned to a default bus named "default".
    - A warning is printed for each generator missing required data.
    - A summary of skipped generators is printed at the end.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_generators(network, db)
    """
    empty_generators = []
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
            print(f"Warning: 'Max Capacity' not found for {gen}")
            p_max = 0.0  # Or continue if you want to skip those too

        # Find associated bus/node
        bus = find_bus_for_object(db, gen, ClassEnum.Generator)
        if bus is None:
            print(f"Warning: No associated bus found for generator {gen}")
            bus = "default"

        network.add("Generator", gen, bus=bus, p_nom=p_max)

    # Report skipped generators
    if empty_generators:
        print(f"\nSkipped {len(empty_generators)} generators with no properties:")
        for g in empty_generators:
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

            # Get properties from PLEXOS
            max_volume = get_prop("Max Volume")  # in 1000 m³
            min_volume = get_prop("Min Volume")  # in 1000 m³
            initial_volume = get_prop("Initial Volume")  # in 1000 m³
            natural_inflow = get_prop("Natural Inflow")  # in cumec

            # Calculate energy capacity (e.g., using max_volume - min_volume)
            if max_volume is not None and min_volume is not None:
                energy_capacity = max_volume - min_volume  # in 1000 m³
            else:
                logger.warning(f"Skipping {su}: missing volume information")
                skipped_units.append(su)
                continue

            # Assume power capacity is proportional to energy capacity if not provided
            power_capacity = energy_capacity / 10  # Example: arbitrary ratio

            # Default efficiencies if not provided
            efficiency_store = 0.9  # 90% efficiency
            efficiency_dispatch = 0.9  # 90% efficiency

            # Add storage unit to the PyPSA network
            network.storage_units.loc[su] = {
                "bus": bus,
                "p_nom": power_capacity,
                "max_hours": energy_capacity / power_capacity,
                "efficiency_store": efficiency_store,
                "efficiency_dispatch": efficiency_dispatch,
                "carrier": "hydro",  # Assuming hydro storage
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


def add_lines(network: Network, db: PlexosDB):
    """
    Adds transmission lines to the given network based on data from the database.

    This function retrieves line objects from the database, extracts their properties,
    and adds them to the network as transmission lines. It ensures that the lines
    connect valid buses in the network.

    Parameters
    ----------
    network : pypsa.Network
        The network object to which the transmission lines will be added.
    db : Database
        The database object containing line data and their properties.

    Notes
    -----
    - The function retrieves the "From Nodes" and "To Nodes" for each line to determine
      the buses the line connects.
    - Default values are used for `length`, `resistance (r)`, `reactance (x)`, and
      `thermal rating (s_nom)` if these properties are not found in the database.

    Default Values
    --------------
    - Length: 1
    - Resistance (r): 0.05
    - Reactance (x): 0.2
    - Thermal Rating (s_nom): 100

    Prints
    ------
    str
        A message indicating the number of transmission lines added to the network.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> add_lines(network, db)
    Added 10 transmission lines
    """
    lines = db.list_objects_by_class(ClassEnum.Line)
    for line in lines:
        props = db.get_object_properties(ClassEnum.Line, line)
        node_from = next(
            (
                m["name"]
                for m in db.get_memberships_system(line, object_class=ClassEnum.Line)
                if m["collection_name"] == "From Nodes"
            ),
            None,
        )
        node_to = next(
            (
                m["name"]
                for m in db.get_memberships_system(line, object_class=ClassEnum.Line)
                if m["collection_name"] == "To Nodes"
            ),
            None,
        )
        length = next(
            (float(p["value"]) for p in props if p["property"] == "Length"), 1
        )
        r = next(
            (float(p["value"]) for p in props if p["property"] == "Resistance"), 0.05
        )
        x = next(
            (float(p["value"]) for p in props if p["property"] == "Reactance"), 0.2
        )
        s_nom = next(
            (float(p["value"]) for p in props if p["property"] == "Thermal Rating"), 100
        )

        if node_from in network.buses and node_to in network.buses:
            network.add(
                "Line",
                name=line,
                bus0=node_from,
                bus1=node_to,
                length=length,
                r=r,
                x=x,
                s_nom=s_nom,
            )
    print(f"Added {len(lines)} transmission lines")


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
                        network.generators.loc[membership["name"], "p_nom_max"] = rhs
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
