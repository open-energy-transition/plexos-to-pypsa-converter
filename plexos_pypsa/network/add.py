import logging

from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
