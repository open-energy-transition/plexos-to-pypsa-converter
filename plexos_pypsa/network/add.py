from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import find_bus_for_object


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


# --- Add Transmission Lines ---
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
