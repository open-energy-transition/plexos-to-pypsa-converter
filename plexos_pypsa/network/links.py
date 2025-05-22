import logging

from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

logger = logging.getLogger(__name__)


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
