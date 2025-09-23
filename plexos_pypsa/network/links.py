import logging

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

from plexos_pypsa.db.parse import (
    get_dataid_timeslice_map,
    get_property_active_mask,
    read_timeslice_activity,
)

logger = logging.getLogger(__name__)


def add_links(network: Network, db: PlexosDB):
    """
    Adds transmission links to the given network based on data from the database.

    This function retrieves line objects from the database, extracts their properties,
    and adds them to the network as links. It ensures that the links connect valid buses.

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

        # Ramp limits
        def get_prop(prop, cond=lambda v: True):
            vals = [
                float(p["value"])
                for p in props
                if p["property"] == prop and cond(float(p["value"]))
            ]
            return max(vals) if vals else None

        ramp_limit_up = get_prop("Max Ramp Up", lambda v: v > 0)
        ramp_limit_down = get_prop("Max Ramp Down", lambda v: v > 0)

        # Add link (do not set p_nom, p_min_pu, p_max_pu here)
        network.add(
            "Link",
            name=line,
            bus0=node_from,
            bus1=node_to,
        )
        print(f"- Added link {line} to buses {node_from} and {node_to}")

        # Set ramp limits if available
        if ramp_limit_up is not None:
            network.links.loc[line, "ramp_limit_up"] = ramp_limit_up
        if ramp_limit_down is not None:
            network.links.loc[line, "ramp_limit_down"] = ramp_limit_down
    print(f"Added {len(lines)} links")


def parse_lines_flow(db: PlexosDB, network, timeslice_csv=None):
    """
    Parse Min/Max Flow and Min/Max Rating for lines from the PlexosDB and return:
    - min_flow: index=network.snapshots, columns=line names
    - max_flow: index=network.snapshots, columns=line names
    - fallback_val: dict mapping line name to fallback value used for p_nom

    Uses time-specified values if available; otherwise, defaults to non-time-specified values.
    If a property is linked to a timeslice, use the timeslice activity to set the property for the relevant snapshots (takes precedence over t_date_from/t_date_to).
    """
    snapshots = network.snapshots
    lines = network.links.index
    timeslice_activity = None
    if timeslice_csv is not None:
        timeslice_activity = read_timeslice_activity(timeslice_csv, snapshots)
    dataid_to_timeslice = (
        get_dataid_timeslice_map(db) if timeslice_csv is not None else {}
    )

    # 1. Get line object_id and name
    line_query = """
        SELECT o.object_id, o.name
        FROM t_object o
        JOIN t_class c ON o.class_id = c.class_id
        WHERE c.name = 'Line'
    """
    line_rows = db.query(line_query)
    line_df = pd.DataFrame(line_rows, columns=["object_id", "line"])

    # 2. Get data_id for each line (child_object_id) from t_membership
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

    # 3. Get property values for each data_id (Min/Max Flow, Min/Max Rating), with t_date_from/to
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
        WHERE p.name IN ('Min Flow', 'Max Flow', 'Min Rating', 'Max Rating')
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

    # 4. Merge: line object_id -> data_id -> property info
    merged = pd.merge(prop_df, membership_df, on="membership_id", how="inner")
    merged = pd.merge(
        merged,
        line_df,
        left_on="child_object_id",
        right_on="object_id",
        how="inner",
    )

    # 5. Collect line Series in dictionaries for efficient concatenation
    min_flow_series = {}
    max_flow_series = {}
    fallback_val_dict = {}

    def build_ts(props, prop_name, fallback=None):
        ts = pd.Series(index=snapshots, dtype=float)
        # 1. Apply all time/timeslice-specific entries first
        for _, row in props[props["property"] == prop_name].iterrows():
            is_time_specific = (
                pd.notnull(row.get("t_date_from"))
                or pd.notnull(row.get("t_date_to"))
                or (
                    timeslice_activity is not None
                    and dataid_to_timeslice is not None
                    and row["data_id"] in dataid_to_timeslice
                )
            )
            if is_time_specific:
                mask = get_property_active_mask(
                    row,
                    snapshots,
                    timeslice_activity=timeslice_activity,
                    dataid_to_timeslice=dataid_to_timeslice,
                )
                ts.loc[mask & ts.isnull()] = float(row["value"])
        # 2. Fill remaining NaNs with static (always-active) entries
        for _, row in props[props["property"] == prop_name].iterrows():
            is_time_specific = (
                pd.notnull(row.get("t_date_from"))
                or pd.notnull(row.get("t_date_to"))
                or (
                    timeslice_activity is not None
                    and dataid_to_timeslice is not None
                    and row["data_id"] in dataid_to_timeslice
                )
            )
            if not is_time_specific:
                ts.loc[ts.isnull()] = float(row["value"])
        # 3. Fallback
        if fallback is not None:
            ts = ts.fillna(fallback)
        return ts

    for line in lines:
        props = merged[merged["line"] == line]
        min_ts = build_ts(props, "Min Flow")
        min_rating_ts = build_ts(props, "Min Rating")
        max_ts = build_ts(props, "Max Flow")
        max_rating_ts = build_ts(props, "Max Rating")
        # Prefer rating if available, otherwise flow
        min_final = min_rating_ts.combine_first(min_ts)
        max_final = max_rating_ts.combine_first(max_ts)
        # Fallback: use first available value
        max_fallback = (
            max_final.dropna().iloc[0] if not max_final.dropna().empty else None
        )
        min_flow_series[line] = min_final
        max_flow_series[line] = max_final
        fallback_val_dict[line] = max_fallback

    min_flow_df = pd.DataFrame(min_flow_series, index=snapshots)
    max_flow_df = pd.DataFrame(max_flow_series, index=snapshots)

    return min_flow_df, max_flow_df, fallback_val_dict


def set_link_flows(network: Network, db: PlexosDB, timeslice_csv=None):
    """
    Set the flow limits for links in the network based on data from the database.

    This function retrieves the Min/Max Flow and Min/Max Rating for each link from the
    database and sets the corresponding attributes in the network.

    Parameters
    ----------
    network : pypsa.Network
        The network object to which the links belong.
    db : Database
        The database object containing line data and their properties.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-based property mapping.

    Notes
    -----
    - The function uses the `parse_lines_flow` function to retrieve flow data.
    - It sets `p_min_pu` and `p_max_pu` for each link based on the retrieved data.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> set_link_flows(network, db, "timeslice.csv")
    Set flow limits for 10 links
    """
    min_flow_df, max_flow_df, fallback_val_dict = parse_lines_flow(
        db, network, timeslice_csv
    )

    # Set p_nom based on fallback_val_dict
    for line in network.links.index:
        fallback_val = fallback_val_dict.get(line, None)
        if fallback_val is not None:
            network.links.loc[line, "p_nom"] = fallback_val
            print(f"Set p_nom for link {line} to {fallback_val}.")
        else:
            logger.warning(
                f"Link {line} has no fallback value for p_nom, cannot set p_min_pu and p_max_pu."
            )

    # Assign p_min_pu and p_max_pu to network.links_t, filling NaNs and infs with safe defaults
    p_min_pu = min_flow_df.divide(network.links["p_nom"], axis=1)
    p_max_pu = max_flow_df.divide(network.links["p_nom"], axis=1)

    # Replace inf/-inf with 0 for p_min_pu and 1 for p_max_pu
    p_min_pu = p_min_pu.replace([float("inf"), float("-inf")], 0).fillna(0)
    p_max_pu = p_max_pu.replace([float("inf"), float("-inf")], 1).fillna(1)

    network.links_t.p_min_pu = p_min_pu
    network.links_t.p_max_pu = p_max_pu

    print(f"Set flow limits for {len(network.links)} links")


def port_links(network: Network, db: PlexosDB, timeslice_csv=None, target_node=None):
    """
    Comprehensive function to add links and set all their properties in the PyPSA network.

    This function combines all link-related operations:
    - Adds transmission links from the Plexos database
    - Sets link flow limits (p_min_pu and p_max_pu)
    - Optionally reassigns all links to a specific node

    Parameters
    ----------
    network : Network
        The PyPSA network to which links will be added.
    db : PlexosDB
        The Plexos database containing link/transmission data.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent link properties.
    target_node : str, optional
        If specified, all links will be reassigned to this node after setup.
        This is useful when demand is aggregated to a single node.

    Returns
    -------
    dict or None
        If target_node is specified, returns summary information about reassignment.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> port_links(network, db)

    # With node reassignment for aggregated demand
    >>> reassignment_info = port_links(network, db, target_node="Load_Aggregate")
    """
    print("Starting link porting process...")

    # Step 1: Add links
    print("1. Adding links...")
    add_links(network, db)

    # Step 2: Set link flow limits
    print("2. Setting link flow limits...")
    set_link_flows(network, db, timeslice_csv=timeslice_csv)

    # Step 3: Reassign links to target node if specified
    if target_node:
        print(f"3. Reassigning links to target node: {target_node}")
        return reassign_links_to_node(network, target_node)
    else:
        print("3. Skipping link reassignment (no target node specified)")

    print(f"Link porting complete! Added {len(network.links)} links.")


def reassign_links_to_node(network: Network, target_node: str):
    """
    Reassign all links to connect a specific node.

    This is useful when demand is aggregated to a single node and all links
    need to be connected to the same node for a meaningful optimization.
    For links between different buses, both bus0 and bus1 are set to the target node.

    Parameters
    ----------
    network : Network
        The PyPSA network containing links.
    target_node : str
        Name of the node to assign all links to.

    Returns
    -------
    dict
        Summary information about the reassignment.
    """
    if target_node not in network.buses.index:
        raise ValueError(f"Target node '{target_node}' not found in network buses")

    if len(network.links) == 0:
        print("No links found in network. Skipping link reassignment.")
        return {
            "reassigned_count": 0,
            "target_node": target_node,
            "original_connections": [],
        }

    original_bus0 = network.links["bus0"].copy()
    original_bus1 = network.links["bus1"].copy()
    original_connections = [(b0, b1) for b0, b1 in zip(original_bus0, original_bus1)]
    unique_original_connections = list(set(original_connections))

    # Reassign all links to connect the target node
    # Note: In aggregated scenarios, all links become self-loops on the aggregate node
    network.links["bus0"] = target_node
    network.links["bus1"] = target_node

    reassigned_count = len(network.links)
    print(f"Reassigned {reassigned_count} links to node '{target_node}'")
    print(f"  - Originally {len(unique_original_connections)} unique connections")
    print(f"  - All links now connect {target_node} to {target_node} (self-loops)")

    return {
        "reassigned_count": reassigned_count,
        "target_node": target_node,
        "original_connections": unique_original_connections,
    }
