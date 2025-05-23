import logging

import pandas as pd  # type: ignore
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


def parse_lines_flow(db: PlexosDB, network):
    """
    Parse Min/Max Flow and Min/Max Rating for lines from the PlexosDB and return two DataFrames:
    - min_flow: index=network.snapshots, columns=line names
    - max_flow: index=network.snapshots, columns=line names

    Uses time-specified values if available; otherwise, defaults to non-time-specified values.
    If no non-time-specified value exists, uses the first value in the properties (even if time-specified).
    If a Rating exists, it replaces the Flow value.
    """

    snapshots = network.snapshots
    lines = network.links.index

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

    def extract_time_series(props, flow_name, rating_name):
        ts = pd.Series(index=snapshots, dtype=float)
        # Prefer rating if available, otherwise flow
        # 1. Try time-specified rating
        time_rating = props[
            (props["property"] == rating_name) & props["t_date_from"].notnull()
        ]
        # 2. Try time-specified flow
        time_flow = props[
            (props["property"] == flow_name) & props["t_date_from"].notnull()
        ]
        # 3. Try non-time-specified rating
        non_time_rating = props[
            (props["property"] == rating_name) & props["t_date_from"].isnull()
        ]
        # 4. Try non-time-specified flow
        non_time_flow = props[
            (props["property"] == flow_name) & props["t_date_from"].isnull()
        ]
        # 5. All rating
        all_rating = props[props["property"] == rating_name]
        # 6. All flow
        all_flow = props[props["property"] == flow_name]

        # Fill time-specified rating
        for _, p in time_rating.iterrows():
            t_from = (
                pd.to_datetime(p["t_date_from"])
                if pd.notnull(p["t_date_from"])
                else None
            )
            t_to = (
                pd.to_datetime(p["t_date_to"]) if pd.notnull(p["t_date_to"]) else None
            )
            if t_from is None:
                continue
            if t_to:
                mask = (snapshots >= t_from) & (snapshots <= t_to)
            else:
                mask = snapshots == t_from
            ts.loc[mask] = float(p["value"])
        # Fill time-specified flow only where ts is still nan
        for _, p in time_flow.iterrows():
            t_from = (
                pd.to_datetime(p["t_date_from"])
                if pd.notnull(p["t_date_from"])
                else None
            )
            t_to = (
                pd.to_datetime(p["t_date_to"]) if pd.notnull(p["t_date_to"]) else None
            )
            if t_from is None:
                continue
            if t_to:
                mask = (snapshots >= t_from) & (snapshots <= t_to)
            else:
                mask = snapshots == t_from
            ts.loc[mask & ts.isnull()] = float(p["value"])
        # Fill with non-time-specified rating if any
        if ts.isnull().any() and not non_time_rating.empty:
            default_val = float(non_time_rating.iloc[0]["value"])
            ts = ts.fillna(default_val)
        # Fill with non-time-specified flow if any
        if ts.isnull().any() and not non_time_flow.empty:
            default_val = float(non_time_flow.iloc[0]["value"])
            ts = ts.fillna(default_val)
        # Fallback: any rating
        if ts.isnull().any() and not all_rating.empty:
            fallback_val = float(all_rating.iloc[0]["value"])
            ts = ts.fillna(fallback_val)
        # Fallback: any flow
        if ts.isnull().any() and not all_flow.empty:
            fallback_val = float(all_flow.iloc[0]["value"])
            ts = ts.fillna(fallback_val)
        # If still nan, fill with nan
        return ts

    for line in lines:
        props = merged[merged["line"] == line]
        min_flow_series[line] = extract_time_series(props, "Min Flow", "Min Rating")
        max_flow_series[line] = extract_time_series(props, "Max Flow", "Max Rating")

    min_flow_df = pd.DataFrame(min_flow_series, index=snapshots)
    max_flow_df = pd.DataFrame(max_flow_series, index=snapshots)

    return min_flow_df, max_flow_df
