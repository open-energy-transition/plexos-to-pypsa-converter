import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore


def parse_generator_ratings(db: PlexosDB, network):
    """
    Parse generator ratings from the PlexosDB using SQL and return a DataFrame with index=network.snapshots, columns=generator names.
    If no t_date_to is given, the rating only lasts for the snapshot at t_date_from.
    All snapshots with no rating are filled with Max Capacity.
    """
    snapshots = network.snapshots

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

    # 3. Get property values for each data_id (Rating or Max Capacity), with t_date_from/to
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
        WHERE p.name = 'Rating' OR p.name = 'Max Capacity'
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

    # 5. Create DataFrame with index=snapshots, columns=generator names
    result = pd.DataFrame(index=snapshots)

    for gen in gen_df["generator"]:
        props = merged[merged["generator"] == gen]

        # Get all ratings
        ratings = []
        for _, p in props[props["property"] == "Rating"].iterrows():
            t_from = (
                pd.to_datetime(p["t_date_from"])
                if pd.notnull(p["t_date_from"])
                else None
            )
            t_to = (
                pd.to_datetime(p["t_date_to"]) if pd.notnull(p["t_date_to"]) else None
            )
            ratings.append(
                {
                    "value": float(p["value"]),
                    "from": t_from,
                    "to": t_to,
                }
            )

        # If no ratings, fallback to Max Capacity
        if not ratings:
            maxcap_row = props[props["property"] == "Max Capacity"]
            maxcap = (
                float(maxcap_row["value"].iloc[0]) if not maxcap_row.empty else None
            )
            result[gen] = pd.Series(maxcap, index=snapshots)
            continue

        # Sort by from date
        ratings = sorted(ratings, key=lambda x: x["from"] or pd.Timestamp.min)
        ts = pd.Series(index=snapshots, dtype=float)
        for r in ratings:
            if r["from"] is None:
                continue
            if r["to"]:
                mask = (snapshots >= r["from"]) & (snapshots <= r["to"])
            else:
                # Only set the snapshot that matches t_date_from
                mask = snapshots == r["from"]
            ts.loc[mask] = r["value"]
        # Fill any NaN with Max Capacity
        if ts.isnull().any():
            maxcap_row = props[props["property"] == "Max Capacity"]
            maxcap = (
                float(maxcap_row["value"].iloc[0]) if not maxcap_row.empty else None
            )
            ts = ts.fillna(maxcap)
        result[gen] = ts

    return result
