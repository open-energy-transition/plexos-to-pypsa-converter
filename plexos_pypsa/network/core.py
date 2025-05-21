import logging
import os

import pandas as pd  # type: ignore
from plexosdb import PlexosDB  # type: ignore
from plexosdb.enums import ClassEnum  # type: ignore
from pypsa import Network  # type: ignore

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
