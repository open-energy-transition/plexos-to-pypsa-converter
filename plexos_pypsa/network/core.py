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

        # add bus to network, set carrier as "AC"
        # NOTE: skipping adding v_nom because none of the AEMO nodes have voltage values
        network.add("Bus", name=node, carrier="AC")
    print(f"Added {len(nodes)} buses")


def add_snapshots(network: Network, path: str):
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
