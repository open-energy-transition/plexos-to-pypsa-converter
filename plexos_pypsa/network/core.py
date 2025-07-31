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
        # add bus to network, set carrier as "AC"
        network.add("Bus", name=node, carrier="AC")
    print(f"Added {len(nodes)} buses")


def add_snapshots(network: Network, path: str):
    """
    Reads demand data to determine time resolution and creates unified time series
    to set as the network snapshots. Handles both directory-based and single CSV formats.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    path : str
        Path to the folder containing raw demand data files, or path to a single CSV file.
    """
    # Check if path is a single file or directory
    if os.path.isfile(path) and path.endswith(".csv"):
        # Single CSV file format (CAISO/SEM style)
        df = pd.read_csv(path)

        # Check if Period column exists (indicates sub-daily resolution)
        if "Period" in df.columns:
            # Create datetime from Year, Month, Day
            df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

            # Determine resolution from Period column
            max_period = df["Period"].max()
            min_period = df["Period"].min()

            # For CAISO/SEM format, periods typically start at 1 and go to 24 (representing hours)
            if min_period == 1 and max_period == 24:
                # 24 periods = hourly resolution
                resolution = 60  # 60 minutes per hour
            elif min_period == 1 and max_period == 48:
                # 48 periods = 30-minute resolution
                resolution = 30
            elif min_period == 1 and max_period == 96:
                # 96 periods = 15-minute resolution
                resolution = 15
            elif min_period == 1 and max_period == 4:
                # 4 periods = 6-hour resolution
                resolution = 6 * 60  # 360 minutes
            else:
                # Calculate resolution based on periods per day
                resolution = int(24 * 60 / max_period)  # minutes per period

            print(
                f"  - Detected {max_period} periods per day, using {resolution}-minute resolution"
            )

            # Create time series with proper resolution
            unique_dates = (
                df[["Year", "Month", "Day"]]
                .drop_duplicates()
                .sort_values(["Year", "Month", "Day"])
            )
            all_times = []

            for _, row in unique_dates.iterrows():
                import datetime

                date = datetime.datetime(
                    year=int(row["Year"]), month=int(row["Month"]), day=int(row["Day"])
                )
                # Create periods starting from the beginning of the day
                # Period 1 = 00:00, Period 2 = 01:00 (for hourly), etc.
                daily_times = pd.date_range(
                    start=date, periods=max_period, freq=f"{resolution}min"
                )
                all_times.extend(daily_times.tolist())

            # Set the time series as network snapshots
            network.set_snapshots(sorted(all_times))

        else:
            # Simple daily or other resolution without Period column
            df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])
            unique_dates_list = sorted(df["datetime"].unique())
            network.set_snapshots(unique_dates_list)

    elif os.path.isdir(path):
        # Directory with multiple CSV files
        all_times_list = []

        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)

                # Check if this is a CAISO/SEM format file (has Period column)
                if "Period" in df.columns:
                    print(f"  - Processing CAISO/SEM format file: {file}")
                    # Create datetime from Year, Month, Day
                    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

                    # Determine resolution from Period column
                    max_period = df["Period"].max()
                    min_period = df["Period"].min()

                    # For CAISO/SEM format, periods typically start at 1 and go to 24 (representing hours)
                    if min_period == 1 and max_period == 24:
                        # 24 periods = hourly resolution
                        resolution = 60  # 60 minutes per hour
                    elif min_period == 1 and max_period == 48:
                        # 48 periods = 30-minute resolution
                        resolution = 30
                    elif min_period == 1 and max_period == 96:
                        # 96 periods = 15-minute resolution
                        resolution = 15
                    elif min_period == 1 and max_period == 4:
                        # 4 periods = 6-hour resolution
                        resolution = 6 * 60  # 360 minutes
                    else:
                        # Calculate resolution based on periods per day
                        resolution = int(24 * 60 / max_period)  # minutes per period

                    print(
                        f"    - Detected {max_period} periods per day, using {resolution}-minute resolution"
                    )

                    # Create time series with proper resolution
                    unique_dates = (
                        df[["Year", "Month", "Day"]]
                        .drop_duplicates()
                        .sort_values(["Year", "Month", "Day"])
                    )

                    for _, row in unique_dates.iterrows():
                        import datetime

                        date = datetime.datetime(
                            year=int(row["Year"]),
                            month=int(row["Month"]),
                            day=int(row["Day"]),
                        )
                        # Create periods starting from the beginning of the day
                        # Period 1 = 00:00, Period 2 = 01:00 (for hourly), etc.
                        daily_times = pd.date_range(
                            start=date, periods=max_period, freq=f"{resolution}min"
                        )
                        all_times_list.append(daily_times)

                else:
                    # Original format: columns represent time periods
                    print(f"  - Processing traditional format file: {file}")
                    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])
                    df.set_index("datetime", inplace=True)

                    # Normalize column names to handle both cases (e.g., 1, 2, ...48 or 01, 02, ...48)
                    df.columns = pd.Index(
                        [
                            str(int(col))
                            if col.strip().isdigit()
                            and col not in {"Year", "Month", "Day"}
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
                        # Default to daily resolution
                        resolution = 24 * 60  # daily
                        print(
                            f"    - File has {len(non_date_columns)} columns, using daily resolution"
                        )

                    # Create a time series for this file
                    times = pd.date_range(
                        start=df.index.min(),
                        end=df.index.max()
                        + pd.Timedelta(days=1)
                        - pd.Timedelta(minutes=resolution),
                        freq=f"{resolution}min",
                    )
                    all_times_list.append(times)

        # Combine all time series into a unified time series
        if all_times_list:
            unified_times = (
                pd.concat([pd.Series(times) for times in all_times_list])
                .drop_duplicates()
                .sort_values()
            )
            # Set the unified time series as the network snapshots
            network.set_snapshots(unified_times.tolist())
        else:
            raise ValueError("No valid CSV files found in directory")
    else:
        raise ValueError(f"Path must be either a CSV file or directory: {path}")


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


def add_carriers(network: Network, db: PlexosDB):
    """
    Adds carriers to the PyPSA network.
    - Always adds 'AC' as a carrier.
    - Adds all fuels from the PlexosDB as carriers.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        A PlexosDB object containing the database connection.
    """
    # Add 'AC' carrier
    if "AC" not in network.carriers.index:
        network.add("Carrier", name="AC")

    # Add "Solar" and "Wind" carriers
    if "Solar" not in network.carriers.index:
        network.add("Carrier", name="Solar")
    if "Wind" not in network.carriers.index:
        network.add("Carrier", name="Wind")

    # Add all fuels as carriers
    fuels = db.list_objects_by_class(ClassEnum.Fuel)
    for fuel in fuels:
        if fuel not in network.carriers.index:
            network.add("Carrier", name=fuel)

    print(f"Added carriers: ['AC', 'Solar', 'Wind'] + {fuels}")


def parse_demand_data(demand_source, bus_mapping=None):
    """
    Parse demand data from various formats and return a standardized DataFrame.

    Parameters
    ----------
    demand_source : str or dict
        - If str: Path to a directory containing individual CSV files per bus/node (original format)
        - If str: Path to a single CSV file containing all demand data
        - If dict: Pre-loaded demand data with custom structure
    bus_mapping : dict, optional
        Mapping from column names in the source data to bus names in the network.
        Example: {"1": "Bus_001", "2": "Bus_002"} or {"Zone1": "Bus_A"}

    Returns
    -------
    pandas.DataFrame
        DataFrame with DatetimeIndex and columns for each bus/load zone.

    Examples
    --------
    # Directory with individual files
    >>> demand_df = parse_demand_data("/path/to/demand/folder")

    # Single CSV file
    >>> demand_df = parse_demand_data("/path/to/single_demand.csv",
    ...                               bus_mapping={"1": "Zone_1", "2": "Zone_2"})
    """

    if isinstance(demand_source, str):
        if os.path.isdir(demand_source):
            # Original format: directory with individual CSV files
            return _parse_demand_directory(demand_source)
        elif os.path.isfile(demand_source):
            # Single CSV file format
            return _parse_demand_single_file(demand_source, bus_mapping)
        else:
            raise ValueError(f"Demand source path does not exist: {demand_source}")
    else:
        raise ValueError("demand_source must be a string path")


def _parse_demand_directory(directory_path):
    """Parse demand data from directory with individual CSV files per bus or CAISO/SEM format files."""
    demand_data = {}

    for file_name in os.listdir(directory_path):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(directory_path, file_name)

        try:
            # First, peek at the file to determine its format
            df_peek = pd.read_csv(file_path, nrows=1)

            # Check if this is a CAISO/SEM format file (has Year, Month, Day, Period columns)
            if all(
                col in df_peek.columns for col in ["Year", "Month", "Day", "Period"]
            ):
                # This is a CAISO/SEM format file - treat it as a single file
                print(f"  - Detected CAISO/SEM format file: {file_name}")
                single_file_data = _parse_demand_single_file(file_path)

                # Merge the data from this file
                for col in single_file_data.columns:
                    # Use filename as prefix for column names to avoid conflicts
                    file_prefix = file_name.replace(".csv", "")
                    column_name = f"{file_prefix}_{col}"
                    demand_data[column_name] = single_file_data[col]

            else:
                # Original format: individual bus files with format {bus}_*.csv
                bus_name = file_name.split("_")[0]

                # Read the load file (original format without Period column)
                df = pd.read_csv(file_path, index_col=["Year", "Month", "Day"])
                df = df.reset_index()
                df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

                # Normalize column names
                df.columns = pd.Index(
                    [
                        str(int(col))
                        if col.strip().isdigit()
                        and col not in {"Year", "Month", "Day", "datetime"}
                        else col
                        for col in df.columns
                    ]
                )

                # Get time columns (exclude datetime columns)
                time_columns = [
                    col
                    for col in df.columns
                    if col not in {"Year", "Month", "Day", "datetime"}
                ]

                # Convert to long format
                df_long = df.melt(
                    id_vars=["datetime"],
                    value_vars=time_columns,
                    var_name="period",
                    value_name="load",
                )

                # Create proper datetime index
                df_long = _create_datetime_index(df_long, len(time_columns))
                demand_data[bus_name] = df_long["load"]

        except Exception as e:
            logger.warning(f"Failed to parse demand file {file_name}: {e}")
            continue

    if demand_data:
        result_df = pd.DataFrame(demand_data)

        # Add metadata - check if any files were CAISO/SEM format
        has_iterations = any("iteration_" in col for col in result_df.columns)
        result_df._format_type = "iteration" if has_iterations else "zone"
        result_df._num_iterations = None

        return result_df
    else:
        raise ValueError("No valid demand files found in directory")


def _parse_demand_single_file(file_path, bus_mapping=None):
    """Parse demand data from a single CSV file containing all load zones."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read demand file {file_path}: {e}")

    # Identify datetime columns
    datetime_cols = []
    if "Year" in df.columns:
        datetime_cols.append("Year")
    if "Month" in df.columns:
        datetime_cols.append("Month")
    if "Day" in df.columns:
        datetime_cols.append("Day")
    if "Period" in df.columns:
        # Period is typically the time period within a day
        period_col = "Period"
    else:
        period_col = None

    # Create datetime column
    if datetime_cols:
        df["datetime"] = pd.to_datetime(df[datetime_cols])
    else:
        raise ValueError("Could not identify datetime columns in demand file")

    # Identify load zone columns (everything except datetime and period columns)
    exclude_cols = datetime_cols + ["datetime"]
    if period_col:
        exclude_cols.append(period_col)

    load_columns = [col for col in df.columns if col not in exclude_cols]

    if not load_columns:
        raise ValueError("No load zone columns found in demand file")

    # Detect if this is an iteration-based format
    # Check if all load columns are numeric strings (indicating iterations)
    is_iteration_format = all(
        col.strip().isdigit()
        or (isinstance(col, (int, float)) and str(col).replace(".", "").isdigit())
        for col in load_columns
    )

    if is_iteration_format:
        print(
            f"  - Detected iteration-based format with {len(load_columns)} iterations"
        )
        # Sort columns numerically for proper iteration order
        load_columns = sorted(load_columns, key=lambda x: int(float(str(x))))
    else:
        print(f"  - Detected zone-based format with {len(load_columns)} zones")

    # Apply bus mapping if provided
    original_columns = load_columns.copy()
    if bus_mapping:
        # Rename columns according to mapping
        df = df.rename(columns=bus_mapping)
        load_columns = [bus_mapping.get(str(col), str(col)) for col in load_columns]

    # If there's a Period column, we need to convert to long format first
    if period_col:
        # For iteration format, we need to handle each iteration separately
        if is_iteration_format:
            demand_data = {}

            # Process each iteration
            for i, orig_col in enumerate(original_columns):
                # Create iteration-specific data
                iteration_data = df[["datetime", period_col, orig_col]].copy()
                iteration_data = iteration_data.sort_values(["datetime", period_col])

                # Create full datetime index
                periods_per_day = (
                    iteration_data.groupby("datetime")[period_col].count().iloc[0]
                )
                min_period = iteration_data[period_col].min()
                max_period = iteration_data[period_col].max()

                # For CAISO/SEM format: periods 1-24 represent hours 0-23
                if min_period == 1 and max_period == 24:
                    resolution = 60  # hourly
                elif min_period == 1 and max_period == 48:
                    resolution = 30  # 30-minute
                elif min_period == 1 and max_period == 96:
                    resolution = 15  # 15-minute
                else:
                    resolution = int(24 * 60 / periods_per_day)  # calculate minutes

                print(
                    f"    - Processing iteration {orig_col}: {periods_per_day} periods/day, {resolution}min resolution"
                )

                # Create time offsets (Period 1 = hour 0, Period 2 = hour 1, etc.)
                iteration_data["time_offset"] = pd.to_timedelta(
                    (iteration_data[period_col] - 1) * resolution, unit="m"
                )
                iteration_data["full_datetime"] = (
                    iteration_data["datetime"] + iteration_data["time_offset"]
                )

                iteration_data = iteration_data.set_index("full_datetime")

                # Store with iteration identifier
                iteration_key = f"iteration_{int(float(str(orig_col)))}"
                demand_data[iteration_key] = iteration_data[orig_col]

        else:
            # Original zone-based processing
            # Melt the DataFrame to long format
            df_long = df.melt(
                id_vars=["datetime"] + ([period_col] if period_col else []),
                value_vars=original_columns,
                var_name="load_zone",
                value_name="load",
            )

            # Group by datetime and load_zone, then create time series
            demand_data = {}
            for zone in load_columns:
                zone_data = df_long[df_long["load_zone"] == zone].copy()
                zone_data = zone_data.sort_values(
                    ["datetime", period_col] if period_col else ["datetime"]
                )

                # Create full datetime index
                if period_col:
                    # Determine resolution from number of periods per day
                    periods_per_day = (
                        zone_data.groupby("datetime")[period_col].count().iloc[0]
                    )
                    min_period = zone_data[period_col].min()
                    max_period = zone_data[period_col].max()

                    # For CAISO/SEM format: periods 1-24 represent hours 0-23
                    if min_period == 1 and max_period == 24:
                        resolution = 60  # hourly
                    elif min_period == 1 and max_period == 48:
                        resolution = 30  # 30-minute
                    elif min_period == 1 and max_period == 96:
                        resolution = 15  # 15-minute
                    else:
                        resolution = int(24 * 60 / periods_per_day)  # calculate minutes

                    # Create time offsets (Period 1 = hour 0, Period 2 = hour 1, etc.)
                    zone_data["time_offset"] = pd.to_timedelta(
                        (zone_data[period_col] - 1) * resolution, unit="m"
                    )
                    zone_data["full_datetime"] = (
                        zone_data["datetime"] + zone_data["time_offset"]
                    )
                else:
                    zone_data["full_datetime"] = zone_data["datetime"]

                zone_data = zone_data.set_index("full_datetime")
                demand_data[zone] = zone_data["load"]
    else:
        # Simple case: each row is a time point, columns are load zones
        df = df.set_index("datetime")
        if is_iteration_format:
            demand_data = {}
            for i, orig_col in enumerate(original_columns):
                iteration_key = f"iteration_{int(float(str(orig_col)))}"
                demand_data[iteration_key] = df[orig_col]
        else:
            demand_data = {col: df[col] for col in load_columns}

    # Add metadata about format type
    demand_df = pd.DataFrame(demand_data)
    demand_df._format_type = "iteration" if is_iteration_format else "zone"
    demand_df._num_iterations = len(load_columns) if is_iteration_format else None

    return demand_df


def _create_datetime_index(df_long, num_periods):
    """Create proper datetime index from melted demand data."""
    if num_periods == 24:
        resolution = 60  # hourly
    elif num_periods == 48:
        resolution = 30  # 30-minute
    elif num_periods == 96:
        resolution = 15  # 15-minute
    else:
        resolution = int(24 * 60 / num_periods)  # calculate minutes

    # Create time offsets
    df_long["time_offset"] = pd.to_timedelta(
        (df_long["period"].astype(int) - 1) * resolution, unit="m"
    )
    df_long["full_datetime"] = df_long["datetime"] + df_long["time_offset"]

    return df_long.set_index("full_datetime")


def add_loads_flexible(
    network: Network,
    demand_source,
    bus_mapping=None,
    target_node=None,
    aggregate_node_name=None,
):
    """
    Flexible function to add loads to the PyPSA network from various demand data formats.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    demand_source : str
        Path to demand data (directory with individual files or single CSV file).
    bus_mapping : dict, optional
        Mapping from demand data column names to network bus names.
    target_node : str, optional
        If specified, all demand will be assigned to this existing node.
        Example: "SEM" to assign all demand to the SEM node.
    aggregate_node_name : str, optional
        If specified, creates a new node with this name and assigns all demand to it.
        Example: "Load_Aggregate" to create an aggregate load node.

    Examples
    --------
    # Original format (directory) - per-node assignment
    >>> add_loads_flexible(network, "/path/to/demand/folder")

    # Single CSV with numbered zones - per-node assignment
    >>> add_loads_flexible(network, "/path/to/demand.csv",
    ...                    bus_mapping={"1": "Bus_001", "2": "Bus_002"})

    # Assign all demand to specific existing node
    >>> add_loads_flexible(network, "/path/to/demand.csv", target_node="SEM")

    # Aggregate all demand to new node
    >>> add_loads_flexible(network, "/path/to/demand.csv",
    ...                    aggregate_node_name="Load_Aggregate")
    """
    print("Parsing demand data...")
    demand_df = parse_demand_data(demand_source, bus_mapping)

    print(f"Found demand data for {len(demand_df.columns)} load zones")
    print(f"Time range: {demand_df.index.min()} to {demand_df.index.max()}")

    # Handle different demand assignment modes
    if target_node is not None:
        # Mode 1: Assign all demand to a specific existing node
        return _add_loads_to_target_node(network, demand_df, target_node)
    elif aggregate_node_name is not None:
        # Mode 2: Create new aggregate node and assign all demand to it
        return _add_loads_to_aggregate_node(network, demand_df, aggregate_node_name)
    else:
        # Mode 3: Default per-node assignment
        return _add_loads_per_node(network, demand_df)


def port_core_network(
    network: Network,
    db: PlexosDB,
    snapshots_source,
    demand_source,
    demand_bus_mapping=None,
    target_node=None,
    aggregate_node_name=None,
    model_name=None,
):
    """
    Comprehensive function to set up the core PyPSA network infrastructure.

    This function combines all core network setup operations:
    - Adds buses from the Plexos database
    - Sets up time snapshots from demand data
    - Adds carriers (AC, Solar, Wind, and all fuels from database)
    - Adds loads with flexible demand data parsing

    Parameters
    ----------
    network : Network
        The PyPSA network to set up.
    db : PlexosDB
        The Plexos database containing network data.
    snapshots_source : str
        Path to demand data for creating time snapshots.
    demand_source : str
        Path to demand data (directory with individual files or single CSV file).
    demand_bus_mapping : dict, optional
        Mapping from demand data column names to network bus names.
    target_node : str, optional
        If specified, all demand will be assigned to this existing node.
        Example: "SEM" to assign all demand to the SEM node.
    aggregate_node_name : str, optional
        If specified, creates a new node with this name and assigns all demand to it.
        Example: "Load_Aggregate" to create an aggregate load node.
    model_name : str, optional
        Name of the specific model to use when multiple models exist in the XML file.
        If None and multiple models exist, an error will be raised.

    Returns
    -------
    dict
        Summary information about the load assignment including mode and statistics.

    Examples
    --------
    >>> network = pypsa.Network()
    >>> db = PlexosDB("path/to/file.xml")
    >>> port_core_network(network, db,
    ...                   snapshots_source="/path/to/demand",
    ...                   demand_source="/path/to/demand")

    # With bus mapping for single CSV format
    >>> port_core_network(network, db,
    ...                   snapshots_source="/path/to/demand.csv",
    ...                   demand_source="/path/to/demand.csv",
    ...                   demand_bus_mapping={"1": "Zone_001", "2": "Zone_002"})

    # Assign all demand to specific node (SEM example)
    >>> port_core_network(network, db,
    ...                   snapshots_source="/path/to/demand.csv",
    ...                   demand_source="/path/to/demand.csv",
    ...                   target_node="SEM")

    # Aggregate all demand to new node (CAISO example)
    >>> port_core_network(network, db,
    ...                   snapshots_source="/path/to/demand.csv",
    ...                   demand_source="/path/to/demand.csv",
    ...                   aggregate_node_name="CAISO_Load")
    """
    print("Setting up core network infrastructure...")

    # Check for multiple models and validate model_name if needed
    print("Checking for multiple models in database...")
    models = db.list_objects_by_class(ClassEnum.Model)

    if len(models) > 1:
        if model_name is None:
            raise ValueError(
                f"Multiple models found in XML file: {models}. Please specify a model_name parameter."
            )
        elif model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found in XML file. Available models: {models}"
            )
        else:
            print(f"  Using specified model: {model_name}")
    elif len(models) == 1:
        if model_name is not None and model_name != models[0]:
            raise ValueError(
                f"Model '{model_name}' not found. Only available model: {models[0]}"
            )
        print(f"  Found single model: {models[0]}")
    else:
        print("  No models found in database")
        if model_name is not None:
            raise ValueError(
                f"Model '{model_name}' not found. No models available in XML file."
            )

    # Step 1: Add buses
    print("1. Adding buses...")
    add_buses(network, db)

    # Step 2: Add snapshots
    print("2. Adding snapshots...")
    add_snapshots(network, snapshots_source)

    # Step 3: Add carriers
    print("3. Adding carriers...")
    add_carriers(network, db)

    # Step 4: Add loads with flexible parsing
    print("4. Adding loads...")
    load_summary = add_loads_flexible(
        network, demand_source, demand_bus_mapping, target_node, aggregate_node_name
    )

    print(
        f"Core network setup complete! Network has {len(network.buses)} buses, "
        f"{len(network.snapshots)} snapshots, {len(network.carriers)} carriers, "
        f"and {len(network.loads)} loads."
    )

    return load_summary


def create_bus_mapping_from_csv(csv_path, network_buses=None, auto_detect=True):
    """
    Create a bus mapping dictionary by analyzing a demand CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the demand CSV file.
    network_buses : list, optional
        List of bus names in the network to match against.
    auto_detect : bool, default True
        Whether to attempt automatic detection of load zone patterns.

    Returns
    -------
    dict
        Mapping from CSV column names to suggested bus names.

    Examples
    --------
    >>> mapping = create_bus_mapping_from_csv("demand.csv")
    >>> print(mapping)  # {"1": "Zone_001", "2": "Zone_002", ...}
    """
    try:
        df = pd.read_csv(csv_path, nrows=5)  # Just read header and a few rows
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Identify datetime columns to exclude
    datetime_cols = ["Year", "Month", "Day", "Period", "datetime"]
    load_columns = [col for col in df.columns if col not in datetime_cols]

    mapping = {}

    if auto_detect:
        for col in load_columns:
            if col.isdigit():
                # Numbered zones: "1" -> "Zone_001"
                mapping[col] = f"Zone_{int(col):03d}"
            elif isinstance(col, str):
                # Named zones: keep as is or clean up
                clean_name = col.replace(" ", "_").replace("-", "_")
                mapping[col] = clean_name
            else:
                # Fallback
                mapping[col] = str(col)

    # If network buses provided, try to match
    if network_buses:
        for original_col in list(mapping.keys()):
            suggested_name = mapping[original_col]

            # Look for exact matches first
            if suggested_name in network_buses:
                continue

            # Look for partial matches
            matches = [
                bus
                for bus in network_buses
                if suggested_name.lower() in bus.lower()
                or bus.lower() in suggested_name.lower()
            ]

            if matches:
                mapping[original_col] = matches[0]
                if len(matches) > 1:
                    print(
                        f"Multiple matches for {original_col}: {matches}. Using {matches[0]}"
                    )

    return mapping


def get_demand_format_info(source_path):
    """
    Analyze demand data source and return format information.

    Parameters
    ----------
    source_path : str
        Path to demand data (directory or CSV file).

    Returns
    -------
    dict
        Information about the demand data format including:
        - format_type: "directory" or "single_file"
        - num_load_zones: number of load zones found
        - time_resolution: estimated time resolution
        - sample_columns: sample of load zone column names
        - suggested_mapping: suggested bus mapping (for single files)
    """
    info = {}

    if os.path.isdir(source_path):
        info["format_type"] = "directory"
        csv_files = [f for f in os.listdir(source_path) if f.endswith(".csv")]
        info["num_files"] = len(csv_files)
        info["num_load_zones"] = len(csv_files)
        info["sample_files"] = csv_files[:5]

        # Analyze one file for time resolution
        if csv_files:
            sample_file = os.path.join(source_path, csv_files[0])
            try:
                df = pd.read_csv(sample_file, nrows=5)
                datetime_cols = ["Year", "Month", "Day", "datetime"]
                time_cols = [col for col in df.columns if col not in datetime_cols]
                info["time_resolution"] = f"{len(time_cols)} periods per day"
            except Exception:
                info["time_resolution"] = "unknown"

    elif os.path.isfile(source_path) and source_path.endswith(".csv"):
        info["format_type"] = "single_file"
        try:
            df = pd.read_csv(source_path, nrows=10)
            datetime_cols = ["Year", "Month", "Day", "Period", "datetime"]
            load_columns = [col for col in df.columns if col not in datetime_cols]

            info["num_load_zones"] = len(load_columns)
            info["sample_columns"] = load_columns[:10]
            info["total_rows"] = len(pd.read_csv(source_path, usecols=[df.columns[0]]))

            # Estimate time resolution
            if "Period" in df.columns:
                max_period = df["Period"].max()
                info["time_resolution"] = f"{max_period} periods per day"
            else:
                info["time_resolution"] = "unknown"

            # Generate suggested mapping
            info["suggested_mapping"] = create_bus_mapping_from_csv(source_path)

        except Exception as e:
            info["error"] = str(e)

    else:
        info["format_type"] = "unknown"
        info["error"] = "Path is neither a directory nor a CSV file"

    return info


def _add_loads_per_node(network: Network, demand_df: pd.DataFrame):
    """
    Default mode: Assign loads to individual nodes based on matching.

    For iteration-based formats, creates multiple loads per node (Load1_{Node}, Load2_{Node}, etc.)
    For zone-based formats, creates one load per zone (Load_{Zone})

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.

    Returns
    -------
    dict
        Summary information about the load assignment.
    """
    loads_added = 0
    loads_skipped = 0

    # Check if this is an iteration-based format
    is_iteration_format = getattr(demand_df, "_format_type", None) == "iteration"
    num_iterations = getattr(demand_df, "_num_iterations", None)

    if is_iteration_format:
        print(f"Processing iteration-based format with {num_iterations} iterations")

        # For iteration-based formats, we need to determine which node to assign to
        # For per-node strategy, we'll assign to the first available node or create a default node
        available_buses = list(network.buses.index)

        if not available_buses:
            # No buses available, create a default node
            default_node = "Node_001"
            network.add("Bus", name=default_node, carrier="AC")
            target_node = default_node
            print(f"  - Created default node: {default_node}")
        else:
            # Use the first available bus
            target_node = available_buses[0]
            print(f"  - Assigning iterations to node: {target_node}")

        # Create one load for each iteration
        for col in demand_df.columns:
            if col.startswith("iteration_"):
                iteration_num = col.split("_")[1]
                load_name = f"Load{iteration_num}_{target_node}"

                network.add("Load", name=load_name, bus=target_node)

                # Add the load time series (align with network snapshots)
                load_series = demand_df[col].reindex(network.snapshots).fillna(0)
                network.loads_t.p_set.loc[:, load_name] = load_series

                loads_added += 1
                print(
                    f"  - Added load time series for {load_name} (bus: {target_node})"
                )

    else:
        print("Processing zone-based format")

        # Original zone-based logic
        for load_zone in demand_df.columns:
            # Check if there's a corresponding bus in the network
            if load_zone in network.buses.index:
                bus_name = load_zone
            else:
                # Try to find a matching bus (case-insensitive or partial match)
                matching_buses = [
                    bus
                    for bus in network.buses.index
                    if bus.lower() == load_zone.lower()
                    or load_zone.lower() in bus.lower()
                    or bus.lower() in load_zone.lower()
                ]

                if matching_buses:
                    bus_name = matching_buses[0]
                    if len(matching_buses) > 1:
                        logger.warning(
                            f"Multiple buses match load zone {load_zone}: {matching_buses}. Using {bus_name}"
                        )
                else:
                    logger.warning(f"No bus found for load zone {load_zone}. Skipping.")
                    loads_skipped += 1
                    continue

            # Add the load to the network
            load_name = f"Load_{load_zone}"
            network.add("Load", name=load_name, bus=bus_name)

            # Add the load time series (align with network snapshots)
            load_series = demand_df[load_zone].reindex(network.snapshots).fillna(0)
            network.loads_t.p_set.loc[:, load_name] = load_series

            loads_added += 1
            print(f"  - Added load time series for {load_name} (bus: {bus_name})")

    print(f"Successfully added {loads_added} loads, skipped {loads_skipped} load zones")

    # Report any network buses without loads (only for zone-based format)
    if not is_iteration_format:
        buses_without_loads = [
            bus
            for bus in network.buses.index
            if not any(
                load.startswith("Load_") and network.loads.loc[load, "bus"] == bus
                for load in network.loads.index
            )
        ]

        if buses_without_loads:
            print(
                f"Warning: {len(buses_without_loads)} buses have no loads: {buses_without_loads[:5]}{'...' if len(buses_without_loads) > 5 else ''}"
            )
    else:
        buses_without_loads = []

    return {
        "mode": "per_node",
        "loads_added": loads_added,
        "loads_skipped": loads_skipped,
        "buses_without_loads": len(buses_without_loads),
        "format_type": "iteration" if is_iteration_format else "zone",
        "target_node": target_node if is_iteration_format else None,
    }


def _add_loads_to_target_node(
    network: Network, demand_df: pd.DataFrame, target_node: str
):
    """
    Assign all demand to a specific existing node.

    For iteration-based formats, creates multiple loads (Load1_{target_node}, Load2_{target_node}, etc.)
    For zone-based formats, creates a single aggregated load

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.
    target_node : str
        Name of the existing node to assign all demand to.

    Returns
    -------
    dict
        Summary information about the load assignment.
    """
    # Verify target node exists
    if target_node not in network.buses.index:
        raise ValueError(
            f"Target node '{target_node}' not found in network buses: {list(network.buses.index)}"
        )

    print(f"Assigning all demand to target node: {target_node}")

    # Check if this is an iteration-based format
    is_iteration_format = getattr(demand_df, "_format_type", None) == "iteration"

    loads_added = 0

    if is_iteration_format:
        # Count actual iteration columns (handle both prefixed and non-prefixed)
        iteration_cols = [col for col in demand_df.columns if "iteration_" in col]
        actual_iterations = len(iteration_cols)

        print(f"Processing iteration-based format with {actual_iterations} iterations")

        # Prepare batch data for efficient DataFrame assignment
        load_time_series_data = {}
        load_names = []

        # Create one load for each iteration and prepare time series data
        for col in iteration_cols:
            # Extract iteration number from column name (handle prefixed names)
            if col.startswith("iteration_"):
                iteration_num = col.split("_")[1]
            else:
                # Handle prefixed columns like "filename_iteration_1"
                iteration_num = col.split("iteration_")[1]

            load_name = f"Load{iteration_num}_{target_node}"
            load_names.append(load_name)

            network.add("Load", name=load_name, bus=target_node)

            # Prepare the load time series (align with network snapshots)
            load_series = demand_df[col].reindex(network.snapshots).fillna(0)
            load_time_series_data[load_name] = load_series

            loads_added += 1
            print(f"  - Added load {load_name} to bus {target_node}")

        # Batch assign all load time series to avoid DataFrame fragmentation
        if load_time_series_data:
            load_time_series_df = pd.DataFrame(load_time_series_data)
            # Use pd.concat to efficiently add all columns at once
            network.loads_t.p_set = pd.concat(
                [network.loads_t.p_set, load_time_series_df], axis=1
            )

        # Calculate total demand for reporting
        total_demand = demand_df[iteration_cols].sum(axis=1)
        peak_demand = total_demand.max()

        print(f"  - Total iterations: {actual_iterations}")
        print(f"  - Peak demand (all iterations): {peak_demand:.2f} MW")

        return {
            "mode": "target_node",
            "format_type": "iteration",
            "target_node": target_node,
            "loads_added": loads_added,
            "iterations_processed": actual_iterations,
            "peak_demand": peak_demand,
        }

    else:
        print("Processing zone-based format - creating aggregated load")

        # Sum all demand across zones to create aggregate demand
        total_demand = demand_df.sum(axis=1)

        # Add single aggregate load to the target node
        load_name = f"Load_Aggregate_{target_node}"
        network.add("Load", name=load_name, bus=target_node)

        # Add the aggregated load time series
        load_series = total_demand.reindex(network.snapshots).fillna(0)
        network.loads_t.p_set.loc[:, load_name] = load_series

        loads_added = 1
        print(f"  - Added aggregated load {load_name} to bus {target_node}")
        print(f"  - Total demand: {len(demand_df.columns)} zones aggregated")
        print(f"  - Peak demand: {total_demand.max():.2f} MW")

        return {
            "mode": "target_node",
            "format_type": "zone",
            "target_node": target_node,
            "load_name": load_name,
            "zones_aggregated": len(demand_df.columns),
            "peak_demand": total_demand.max(),
        }


def _add_loads_to_aggregate_node(
    network: Network, demand_df: pd.DataFrame, aggregate_node_name: str
):
    """
    Create a new aggregate node and assign all demand to it.

    For iteration-based formats, creates multiple loads (Load1_{aggregate_node}, Load2_{aggregate_node}, etc.)
    For zone-based formats, creates a single aggregated load

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.
    aggregate_node_name : str
        Name for the new aggregate node.

    Returns
    -------
    dict
        Summary information about the load assignment.
    """
    # Check if aggregate node already exists
    if aggregate_node_name in network.buses.index:
        logger.warning(
            f"Aggregate node '{aggregate_node_name}' already exists. Using existing node."
        )
    else:
        # Create new aggregate bus
        network.add("Bus", name=aggregate_node_name, carrier="AC")
        print(f"Created new aggregate bus: {aggregate_node_name}")

    print(f"Assigning all demand to aggregate node: {aggregate_node_name}")

    # Check if this is an iteration-based format
    is_iteration_format = getattr(demand_df, "_format_type", None) == "iteration"

    loads_added = 0

    if is_iteration_format:
        # Count actual iteration columns (handle both prefixed and non-prefixed)
        iteration_cols = [col for col in demand_df.columns if "iteration_" in col]
        actual_iterations = len(iteration_cols)

        print(f"Processing iteration-based format with {actual_iterations} iterations")

        # Prepare batch data for efficient DataFrame assignment
        load_time_series_data = {}
        load_names = []

        # Create one load for each iteration and prepare time series data
        for col in iteration_cols:
            # Extract iteration number from column name (handle prefixed names)
            if col.startswith("iteration_"):
                iteration_num = col.split("_")[1]
            else:
                # Handle prefixed columns like "filename_iteration_1"
                iteration_num = col.split("iteration_")[1]

            load_name = f"Load{iteration_num}_{aggregate_node_name}"
            load_names.append(load_name)

            network.add("Load", name=load_name, bus=aggregate_node_name)

            # Prepare the load time series (align with network snapshots)
            load_series = demand_df[col].reindex(network.snapshots).fillna(0)
            load_time_series_data[load_name] = load_series

            loads_added += 1
            print(f"  - Added load {load_name} to bus {aggregate_node_name}")

        # Batch assign all load time series to avoid DataFrame fragmentation
        if load_time_series_data:
            load_time_series_df = pd.DataFrame(load_time_series_data)
            # Use pd.concat to efficiently add all columns at once
            network.loads_t.p_set = pd.concat(
                [network.loads_t.p_set, load_time_series_df], axis=1
            )

        # Calculate total demand for reporting
        total_demand = demand_df[iteration_cols].sum(axis=1)
        peak_demand = total_demand.max()

        print(f"  - Total iterations: {actual_iterations}")
        print(f"  - Peak demand (all iterations): {peak_demand:.2f} MW")

        return {
            "mode": "aggregate_node",
            "format_type": "iteration",
            "aggregate_node": aggregate_node_name,
            "loads_added": loads_added,
            "iterations_processed": actual_iterations,
            "peak_demand": peak_demand,
        }

    else:
        print("Processing zone-based format - creating single aggregated load")

        # Sum all demand across zones to create aggregate demand
        total_demand = demand_df.sum(axis=1)

        # Add single aggregate load to the new node
        load_name = "Load_Aggregate"
        network.add("Load", name=load_name, bus=aggregate_node_name)

        # Add the aggregated load time series
        load_series = total_demand.reindex(network.snapshots).fillna(0)
        network.loads_t.p_set.loc[:, load_name] = load_series

        loads_added = 1
        print(f"  - Added aggregated load {load_name} to bus {aggregate_node_name}")
        print(f"  - Total demand: {len(demand_df.columns)} zones aggregated")
        print(f"  - Peak demand: {total_demand.max():.2f} MW")

        return {
            "mode": "aggregate_node",
            "format_type": "zone",
            "aggregate_node": aggregate_node_name,
            "load_name": load_name,
            "zones_aggregated": len(demand_df.columns),
            "peak_demand": total_demand.max(),
        }



def setup_network(
    network: Network,
    db: PlexosDB,
    snapshots_source,
    demand_source,
    target_node=None,
    aggregate_node_name=None,
    demand_bus_mapping=None,
    timeslice_csv=None,
    vre_profiles_path=None,
    model_name=None,
    inflow_path=None,
):
    """
    Unified network setup function that automatically detects the appropriate mode.

    This function intelligently chooses between three setup modes based on parameters:
    1. Per-node mode: Neither target_node nor aggregate_node_name specified (AEMO scenario)
    2. Target node mode: target_node specified (SEM scenario - loads to target, generators/links keep original assignments)  
    3. Aggregation mode: aggregate_node_name specified (CAISO scenario - everything reassigned to aggregate node)

    Parameters
    ----------
    network : Network
        The PyPSA network to set up.
    db : PlexosDB
        The Plexos database containing network data.
    snapshots_source : str
        Path to demand data for creating time snapshots.
    demand_source : str
        Path to demand data (directory with individual files or single CSV file).
    target_node : str, optional
        If specified, all demand will be assigned to this existing node.
        Example: "SEM" to assign all demand to the SEM node.
    aggregate_node_name : str, optional
        If specified, creates a new node with this name and assigns all demand,
        generators, and links to it. Example: "CAISO_Load_Aggregate".
    demand_bus_mapping : dict, optional
        Mapping from demand data column names to network bus names.
    timeslice_csv : str, optional
        Path to the timeslice CSV file for time-dependent properties.
    vre_profiles_path : str, optional
        Path to the folder containing VRE generation profile files.
    model_name : str, optional
        Name of the specific model to use when multiple models exist in the XML file.
        If None and multiple models exist, an error will be raised.
    inflow_path : str, optional
        Path to the folder containing hydro inflow data files for storage units.
        If provided, natural inflows will be processed and added to hydro storage.

    Returns
    -------
    dict
        Summary information about the network setup including mode and statistics.

    Raises
    ------
    ValueError
        If both target_node and aggregate_node_name are specified.

    Examples
    --------
    # Per-node mode (traditional AEMO)
    >>> setup_network(network, db, snapshots_source=path, demand_source=path)
    
    # Target node mode (SEM scenario)
    >>> setup_network(network, db, snapshots_source=path, demand_source=path, 
    ...               target_node="SEM")
    
    # Aggregation mode (CAISO scenario)
    >>> setup_network(network, db, snapshots_source=path, demand_source=path,
    ...               aggregate_node_name="CAISO_Load_Aggregate")
    """
    # Validate parameter combinations
    if target_node is not None and aggregate_node_name is not None:
        raise ValueError("Cannot specify both target_node and aggregate_node_name. Choose one mode.")

    # Detect mode and print status
    if aggregate_node_name is not None:
        mode = "aggregation"
        print(f"Setting up network with demand aggregation to new node: {aggregate_node_name}")
    elif target_node is not None:
        mode = "target_node"
        print(f"Setting up network with all demand assigned to target node: {target_node}")
    else:
        mode = "per_node"
        print("Setting up network with per-node demand assignment")

    # Check for multiple models and validate model_name if needed
    print("Checking for multiple models in database...")
    models = db.list_objects_by_class(ClassEnum.Model)

    if len(models) > 1:
        if model_name is None:
            raise ValueError(
                f"Multiple models found in XML file: {models}. Please specify a model_name parameter."
            )
        elif model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found in XML file. Available models: {models}"
            )
        else:
            print(f"  Using specified model: {model_name}")
    elif len(models) == 1:
        if model_name is not None and model_name != models[0]:
            raise ValueError(
                f"Model '{model_name}' not found. Only available model: {models[0]}"
            )
        print(f"  Found single model: {models[0]}")
    else:
        print("  No models found in database")
        if model_name is not None:
            raise ValueError(
                f"Model '{model_name}' not found. No models available in XML file."
            )

    # Import required modules (avoid circular imports)
    from plexos_pypsa.network.generators import port_generators, reassign_generators_to_node
    from plexos_pypsa.network.links import port_links, reassign_links_to_node
    from plexos_pypsa.network.storage import add_storage, add_hydro_inflows

    # Step 1: Set up core network (port_core_network handles demand assignment logic)
    print("=" * 60)
    print("STEP 1: Setting up core network")
    print("=" * 60)
    load_summary = port_core_network(
        network,
        db,
        snapshots_source=snapshots_source,
        demand_source=demand_source,
        demand_bus_mapping=demand_bus_mapping,
        target_node=target_node,
        aggregate_node_name=aggregate_node_name,
        model_name=model_name,
    )

    # Step 2: Add storage (batteries, hydro, pumped hydro)
    print("\n" + "=" * 60)
    print("STEP 2: Adding storage units")
    print("=" * 60)
    add_storage(network, db, timeslice_csv)
    
    # For aggregation mode, reassign all storage units to the aggregate node
    if mode == "aggregation":
        print(f"Reassigning all storage units to aggregate node: {aggregate_node_name}")
        for storage_name in network.storage_units.index:
            network.storage_units.loc[storage_name, "bus"] = aggregate_node_name

    # Step 2b: Add hydro inflows if path provided
    if inflow_path and os.path.exists(inflow_path):
        print("\n" + "=" * 60)
        print("STEP 2b: Adding hydro inflows")
        print("=" * 60)
        add_hydro_inflows(network, db, inflow_path)
    elif inflow_path:
        print(f"\nWarning: Inflow path specified but not found: {inflow_path}")
        print("Skipping hydro inflow processing")
    else:
        print("\nNo inflow path specified - storage units will not have natural inflows")

    # Step 3: Add generators
    print("\n" + "=" * 60)
    print("STEP 3: Adding generators")
    print("=" * 60)
    port_generators(
        network, db, timeslice_csv=timeslice_csv, vre_profiles_path=vre_profiles_path
    )
    
    # For aggregation mode, reassign all generators to the aggregate node
    generator_summary = None
    if mode == "aggregation":
        print(f"Reassigning all generators to aggregate node: {aggregate_node_name}")
        generator_summary = reassign_generators_to_node(network, aggregate_node_name)

    # Step 4: Add links/lines
    print("\n" + "=" * 60)
    print("STEP 4: Adding links")
    print("=" * 60)
    port_links(network, db)
    
    # For aggregation mode, reassign all links to/from the aggregate node
    link_summary = None
    if mode == "aggregation":
        print(f"Reassigning all links to/from aggregate node: {aggregate_node_name}")
        link_summary = reassign_links_to_node(network, aggregate_node_name)


    print("\n" + "=" * 60)
    print(f"NETWORK SETUP COMPLETE ({mode.upper()} MODE)")
    print("=" * 60)
    print(f"Final network summary:")
    print(f"  Buses: {len(network.buses)}")
    print(f"  Generators: {len(network.generators)}")
    print(f"  Links: {len(network.links)}")
    print(f"  Batteries: {len(network.storage_units)}")
    print(f"  Loads: {len(network.loads)}")
    print(f"  Snapshots: {len(network.snapshots)}")

    # Add mode information to summary
    load_summary["mode"] = mode
    if mode == "aggregation":
        load_summary["aggregate_node_name"] = aggregate_node_name
        load_summary["generator_summary"] = generator_summary
        load_summary["link_summary"] = link_summary
    elif mode == "target_node":
        load_summary["target_node"] = target_node

    return load_summary
