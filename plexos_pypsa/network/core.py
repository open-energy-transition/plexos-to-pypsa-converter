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
        try:
            props = db.get_object_properties(ClassEnum.Node, node)
        except Exception:
            # If node has no properties, just proceed
            props = []
        # add bus to network, set carrier as "AC"
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
    network.set_snapshots(unified_times.tolist())


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
    """Parse demand data from directory with individual CSV files per bus."""
    demand_data = {}

    for file_name in os.listdir(directory_path):
        if not file_name.endswith(".csv"):
            continue

        # Extract bus name from filename (assumes format: {bus}_*.csv)
        bus_name = file_name.split("_")[0]
        file_path = os.path.join(directory_path, file_name)

        try:
            # Read the load file
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
        return pd.DataFrame(demand_data)
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

    # Apply bus mapping if provided
    if bus_mapping:
        # Rename columns according to mapping
        df = df.rename(columns=bus_mapping)
        load_columns = [bus_mapping.get(col, col) for col in load_columns]

    # If there's a Period column, we need to convert to long format first
    if period_col:
        # Melt the DataFrame to long format
        df_long = df.melt(
            id_vars=["datetime"] + ([period_col] if period_col else []),
            value_vars=load_columns,
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
                if periods_per_day == 24:
                    resolution = 60  # hourly
                elif periods_per_day == 48:
                    resolution = 30  # 30-minute
                elif periods_per_day == 96:
                    resolution = 15  # 15-minute
                else:
                    resolution = int(24 * 60 / periods_per_day)  # calculate minutes

                # Create time offsets
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
        demand_data = {col: df[col] for col in load_columns}

    return pd.DataFrame(demand_data)


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


def add_loads_flexible(network: Network, demand_source, bus_mapping=None):
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

    Examples
    --------
    # Original format (directory)
    >>> add_loads_flexible(network, "/path/to/demand/folder")

    # Single CSV with numbered zones
    >>> add_loads_flexible(network, "/path/to/demand.csv",
    ...                    bus_mapping={"1": "Bus_001", "2": "Bus_002"})

    # Single CSV with named zones
    >>> add_loads_flexible(network, "/path/to/demand.csv",
    ...                    bus_mapping={"Zone_A": "Bus_A", "Zone_B": "Bus_B"})
    """
    print("Parsing demand data...")
    demand_df = parse_demand_data(demand_source, bus_mapping)

    print(f"Found demand data for {len(demand_df.columns)} load zones")
    print(f"Time range: {demand_df.index.min()} to {demand_df.index.max()}")

    loads_added = 0
    loads_skipped = 0

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

    # Report any network buses without loads
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


def port_core_network(
    network: Network,
    db: PlexosDB,
    snapshots_source,
    demand_source,
    demand_bus_mapping=None,
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
    """
    print("Setting up core network infrastructure...")

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
    add_loads_flexible(network, demand_source, demand_bus_mapping)

    print(
        f"Core network setup complete! Network has {len(network.buses)} buses, "
        f"{len(network.snapshots)} snapshots, {len(network.carriers)} carriers, "
        f"and {len(network.loads)} loads."
    )


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
