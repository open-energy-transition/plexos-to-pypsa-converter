import datetime
import logging
from pathlib import Path

import pandas as pd
from plexosdb import PlexosDB
from plexosdb.enums import ClassEnum
from pypsa import Network

from plexos_to_pypsa_converter.db.csv_readers import load_static_properties

# PlexosDB-only imports (archived, not used by CSV workflow)
# These are only needed for the old setup_network() function
try:
    from plexos_to_pypsa_converter.network.constraints import add_constraints_enhanced
except ImportError:
    add_constraints_enhanced = None

try:
    from plexos_to_pypsa_converter.network.generators import (
        port_generators,
        reassign_generators_to_node,
    )
except ImportError:
    port_generators = None
    reassign_generators_to_node = None

try:
    from plexos_to_pypsa_converter.network.lines import port_lines
except ImportError:
    port_lines = None

try:
    from plexos_to_pypsa_converter.network.links import (
        port_links,
        reassign_links_to_node,
    )
except ImportError:
    port_links = None
    reassign_links_to_node = None

try:
    from plexos_to_pypsa_converter.network.storage import (
        add_hydro_inflows,
        add_storage,
    )
except ImportError:
    add_hydro_inflows = None
    add_storage = None

logger = logging.getLogger(__name__)


def add_buses(network: Network, db: PlexosDB):
    """Add buses to the given network based on the nodes retrieved from the database.

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
    """Read demand data to determine time resolution and create unified time series
    to set as the network snapshots. Handle both directory-based and single CSV formats.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    path : str
        Path to the folder containing raw demand data files, or path to a single CSV file.
    """
    # Check if path is a single file or directory
    path_obj = Path(path)
    if path_obj.is_file() and path.endswith(".csv"):
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

    elif path_obj.is_dir():
        # Directory with multiple CSV files
        all_times_list = []

        for file in path_obj.iterdir():
            if file.name.endswith(".csv"):
                file_path = str(file)
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
            msg = "No valid CSV files found in directory"
            raise ValueError(msg)
    else:
        msg = f"Path must be either a CSV file or directory: {path}"
        raise ValueError(msg)


def add_loads(network: Network, path: str):
    """Add loads to the PyPSA network for each bus based on the corresponding {bus}...csv file.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    path : str
        Path to the folder containing raw demand data files.
    """
    for bus in network.buses.index:
        # Find the corresponding load file for the bus
        path_obj = Path(path)
        file_name = next(
            (
                f.name
                for f in path_obj.iterdir()
                if f.name.startswith(f"{bus}_") and f.name.endswith(".csv")
            ),
            None,
        )
        if file_name is None:
            print(f"Warning: No load file found for bus {bus}")
            continue
        file_path = str(path_obj / file_name)

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
            msg = "Unsupported resolution."
            raise ValueError(msg)

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


def discover_carriers_from_db(db: PlexosDB) -> list:
    """Automatically discover all carriers needed for PyPSA network by analyzing PLEXOS database.

    This function examines multiple PLEXOS classes to build a comprehensive list of carriers:
    - Core carriers (AC, conversion)
    - All fuels from Fuel class
    - Renewable carriers derived from generator names
    - Multi-sector carriers (Gas, conversion types)
    - Technology-specific carriers from generator analysis

    Parameters
    ----------
    db : PlexosDB
        PLEXOS database connection

    Returns
    -------
    list
        Complete list of carrier names needed for the network
    """
    carriers = set()

    # 1. Core carriers (always needed)
    carriers.add("AC")  # For electrical buses
    carriers.add("conversion")  # For generators-as-links

    # 2. PLEXOS Fuels (from database)
    try:
        fuels = db.list_objects_by_class(ClassEnum.Fuel)
        carriers.update(fuels)
        print(f"  Found {len(fuels)} fuel carriers from database")
    except Exception as e:
        logger.warning(f"Could not query fuels from database: {e}")

    # 3. Renewable carriers (from generator analysis)
    try:
        generators = db.list_objects_by_class(ClassEnum.Generator)
        renewable_carriers = set()

        for gen_name in generators:
            gen_lower = gen_name.lower()
            if "wind" in gen_lower:
                if "offshore" in gen_lower:
                    renewable_carriers.add("Wind Offshore")
                elif "onshore" in gen_lower:
                    renewable_carriers.add("Wind Onshore")
                else:
                    renewable_carriers.add("Wind")
            elif "solar" in gen_lower or "pv" in gen_lower:
                renewable_carriers.add("Solar")
                if "pv" in gen_lower:
                    renewable_carriers.add("Solar PV")
            elif "hydro" in gen_lower:
                renewable_carriers.add("Hydro")

        carriers.update(renewable_carriers)
        print(
            f"  Found {len(renewable_carriers)} renewable carriers: {sorted(renewable_carriers)}"
        )

    except Exception as e:
        logger.warning(f"Could not analyze generators for renewable carriers: {e}")

    # 4. Multi-sector carriers (gas, conversion types)
    try:
        # Always add core multi-sector carriers since they may be used by multi-sector models
        # Even if we can't detect the specific classes, the conversion carriers are needed
        carriers.add("Gas")
        carriers.add("Gas2Electric")  # For gas-to-electric conversion links
        carriers.add("conversion")  # For generators-as-links conversion
        print("  Added core multi-sector carriers: Gas, Gas2Electric, conversion")

        # Try to detect additional multi-sector components with various class name patterns
        multi_sector_carriers = set()

        # Check for gas components with various possible class names
        possible_gas_classes = [
            "Gas_Node",
            "Gas_Pipeline",
            "Gas_Storage",
            "Gas_Field",
            "Gas_Plant",
            "Gas_Demand",
            "GasNode",
            "GasPipeline",
            "GasStorage",
            "GasField",
            "GasPlant",
            "GasDemand",
        ]

        for class_name in possible_gas_classes:
            try:
                class_enum = getattr(ClassEnum, class_name, None)
                if class_enum and db.list_objects_by_class(class_enum):
                    multi_sector_carriers.update(["Natural Gas", "Gas Network"])
                    break
            except Exception:
                logger.debug(f"Gas class {class_name} not available, trying next")
                continue

        # Check for flow network components
        possible_flow_classes = [
            "Flow_Node",
            "Flow_Path",
            "Flow_Storage",
            "FlowNode",
            "FlowPath",
            "FlowStorage",
        ]

        for class_name in possible_flow_classes:
            try:
                class_enum = getattr(ClassEnum, class_name, None)
                if class_enum and db.list_objects_by_class(class_enum):
                    multi_sector_carriers.update(
                        ["Hydrogen", "Ammonia", "Electric2Hydrogen", "Hydrogen2Ammonia"]
                    )
                    break
            except Exception:
                logger.debug(f"Flow class {class_name} not available, trying next")
                continue

        if multi_sector_carriers:
            carriers.update(multi_sector_carriers)
            print(
                f"  Found additional multi-sector carriers: {sorted(multi_sector_carriers)}"
            )

    except Exception as e:
        logger.warning(f"Could not analyze multi-sector components: {e}")
        # Ensure essential multi-sector carriers are still added even if detection fails
        carriers.update(["Gas", "Gas2Electric", "conversion"])

    # 5. Technology-specific carriers (from detailed generator analysis)
    try:
        generators = db.list_objects_by_class(ClassEnum.Generator)
        tech_carriers = set()
        generator_name_carriers = set()

        for gen_name in generators:
            gen_lower = gen_name.lower()
            # Gas technologies
            if "ccgt" in gen_lower or "combined cycle" in gen_lower:
                tech_carriers.add("Natural Gas CCGT")
            elif "ocgt" in gen_lower or "open cycle" in gen_lower:
                tech_carriers.add("Natural Gas OCGT")
            elif "gas" in gen_lower and "natural" not in gen_lower:
                tech_carriers.add("Natural Gas")

            # Coal technologies
            if "lignite" in gen_lower:
                tech_carriers.add("Lignite")
            elif "hard coal" in gen_lower:
                tech_carriers.add("Hard Coal")
            elif "coal" in gen_lower:
                tech_carriers.add("Coal")

            # Biomass technologies
            if "biomass" in gen_lower:
                if "waste" in gen_lower:
                    tech_carriers.add("Biomass Waste")
                else:
                    tech_carriers.add("Biomass")

            # Other technologies
            if "nuclear" in gen_lower:
                tech_carriers.add("Nuclear")
            elif "oil" in gen_lower:
                tech_carriers.add("Oil")
            elif "solids" in gen_lower:
                tech_carriers.add("Solids Fired")

            # Individual generator names (for generators-as-links functionality)
            # This ensures generator names are available as carriers when needed
            generator_name_carriers.add(gen_name)

        carriers.update(tech_carriers)
        carriers.update(generator_name_carriers)

        if tech_carriers:
            print(f"  Found {len(tech_carriers)} technology-specific carriers")
        print(
            f"  Found {len(generator_name_carriers)} generator-name carriers (for generators-as-links)"
        )

    except Exception as e:
        logger.warning(f"Could not analyze generators for technology carriers: {e}")

    carriers_list = sorted(carriers)
    print(f"  Total carriers discovered: {len(carriers_list)}")
    return carriers_list


def add_carriers(network: Network, db: PlexosDB):
    """Add carriers to the PyPSA network using automatic discovery from PLEXOS database.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    db : PlexosDB
        A PlexosDB object containing the database connection.
    """
    # Discover all carriers from database
    carriers_to_add = discover_carriers_from_db(db)

    # Add each carrier if not already present
    added_carriers = []
    for carrier in carriers_to_add:
        if carrier not in network.carriers.index:
            network.add("Carrier", name=carrier)
            added_carriers.append(carrier)

    print(
        f"Added {len(added_carriers)} carriers automatically: {sorted(added_carriers)}"
    )


def parse_demand_data(demand_source, bus_mapping=None):
    """Parse demand data from various formats and return a standardized DataFrame.

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
        path_obj = Path(demand_source)
        if path_obj.is_dir():
            # Original format: directory with individual CSV files
            return _parse_demand_directory(demand_source)
        elif path_obj.is_file():
            # Single CSV file format
            return _parse_demand_single_file(demand_source, bus_mapping)
        else:
            msg = f"Demand source path does not exist: {demand_source}"
            raise ValueError(msg)
    else:
        msg = "demand_source must be a string path"
        raise TypeError(msg)


def _parse_demand_directory(directory_path):
    """Parse demand data from directory with individual CSV files per bus or CAISO/SEM format files."""
    demand_data = {}
    path_obj = Path(directory_path)

    for file in path_obj.iterdir():
        if not file.name.endswith(".csv"):
            continue

        file_path = str(file)

        try:
            # First, peek at the file to determine its format
            df_peek = pd.read_csv(file_path, nrows=1)

            # Check if this is a CAISO/SEM format file (has Year, Month, Day, Period columns)
            if all(
                col in df_peek.columns for col in ["Year", "Month", "Day", "Period"]
            ):
                # This is a CAISO/SEM format file - treat it as a single file
                print(f"  - Detected CAISO/SEM format file: {file.name}")
                single_file_data = _parse_demand_single_file(file_path)

                # Merge the data from this file
                for col in single_file_data.columns:
                    # Use filename as prefix for column names to avoid conflicts
                    file_prefix = file.stem
                    column_name = f"{file_prefix}_{col}"
                    demand_data[column_name] = single_file_data[col]

            else:
                # Original format: individual bus files with format {bus}_*.csv
                bus_name = file.stem.split("_")[0]

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
            logger.warning(f"Failed to parse demand file {file.name}: {e}")
            continue

    if demand_data:
        result_df = pd.DataFrame(demand_data)

        # Add metadata - check if any files were CAISO/SEM format
        has_iterations = any("iteration_" in col for col in result_df.columns)
        result_df._format_type = "iteration" if has_iterations else "zone"
        result_df._num_iterations = None

        return result_df
    else:
        msg = "No valid demand files found in directory"
        raise ValueError(msg)


def _parse_demand_single_file(file_path, bus_mapping=None):
    """Parse demand data from a single CSV file containing all load zones."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        msg = f"Failed to read demand file {file_path}: {e}"
        raise ValueError(msg) from e

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
        msg = "Could not identify datetime columns in demand file"
        raise ValueError(msg)

    # Identify load zone columns (everything except datetime and period columns)
    exclude_cols = datetime_cols + ["datetime"]
    if period_col:
        exclude_cols.append(period_col)

    load_columns = [col for col in df.columns if col not in exclude_cols]

    if not load_columns:
        msg = "No load zone columns found in demand file"
        raise ValueError(msg)

    # Detect if this is an iteration-based format
    # Check if all load columns are numeric strings (indicating iterations)
    is_iteration_format = all(
        col.strip().isdigit()
        or (isinstance(col, int | float) and str(col).replace(".", "").isdigit())
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
            for _i, orig_col in enumerate(original_columns):
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
            for _i, orig_col in enumerate(original_columns):
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
    load_scenario=None,
):
    """Flexible function to add loads to the PyPSA network from various demand data formats.

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
    load_scenario : str, optional
        For demand data with multiple scenarios (iterations), specify which scenario
        to use. If None, defaults to first scenario. Example: "iteration_1"

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
        return _add_loads_to_target_node(network, demand_df, target_node, load_scenario)
    elif aggregate_node_name is not None:
        # Mode 2: Create new aggregate node and assign all demand to it
        return _add_loads_to_aggregate_node(
            network, demand_df, aggregate_node_name, load_scenario
        )
    else:
        # Mode 3: Default per-node assignment
        return _add_loads_per_node(network, demand_df, load_scenario)


def port_core_network(
    network: Network,
    db: PlexosDB,
    snapshots_source,
    demand_source,
    demand_bus_mapping=None,
    target_node=None,
    aggregate_node_name=None,
    model_name=None,
    load_scenario=None,
    demand_target_node=None,
):
    """Comprehensive function to set up the core PyPSA network infrastructure.

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
    load_scenario : str, optional
        For demand data with multiple scenarios (iterations), specify which scenario
        to use. If None, defaults to first scenario. Example: "iteration_1"

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
            msg = f"Multiple models found in XML file: {models}. Please specify a model_name parameter."
            raise ValueError(msg)
        elif model_name not in models:
            msg = f"Model '{model_name}' not found in XML file. Available models: {models}"
            raise ValueError(msg)
        else:
            print(f"  Using specified model: {model_name}")
    elif len(models) == 1:
        if model_name is not None and model_name != models[0]:
            msg = f"Model '{model_name}' not found. Only available model: {models[0]}"
            raise ValueError(msg)
        print(f"  Found single model: {models[0]}")
    else:
        print("  No models found in database")
        if model_name is not None:
            msg = f"Model '{model_name}' not found. No models available in XML file."
            raise ValueError(msg)

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
    # Determine demand target: demand_target_node takes precedence
    effective_target = demand_target_node if demand_target_node else target_node
    effective_aggregate = None if demand_target_node else aggregate_node_name

    load_summary = add_loads_flexible(
        network,
        demand_source,
        demand_bus_mapping,
        effective_target,
        effective_aggregate,
        load_scenario,
    )

    print(
        f"Core network setup complete! Network has {len(network.buses)} buses, "
        f"{len(network.snapshots)} snapshots, {len(network.carriers)} carriers, "
        f"and {len(network.loads)} loads."
    )

    return load_summary


def create_bus_mapping_from_csv(csv_path, network_buses=None, auto_detect=True):
    """Create a bus mapping dictionary by analyzing a demand CSV file.

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
        msg = f"Failed to read CSV file: {e}"
        raise ValueError(msg) from e

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
    """Analyze demand data source and return format information.

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
    path_obj = Path(source_path)

    if path_obj.is_dir():
        info["format_type"] = "directory"
        csv_files = [f.name for f in path_obj.iterdir() if f.name.endswith(".csv")]
        info["num_files"] = len(csv_files)
        info["num_load_zones"] = len(csv_files)
        info["sample_files"] = csv_files[:5]

        # Analyze one file for time resolution
        if csv_files:
            sample_file = str(path_obj / csv_files[0])
            try:
                df = pd.read_csv(sample_file, nrows=5)
                datetime_cols = ["Year", "Month", "Day", "datetime"]
                time_cols = [col for col in df.columns if col not in datetime_cols]
                info["time_resolution"] = f"{len(time_cols)} periods per day"
            except Exception:
                info["time_resolution"] = "unknown"

    elif path_obj.is_file() and source_path.endswith(".csv"):
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


def _normalize_scenario_name(
    user_input: str | int, available_columns: list
) -> str | None:
    """Normalize user scenario input to match actual DataFrame column names.

    Accepts either 1, "1", or "iteration_1" and returns the actual column name.
    Handles columns with filename prefixes (e.g., "Load_2024 0308_iteration_1").
    Returns None if scenario not found.

    Parameters
    ----------
    user_input : str | int
        User-provided scenario name (e.g., 1, "1", or "iteration_1")
    available_columns : list
        List of available column names in the DataFrame

    Returns
    -------
    str or None
        Normalized column name if found, None otherwise

    Raises
    ------
    ValueError
        If multiple columns match (ambiguous scenario)

    Examples
    --------
    >>> # Simple case (no prefix)
    >>> cols = ["iteration_1", "iteration_2", "iteration_3"]
    >>> _normalize_scenario_name(1, cols)
    'iteration_1'
    >>> _normalize_scenario_name("iteration_2", cols)
    'iteration_2'

    >>> # With filename prefix (CAISO LoadProfile case)
    >>> cols = ["Load_2024 0308_iteration_1", "Load_2024 0308_iteration_2"]
    >>> _normalize_scenario_name(1, cols)
    'Load_2024 0308_iteration_1'
    >>> _normalize_scenario_name("iteration_1", cols)
    'Load_2024 0308_iteration_1'
    """
    # Convert integer input to string
    user_input = str(user_input)

    # If exact match, return as-is
    if user_input in available_columns:
        return user_input

    # Try adding "iteration_" prefix
    prefixed = f"iteration_{user_input}"
    if prefixed in available_columns:
        return prefixed

    # Try removing "iteration_" prefix (in case user typed it but columns don't have it)
    if user_input.startswith("iteration_"):
        unprefixed = user_input.replace("iteration_", "", 1)
        if unprefixed in available_columns:
            return unprefixed

    # Try suffix matching (handles filename-prefixed columns like "Load_2024 0308_iteration_1")
    # This allows scenario=1 to match columns with arbitrary prefixes
    for suffix in [f"_{prefixed}", f"_{user_input}"]:
        matches = [col for col in available_columns if col.endswith(suffix)]
        if len(matches) == 1:
            return matches[0]  # Unambiguous match
        elif len(matches) > 1:
            # Show first 5 matches for debugging
            match_preview = matches[:5]
            if len(matches) > 5:
                match_preview_str = f"{match_preview} (showing 5 of {len(matches)})"
            else:
                match_preview_str = str(matches)
            msg = (
                f"Ambiguous scenario '{user_input}': found {len(matches)} matching columns: {match_preview_str}. "
                f"Please specify the full column name to disambiguate."
            )
            raise ValueError(msg)

    # Not found
    return None


def _add_loads_per_node(
    network: Network, demand_df: pd.DataFrame, load_scenario: str | None = None
):
    """Assign loads to individual nodes based on matching (default mode).

    For iteration-based formats with load_scenario specified or defaulted, creates single load
    For iteration-based formats without scenario selection (legacy), creates multiple loads per node
    For zone-based formats, creates one load per zone (Load_{Zone})

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.
    load_scenario : str, optional
        For iteration-based formats, specify which scenario to use. If None, defaults to first scenario.

    Returns
    -------
    dict
        Summary information about the load assignment.
    """
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
            print(f"  - Assigning scenario to node: {target_node}")

        # Count iteration columns
        iteration_cols = [col for col in demand_df.columns if "iteration_" in col]

        # Scenario selection logic
        if load_scenario is None:
            # Default to first scenario
            selected_scenario = iteration_cols[0]
            # Extract numeric part for user-friendly message
            first_num = iteration_cols[0].replace("iteration_", "")
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(
                f"  - No load_scenario specified, defaulting to first scenario: {selected_scenario}"
            )
            print(
                f"  - To select a different scenario, use load_scenario parameter (e.g., load_scenario='{first_num}')"
            )
        else:
            # Normalize user input to match column names
            selected_scenario = _normalize_scenario_name(load_scenario, iteration_cols)
            if selected_scenario is None:
                msg = f"Specified load_scenario '{load_scenario}' not found. Available scenarios: {iteration_cols}"
                raise ValueError(msg)
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(f"  - Using user-specified scenario: {selected_scenario}")

        # Create single load with selected scenario
        load_name = f"Load_{target_node}"
        network.add("Load", name=load_name, bus=target_node)

        # Add the load time series (align with network snapshots)
        load_series = demand_df[selected_scenario].reindex(network.snapshots).fillna(0)
        network.loads_t.p_set.loc[:, load_name] = load_series

        print(f"  - Added load {load_name} using scenario {selected_scenario}")

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

            print(f"  - Added load time series for {load_name} (bus: {bus_name})")

    print(
        f"Successfully added {len(network.loads)} loads, skipped {loads_skipped} load zones"
    )

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

    if is_iteration_format:
        return {
            "mode": "per_node",
            "format_type": "iteration",
            "target_node": target_node,
            "load_name": load_name,
            "scenario_selected": selected_scenario,
            "scenarios_available": iteration_cols,
        }
    else:
        return {
            "mode": "per_node",
            "loads_skipped": loads_skipped,
            "buses_without_loads": len(buses_without_loads),
            "format_type": "zone",
        }


def _add_loads_to_target_node(
    network: Network,
    demand_df: pd.DataFrame,
    target_node: str,
    load_scenario: str | None = None,
):
    """Assign all demand to a specific existing node.

    For iteration-based formats with load_scenario specified or defaulted, creates single load
    For zone-based formats, creates a single aggregated load

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.
    target_node : str
        Name of the existing node to assign all demand to.
    load_scenario : str, optional
        For iteration-based formats, specify which scenario to use. If None, defaults to first scenario.

    Returns
    -------
    dict
        Summary information about the load assignment.
    """
    # Verify target node exists
    if target_node not in network.buses.index:
        msg = f"Target node '{target_node}' not found in network buses: {list(network.buses.index)}"
        raise ValueError(msg)

    print(f"Assigning all demand to target node: {target_node}")

    # Check if this is an iteration-based format
    is_iteration_format = getattr(demand_df, "_format_type", None) == "iteration"

    if is_iteration_format:
        # Count actual iteration columns (handle both prefixed and non-prefixed)
        iteration_cols = [col for col in demand_df.columns if "iteration_" in col]
        actual_iterations = len(iteration_cols)

        print(f"Processing iteration-based format with {actual_iterations} iterations")

        # Scenario selection logic
        if load_scenario is None:
            # Default to first scenario
            selected_scenario = iteration_cols[0]
            # Extract numeric part for user-friendly message
            first_num = iteration_cols[0].replace("iteration_", "")
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(
                f"  - No load_scenario specified, defaulting to first scenario: {selected_scenario}"
            )
            print(
                f"  - To select a different scenario, use load_scenario parameter (e.g., load_scenario='{first_num}')"
            )
        else:
            # Normalize user input to match column names
            selected_scenario = _normalize_scenario_name(load_scenario, iteration_cols)
            if selected_scenario is None:
                msg = f"Specified load_scenario '{load_scenario}' not found. Available scenarios: {iteration_cols}"
                raise ValueError(msg)
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(f"  - Using user-specified scenario: {selected_scenario}")

        # Create single load with selected scenario
        load_name = f"Load_{target_node}"
        network.add("Load", name=load_name, bus=target_node)

        # Add the load time series (align with network snapshots)
        load_series = demand_df[selected_scenario].reindex(network.snapshots).fillna(0)
        network.loads_t.p_set.loc[:, load_name] = load_series

        peak_demand = load_series.max()

        print(f"  - Added load {load_name} using scenario {selected_scenario}")
        print(f"  - Peak demand: {peak_demand:.2f} MW")

        return {
            "mode": "target_node",
            "format_type": "iteration",
            "target_node": target_node,
            "load_name": load_name,
            "scenario_selected": selected_scenario,
            "scenarios_available": iteration_cols,
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
    network: Network,
    demand_df: pd.DataFrame,
    aggregate_node_name: str,
    load_scenario: str | None = None,
):
    """Create a new aggregate node and assign all demand to it.

    For iteration-based formats with load_scenario specified or defaulted, creates single load
    For zone-based formats, creates a single aggregated load

    Parameters
    ----------
    network : Network
        The PyPSA network object.
    demand_df : DataFrame
        DataFrame with demand time series for each load zone/iteration.
    aggregate_node_name : str
        Name for the new aggregate node.
    load_scenario : str, optional
        For iteration-based formats, specify which scenario to use. If None, defaults to first scenario.

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

    if is_iteration_format:
        # Count actual iteration columns (handle both prefixed and non-prefixed)
        iteration_cols = [col for col in demand_df.columns if "iteration_" in col]
        actual_iterations = len(iteration_cols)

        print(f"Processing iteration-based format with {actual_iterations} iterations")

        # Scenario selection logic
        if load_scenario is None:
            # Default to first scenario
            selected_scenario = iteration_cols[0]
            # Extract numeric part for user-friendly message
            first_num = iteration_cols[0].replace("iteration_", "")
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(
                f"  - No load_scenario specified, defaulting to first scenario: {selected_scenario}"
            )
            print(
                f"  - To select a different scenario, use load_scenario parameter (e.g., load_scenario='{first_num}')"
            )
        else:
            # Normalize user input to match column names
            selected_scenario = _normalize_scenario_name(load_scenario, iteration_cols)
            if selected_scenario is None:
                msg = f"Specified load_scenario '{load_scenario}' not found. Available scenarios: {iteration_cols}"
                raise ValueError(msg)
            print(f"  - Multiple load scenarios detected: {iteration_cols}")
            print(f"  - Using user-specified scenario: {selected_scenario}")

        # Create single load with selected scenario
        load_name = f"Load_{aggregate_node_name}"
        network.add("Load", name=load_name, bus=aggregate_node_name)

        # Add the load time series (align with network snapshots)
        load_series = demand_df[selected_scenario].reindex(network.snapshots).fillna(0)
        network.loads_t.p_set.loc[:, load_name] = load_series

        peak_demand = load_series.max()

        print(f"  - Added load {load_name} using scenario {selected_scenario}")
        print(f"  - Peak demand: {peak_demand:.2f} MW")

        return {
            "mode": "aggregate_node",
            "format_type": "iteration",
            "aggregate_node": aggregate_node_name,
            "load_name": load_name,
            "scenario_selected": selected_scenario,
            "scenarios_available": iteration_cols,
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


def _add_loads_with_participation_factors(
    network: Network,
    demand_df: pd.DataFrame,
    csv_dir: str | Path,
    load_scenario: str | None = None,
) -> dict:
    """Add loads using participation factors (unified strategy for CAISO and NREL-118 patterns).

    Auto-detects model structure and applies appropriate distribution:
    - CAISO pattern: System load -> Regions (region-level factors in Region.csv)
    - NREL-118 pattern: Regional loads -> Nodes (node-level factors in Node.csv)

    Parameters
    ----------
    network : Network
        PyPSA network
    demand_df : pd.DataFrame
        Demand data (system-wide or pre-loaded)
    csv_dir : str | Path
        Directory containing Node.csv and Region.csv
    load_scenario : str, optional
        Scenario selection for iteration-based formats

    Returns
    -------
    dict
        Summary with load information

    Examples
    --------
    CAISO IRP23:
        - Single system load file
        - Region.csv has "Load" column (factors sum to 1.0)
        - Distributes to 4 nodes: CIPB, CIPV, CISC, CISD

    NREL-118:
        - Per-region load files (Load R1, Load R2, Load R3)
        - Node.csv has "Load Participation Factor" column
        - Distributes to 90 nodes across 3 regions
    """
    csv_dir = Path(csv_dir)

    # Load metadata
    try:
        node_df = load_static_properties(csv_dir, "Node")
        region_df = load_static_properties(csv_dir, "Region")
    except Exception as e:
        msg = f"Failed to load Node.csv or Region.csv: {e}"
        raise ValueError(msg) from e

    # Detect model pattern
    has_node_factors = "Load Participation Factor" in node_df.columns
    has_region_factors = "Load" in region_df.columns
    has_region_load_files = "Load.Data File" in region_df.columns

    logger.info("Detecting load distribution pattern...")
    logger.info(f"  Node-level factors: {has_node_factors}")
    logger.info(f"  Region-level factors: {has_region_factors}")
    logger.info(f"  Region load files: {has_region_load_files}")

    # Route to appropriate sub-strategy
    if has_region_load_files and has_node_factors:
        # NREL-118 pattern: Regional loads -> Nodes
        logger.info(
            "Pattern detected: Regional loads with node-level participation factors (NREL-118 style)"
        )
        return _distribute_regional_loads_to_nodes(
            network, node_df, region_df, csv_dir, load_scenario
        )

    elif has_region_factors and not has_region_load_files:
        # CAISO pattern: System load -> Regions
        logger.info(
            "Pattern detected: System load with region-level participation factors (CAISO style)"
        )
        return _distribute_system_load_to_regions(
            network, demand_df, node_df, region_df, load_scenario
        )

    else:
        msg = (
            "Cannot determine load distribution pattern. Expected either:\n"
            "  1. Region.csv with 'Load.Data File' + Node.csv with 'Load Participation Factor' (NREL-118)\n"
            "  2. Region.csv with 'Load' column without 'Load.Data File' (CAISO)"
        )
        raise ValueError(msg)


def _distribute_system_load_to_regions(
    network: Network,
    demand_df: pd.DataFrame,
    node_df: pd.DataFrame,
    region_df: pd.DataFrame,
    load_scenario: str | None = None,
) -> dict:
    """Distribute system-wide load to regions using region-level participation factors (CAISO pattern).

    Parameters
    ----------
    network : Network
        PyPSA network
    demand_df : pd.DataFrame
        System-wide demand data
    node_df : pd.DataFrame
        Node metadata with Region column
    region_df : pd.DataFrame
        Region metadata with Load column (participation factors)
    load_scenario : str, optional
        Scenario selection

    Returns
    -------
    dict
        Summary information
    """
    # Get system load (scenario already selected by parse_demand_data)
    if isinstance(demand_df, pd.Series):
        system_load = demand_df
    elif len(demand_df.columns) == 1:
        system_load = demand_df.iloc[:, 0]
    # Multiple columns - need scenario selection
    elif load_scenario:
        # Use _normalize_scenario_name to handle filename prefixes
        selected_scenario = _normalize_scenario_name(
            load_scenario, demand_df.columns.tolist()
        )
        if selected_scenario is None:
            msg = f"Scenario '{load_scenario}' not found in columns: {demand_df.columns.tolist()}"
            raise ValueError(msg)
        system_load = demand_df[selected_scenario]
        logger.info(f"Using load scenario: {selected_scenario}")
    else:
        # Default to first column
        system_load = demand_df.iloc[:, 0]
        logger.warning(
            f"Multiple load columns found, using first: {demand_df.columns[0]}"
        )

    # Get regions with participation factors
    regions_with_load = region_df[region_df["Load"].notna()].copy()

    if regions_with_load.empty:
        msg = "No regions found with Load participation factors in Region.csv"
        raise ValueError(msg)

    # Validate factors sum to approximately 1.0
    total_factor = regions_with_load["Load"].astype(float).sum()
    if not (0.99 <= total_factor <= 1.01):
        logger.warning(
            f"Region load participation factors sum to {total_factor:.6f}, expected ~1.0. "
            f"Normalizing factors."
        )
        # Normalize
        regions_with_load["Load"] = (
            regions_with_load["Load"].astype(float) / total_factor
        )

    loads_added = 0
    peak_demand_total = 0

    for region_name, region_row in regions_with_load.iterrows():
        factor = float(region_row["Load"])

        # Find node for this region (expect 1:1 mapping)
        nodes_in_region = node_df[node_df["Region"] == region_name]

        if nodes_in_region.empty:
            logger.warning(f"No nodes found for region {region_name}, skipping")
            continue

        if len(nodes_in_region) > 1:
            logger.warning(
                f"Multiple nodes found for region {region_name}: {nodes_in_region.index.tolist()}. "
                f"Using first node: {nodes_in_region.index[0]}"
            )

        node_name = nodes_in_region.index[0]

        # Check if bus exists in network
        if node_name not in network.buses.index:
            logger.warning(
                f"Bus {node_name} not in network, skipping load for region {region_name}"
            )
            continue

        # Create load
        load_name = f"Load_{region_name}"
        regional_load = system_load * factor

        network.add(
            "Load",
            load_name,
            bus=node_name,
            p_set=regional_load,
        )

        loads_added += 1
        peak_demand_total += regional_load.max()

        logger.info(
            f"  Added {load_name} to bus {node_name}: "
            f"factor={factor:.4f} ({factor * 100:.2f}%), peak={regional_load.max():.1f} MW"
        )

    logger.info(
        f"Added {loads_added} loads with total peak demand: {peak_demand_total:.1f} MW"
    )

    return {
        "strategy": "participation_factors (system->regions)",
        "loads_added": loads_added,
        "regions": len(regions_with_load),
        "peak_demand": peak_demand_total,
    }


def _resolve_data_file_reference(data_file_ref: str, csv_dir: Path) -> Path | None:
    """Resolve a Data File reference to an absolute file path.

    Parameters
    ----------
    data_file_ref : str
        Data File reference like "Data File.Load R1"
    csv_dir : Path
        Directory containing CSV files (e.g., .../csvs_from_xml/System/)

    Returns
    -------
    Path or None
        Absolute path to the data file, or None if not found

    Examples
    --------
    >>> ref = "Data File.Load R1"
    >>> csv_dir = Path("nrel-118/csvs_from_xml/System")
    >>> path = _resolve_data_file_reference(ref, csv_dir)
    >>> # Returns: nrel-118/Input files/RT/Load/LoadR1RT.csv
    """
    try:
        # Load Data File.csv
        data_file_df = load_static_properties(csv_dir, "Data File")

        if data_file_df.empty:
            logger.warning("Data File.csv not found or empty")
            return None

        # Strip "Data File." prefix from reference
        if data_file_ref.startswith("Data File."):
            object_name = data_file_ref.replace("Data File.", "", 1)
        else:
            object_name = data_file_ref

        # Look up in Data File.csv
        if object_name not in data_file_df.index:
            logger.warning(
                f"Data File object '{object_name}' not found in Data File.csv"
            )
            return None

        # Get filename from "Filename(text)" column
        if "Filename(text)" not in data_file_df.columns:
            logger.warning("Data File.csv missing 'Filename(text)' column")
            return None

        filename_text = data_file_df.at[object_name, "Filename(text)"]

        if pd.isna(filename_text):
            logger.warning(f"Data File object '{object_name}' has no Filename(text)")
            return None

        # Convert backslashes to forward slashes (Windows to Unix paths)
        filename_text = str(filename_text).replace("\\", "/")

        # Resolve relative to model base directory (parent of csvs_from_xml)
        # csv_dir is typically: .../model_name/csvs_from_xml/System/
        # model_base is: .../model_name/
        csv_parent = csv_dir.parent  # .../csvs_from_xml/System -> .../csvs_from_xml
        if csv_parent.name == "csvs_from_xml":
            model_base = csv_parent.parent
        else:
            # Fallback: assume csv_dir is directly under model base
            model_base = csv_dir.parent

        file_path = model_base / filename_text

        if not file_path.exists():
            logger.warning(f"Data File path does not exist: {file_path}")
            return None
        else:
            return file_path

    except Exception as e:
        logger.warning(f"Failed to resolve Data File reference '{data_file_ref}': {e}")
        return None


def _load_regional_load_file(file_path: Path) -> pd.Series | None:
    """Load a regional load file (simple DATETIME/value format).

    Parameters
    ----------
    file_path : Path
        Path to regional load CSV file

    Returns
    -------
    pd.Series or None
        Time series of load values with DatetimeIndex, or None if failed

    Examples
    --------
    File format:
        "DATETIME","value"
        "1/1/24 1:00",5465.73
        "1/1/24 2:00",4994.54
    """
    try:
        # Read CSV
        df = pd.read_csv(file_path)

        # Check for required columns
        if "DATETIME" not in df.columns or "value" not in df.columns:
            logger.warning(
                f"Regional load file missing required columns (DATETIME, value): {file_path}"
            )
            return None

        # Parse datetime column
        df["datetime"] = pd.to_datetime(df["DATETIME"], format="%m/%d/%y %H:%M")

        # Set index and return as Series
        df = df.set_index("datetime")
        load_series = df["value"]

        logger.info(
            f"Loaded regional load file: {file_path.name} ({len(load_series)} timesteps)"
        )

    except Exception as e:
        logger.warning(f"Failed to load regional load file {file_path}: {e}")
        return None
    else:
        return load_series


def _distribute_regional_loads_to_nodes(
    network: Network,
    node_df: pd.DataFrame,
    region_df: pd.DataFrame,
    csv_dir: Path,
    load_scenario: str | None = None,
) -> dict:
    """Distribute regional loads to nodes using node-level participation factors (NREL-118 pattern).

    Parameters
    ----------
    network : Network
        PyPSA network
    node_df : pd.DataFrame
        Node metadata with Load Participation Factor and Region columns
    region_df : pd.DataFrame
        Region metadata with Load.Data File references
    csv_dir : Path
        Directory containing CSV files
    load_scenario : str, optional
        Scenario selection

    Returns
    -------
    dict
        Summary information
    """
    # Get regions with load files
    regions_with_loads = region_df[region_df["Load.Data File"].notna()].copy()

    if regions_with_loads.empty:
        msg = "No regions found with Load.Data File references in Region.csv"
        raise ValueError(msg)

    loads_added = 0
    peak_demand_total = 0

    for region_name, region_row in regions_with_loads.iterrows():
        # Get nodes in this region with participation factors
        nodes_in_region = node_df[
            (node_df["Region"] == region_name)
            & (node_df["Load Participation Factor"].notna())
        ].copy()

        if nodes_in_region.empty:
            logger.warning(
                f"No nodes with participation factors found for region {region_name}, skipping"
            )
            continue

        # Get participation factors and normalize
        factors = nodes_in_region["Load Participation Factor"].astype(float)
        total_factor = factors.sum()

        if total_factor == 0:
            logger.warning(
                f"Region {region_name} has zero total participation factor, skipping"
            )
            continue

        if not (0.95 <= total_factor <= 1.05):
            logger.warning(
                f"Region {region_name}: participation factors sum to {total_factor:.6f}, expected ~1.0. "
                f"Normalizing."
            )

        normalized_factors = factors / total_factor

        # Load regional load profile
        data_file_ref = region_row["Load.Data File"]

        # Resolve Data File reference to actual file path
        file_path = _resolve_data_file_reference(data_file_ref, csv_dir)
        if file_path is None:
            logger.warning(
                f"Region {region_name}: Could not resolve Data File reference '{data_file_ref}', skipping"
            )
            continue

        # Load the regional load file
        regional_load = _load_regional_load_file(file_path)
        if regional_load is None:
            logger.warning(
                f"Region {region_name}: Could not load regional load file, skipping"
            )
            continue

        # Distribute regional load to nodes using participation factors
        for node_name, factor in zip(
            nodes_in_region.index, normalized_factors, strict=False
        ):
            # Check if bus exists in network
            if node_name not in network.buses.index:
                logger.warning(
                    f"Bus {node_name} not in network, skipping load for this node"
                )
                continue

            # Calculate node load
            node_load = regional_load * factor

            # Align with network snapshots
            node_load_aligned = node_load.reindex(network.snapshots).fillna(0)

            # Create load
            load_name = f"Load_{node_name}"
            network.add(
                "Load",
                load_name,
                bus=node_name,
                p_set=node_load_aligned,
            )

            loads_added += 1
            peak_demand_total += node_load_aligned.max()

            logger.info(
                f"  Added {load_name} to bus {node_name}: "
                f"factor={factor:.4f} ({factor * 100:.2f}%), peak={node_load_aligned.max():.1f} MW"
            )

    if loads_added == 0:
        logger.warning(
            "No loads added. Check that regional load files exist and are accessible."
        )
    else:
        logger.info(
            f"Added {loads_added} loads across {len(regions_with_loads)} regions "
            f"with total peak demand: {peak_demand_total:.1f} MW"
        )

    return {
        "strategy": "participation_factors (regions->nodes)",
        "loads_added": loads_added,
        "regions": len(regions_with_loads),
        "peak_demand": peak_demand_total,
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
    transmission_as_lines=False,
    load_scenario=None,
    demand_target_node=None,
):
    """Unified network setup function that automatically detects the appropriate mode.

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
    transmission_as_lines : bool, optional
        If True, convert PLEXOS Line objects to PyPSA Lines with electrical impedance.
        If False (default), use existing Links behavior for backward compatibility.
    load_scenario : str, optional
        For demand data with multiple scenarios (iterations), specify which scenario
        to use. If None, defaults to first scenario. Example: "iteration_1"
    demand_target_node : str, optional
        Specific node to assign ALL demand to, regardless of setup mode.
        This allows keeping generators on their original nodes while
        consolidating all demand to one node. Takes precedence over target_node.

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
        msg = (
            "Cannot specify both target_node and aggregate_node_name. Choose one mode."
        )
        raise ValueError(msg)

    # Detect mode and print status
    if aggregate_node_name is not None:
        mode = "aggregation"
        print(
            f"Setting up network with demand aggregation to new node: {aggregate_node_name}"
        )
    elif target_node is not None:
        mode = "target_node"
        print(
            f"Setting up network with all demand assigned to target node: {target_node}"
        )
    else:
        mode = "per_node"
        print("Setting up network with per-node demand assignment")

    # Check for multiple models and validate model_name if needed
    print("Checking for multiple models in database...")
    models = db.list_objects_by_class(ClassEnum.Model)

    if len(models) > 1:
        if model_name is None:
            msg = f"Multiple models found in XML file: {models}. Please specify a model_name parameter."
            raise ValueError(msg)
        elif model_name not in models:
            msg = f"Model '{model_name}' not found in XML file. Available models: {models}"
            raise ValueError(msg)
        else:
            print(f"  Using specified model: {model_name}")
    elif len(models) == 1:
        if model_name is not None and model_name != models[0]:
            msg = f"Model '{model_name}' not found. Only available model: {models[0]}"
            raise ValueError(msg)
        print(f"  Found single model: {models[0]}")
    else:
        print("  No models found in database")
        if model_name is not None:
            msg = f"Model '{model_name}' not found. No models available in XML file."
            raise ValueError(msg)

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
        load_scenario=load_scenario,
        demand_target_node=demand_target_node,
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
    if inflow_path and Path(inflow_path).exists():
        print("\n" + "=" * 60)
        print("STEP 2b: Adding hydro inflows")
        print("=" * 60)
        add_hydro_inflows(network, db, inflow_path)
    elif inflow_path:
        print(f"\nWarning: Inflow path specified but not found: {inflow_path}")
        print("Skipping hydro inflow processing")
    else:
        print(
            "\nNo inflow path specified - storage units will not have natural inflows"
        )

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

    # Step 4: Add transmission (lines or links)
    print("\n" + "=" * 60)
    if transmission_as_lines:
        print("STEP 4: Adding transmission lines (with electrical impedance)")
        print("=" * 60)
        port_lines(network, db, timeslice_csv=timeslice_csv)

        # Note: Lines don't need reassignment - they maintain physical connections
        # even in aggregation mode (self-loops on aggregate node are handled in optimization)
        link_summary = None
        if mode == "aggregation":
            print(
                f"Note: Lines maintain original bus connections for {aggregate_node_name} aggregation"
            )
    else:
        print("STEP 4: Adding transmission links (legacy mode)")
        print("=" * 60)
        port_links(network, db)

        # For aggregation mode, reassign all links to/from the aggregate node
        link_summary = None
        if mode == "aggregation":
            print(
                f"Reassigning all links to/from aggregate node: {aggregate_node_name}"
            )
            link_summary = reassign_links_to_node(network, aggregate_node_name)

    # Add constraints as final step
    print("\n" + "=" * 60)
    print("STEP 5: Adding PLEXOS constraints")
    print("=" * 60)

    try:
        constraint_results = add_constraints_enhanced(network, db, verbose=True)
        print(
            f" Constraint porting completed: {constraint_results['implemented']} implemented, {constraint_results['skipped']} skipped"
        )
    except Exception as e:
        print(f"  Constraint porting failed: {e}")
        constraint_results = {"implemented": 0, "skipped": 0, "warnings": []}

    print("\n" + "=" * 60)
    print(f"NETWORK SETUP COMPLETE ({mode.upper()} MODE)")
    print("=" * 60)
    print("Final network summary:")
    print(f"  Buses: {len(network.buses)}")
    print(f"  Generators: {len(network.generators)}")
    if transmission_as_lines:
        print(f"  Transmission Lines: {len(network.lines)}")
        print(f"  Links: {len(network.links)}")
    else:
        print(f"  Links (incl. transmission): {len(network.links)}")
    print(f"  Batteries: {len(network.storage_units)}")
    print(f"  Loads: {len(network.loads)}")
    print(f"  Snapshots: {len(network.snapshots)}")
    print(f"  Constraints: {constraint_results['implemented']} implemented")

    # Add mode information to summary
    load_summary["mode"] = mode
    load_summary["constraint_results"] = constraint_results
    if mode == "aggregation":
        load_summary["aggregate_node_name"] = aggregate_node_name
        load_summary["generator_summary"] = generator_summary
        load_summary["link_summary"] = link_summary
    elif mode == "target_node":
        load_summary["target_node"] = target_node

    return load_summary
