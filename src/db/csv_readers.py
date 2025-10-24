"""CSV-based readers for COAD-exported PLEXOS models.

This module provides functions to load and parse COAD CSV exports, replacing
the PlexosDB database query approach with direct CSV reading.

The COAD export structure consists of:
1. Static property CSVs: Generator.csv, Fuel.csv, Node.csv, Line.csv, etc.
   - One CSV per PLEXOS class
   - Rows = objects, columns = properties
   - Relationships stored as column values (e.g., Generator.csv has "Node" and "Fuel" columns)

2. Time-varying properties CSV: Time varying properties.csv
   - Contains properties with temporal metadata (date ranges, timeslices)
   - Columns: class, object, property, value, date_from, date_to, timeslice, data_id
   - One row per property value with its temporal constraints
"""

import ast
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_datetime(value: Any) -> pd.Timestamp | None:
    """Ensure a value is a pandas Timestamp or None.

    Handles various input types:
    - Already a Timestamp: returns as-is
    - String: converts to Timestamp
    - NaT/NaN: returns None
    - None: returns None

    Parameters
    ----------
    value : Any
        Value to convert to Timestamp

    Returns
    -------
    pd.Timestamp | None
        Converted Timestamp or None if value is null

    Examples
    --------
    >>> ensure_datetime("2024-01-01")
    Timestamp('2024-01-01 00:00:00')
    >>> ensure_datetime(pd.Timestamp("2024-01-01"))
    Timestamp('2024-01-01 00:00:00')
    >>> ensure_datetime(None)
    None
    >>> ensure_datetime(pd.NaT)
    None
    """
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.to_datetime(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert value to datetime: {value}")
        return None


def parse_numeric_value(
    value: Any, use_first: bool = True, strategy: str = "first"
) -> float | None:
    """Parse a numeric value that might be a string representation of a list.

    When COAD exports PLEXOS data to CSV, properties with multiple values
    (e.g., multiple generator units, multiple operating conditions) are stored as
    string representations of lists. This function handles various formats and
    provides multiple strategies for aggregating multi-value properties.

    Parameters
    ----------
    value : Any
        Value to parse (could be number, string, or string representation of list)
    use_first : bool, default True
        DEPRECATED: Use strategy parameter instead.
        If value is a list, whether to use first element (True) or sum all (False).
    strategy : str, default "first"
        Strategy for handling multi-value properties:
        - "first": Use first value (default, backward compatible)
        - "min": Use minimum value (conservative for max limits)
        - "max": Use maximum value (conservative for min limits)
        - "sum": Sum all values (for additive properties like capacity)
        - "average": Average all values (for representative physical parameters)

    Returns
    -------
    float | None
        Parsed numeric value, or None if parsing fails

    Examples
    --------
    Parse a simple string number:
    >>> parse_numeric_value("55.197")
    55.197

    Parse a list string with different strategies:
    >>> parse_numeric_value("['100', '150', '200']", strategy="first")
    100.0
    >>> parse_numeric_value("['100', '150', '200']", strategy="min")
    100.0
    >>> parse_numeric_value("['100', '150', '200']", strategy="max")
    200.0
    >>> parse_numeric_value("['100', '150', '200']", strategy="sum")
    450.0
    >>> parse_numeric_value("['100', '150', '200']", strategy="average")
    150.0

    Handle already-parsed numeric values:
    >>> parse_numeric_value(55.197)
    55.197

    Handle None or empty strings:
    >>> parse_numeric_value(None)
    None
    >>> parse_numeric_value("")
    None
    """
    if value is None or value == "":
        return None

    # Already a number
    if isinstance(value, int | float):
        return float(value)

    # Try direct conversion
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # Try parsing as list
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Convert all elements to float
                float_values = [float(x) for x in parsed]

                # Apply strategy (use_first param takes precedence for backward compatibility)
                if not use_first and strategy == "first":
                    strategy = "sum"

                if strategy == "first":
                    result = float_values[0]
                elif strategy == "min":
                    result = min(float_values)
                elif strategy == "max":
                    result = max(float_values)
                elif strategy == "sum":
                    result = sum(float_values)
                elif strategy == "average":
                    result = sum(float_values) / len(float_values)
                else:
                    logger.warning(
                        f"Unknown strategy '{strategy}', using 'first'. "
                        f"Valid strategies: first, min, max, sum, average"
                    )
                    result = float_values[0]

                # Log when multi-value property is encountered
                if len(float_values) > 1:
                    logger.debug(
                        f"Multi-value property {float_values} resolved to {result} using strategy '{strategy}'"
                    )

                return result
        except (ValueError, SyntaxError, TypeError):
            pass

    logger.warning(f"Could not parse numeric value: {value}")
    return None


def load_static_properties(csv_dir: str | Path, class_name: str) -> pd.DataFrame:
    """Load static properties for a given PLEXOS class from COAD CSV export.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    class_name : str
        PLEXOS class name (e.g., 'Generator', 'Fuel', 'Node', 'Line')

    Returns
    -------
    pd.DataFrame
        DataFrame with object names as index and properties as columns.
        Returns empty DataFrame if file doesn't exist.

    Examples
    --------
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> generators.loc["AA1", "Max Capacity"]
    21.0
    >>> generators.loc["AA1", "Node"]
    'SEM'
    """
    csv_path = Path(csv_dir) / f"{class_name}.csv"

    if not csv_path.exists():
        print(f"Warning: CSV file not found: {csv_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, index_col="object")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()
    else:
        print(f"Loaded {len(df)} {class_name} objects from {csv_path.name}")
        return df


def load_time_varying_properties(
    csv_dir: str | Path,
    class_name: str | None = None,
    property_name: str | None = None,
    object_name: str | None = None,
) -> pd.DataFrame:
    """Load time-varying properties from COAD CSV export with optional filtering.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports
    class_name : str, optional
        Filter by PLEXOS class (e.g., 'Generator', 'Fuel')
    property_name : str, optional
        Filter by property name (e.g., 'Rating', 'Price')
    object_name : str, optional
        Filter by specific object name

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: class, object, property, value, date_from, date_to, timeslice, data_id.
        Returns empty DataFrame if file doesn't exist.

    Examples
    --------
    >>> # Load all time-varying properties
    >>> all_props = load_time_varying_properties(csv_dir)

    >>> # Load only Generator ratings
    >>> gen_ratings = load_time_varying_properties(
    ...     csv_dir, class_name="Generator", property_name="Rating"
    ... )

    >>> # Load all properties for a specific generator
    >>> aa1_props = load_time_varying_properties(
    ...     csv_dir, class_name="Generator", object_name="AA1"
    ... )
    """
    csv_path = Path(csv_dir) / "Time varying properties.csv"

    if not csv_path.exists():
        print(f"Warning: Time varying properties CSV not found: {csv_path}")
        print(
            "  This is expected for older COAD exports. Only static properties available."
        )
        return pd.DataFrame(
            columns=[
                "class",
                "object",
                "property",
                "value",
                "date_from",
                "date_to",
                "timeslice",
                "data_id",
            ]
        )

    try:
        # Read CSV and parse date columns directly
        df = pd.read_csv(
            csv_path,
            parse_dates=["date_from", "date_to"],
            date_format="ISO8601",
        )

        # Apply filters
        if class_name is not None:
            df = df[df["class"] == class_name]
        if property_name is not None:
            df = df[df["property"] == property_name]
        if object_name is not None:
            df = df[df["object"] == object_name]

    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame(
            columns=[
                "class",
                "object",
                "property",
                "value",
                "date_from",
                "date_to",
                "timeslice",
                "data_id",
            ]
        )
    else:
        if class_name or property_name or object_name:
            filters = []
            if class_name:
                filters.append(f"class={class_name}")
            if property_name:
                filters.append(f"property={property_name}")
            if object_name:
                filters.append(f"object={object_name}")
            print(f"Loaded {len(df)} time-varying properties ({', '.join(filters)})")
        else:
            print(f"Loaded {len(df)} time-varying properties from {csv_path.name}")

        return df


def find_bus_for_object_csv(static_df: pd.DataFrame, object_name: str) -> str | None:
    """Find the associated bus (Node) for a given object using static CSV data.

    This replaces db.parse.find_bus_for_object() with CSV-based lookup.

    Parameters
    ----------
    static_df : pd.DataFrame
        DataFrame from load_static_properties() for the object's class (e.g., Generator.csv)
    object_name : str
        Name of the object (e.g., generator name)

    Returns
    -------
    str | None
        The name of the associated Node (bus), or None if not found.

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> bus = find_bus_for_object_csv(generators, "AA1")
    >>> print(bus)
    'SEM'
    """
    if object_name not in static_df.index:
        print(f"Warning: Object '{object_name}' not found in static properties")
        return None

    if "Node" not in static_df.columns:
        print(
            f"Warning: 'Node' column not found in static properties for {object_name}"
        )
        return None

    node_value = static_df.loc[object_name, "Node"]

    # Handle empty/NaN values
    if pd.isna(node_value) or node_value == "":
        print(f"Warning: No Node found for object '{object_name}'")
        return None

    return str(node_value)


def find_fuel_for_generator_csv(
    static_df: pd.DataFrame, generator_name: str
) -> str | None:
    """Find the associated fuel for a given generator using static CSV data.

    Handles multi-fuel generators by removing duplicates and using the first fuel.
    When multiple unique fuels exist, uses the primary (first) fuel and logs a message.

    This replaces db.parse.find_fuel_for_generator() with CSV-based lookup.

    Parameters
    ----------
    static_df : pd.DataFrame
        DataFrame from load_static_properties(csv_dir, "Generator")
    generator_name : str
        Name of the generator

    Returns
    -------
    str | None
        The name of the associated Fuel (primary fuel if multiple), or None if not found.

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> fuel = find_fuel_for_generator_csv(generators, "AA1")
    >>> print(fuel)
    None  # Hydro generators may not have fuel

    >>> fuel = find_fuel_for_generator_csv(generators, "MP1")
    >>> print(fuel)
    'ROI Coal'  # First fuel from ['ROI Coal', 'ROI Coal', 'ROI Oil']
    """
    if generator_name not in static_df.index:
        print(f"Warning: Generator '{generator_name}' not found in static properties")
        return None

    if "Fuel" not in static_df.columns:
        print("Warning: 'Fuel' column not found in Generator static properties")
        return None

    fuel_value = static_df.loc[generator_name, "Fuel"]

    # Handle empty/NaN values
    if pd.isna(fuel_value) or fuel_value == "":
        return None

    # Handle multi-fuel lists: "['ROI Gas', 'ROI Gas']" or "['ROI Coal', 'ROI Oil']"
    if isinstance(fuel_value, str) and fuel_value.startswith("["):
        try:
            fuel_list = ast.literal_eval(fuel_value)
            if isinstance(fuel_list, list) and len(fuel_list) > 0:
                # Remove duplicates while preserving order
                unique_fuels = []
                for fuel in fuel_list:
                    if fuel not in unique_fuels:
                        unique_fuels.append(fuel)

                # If only one unique fuel remains, use it
                if len(unique_fuels) == 1:
                    return str(unique_fuels[0])

                # If multiple unique fuels, use first and log
                if len(unique_fuels) > 1:
                    primary_fuel = str(unique_fuels[0])
                    logger.info(
                        f"Generator '{generator_name}' has multiple fuels {unique_fuels}. "
                        f"Using primary fuel: '{primary_fuel}'"
                    )
                    return primary_fuel
        except (ValueError, SyntaxError) as e:
            logger.warning(
                f"Failed to parse Fuel list for generator '{generator_name}': {e}"
            )

    return str(fuel_value)


def get_dataid_timeslice_map_csv(time_varying_df: pd.DataFrame) -> dict[int, list[str]]:
    """Build a mapping from data_id to timeslice name(s) from time-varying properties CSV.

    This replaces db.parse.get_dataid_timeslice_map() with CSV-based approach.

    Parameters
    ----------
    time_varying_df : pd.DataFrame
        DataFrame from load_time_varying_properties()

    Returns
    -------
    dict[int, list[str]]
        Dictionary mapping data_id to list of timeslice names

    Examples
    --------
    >>> time_varying = load_time_varying_properties(csv_dir)
    >>> mapping = get_dataid_timeslice_map_csv(time_varying)
    >>> mapping[12345]
    ['M1', 'M2', 'M3']
    """
    dataid_to_timeslice: dict[int, list[str]] = {}

    # Filter to rows that have a timeslice defined
    has_timeslice = time_varying_df[time_varying_df["timeslice"].notna()].copy()

    for _, row in has_timeslice.iterrows():
        data_id = int(row["data_id"])
        timeslice = str(row["timeslice"])

        if data_id not in dataid_to_timeslice:
            dataid_to_timeslice[data_id] = []

        # Avoid duplicates
        if timeslice not in dataid_to_timeslice[data_id]:
            dataid_to_timeslice[data_id].append(timeslice)

    return dataid_to_timeslice


def load_all_static_csvs(csv_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all standard PLEXOS class CSVs from a COAD export directory.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping class names to DataFrames.
        Keys: 'Generator', 'Fuel', 'Node', 'Line', 'Storage', 'Battery', etc.

    Examples
    --------
    >>> csv_dir = "models/sem-2024/SEM Forecast model/"
    >>> all_csvs = load_all_static_csvs(csv_dir)
    >>> generators = all_csvs.get("Generator", pd.DataFrame())
    >>> fuels = all_csvs.get("Fuel", pd.DataFrame())
    """
    standard_classes = [
        "Generator",
        "Fuel",
        "Node",
        "Line",
        "Storage",
        "Battery",
        "Region",
        "Zone",
        "Emission",
        "Company",
        "Constraint",
        "Timeslice",
        "Data File",
        "Variable",
        "Model",
        "Horizon",
        "Report",
        "Production",
        "Scenario",
        "Market",
        "Performance",
        "PASA",
        "Facility",
        "Process",
        "Commodity",
        "Flow Node",
        "Flow Path",
        "Flow Network",
    ]

    all_csvs = {}

    for class_name in standard_classes:
        df = load_static_properties(csv_dir, class_name)
        if not df.empty:
            all_csvs[class_name] = df

    print(f"\nLoaded {len(all_csvs)} static CSV files from {csv_dir}")

    return all_csvs


def get_property_from_static_csv(
    static_df: pd.DataFrame,
    object_name: str,
    property_name: str,
    default: Any = None,
) -> Any:
    """Get a property value for an object from static CSV data.

    Parameters
    ----------
    static_df : pd.DataFrame
        DataFrame from load_static_properties()
    object_name : str
        Name of the object
    property_name : str
        Name of the property to retrieve
    default : Any, optional
        Default value to return if property not found

    Returns
    -------
    Any
        Property value, or default if not found

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> max_cap = get_property_from_static_csv(generators, "AA1", "Max Capacity", default=0.0)
    >>> print(max_cap)
    21.0
    """
    if object_name not in static_df.index:
        return default

    if property_name not in static_df.columns:
        return default

    value = static_df.loc[object_name, property_name]

    if pd.isna(value) or value == "":
        return default

    return value


def get_numeric_property_from_static_csv(
    static_df: pd.DataFrame,
    object_name: str,
    property_name: str,
    default: float | None = None,
    strategy: str = "first",
) -> float | None:
    """Get a numeric property value with multi-value handling strategy.

    This is a convenience function that combines get_property_from_static_csv()
    with parse_numeric_value() to handle numeric properties that may have
    multiple values in the CSV.

    Parameters
    ----------
    static_df : pd.DataFrame
        DataFrame from load_static_properties()
    object_name : str
        Name of the object
    property_name : str
        Name of the property to retrieve
    default : float | None, optional
        Default value to return if property not found or cannot be parsed
    strategy : str, default "first"
        Strategy for handling multi-value properties:
        - "first": Use first value (default)
        - "min": Use minimum value (conservative for max limits)
        - "max": Use maximum value (conservative for min limits)
        - "sum": Sum all values (for additive properties)
        - "average": Average all values (for physical parameters)

    Returns
    -------
    float | None
        Parsed numeric value, or default if not found/cannot be parsed

    Examples
    --------
    >>> lines = load_static_properties(csv_dir, "Line")
    >>> # Single value
    >>> max_flow = get_numeric_property_from_static_csv(lines, "line001", "Max Flow")
    >>> # Multi-value with strategy
    >>> max_flow = get_numeric_property_from_static_csv(
    ...     lines, "line001", "Max Flow", strategy="min"
    ... )
    """
    raw_value = get_property_from_static_csv(
        static_df, object_name, property_name, default=None
    )

    if raw_value is None:
        return default

    parsed = parse_numeric_value(raw_value, strategy=strategy)
    return parsed if parsed is not None else default


def list_objects_by_class_csv(static_df: pd.DataFrame) -> list[str]:
    """Get list of object names for a given class from static CSV.

    This replaces db.list_objects_by_class() with CSV-based approach.

    Parameters
    ----------
    static_df : pd.DataFrame
        DataFrame from load_static_properties()

    Returns
    -------
    list[str]
        List of object names (DataFrame index)

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> gen_list = list_objects_by_class_csv(generators)
    >>> print(len(gen_list))
    116
    """
    return static_df.index.tolist()


def find_bus_for_storage_via_generators_csv(
    storage_df: pd.DataFrame, generator_df: pd.DataFrame, storage_name: str
) -> tuple[str, str] | None:
    """Find bus for storage unit by checking connected generators using CSV data.

    PLEXOS storage units are often connected to generators as "Head Storage" or "Tail Storage"
    rather than directly to nodes. This function finds the bus by looking up connected generators.

    This replaces db.parse.find_bus_for_storage_via_generators() with CSV-based approach.

    Parameters
    ----------
    storage_df : pd.DataFrame
        DataFrame from load_static_properties(csv_dir, "Storage")
    generator_df : pd.DataFrame
        DataFrame from load_static_properties(csv_dir, "Generator")
    storage_name : str
        Name of the storage unit

    Returns
    -------
    tuple[str, str] | None
        (bus_name, primary_generator_name) or None if not found

    Notes
    -----
    In PLEXOS, the relationship is stored in Generator.csv under the "Storage" column,
    which contains the storage unit name if that generator is connected to storage.

    Examples
    --------
    >>> storage = load_static_properties(csv_dir, "Storage")
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> result = find_bus_for_storage_via_generators_csv(storage, generators, "HEAD")
    >>> if result:
    ...     bus_name, gen_name = result
    ...     print(f"Storage HEAD connected to {gen_name} at bus {bus_name}")
    """
    if "Storage" not in generator_df.columns:
        print(f"No 'Storage' column in Generator CSV for storage {storage_name}")
        return None

    # Find generators connected to this storage
    connected_gens = generator_df[generator_df["Storage"] == storage_name]

    if connected_gens.empty:
        print(f"No connected generators found for storage {storage_name}")
        return None

    # Try each connected generator to find its bus
    for gen_name in connected_gens.index:
        gen_bus = find_bus_for_object_csv(generator_df, gen_name)
        if gen_bus:
            print(
                f"Storage {storage_name}: found bus '{gen_bus}' via generator '{gen_name}'"
            )
            return gen_bus, gen_name

    # No successful bus connection found
    print(
        f"Could not find bus for storage {storage_name} via any of its generators: {list(connected_gens.index)}"
    )
    return None


def get_emission_rates_csv(
    generator_df: pd.DataFrame, generator_name: str
) -> dict[str, tuple[float, str]]:
    """Extract emission rate properties for a specific generator from CSV data.

    This replaces db.parse.get_emission_rates() with CSV-based approach.

    Parameters
    ----------
    generator_df : pd.DataFrame
        DataFrame from load_static_properties(csv_dir, "Generator")
    generator_name : str
        Name of the generator

    Returns
    -------
    dict[str, tuple[float, str]]
        Dictionary of emission properties {emission_type: (rate, unit)}
        Empty dict if generator not found or no emission properties

    Notes
    -----
    Looks for columns in Generator.csv that contain "emission", "co2", "so2", "nox", etc.

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> emissions = get_emission_rates_csv(generators, "Coal_Plant_1")
    >>> print(emissions)
    {'co2_emission_rate': (0.95, 'tCO2/MWh'), 'nox_emission_rate': (0.002, 'kg/MWh')}
    """
    emission_props: dict[str, tuple[float, str]] = {}

    if generator_name not in generator_df.index:
        print(f"Generator '{generator_name}' not found in CSV")
        return emission_props

    # Get row for this generator
    gen_row = generator_df.loc[generator_name]

    # Search for emission-related columns
    emission_keywords = ["emission", "co2", "so2", "nox", "Co2"]

    for col_name in generator_df.columns:
        col_lower = col_name.lower()
        if any(keyword.lower() in col_lower for keyword in emission_keywords):
            value = gen_row[col_name]
            if pd.notna(value) and value != "":
                try:
                    # Attempt to convert to float
                    rate = float(value)
                    # Unit is not typically stored in COAD CSV, use generic
                    unit = "unit"
                    emission_props[col_lower] = (rate, unit)
                except (ValueError, TypeError):
                    # Skip if can't convert to float
                    continue

    return emission_props


def read_plexos_input_csv(
    file_path: str | Path,
    object_name: str | None = None,
    scenario: str | int | None = None,
    resolution: str = "hourly",
    snapshots: pd.DatetimeIndex | None = None,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """Read and parse PLEXOS input CSV files in various formats.

    This is a generalized function that handles all PLEXOS input CSV formats:
    - Periods in Columns (most common): Year/Month/Day/Period + value columns
    - Periods in Columns (annual): Month/Day/Period + value columns (no Year)
    - Bands in Columns: Datetime + band columns
    - Names in Columns: Datetime/Pattern + object name columns
    - Stochastic scenarios: Multiple scenario columns (1, 2, 3, 4, 5)

    PLEXOS Input CSV Format Support:
    1. Periods in Columns:
       - Columns: Year, Month, Day, Period (hour), then value columns
       - OR: Columns: Datetime, then period/hour columns (1, 2, 3, ..., 24/48)
       - Supports up and down scaling
       - Supports name column for identifying object
       - Does not support band column
       - Does not support datetime date format (uses Year/Month/Day/Period)

    2. Bands in Columns:
       - Columns: Datetime, then band columns
       - Does not support up and down scaling
       - Supports name column
       - Supports pattern or period column
       - Supports datetime date format

    3. Names in Columns:
       - Columns: Datetime/Pattern/Period, then object name columns
       - Does not support up and down scaling
       - Supports pattern or period column
       - Supports datetime date format
       - PLEXOS looks for object name in column headers

    Date Format Options:
    - Year/Month/Day/Period: Most common, Period=1 means hour 1 (00:00-01:00)
    - Datetime: ISO format datetime string
    - Timeslice/Pattern: Named time periods

    Parameters
    ----------
    file_path : str | Path
        Path to the PLEXOS input CSV file
    object_name : str, optional
        Name of specific object to extract (for Names in Columns format).
        If None, returns all columns.
    scenario : str | int, optional
        Scenario number to extract (for stochastic models with multiple scenarios).
        Common values: "1", "2", "3", "4", "5" or 1, 2, 3, 4, 5.
        If None, uses first scenario or averages all scenarios.
    resolution : str, default "hourly"
        Time resolution: "hourly" (60min), "half_hourly" (30min), "5min", etc.
    snapshots : pd.DatetimeIndex, optional
        Network snapshots for tiling annual profiles. When provided with Month/Day/Period
        format (no Year), the annual pattern is tiled across all years in snapshots
        and interpolated to match snapshot resolution.
    interpolation_method : str, default "linear"
        Interpolation method for sparse data when tiling annual profiles.
        Options: "linear", "ffill", "nearest". Only used when snapshots is provided
        and data is detected as annual profile (Year=1900).

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and value column(s).
        - If object_name specified: single column with that object's data
        - If multiple objects: multiple columns, one per object
        - If stochastic scenarios: columns for each scenario or selected scenario

    Examples
    --------
    Read SEM wind profile (Periods in Columns, stochastic):
    >>> df = read_plexos_input_csv(
    ...     "ROI Wind_5base years - 2018-2033.csv",
    ...     scenario="1"
    ... )

    Read AEMO demand (Names in Columns):
    >>> df = read_plexos_input_csv(
    ...     "demand.csv",
    ...     object_name="NSW1"
    ... )

    Read VRE profile (Periods in Columns):
    >>> df = read_plexos_input_csv(
    ...     "Gen_Solar.csv"
    ... )

    Notes
    -----
    - Automatically detects format based on column names
    - Handles various date formats (Year/Month/Day/Period, Datetime, etc.)
    - Supports hourly (24 periods) and half-hourly (48 periods) resolutions
    - For stochastic models, can select specific scenario or average all
    - Returns data indexed by datetime for easy PyPSA integration
    """
    df = pd.read_csv(file_path)

    # Detect format and parse accordingly
    format_type = _detect_plexos_csv_format(df)

    if format_type == "periods_in_columns_ymd":
        result = _parse_periods_in_columns_ymd(df, scenario)

        # If snapshots provided and data is annual profile (Year=1900), tile it across years
        if snapshots is not None and result.index.year.min() == 1900:
            result = _tile_annual_profile_across_years(
                result, snapshots, interpolation_method
            )

        return result
    elif format_type == "periods_in_columns_ymd_numeric":
        return _parse_periods_in_columns_ymd_numeric(df, scenario)
    elif format_type == "periods_in_columns_datetime":
        return _parse_periods_in_columns_datetime(df, object_name, scenario)
    elif format_type == "names_in_columns":
        return _parse_names_in_columns(df, object_name, scenario)
    elif format_type == "bands_in_columns":
        return _parse_bands_in_columns(df)
    else:
        msg = f"Could not detect PLEXOS CSV format for file: {file_path}"
        raise ValueError(msg)


def _detect_plexos_csv_format(df: pd.DataFrame) -> str:
    """Detect which PLEXOS input CSV format is being used.

    Returns
    -------
    str
        One of: "periods_in_columns_ymd", "periods_in_columns_datetime",
        "names_in_columns", "bands_in_columns"
    """
    columns_lower = [col.lower() for col in df.columns]

    # Check for Year/Month/Day/Period format (SEM-style)
    if all(col in columns_lower for col in ["year", "month", "day", "period"]):
        return "periods_in_columns_ymd"

    # Check for Month/Day/Period format WITHOUT Year (annual repeating profile)
    # Used by CAISO hydro dispatch and other annual patterns
    if (
        all(col in columns_lower for col in ["month", "day", "period"])
        and "year" not in columns_lower
    ):
        return "periods_in_columns_ymd"  # Same format type - parser will inject default year

    # Check for Year/Month/Day + numeric period columns (AEMO-style without Period column)
    if all(col in columns_lower for col in ["year", "month", "day"]):
        # Check if we have numeric columns (01, 02, ..., 24/48)
        numeric_cols = [col for col in df.columns if str(col).strip().isdigit()]
        if len(numeric_cols) in [24, 48]:
            return "periods_in_columns_ymd_numeric"

    # Check for Datetime + numeric period columns (AEMO-style VRE profiles)
    if "datetime" in columns_lower:
        # Check if we have numeric columns (1, 2, 3, ..., 24 or 48)
        numeric_cols = [col for col in df.columns if str(col).strip().isdigit()]
        if len(numeric_cols) in [24, 48]:
            return "periods_in_columns_datetime"

        # Check if we have object name columns (Names in Columns format)
        non_date_cols = [
            col
            for col in df.columns
            if col.lower() not in ["datetime", "date", "timeslice", "pattern"]
        ]
        if len(non_date_cols) > 0:
            return "names_in_columns"

    # Check for Pattern/Timeslice + object columns
    if any(col in columns_lower for col in ["pattern", "timeslice"]):
        return "names_in_columns"

    # Check for band columns
    if any("band" in col.lower() for col in df.columns):
        return "bands_in_columns"

    # Default to names in columns if we have datetime-like first column
    return "names_in_columns"


def _parse_periods_in_columns_ymd(
    df: pd.DataFrame, scenario: str | int | None = None, default_year: int = 1900
) -> pd.DataFrame:
    """Parse Periods in Columns format with Year/Month/Day/Period.

    Used by SEM and other models with stochastic scenarios.
    Also handles Month/Day/Period format (no Year) for annual repeating profiles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Year/Month/Day/Period or Month/Day/Period columns
    scenario : str | int, optional
        Scenario number to extract for stochastic models
    default_year : int, default 1900
        Year to use when Year column is missing (for annual repeating profiles).
        Using 1900 as marker for annual profiles that need tiling.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and value column(s)
    """
    # Handle missing Year column (annual repeating profile)
    if "Year" not in df.columns:
        df = df.copy()  # Don't modify original
        df["Year"] = default_year

    # Create datetime from Year, Month, Day, Period
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]]) + pd.to_timedelta(
        (df["Period"] - 1), unit="h"
    )

    # Find value columns (numeric scenario columns OR named value columns like "Value")
    # First try to find numeric scenario columns (1, 2, 3, 4, 5)
    value_cols = [
        col
        for col in df.columns
        if col not in ["Year", "Month", "Day", "Period", "datetime"]
        and str(col).replace(".", "").isdigit()
    ]

    # If no numeric columns, look for other value columns (e.g., "Value", "value")
    if not value_cols:
        value_cols = [
            col
            for col in df.columns
            if col not in ["Year", "Month", "Day", "Period", "datetime"]
        ]

    if not value_cols:
        msg = "No value columns found in Periods in Columns format"
        raise ValueError(msg)

    # If scenario specified, select that column
    if scenario is not None:
        scenario_col = str(scenario)
        if scenario_col in value_cols:
            result = df[["datetime", scenario_col]].copy()
            result.columns = ["datetime", "value"]
        else:
            msg = f"Scenario '{scenario}' not found. Available: {value_cols}"
            raise ValueError(msg)
    else:
        # Return all scenarios
        result = df[["datetime"] + value_cols].copy()

    result.set_index("datetime", inplace=True)
    return result


def _parse_periods_in_columns_ymd_numeric(
    df: pd.DataFrame, scenario: str | int | None = None
) -> pd.DataFrame:
    """Parse Periods in Columns format with Year/Month/Day + numeric period columns.

    Used by AEMO trace files with Year, Month, Day + columns (01, 02, ..., 48).
    Similar to _parse_periods_in_columns_ymd but without a Period column.
    """
    # Create base datetime from Year, Month, Day (start of day)
    df["datetime"] = pd.to_datetime(df[["Year", "Month", "Day"]])

    # Find numeric period columns (01, 02, 03, ..., 24/48)
    numeric_cols = sorted(
        [col for col in df.columns if str(col).strip().isdigit()], key=lambda x: int(x)
    )

    if not numeric_cols:
        msg = "No numeric period columns found in Year/Month/Day format"
        raise ValueError(msg)

    # Determine resolution (hourly = 24 columns, half-hourly = 48 columns)
    if len(numeric_cols) == 24:
        resolution_minutes = 60
    elif len(numeric_cols) == 48:
        resolution_minutes = 30
    else:
        resolution_minutes = 60  # Default to hourly

    # Melt the dataframe to long format
    df_long = df.melt(
        id_vars=["datetime"],
        value_vars=numeric_cols,
        var_name="period",
        value_name="value",
    )

    # Convert period number to timedelta (period 1 = 00:00-00:30 or 00:00-01:00)
    df_long["period"] = pd.to_timedelta(
        (df_long["period"].astype(int) - 1) * resolution_minutes, unit="min"
    )

    # Add period offset to datetime
    df_long["datetime"] = df_long["datetime"] + df_long["period"]

    # Select and rename columns
    result = df_long[["datetime", "value"]].copy()
    result.set_index("datetime", inplace=True)

    return result


def _tile_annual_profile_across_years(
    df: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """Tile annual profile (Year=1900) across all years in snapshots.

    For Month/Day/Period formats without Year, the data represents an annual
    repeating pattern. This function tiles that pattern across all modeled years
    and interpolates to match snapshot resolution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index (Year=1900) and value column(s)
    snapshots : pd.DatetimeIndex
        Network snapshots to align with
    interpolation_method : str, default "linear"
        Interpolation method: "linear", "ffill", or "nearest"

    Returns
    -------
    pd.DataFrame
        DataFrame reindexed to snapshots with tiled and interpolated data

    Examples
    --------
    >>> # Monthly sparse data for one year
    >>> df = pd.DataFrame({
    ...     'value': [100, 120, 150, 180, 200, 180, 150, 140, 130, 120, 110, 100]
    ... }, index=pd.date_range('1900-01-01', periods=12, freq='M'))
    >>> snapshots = pd.date_range('2024-01-01', '2026-12-31', freq='H')
    >>> tiled = _tile_annual_profile_across_years(df, snapshots)
    """
    # Extract year range from snapshots
    min_year = snapshots.year.min()
    max_year = snapshots.year.max()

    # Tile pattern across years
    tiled_dfs = []
    for year in range(min_year, max_year + 1):
        year_df = df.copy()
        # Shift datetime from 1900 to actual year
        year_offset = year - 1900
        year_df.index = year_df.index + pd.DateOffset(years=year_offset)
        tiled_dfs.append(year_df)

    combined = pd.concat(tiled_dfs)

    # Interpolate to match snapshots resolution
    if interpolation_method == "linear":
        result = combined.reindex(snapshots).interpolate(method="linear")
    elif interpolation_method == "ffill":
        result = combined.reindex(snapshots, method="ffill")
    else:  # nearest
        result = combined.reindex(snapshots, method="nearest")

    return result


def _parse_periods_in_columns_datetime(
    df: pd.DataFrame, object_name: str | None = None, scenario: str | int | None = None
) -> pd.DataFrame:
    """Parse Periods in Columns format with Datetime + hour columns.

    Used by AEMO and other models with hourly/half-hourly resolution.
    """
    df["datetime"] = pd.to_datetime(
        df["datetime" if "datetime" in df.columns else "Datetime"]
    )

    # Find numeric period columns (1, 2, 3, ..., 24/48)
    numeric_cols = sorted(
        [col for col in df.columns if str(col).strip().isdigit()], key=lambda x: int(x)
    )

    if not numeric_cols:
        msg = "No numeric period columns found"
        raise ValueError(msg)

    # Determine resolution (hourly = 24 columns, half-hourly = 48 columns)
    if len(numeric_cols) == 24:
        resolution_minutes = 60
    elif len(numeric_cols) == 48:
        resolution_minutes = 30
    else:
        resolution_minutes = 60  # Default to hourly

    # Melt the dataframe to long format
    df_long = df.melt(
        id_vars=["datetime"],
        value_vars=numeric_cols,
        var_name="period",
        value_name="value",
    )

    # Convert period to timedelta
    df_long["period"] = pd.to_timedelta(
        (df_long["period"].astype(int) - 1) * resolution_minutes, unit="m"
    )

    # Create full datetime
    df_long["datetime"] = df_long["datetime"].dt.floor("D") + df_long["period"]
    df_long.drop(columns=["period"], inplace=True)

    df_long.set_index("datetime", inplace=True)
    return df_long


def _parse_names_in_columns(
    df: pd.DataFrame, object_name: str | None = None, scenario: str | int | None = None
) -> pd.DataFrame:
    """Parse Names in Columns format.

    Object names are column headers, rows are time periods.
    """
    # Find datetime column
    datetime_col = None
    for col in ["datetime", "Datetime", "Date", "DATE"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        # Try Pattern or Timeslice column
        for col in ["Pattern", "pattern", "Timeslice", "timeslice"]:
            if col in df.columns:
                # For now, skip pattern-based parsing (would need timeslice mapping)
                msg = "Pattern/Timeslice-based CSV parsing not yet implemented. Please use datetime-based CSVs."
                raise NotImplementedError(msg)

        msg = "Could not find datetime column in Names in Columns format"
        raise ValueError(msg)

    df["datetime"] = pd.to_datetime(df[datetime_col])

    # Get object columns (everything except datetime-related columns)
    meta_cols = [
        datetime_col,
        "datetime",
        "Pattern",
        "pattern",
        "Timeslice",
        "timeslice",
        "Iteration",
        "iteration",
    ]
    object_cols = [col for col in df.columns if col not in meta_cols]

    if not object_cols:
        msg = "No object columns found in Names in Columns format"
        raise ValueError(msg)

    # If object_name specified, select that column
    if object_name is not None:
        if object_name not in object_cols:
            msg = f"Object '{object_name}' not found. Available: {object_cols}"
            raise ValueError(msg)
        result = df[["datetime", object_name]].copy()
        result.columns = ["datetime", "value"]
    else:
        # Return all object columns
        result = df[["datetime"] + object_cols].copy()

    result.set_index("datetime", inplace=True)
    return result


def _parse_bands_in_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Bands in Columns format.

    Each column represents a different band (e.g., different cost tranches).
    """
    # Find datetime column
    datetime_col = None
    for col in ["datetime", "Datetime", "Date", "DATE"]:
        if col in df.columns:
            datetime_col = col
            break

    if datetime_col is None:
        msg = "Could not find datetime column in Bands in Columns format"
        raise ValueError(msg)

    df["datetime"] = pd.to_datetime(df[datetime_col])

    # Get band columns (columns with 'band' in name or numeric columns)
    band_cols = [col for col in df.columns if col not in {datetime_col, "datetime"}]

    if not band_cols:
        msg = "No band columns found in Bands in Columns format"
        raise ValueError(msg)

    result = df[["datetime"] + band_cols].copy()
    result.set_index("datetime", inplace=True)
    return result


def load_timeslice_definitions(csv_dir: str | Path) -> pd.DataFrame:
    """Load timeslice definitions from Timeslice.csv with pattern strings.

    This loads the Timeslice.csv file that contains pattern definitions
    (e.g., "M6-9,H16-22" for summer peak hours). These patterns can then
    be parsed using the timeslice_parser module.

    Parameters
    ----------
    csv_dir : str | Path
        Directory containing COAD CSV exports

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: object, category, Include, Include(text)
        Returns empty DataFrame if file doesn't exist.

    Examples
    --------
    >>> timeslice_defs = load_timeslice_definitions(csv_dir)
    >>> timeslice_defs.loc[timeslice_defs["object"] == "SUMMER PEAK", "Include(text)"]
    'M4-9,H8-19'

    Notes
    -----
    This is different from the timeslice activity CSV (DATETIME,NAME,TIMESLICE format)
    used by AEMO models. This function loads the pattern definition CSV used by
    SEM, CAISO, and other models.
    """
    csv_path = Path(csv_dir) / "Timeslice.csv"

    if not csv_path.exists():
        logger.info(f"Timeslice.csv not found at {csv_path}")
        return pd.DataFrame(columns=["object", "category", "Include", "Include(text)"])

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception(f"Error loading {csv_path}")
        return pd.DataFrame(columns=["object", "category", "Include", "Include(text)"])
    else:
        logger.info(f"Loaded {len(df)} timeslice definitions from {csv_path.name}")
        return df
