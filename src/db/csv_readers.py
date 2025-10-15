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


def parse_numeric_value(value: Any, use_first: bool = True) -> float | None:
    """Parse a numeric value that might be a string representation of a list.

    When COAD exports PLEXOS data to CSV, properties with multiple values
    (e.g., multiple generator units) are stored as string representations of lists.
    This function handles various formats:
    - Direct numeric values: 55.197
    - String numbers: "55.197"
    - String representations of lists: "['55.197', '62.606']"

    Parameters
    ----------
    value : Any
        Value to parse (could be number, string, or string representation of list)
    use_first : bool, default True
        If value is a list, whether to use first element (True) or sum all (False).
        - Use True for properties that represent a characteristic (e.g., efficiency)
        - Use False for properties that represent a total (e.g., total capacity)

    Returns
    -------
    float | None
        Parsed numeric value, or None if parsing fails

    Examples
    --------
    Parse a simple string number:
    >>> parse_numeric_value("55.197")
    55.197

    Parse a list string, taking first value:
    >>> parse_numeric_value("['55.197', '62.606']")
    55.197

    Parse a list string, summing all values:
    >>> parse_numeric_value("['55.197', '62.606']", use_first=False)
    117.803

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
                if use_first:
                    return float_values[0]
                else:
                    return sum(float_values)
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
        The name of the associated Fuel, or None if not found.

    Examples
    --------
    >>> generators = load_static_properties(csv_dir, "Generator")
    >>> fuel = find_fuel_for_generator_csv(generators, "AA1")
    >>> print(fuel)
    None  # Hydro generators may not have fuel
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
