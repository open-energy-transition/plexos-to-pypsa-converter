from pathlib import Path

import pandas as pd


def load_country_demand_profiles(
    csv_dir: str,
    file_pattern: str = "{country}_*Demand*.csv",
    sector: str = "electricity",
    time_columns: list | None = None,
) -> dict[str, pd.DataFrame]:
    """Load country-specific demand profiles from CSV files.

    Generic function for loading hourly demand data organized by country.
    Supports various file naming conventions and data formats.

    Parameters
    ----------
    csv_dir : str
        Directory containing country demand CSV files
    file_pattern : str, default "{country}_*Demand*.csv"
        Glob pattern for matching demand files. Use {country} placeholder
        for country code extraction
    sector : str, default "electricity"
        Sector identifier for logging and validation
    time_columns : list, optional
        List of hour columns (1-24). If None, auto-detects numeric columns

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping country codes to demand DataFrames with hourly profiles

    Examples
    --------
    >>> gas_demands = load_country_demand_profiles(
    ...     "/path/to/gas/demand", "*_Total_Demand_2030.csv", sector="gas"
    ... )
    >>> elec_demands = load_country_demand_profiles(
    ...     "/path/to/elec/demand", "*_Demand 2030.csv", sector="electricity"
    ... )
    """
    if not Path(csv_dir).exists():
        msg = f"CSV directory not found: {csv_dir}"
        raise FileNotFoundError(msg)

    # Convert pattern to glob pattern
    if "{country}" in file_pattern:
        glob_pattern = file_pattern.replace("{country}", "*")
    else:
        glob_pattern = file_pattern

    demand_profiles = {}

    # Use pathlib for better handling of complex paths with spaces and special characters
    csv_dir_path = Path(csv_dir)
    csv_files = list(csv_dir_path.glob(glob_pattern))
    csv_files = [str(f) for f in csv_files]  # Convert back to strings

    if not csv_files:
        # List what files are actually available for debugging
        try:
            available_files = list(csv_dir_path.iterdir())
            available_names = [f.name for f in available_files if f.is_file()]
            print(
                f"  Available files in directory: {available_names[:10]}..."
            )  # Show first 10
            csv_files_in_dir = [f.name for f in available_files if f.suffix == ".csv"]
            print(
                f"  CSV files found: {csv_files_in_dir[:5]}..."
            )  # Show first 5 CSV files
        except Exception as e:
            print(f"  Could not list directory contents: {e}")

        msg = f"No {sector} demand CSV files found with pattern: {glob_pattern} in directory: {csv_dir}"
        raise FileNotFoundError(msg)

    print(f"Loading {sector} demand profiles from {len(csv_files)} CSV files...")

    for csv_file in sorted(csv_files):
        filename = Path(csv_file).name

        # Extract country code from filename
        country_code = _extract_country_code(filename)
        if not country_code:
            print(
                f"  Warning: Could not extract country code from {filename}, skipping"
            )
            continue

        try:
            # Load CSV with flexible parsing
            df = pd.read_csv(csv_file)

            # Convert to hourly time series
            hourly_profile = _convert_to_hourly_timeseries(df, time_columns)

            if hourly_profile is not None and len(hourly_profile) > 0:
                demand_profiles[country_code] = hourly_profile
                print(f"   Loaded {country_code}: {len(hourly_profile)} hourly values")
            else:
                print(f"  Warning: Empty profile for {country_code}, skipping")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")
            continue

    print(
        f"Successfully loaded {sector} demand profiles for {len(demand_profiles)} countries"
    )
    return demand_profiles


def load_vre_profiles_by_country(
    csv_dir: str,
    wind_file: str = "EU-28 Wind Power Profiles.csv",
    solar_file: str = "SolarGeneration_EU28.csv",
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load Variable Renewable Energy (VRE) profiles by country and technology.

    Parameters
    ----------
    csv_dir : str
        Directory containing VRE profile CSV files
    wind_file : str
        Filename for wind generation profiles
    solar_file : str
        Filename for solar generation profiles

    Returns
    -------
    Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary: {country: {technology: DataFrame}}
        Technologies: 'wind_onshore', 'wind_offshore', 'solar_pv'
    """
    vre_profiles = {}

    # Load wind profiles
    wind_path = Path(csv_dir) / wind_file
    if wind_path.exists():
        print(f"Loading wind profiles from {wind_file}...")
        wind_data = pd.read_csv(wind_path)

        # Parse wind profiles by country and technology
        wind_profiles = _parse_vre_profiles(wind_data, "wind")

        for country, profiles in wind_profiles.items():
            if country not in vre_profiles:
                vre_profiles[country] = {}
            vre_profiles[country].update(profiles)

        print(f"   Loaded wind profiles for {len(wind_profiles)} countries")

    # Load solar profiles
    solar_path = Path(csv_dir) / solar_file
    if solar_path.exists():
        print(f"Loading solar profiles from {solar_file}...")
        solar_data = pd.read_csv(solar_path)

        # Parse solar profiles by country
        solar_profiles = _parse_vre_profiles(solar_data, "solar")

        for country, profiles in solar_profiles.items():
            if country not in vre_profiles:
                vre_profiles[country] = {}
            vre_profiles[country].update(profiles)

        print(f"   Loaded solar profiles for {len(solar_profiles)} countries")

    return vre_profiles


def load_cross_border_infrastructure(
    csv_file: str, infrastructure_type: str = "gas_flow"
) -> pd.DataFrame:
    """Load cross-border infrastructure data from CSV file.

    Parameters
    ----------
    csv_file : str
        Path to infrastructure CSV file
    infrastructure_type : str
        Type of infrastructure ('gas_flow', 'electricity_transmission', etc.)

    Returns
    -------
    pd.DataFrame
        Infrastructure data with standardized format
    """
    if not Path(csv_file).exists():
        msg = f"Infrastructure CSV file not found: {csv_file}"
        raise FileNotFoundError(msg)

    print(f"Loading {infrastructure_type} infrastructure from {Path(csv_file).name}...")

    df = pd.read_csv(csv_file)

    # Standardize infrastructure data format
    infrastructure_data = _standardize_infrastructure_format(df, infrastructure_type)

    print(f"   Loaded {len(infrastructure_data)} {infrastructure_type} connections")
    return infrastructure_data


def load_marei_gas_pricing(
    csv_dir: str, pricing_scheme: str = "Production"
) -> dict[str, pd.DataFrame]:
    """Load MaREI-specific gas pricing data.

    Parameters
    ----------
    csv_dir : str
        Directory containing gas pricing CSV files
    pricing_scheme : str
        Pricing scheme to load ('Production', 'Postage', 'Trickle', 'Uniform')

    Returns
    -------
    Dict[str, pd.DataFrame]
        Gas pricing data by type (production_prices, postage_charges, etc.)
    """
    pricing_files = {
        "Production": "Production Prices.csv",
        "Postage": "Postage Stamp Charges.csv",
        "Trickle": "Trickle Charges.csv",
        "Uniform": "Uniform Charges.csv",
    }

    if pricing_scheme not in pricing_files:
        msg = f"Unknown pricing scheme: {pricing_scheme}. Available: {list(pricing_files.keys())}"
        raise ValueError(msg)

    pricing_data = {}
    pricing_file = pricing_files[pricing_scheme]
    csv_path = Path(csv_dir) / pricing_file

    if csv_path.exists():
        print(f"Loading {pricing_scheme} gas pricing from {pricing_file}...")
        df = pd.read_csv(csv_path)
        pricing_data[pricing_scheme.lower()] = df
        print(f"   Loaded {len(df)} pricing entries")
    else:
        print(f"  Warning: Pricing file not found: {csv_path}")

    return pricing_data


def load_marei_infrastructure_scenarios(
    csv_dir: str, scenario: str = "PCI"
) -> dict[str, pd.DataFrame]:
    """Load MaREI infrastructure scenario data.

    Parameters
    ----------
    csv_dir : str
        Directory containing infrastructure scenario CSV files
    scenario : str
        Infrastructure scenario ('PCI', 'High', 'Low')

    Returns
    -------
    Dict[str, pd.DataFrame]
        Infrastructure data by type (flow, storage_cap, storage_inj, etc.)
    """
    scenario_files = {
        "flow": f"{scenario}_Flow.csv",
        "storage_cap": f"{scenario}_Storage_Cap.csv",
        "storage_inj": f"{scenario}_Storage_Inj.csv",
        "storage_with": f"{scenario}_Storage_With.csv",
        "lng": f"{scenario}_LNG.csv",
        "backflow": f"{scenario}_Backflow.csv",
    }

    scenario_data = {}

    for data_type, filename in scenario_files.items():
        csv_path = Path(csv_dir) / filename
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                scenario_data[data_type] = df
                print(f"   Loaded {scenario} {data_type}: {len(df)} entries")
            except Exception as e:
                print(f"  Warning: Error loading {filename}: {e}")
        else:
            print(f"  Info: {filename} not found (optional)")

    return scenario_data


def load_marei_full_dataset(
    csv_base_dir: str, scenario: str = "PCI", pricing_scheme: str = "Production"
) -> dict[str, dict[str, pd.DataFrame] | pd.DataFrame]:
    """Load complete MaREI dataset with all CSV files.

    Parameters
    ----------
    csv_base_dir : str
        Base directory containing all MaREI CSV subdirectories
    scenario : str
        Infrastructure scenario ('PCI', 'High', 'Low')
    pricing_scheme : str
        Gas pricing scheme ('Production', 'Postage', 'Trickle', 'Uniform')

    Returns
    -------
    Dict[str, Union[Dict[str, pd.DataFrame], pd.DataFrame]]
        Complete dataset organized by data type
    """
    print(
        f"Loading complete MaREI dataset (Scenario: {scenario}, Pricing: {pricing_scheme})..."
    )

    full_dataset = {}

    # Load gas demand profiles
    gas_demand_dir = Path(csv_base_dir) / "Gas Demand"
    if gas_demand_dir.exists():
        full_dataset["gas_demand"] = load_country_demand_profiles(
            gas_demand_dir, "*_Total_Demand_2030.csv", sector="gas"
        )

    # Load electricity demand profiles
    elec_demand_dir = Path(csv_base_dir) / "Load Files 2012"
    if elec_demand_dir.exists():
        full_dataset["electricity_demand"] = load_country_demand_profiles(
            elec_demand_dir, "*_Demand 2030.csv", sector="electricity"
        )

    # Load VRE profiles
    wind_dir = Path(csv_base_dir) / "Normalised Wind Generation"
    solar_dir = Path(csv_base_dir) / "Normalised Solar Generation"
    if wind_dir.exists() or solar_dir.exists():
        # Try wind directory first, fall back to base directory
        vre_dir = str(wind_dir if wind_dir.exists() else Path(csv_base_dir))
        full_dataset["vre_profiles"] = load_vre_profiles_by_country(vre_dir)

    # Load infrastructure scenarios
    infra_dir = Path(csv_base_dir) / "Infrastructure Scenarios"
    if infra_dir.exists():
        full_dataset["infrastructure"] = load_marei_infrastructure_scenarios(
            infra_dir, scenario
        )

    # Load gas infrastructure base data
    gas_infra_dir = Path(csv_base_dir) / "Gas Infrastructure"
    if gas_infra_dir.exists():
        full_dataset["gas_infrastructure"] = load_marei_infrastructure_scenarios(
            gas_infra_dir, scenario
        )

    # Load gas pricing data
    pricing_dir = Path(csv_base_dir) / "Gas Pricing"
    if pricing_dir.exists():
        full_dataset["gas_pricing"] = load_marei_gas_pricing(
            pricing_dir, pricing_scheme
        )

    # Load hydro profiles
    hydro_file = Path(csv_base_dir) / "Hydro Monthly Profiles.csv"
    if hydro_file.exists():
        full_dataset["hydro_profiles"] = pd.read_csv(hydro_file)
        print("   Loaded hydro monthly profiles")

    print(f" Complete MaREI dataset loaded with {len(full_dataset)} data categories")
    return full_dataset


# Helper functions


def _extract_country_code(filename: str) -> str | None:
    """Extract 2-letter country code from filename."""
    # Try common patterns: AT_*, BE_*, etc.
    parts = filename.split("_")
    if len(parts) > 0:
        potential_code = parts[0]
        if len(potential_code) == 2 and potential_code.isalpha():
            return potential_code.upper()

    # Try other patterns if needed
    if filename.startswith(("CH_", "NI_", "NO_")):
        return filename[:2].upper()

    return None


def _convert_to_hourly_timeseries(
    df: pd.DataFrame, time_columns: list | None = None
) -> pd.DataFrame | None:
    """Convert CSV data to hourly time series format."""
    if df.empty:
        return None

    # Auto-detect hour columns if not specified
    if time_columns is None:
        # Look for numeric columns (typically hour columns 1-24)
        numeric_cols = [col for col in df.columns if str(col).isdigit()]
        if numeric_cols:
            time_columns = sorted(numeric_cols, key=int)
        else:
            # Fall back to all numeric columns except date columns
            time_columns = [
                col
                for col in df.columns
                if col not in ["Year", "Month", "Day"]
                and pd.api.types.is_numeric_dtype(df[col])
            ]

    if not time_columns:
        print("    Warning: No time columns found")
        return None

    if all(col in df.columns for col in ["Year", "Month", "Day"]):
        try:
            # Create hourly datetime index
            hourly_data = []
            for _, row in df.iterrows():
                year, month, day = int(row["Year"]), int(row["Month"]), int(row["Day"])
                for hour_col in time_columns:
                    hour = (
                        int(hour_col)
                        if str(hour_col).isdigit()
                        else int(hour_col.split("_")[-1])
                    )
                    dt = pd.Timestamp(
                        year=year, month=month, day=day, hour=hour - 1
                    )  # Convert to 0-23
                    hourly_data.append({"datetime": dt, "value": row[hour_col]})

            if hourly_data:
                hourly_df = pd.DataFrame(hourly_data)
                hourly_df.set_index("datetime", inplace=True)
                return hourly_df["value"]

        except Exception as e:
            print(f"    Warning: Could not create datetime index: {e}")

    # Fall back to simple concatenation of hourly values
    try:
        hourly_values = []
        for _, row in df.iterrows():
            hourly_values.extend([row[hour_col] for hour_col in time_columns])

        return pd.Series(hourly_values, name="demand")
    except Exception as e:
        print(f"    Error converting to hourly series: {e}")
        return None


def _parse_vre_profiles(
    df: pd.DataFrame, technology: str
) -> dict[str, dict[str, pd.DataFrame]]:
    """Parse VRE profiles by country and technology from wide-format CSV."""
    profiles = {}

    # Look for country-specific columns
    country_columns = {}

    for col in df.columns:
        if col == "Datetime":
            continue

        # Parse column name for country and technology info
        # Expected format: "Technology-COUNTRY" (e.g., "Wind Onshore-AT")
        if "-" in col:
            tech_part, country = col.split("-", 1)
            tech_part = tech_part.strip()
            country = country.strip()

            # Standardize technology names
            if "wind onshore" in tech_part.lower():
                tech_name = "wind_onshore"
            elif "wind offshore" in tech_part.lower():
                tech_name = "wind_offshore"
            elif "solar" in tech_part.lower():
                tech_name = "solar_pv"
            else:
                tech_name = tech_part.lower().replace(" ", "_")

            if country not in country_columns:
                country_columns[country] = {}
            country_columns[country][tech_name] = col

    # Extract profiles for each country and technology
    for country, tech_cols in country_columns.items():
        profiles[country] = {}

        for tech_name, col_name in tech_cols.items():
            try:
                profile_data = df[col_name].copy()

                # Handle datetime index if available
                if "Datetime" in df.columns:
                    profile_data.index = pd.to_datetime(df["Datetime"])

                profiles[country][tech_name] = profile_data
            except Exception as e:
                print(f"    Warning: Error parsing {tech_name} for {country}: {e}")

    return profiles


def _standardize_infrastructure_format(
    df: pd.DataFrame, infrastructure_type: str
) -> pd.DataFrame:
    """Standardize infrastructure data format for consistent processing."""
    # This function would be expanded based on specific infrastructure data formats
    # For now, return the data as-is with basic validation

    if df.empty:
        return df

    # Basic validation and cleaning
    standardized_df = df.copy()

    # Remove any completely empty rows/columns
    standardized_df = standardized_df.dropna(how="all")
    standardized_df = standardized_df.loc[
        :, ~standardized_df.columns.str.contains("^Unnamed")
    ]

    return standardized_df
