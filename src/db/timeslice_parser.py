"""Parser for PLEXOS timeslice pattern strings.

This module implements the PLEXOS pattern language for defining time periods.
Patterns are string expressions made up of statements separated by commas (AND) and semicolons (OR).

Pattern Symbols:
    H (1-24): Hour of day (1=midnight to 1am, 24=11pm to midnight)
    W (1-7): Day of week (1=Sunday, 2=Monday, ..., 7=Saturday)
    D (1-31): Day of month
    M (1-12): Month of year (1=January, 2=February, ...)
    P (1-N): Trading period of day (depends on horizon settings)
    Q (1-4): Quarter of year (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
    K (1-53): Week of year (ISO 8601 numbering)
    !: NOT operator (negates the pattern)

Operators:
    , (comma): AND logic - all conditions must be met
    ; (semicolon): OR logic - any condition can be met

Examples:
    "M6-9,H16-22" -> Summer months (Jun-Sep) AND peak hours (4pm-10pm)
    "W1;W2-7,H7-22" -> Sunday OR (weekdays AND hours 7am-10pm)
    "!H1-6" -> All hours EXCEPT 1-6
    "M4-9,H1-3,24" -> Months Apr-Sep AND (hours 1-3 OR hour 24)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parse_plexos_pattern(
    pattern: str, snapshots: pd.DatetimeIndex, trading_periods_per_day: int = 24
) -> pd.Series:
    """Parse a PLEXOS pattern string and return a boolean mask for snapshots.

    Parameters
    ----------
    pattern : str
        PLEXOS pattern string (e.g., "M6-9,H16-22", "W2-7,H7-22")
    snapshots : pd.DatetimeIndex
        Model snapshots to evaluate pattern against
    trading_periods_per_day : int, default 24
        Number of trading periods per day (for P symbol). Default 24 for hourly.

    Returns
    -------
    pd.Series
        Boolean mask indexed by snapshots, True where pattern matches

    Examples
    --------
    Summer peak hours:
    >>> pattern = "M6-9,H16-22"
    >>> mask = parse_plexos_pattern(pattern, snapshots)

    Weekday peak OR weekend:
    >>> pattern = "W2-7,H7-22;W1"
    >>> mask = parse_plexos_pattern(pattern, snapshots)

    Notes
    -----
    - Patterns are case-insensitive
    - Spaces in patterns are ignored
    - Comma (,) = AND logic between statements
    - Semicolon (;) = OR logic between clauses
    """
    if not pattern or pd.isna(pattern):
        logger.warning("Empty or null pattern provided, returning all False")
        return pd.Series(False, index=snapshots)

    # Normalize pattern: uppercase, remove extra spaces
    pattern = str(pattern).upper().replace(" ", "")

    # Split by semicolon (OR logic)
    or_clauses = pattern.split(";")
    result_mask = pd.Series(False, index=snapshots)

    for clause in or_clauses:
        if not clause.strip():
            continue

        # Parse statements from this clause
        # Need to handle "M4-9,H1-3,24" correctly where "H1-3,24" is a single statement
        statements = _extract_statements(clause)
        clause_mask = pd.Series(True, index=snapshots)

        for stmt in statements:
            if not stmt:
                continue

            try:
                stmt_mask = _parse_single_statement(
                    stmt, snapshots, trading_periods_per_day
                )
                clause_mask &= stmt_mask
            except Exception:
                logger.exception(f"Failed to parse statement: {stmt}")
                # On error, mark this clause as invalid (False)
                clause_mask = pd.Series(False, index=snapshots)
                break

        result_mask |= clause_mask

    return result_mask


def _extract_statements(clause: str) -> list[str]:
    """Extract individual statements from a clause.

    Handles cases like "M4-9,H1-3,24" where "H1-3,24" is a single statement.

    Parameters
    ----------
    clause : str
        Clause string with comma-separated statements

    Returns
    -------
    list[str]
        List of statements (e.g., ["M4-9", "H1-3,24"])

    Examples
    --------
    >>> _extract_statements("M4-9,H1-3,24")
    ['M4-9', 'H1-3,24']
    >>> _extract_statements("M6-9,H16-22")
    ['M6-9', 'H16-22']
    >>> _extract_statements("W2-7,H7-22")
    ['W2-7', 'H7-22']
    """
    statements = []
    current_stmt = ""
    i = 0

    while i < len(clause):
        char = clause[i]

        if char == ",":
            # Check if next character is a symbol letter (start of new statement)
            # or a number (continuation of current statement's range)
            if i + 1 < len(clause):
                next_char = clause[i + 1]
                if next_char.isalpha() or next_char == "!":
                    # New statement starts
                    if current_stmt:
                        statements.append(current_stmt.strip())
                    current_stmt = ""
                    i += 1
                    continue
                else:
                    # Continuation of current statement's range
                    current_stmt += char
            else:
                # End of clause
                if current_stmt:
                    statements.append(current_stmt.strip())
                current_stmt = ""
        else:
            current_stmt += char

        i += 1

    # Add last statement
    if current_stmt:
        statements.append(current_stmt.strip())

    return statements


def _parse_single_statement(
    stmt: str, snapshots: pd.DatetimeIndex, trading_periods_per_day: int = 24
) -> pd.Series:
    """Parse a single statement like 'H1-6', 'M4-9', 'W2-7', or '!H1-6'.

    Parameters
    ----------
    stmt : str
        Single pattern statement
    snapshots : pd.DatetimeIndex
        Model snapshots
    trading_periods_per_day : int, default 24
        Number of trading periods per day

    Returns
    -------
    pd.Series
        Boolean mask for this statement
    """
    # Handle NOT operator
    is_negated = stmt.startswith("!")
    if is_negated:
        stmt = stmt[1:]

    if not stmt:
        msg = "Empty statement after removing NOT operator"
        raise ValueError(msg)

    # Extract symbol (first character)
    symbol = stmt[0].upper()
    range_str = stmt[1:]

    if not range_str:
        msg = f"No range provided for symbol {symbol}"
        raise ValueError(msg)

    # Parse the range values
    try:
        values = _parse_range(range_str)
    except Exception:
        logger.exception(f"Failed to parse range: {range_str}")
        raise

    # Apply to snapshots based on symbol
    mask = _apply_symbol_filter(symbol, values, snapshots, trading_periods_per_day)

    # Apply negation if needed
    return ~mask if is_negated else mask


def _parse_range(range_str: str) -> set[int]:
    """Parse a range string like '4-9', '1,5,9', '1-3,10-12', or '1-3,24'.

    Parameters
    ----------
    range_str : str
        Range specification (e.g., "4-9", "1,5,9", "1-3,10-12")

    Returns
    -------
    set[int]
        Set of integer values in the range

    Examples
    --------
    >>> _parse_range("4-9")
    {4, 5, 6, 7, 8, 9}
    >>> _parse_range("1,5,9")
    {1, 5, 9}
    >>> _parse_range("1-3,10-12")
    {1, 2, 3, 10, 11, 12}
    >>> _parse_range("1-3,24")
    {1, 2, 3, 24}
    """
    values = set()
    parts = range_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range notation (e.g., "4-9")
            range_parts = part.split("-")
            if len(range_parts) != 2:
                msg = f"Invalid range format: {part}"
                raise ValueError(msg)

            try:
                start = int(range_parts[0])
                end = int(range_parts[1])
            except ValueError:
                msg = f"Non-integer values in range: {part}"
                raise ValueError(msg) from None

            if start > end:
                msg = f"Invalid range: start ({start}) > end ({end})"
                raise ValueError(msg)

            values.update(range(start, end + 1))
        else:
            # Single value (e.g., "5", "24")
            try:
                values.add(int(part))
            except ValueError:
                msg = f"Non-integer value: {part}"
                raise ValueError(msg) from None

    return values


def _apply_symbol_filter(
    symbol: str,
    values: set[int],
    snapshots: pd.DatetimeIndex,
    trading_periods_per_day: int,
) -> pd.Series:
    """Apply a symbol filter (H, W, D, M, P, Q, K) to snapshots.

    Parameters
    ----------
    symbol : str
        Pattern symbol (H, W, D, M, P, Q, K)
    values : set[int]
        Values to match
    snapshots : pd.DatetimeIndex
        Model snapshots
    trading_periods_per_day : int
        Trading periods per day (for P symbol)

    Returns
    -------
    pd.Series
        Boolean mask
    """
    if symbol == "H":
        # Hour of day (1-24)
        # PLEXOS convention: H1 = midnight to 1am (hour 0), H24 = 11pm to midnight (hour 23)
        # Convert PLEXOS hour (1-24) to Python hour (0-23)
        python_hours = {(h - 1) % 24 for h in values}
        mask = pd.Series([dt.hour in python_hours for dt in snapshots], index=snapshots)

    elif symbol == "W":
        # Day of week (1-7)
        # PLEXOS convention: 1=Sunday, 2=Monday, ..., 7=Saturday
        # Python convention: Monday=0, Sunday=6
        # Convert PLEXOS day (1-7) to Python day (0-6)
        python_days = {
            (w - 1) % 7 for w in values
        }  # 1->0 (Sun), 2->1 (Mon), ..., 7->6 (Sat)
        mask = pd.Series(
            [dt.dayofweek in python_days for dt in snapshots], index=snapshots
        )

    elif symbol == "D":
        # Day of month (1-31)
        mask = pd.Series([dt.day in values for dt in snapshots], index=snapshots)

    elif symbol == "M":
        # Month of year (1-12)
        mask = pd.Series([dt.month in values for dt in snapshots], index=snapshots)

    elif symbol == "P":
        # Trading period of day (1-N)
        # Depends on trading period resolution (hourly = 24, half-hourly = 48, etc.)
        # P1 = first period of day (midnight), P2 = second period, etc.
        period_duration_minutes = 24 * 60 / trading_periods_per_day
        # Calculate period number for each snapshot
        periods = [
            int((dt.hour * 60 + dt.minute) / period_duration_minutes) + 1
            for dt in snapshots
        ]
        mask = pd.Series([p in values for p in periods], index=snapshots)

    elif symbol == "Q":
        # Quarter of year (1-4)
        # Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
        mask = pd.Series([dt.quarter in values for dt in snapshots], index=snapshots)

    elif symbol == "K":
        # Week of year (1-53, ISO 8601)
        # ISO 8601: Week 1 is the first week containing Thursday of the year
        mask = pd.Series(
            [dt.isocalendar()[1] in values for dt in snapshots], index=snapshots
        )

    else:
        logger.warning(f"Unknown pattern symbol: {symbol}, treating as always False")
        mask = pd.Series(False, index=snapshots)

    return mask


def load_and_parse_timeslice_patterns(
    timeslice_df: pd.DataFrame,
    snapshots: pd.DatetimeIndex,
    trading_periods_per_day: int = 24,
) -> pd.DataFrame:
    """Load timeslice definitions and parse patterns into activity matrix.

    This function processes a Timeslice.csv DataFrame with pattern definitions
    (Include(text) column) and converts them to a boolean activity matrix.

    Parameters
    ----------
    timeslice_df : pd.DataFrame
        DataFrame with timeslice definitions (must have 'object' and 'Include(text)' columns)
    snapshots : pd.DatetimeIndex
        Model snapshots
    trading_periods_per_day : int, default 24
        Trading periods per day

    Returns
    -------
    pd.DataFrame
        Boolean activity matrix with index=snapshots, columns=timeslice names

    Examples
    --------
    >>> timeslice_df = pd.read_csv("Timeslice.csv")
    >>> activity = load_and_parse_timeslice_patterns(timeslice_df, network.snapshots)
    >>> activity["SUMMER PEAK"]  # Boolean series for summer peak timeslice
    """
    if "object" not in timeslice_df.columns:
        msg = "Timeslice DataFrame must have 'object' column"
        raise ValueError(msg)

    # Try different column name variations for pattern text
    pattern_col = None
    for col in ["Include(text)", "Include (text)", "pattern", "Pattern"]:
        if col in timeslice_df.columns:
            pattern_col = col
            break

    if pattern_col is None:
        msg = "Timeslice DataFrame must have 'Include(text)' or 'pattern' column"
        raise ValueError(msg)

    # Initialize activity matrix
    timeslice_names = timeslice_df["object"].tolist()
    activity = pd.DataFrame(False, index=snapshots, columns=timeslice_names)

    # Parse each timeslice pattern
    for _, row in timeslice_df.iterrows():
        timeslice_name = row["object"]
        pattern = row[pattern_col]

        if pd.isna(pattern) or pattern == "":
            logger.warning(
                f"Timeslice '{timeslice_name}' has no pattern, treating as always inactive"
            )
            continue

        try:
            mask = parse_plexos_pattern(pattern, snapshots, trading_periods_per_day)
            activity[timeslice_name] = mask
            logger.info(
                f"Parsed timeslice '{timeslice_name}': {mask.sum()}/{len(mask)} active periods"
            )
        except Exception:
            logger.exception(
                f"Failed to parse pattern for timeslice '{timeslice_name}': {pattern}"
            )
            # Leave as False (inactive) on error

    return activity
