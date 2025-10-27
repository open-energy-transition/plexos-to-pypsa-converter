"""Investment-period helper utilities for multi-horizon model support.

The helpers in this module focus on preparing representative-day profiles and
weightings from long-duration half-hourly (or sub-hourly) time-series datasets.
They are intentionally decoupled from specific model IDs so workflows can reuse
them across different PLEXOS exports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TimeClassifier = Callable[[pd.Timestamp], str]


@dataclass(frozen=True, slots=True)
class InvestmentPeriod:
    """Defines a contiguous range of years to be represented as one period."""

    label: str
    start_year: int
    end_year: int

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return True if the timestamp falls inside this period."""
        year = timestamp.year
        return self.start_year <= year <= self.end_year


@dataclass(frozen=True, slots=True)
class RepresentativeProfile:
    """Average intraday trajectory for a specific day-type."""

    name: str
    weight_years: float
    offsets: pd.TimedeltaIndex
    values: pd.DataFrame


def load_timeseries_matrix(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV exported from PLEXOS traces into a long DatetimeIndex table.

    The loader supports both "Period" column formats (Year/Month/Day/Period/value)
    and wide "01-48" column layouts. All numeric columns apart from the date
    scaffolding are preserved.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = {"Year", "Month", "Day"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        msg = f"CSV is missing required columns ({missing}): {csv_path}"
        raise ValueError(msg)

    # Normalise datetime scaffold
    dt_index = pd.to_datetime(df[["Year", "Month", "Day"]])

    if "Period" in df.columns:
        period_series = df["Period"].astype(int)
        resolution = _resolution_from_period(period_series)
        offsets = pd.to_timedelta((period_series - 1) * resolution, unit="m")
        timestamps = dt_index + offsets
        value_cols = [
            col
            for col in df.columns
            if col not in {"Year", "Month", "Day", "Period"}
            and not str(col).startswith("Unnamed")
        ]
        data = df[value_cols].copy()
    else:
        # Columns typically named "01"..."48" or integers
        value_cols = [
            col
            for col in df.columns
            if col not in {"Year", "Month", "Day"}
            and not str(col).startswith("Unnamed")
        ]
        if not value_cols:
            msg = f"No numeric value columns found in {csv_path}"
            raise ValueError(msg)
        melt_df = df.melt(
            id_vars=["Year", "Month", "Day"],
            value_vars=value_cols,
            var_name="_period",
            value_name="_value",
        )
        melt_df["_period"] = melt_df["_period"].astype(int)
        period_series = melt_df["_period"]
        resolution = _resolution_from_period(period_series)
        offsets = pd.to_timedelta((period_series - 1) * resolution, unit="m")
        timestamps = pd.to_datetime(melt_df[["Year", "Month", "Day"]]) + offsets
        data = melt_df[["_value"]].copy()
        data.columns = [csv_path.stem]
        return data.set_index(timestamps).sort_index()

    data = data.set_index(timestamps)
    data.index.name = "timestamp"
    return data.sort_index()


def load_directory_timeseries(
    directory: str | Path,
    pattern: str = "*.csv",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load and concatenate multiple CSV traces from a directory."""
    directory = Path(directory)
    if not directory.exists():
        msg = f"Directory does not exist: {directory}"
        raise FileNotFoundError(msg)

    frames = []
    for idx, csv_path in enumerate(sorted(directory.glob(pattern))):
        if limit is not None and idx >= limit:
            break
        frame = load_timeseries_matrix(csv_path)
        if frame.columns.size == 1:
            frames.append(frame)
        else:
            # Prefix column names with file stem for uniqueness
            renamed = frame.add_prefix(f"{csv_path.stem}__")
            frames.append(renamed)

    if not frames:
        msg = f"No CSV files matched pattern '{pattern}' in {directory}"
        raise ValueError(msg)

    combined = pd.concat(frames, axis=1).sort_index()
    combined.index.name = "timestamp"
    return combined


def profile_to_minutes_dataframe(profile: RepresentativeProfile) -> pd.DataFrame:
    """Return a DataFrame indexed by minutes offset for the given profile."""
    frame = profile.values.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name="value")
    # Ensure index aligns with offsets
    frame.index = profile.offsets
    minutes = frame.index.total_seconds() / 60
    frame.index = minutes
    frame.index.name = "minutes"
    return frame


def write_profile_csv(profile: RepresentativeProfile, target_path: Path) -> None:
    """Persist a representative profile to CSV using minutes offset index."""
    frame = profile_to_minutes_dataframe(profile)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target_path)


def _resolution_from_period(period: pd.Series) -> int:
    """Infer the resolution (minutes) from a period column."""
    max_period = int(period.max())
    if max_period == 24:
        return 60
    if max_period == 48:
        return 30
    if max_period == 96:
        return 15
    # Fallback to generic calculation
    return int(24 * 60 / max_period)


def build_day_type_classifier(
    timeslice_csv: str | Path,
    group_patterns: Mapping[str, Sequence[str]],
    default_group: str = "base",
) -> TimeClassifier:
    """Create a classifier that maps timestamps to day-type groups.

    The classifier uses a month/day lookup table derived from a PLEXOS
    timeslice export. Each NAME entry is matched against provided substring
    patterns to decide the final group label.
    """
    timeslice_df = pd.read_csv(timeslice_csv, parse_dates=["DATETIME"], dayfirst=True)
    if "NAME" not in timeslice_df.columns:
        msg = f"Expected NAME column in timeslice CSV: {timeslice_csv}"
        raise ValueError(msg)

    def assign_group(name: str) -> str:
        for group, patterns in group_patterns.items():
            for pattern in patterns:
                if pattern.lower() in name.lower():
                    return group
        return default_group

    monthday_counter: dict[tuple[int, int], Counter[str]] = {}
    for _, row in timeslice_df.iterrows():
        ts: pd.Timestamp = row["DATETIME"]
        name = str(row["NAME"])
        group = assign_group(name)
        key = (ts.month, ts.day)
        monthday_counter.setdefault(key, Counter()).update([group])

    monthday_mapping: dict[tuple[int, int], str] = {}
    for key, counter in monthday_counter.items():
        group, _ = counter.most_common(1)[0]
        monthday_mapping[key] = group

    def classifier(timestamp: pd.Timestamp) -> str:
        key = (timestamp.month, timestamp.day)
        return monthday_mapping.get(key, default_group)

    return classifier


def compute_period_statistics(
    data: pd.DataFrame,
    periods: Iterable[InvestmentPeriod],
    classifier: TimeClassifier,
) -> dict[str, list[RepresentativeProfile]]:
    """Aggregate intraday profiles and weights for each investment period."""
    results: dict[str, list[RepresentativeProfile]] = {}
    for period in periods:
        mask = data.index.map(period.contains)
        period_data = data.loc[mask]
        if period_data.empty:
            results[period.label] = []
            continue

        # Add helper columns for grouping and averaging
        period_df = period_data.copy()
        period_df["_group"] = period_df.index.map(classifier)
        period_df["_offset"] = period_df.index - period_df.index.floor("D")

        if period_df["_offset"].dt.total_seconds().mod(60).any():
            msg = f"Unexpected non-minute offsets detected in period {period.label}"
            raise ValueError(msg)

        # Compute day counts per group once for weighting
        index_name = period_df.index.name or "index"
        day_lookup = (
            period_df[["_group"]].reset_index().rename(columns={index_name: "datetime"})
        )
        day_lookup["date"] = day_lookup["datetime"].dt.floor("D")
        day_lookup = day_lookup.drop_duplicates(subset=["date"])
        day_counts = day_lookup.groupby("_group")["date"].count().to_dict()

        period_profiles = []
        grouped = period_df.groupby(["_group", "_offset"]).mean(numeric_only=True)
        for group_name in grouped.index.get_level_values(0).unique():
            profile = grouped.xs(group_name, level="_group").sort_index().copy()
            offsets = pd.TimedeltaIndex(profile.index, name="offset")
            profile.index = offsets
            weight_years = day_counts.get(group_name, 0) / 365.0

            period_profiles.append(
                RepresentativeProfile(
                    name=str(group_name),
                    weight_years=weight_years,
                    offsets=offsets,
                    values=profile,
                )
            )

        results[period.label] = period_profiles

    return results
