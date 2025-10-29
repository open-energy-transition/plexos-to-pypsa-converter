"""Investment-period helper utilities for multi-horizon model support.

The helpers in this module focus on preparing representative-day profiles and
weightings from long-duration half-hourly (or sub-hourly) time-series datasets.
They are intentionally decoupled from specific model IDs so workflows can reuse
them across different PLEXOS exports.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pypsa import Network as PyPSANetwork
else:  # pragma: no cover - runtime type fallback
    PyPSANetwork = Any

logger = logging.getLogger(__name__)

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
    base_timestamp: pd.Timestamp | None = None


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
        if data.shape[1] == 1:
            data.columns = [csv_path.stem]
        else:
            prefixed = {
                col: f"{csv_path.stem}__{str(col).strip()}" for col in data.columns
            }
            data = data.rename(columns=prefixed)
    else:
        value_cols = [
            col
            for col in df.columns
            if col not in {"Year", "Month", "Day"}
            and not str(col).startswith("Unnamed")
        ]
        if not value_cols:
            msg = f"No numeric value columns found in {csv_path}"
            raise ValueError(msg)

        numeric_headers = all(str(col).strip().isdigit() for col in value_cols)

        if numeric_headers:
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

        # Fallback: treat as daily profile(s) without explicit period columns.
        data = df[value_cols].copy()
        timestamps = dt_index

        if data.shape[1] == 1:
            data.columns = [csv_path.stem]
        else:
            prefixed = {
                col: f"{csv_path.stem}__{str(col).strip()}" for col in data.columns
            }
            data = data.rename(columns=prefixed)

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
        try:
            frame = load_timeseries_matrix(csv_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "Skipping %s during investment-period aggregation: %s",
                csv_path,
                exc,
            )
            continue
        frames.append(frame)

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


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load investment period metadata JSON written by the aggregation step."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        msg = f"Investment period manifest not found at {manifest_path}"
        raise FileNotFoundError(msg)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_snapshot_multiindex(
    manifest: dict[str, Any],
    period_order: Sequence[str] | None = None,
    representative_order: Sequence[str] | None = None,
    profile_offsets: Mapping[str, pd.TimedeltaIndex] | None = None,
    base_timestamp: pd.Timestamp | None = None,
    period_base_dates: Mapping[str, pd.Timestamp] | None = None,
    profile_base_dates: Mapping[str, pd.Timestamp] | None = None,
) -> tuple[pd.MultiIndex, pd.Series, pd.Series, dict[str, pd.Timestamp]]:
    """Construct MultiIndex snapshots and weighting series from manifest data."""
    if profile_offsets is None:
        msg = "profile_offsets must be provided to build snapshot index"
        raise ValueError(msg)

    if base_timestamp is None:
        base_timestamp = pd.Timestamp("2000-01-01")

    periods = _detect_period_labels(manifest, period_order)
    if period_base_dates is None:
        period_base_dates = _infer_period_base_dates(periods, base_timestamp)
    representative_names = profile_offsets.keys()
    base_datetimes = _assign_base_datetimes(
        representative_names,
        representative_order,
        base_timestamp,
    )

    snapshot_labels: list[tuple[str, pd.Timestamp]] = []
    snapshot_weights: dict[tuple[str, pd.Timestamp], float] = {}
    period_weights: dict[str, float] = {}

    for period in periods:
        groups = _collect_groups(manifest, period, representative_order)
        total_weight = 0.0
        for group_name, weight_years in groups:
            offsets = profile_offsets.get(group_name)
            if offsets is None or len(offsets) == 0:
                continue
            total_weight += weight_years
            per_snapshot_weight = weight_years / len(offsets)
            base_override = None
            if profile_base_dates is not None:
                base_override = profile_base_dates.get(group_name)
            if base_override is not None:
                timestamps = base_override + offsets
            else:
                group_offset = base_datetimes[group_name] - base_timestamp
                period_base = period_base_dates.get(period, base_timestamp)
                timestamps = period_base + group_offset + offsets
            for ts in timestamps:
                key = (period, ts)
                snapshot_labels.append(key)
                snapshot_weights[key] = per_snapshot_weight
        period_weights[period] = total_weight

    snapshots_index = pd.MultiIndex.from_tuples(
        snapshot_labels,
        names=["period", "timestamp"],
    )
    period_series = pd.Series(period_weights, name="period_weight_years")
    snapshot_series = (
        pd.Series(snapshot_weights, name="snapshot_weight_years")
        .reindex(snapshots_index)
        .fillna(0.0)
    )
    return snapshots_index, period_series, snapshot_series, dict(period_base_dates)


def infer_profile_offsets(
    manifest: dict[str, Any],
    base_dir: str | Path,
) -> dict[str, pd.TimedeltaIndex]:
    """Infer intraday offsets for each representative day profile."""
    base_dir = Path(base_dir)
    offsets: dict[str, pd.TimedeltaIndex] = {}
    for group_name, profile_set in manifest.items():
        if group_name.startswith("_") or not isinstance(profile_set, dict):
            continue
        for entries in profile_set.values():
            for entry in entries:
                name = entry["name"]
                if name in offsets:
                    continue
                csv_path = base_dir / entry["csv"]
                df = pd.read_csv(csv_path)
                if "minutes" in df.columns:
                    minutes = df["minutes"].to_numpy()
                else:
                    count = max(len(df), 1)
                    step = 1440 / count
                    minutes = [idx * step for idx in range(count)]
                offsets[name] = pd.to_timedelta(minutes, unit="m")
    return offsets


def load_group_profiles(
    manifest: dict[str, Any],
    base_dir: str | Path,
    group: str,
    profile_offsets: Mapping[str, pd.TimedeltaIndex] | None = None,
    representative_order: Sequence[str] | None = None,
    base_timestamp: pd.Timestamp | None = None,
    period_base_dates: Mapping[str, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Load representative-day profiles for a specific group into MultiIndex frame."""
    base_dir = Path(base_dir)
    group_data = manifest.get(group)
    if not group_data:
        msg = f"Group '{group}' not found in investment manifest keys: {list(manifest.keys())}"
        raise KeyError(msg)

    if profile_offsets is None:
        profile_offsets = infer_profile_offsets(manifest, base_dir)

    if base_timestamp is None:
        base_timestamp = pd.Timestamp("2000-01-01")

    base_datetimes = _assign_base_datetimes(
        profile_offsets.keys(), representative_order, base_timestamp
    )

    if period_base_dates is None:
        period_base_dates = _extract_period_base_dates(
            manifest,
            sorted(group_data.keys()),
            base_timestamp,
        )

    frames: list[pd.DataFrame] = []
    for period_label, entries in group_data.items():
        for entry in entries:
            csv_path = base_dir / entry["csv"]
            df = pd.read_csv(csv_path)
            if "minutes" in df.columns:
                df = df.drop(columns="minutes")

            offsets = profile_offsets.get(entry["name"])
            if offsets is None or len(offsets) == 0:
                continue

            if len(df) != len(offsets):  # pragma: no cover - defensive logging
                logger.warning(
                    "Profile length mismatch for '%s' (%d rows vs %d offsets)",
                    entry["name"],
                    len(df),
                    len(offsets),
                )
                offsets = offsets[: len(df)]

            entry_base = entry.get("base")
            if entry_base is not None:
                base_dt = pd.Timestamp(entry_base)
                timestamps = base_dt + offsets
            else:
                group_offset = base_datetimes[entry["name"]] - base_timestamp
                period_base = period_base_dates.get(period_label, base_timestamp)
                timestamps = period_base + group_offset + offsets
            index = pd.MultiIndex.from_arrays(
                [[period_label] * len(timestamps), timestamps],
                names=["period", "timestamp"],
            )
            df.index = index
            frames.append(df)

    if not frames:
        msg = f"No profiles found for group '{group}' in manifest."
        raise ValueError(msg)

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_index()
    return combined


def configure_investment_periods(
    network: PyPSANetwork,
    manifest: dict[str, Any],
    base_dir: str | Path,
    period_order: Sequence[str] | None = None,
    representative_order: Sequence[str] | None = None,
) -> tuple[
    pd.MultiIndex,
    pd.Series,
    pd.Series,
    dict[str, pd.TimedeltaIndex],
    dict[str, pd.Timestamp],
]:
    """Apply investment-period snapshot structure and weights to a network."""
    default_base = pd.Timestamp("2000-01-01")
    profile_offsets = infer_profile_offsets(manifest, base_dir)
    inferred_periods = _detect_period_labels(manifest, period_order)
    period_base_dates = _extract_period_base_dates(
        manifest,
        inferred_periods,
        default_base,
    )
    profile_base_dates = _extract_profile_bases(manifest)

    (
        snapshots_index,
        period_weights,
        snapshot_weights,
        period_base_dates,
    ) = build_snapshot_multiindex(
        manifest,
        period_order=period_order,
        representative_order=representative_order,
        profile_offsets=profile_offsets,
        period_base_dates=period_base_dates,
        profile_base_dates=profile_base_dates,
    )

    network.set_snapshots(snapshots_index)
    snapshot_df = network.snapshot_weightings.copy()
    snapshot_df = snapshot_df.reindex(snapshots_index)
    snapshot_df["objective"] = snapshot_weights.to_numpy()
    snapshot_df["years"] = snapshot_weights.to_numpy()
    network.snapshot_weightings = snapshot_df

    period_df = pd.DataFrame(
        {
            "objective": period_weights,
            "years": period_weights,
        }
    )
    period_df.index.name = "period"
    network.investment_period_weightings = period_df

    return (
        snapshots_index,
        period_weights,
        snapshot_weights,
        profile_offsets,
        period_base_dates,
    )


def get_snapshot_timestamps(snapshots: pd.Index) -> pd.DatetimeIndex:
    """Return the datetime level of a snapshots index (supports MultiIndex)."""
    if isinstance(snapshots, pd.MultiIndex):
        for name in ("timestamp", "timestep", "time", "datetime"):
            if name in snapshots.names:
                return pd.DatetimeIndex(snapshots.get_level_values(name))
        return pd.DatetimeIndex(snapshots.get_level_values(-1))
    return pd.DatetimeIndex(snapshots)


def _detect_period_labels(
    manifest: dict[str, Any],
    order: Sequence[str] | None,
) -> list[str]:
    labels: set[str] = set()
    for group_name, group_data in manifest.items():
        if group_name.startswith("_") or not isinstance(group_data, dict):
            continue
        labels.update(group_data.keys())
    if order:
        ordered = [label for label in order if label in labels]
        ordered.extend(sorted(label for label in labels if label not in ordered))
        return ordered
    return sorted(labels)


def _collect_groups(
    manifest: dict[str, Any],
    period_label: str,
    representative_order: Sequence[str] | None,
) -> list[tuple[str, float]]:
    weights: dict[str, float] = {}
    for profile_set in manifest.values():
        entries = profile_set.get(period_label, [])
        for entry in entries:
            name = entry["name"]
            weight = float(entry["weight_years"])
            weights[name] = weights.get(name, 0.0) + weight

    if representative_order:
        ordered = [
            (name, weights[name]) for name in representative_order if name in weights
        ]
        remainder = sorted(
            (
                (name, w)
                for name, w in weights.items()
                if name not in representative_order
            ),
            key=lambda item: item[0],
        )
        return ordered + remainder

    return sorted(weights.items(), key=lambda item: item[0])


def _order_representatives(
    names: Iterable[str],
    representative_order: Sequence[str] | None,
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    if representative_order:
        for name in representative_order:
            if name in names and name not in seen:
                ordered.append(name)
                seen.add(name)

    for name in sorted(names):
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    return ordered


def _assign_base_datetimes(
    names: Iterable[str],
    representative_order: Sequence[str] | None,
    base_timestamp: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    ordered = _order_representatives(names, representative_order)
    return {
        name: base_timestamp + pd.to_timedelta(idx, unit="D")
        for idx, name in enumerate(ordered)
    }


def _extract_profile_bases(manifest: dict[str, Any]) -> dict[str, pd.Timestamp]:
    bases: dict[str, pd.Timestamp] = {}
    for group_name, group_data in manifest.items():
        if group_name.startswith("_") or not isinstance(group_data, dict):
            continue
        for entries in group_data.values():
            for entry in entries:
                base_val = entry.get("base")
                if base_val is None:
                    continue
                try:
                    bases[entry["name"]] = pd.Timestamp(base_val)
                except (ValueError, TypeError):  # pragma: no cover - invalid metadata
                    continue
    return bases


def _extract_period_base_dates(
    manifest: dict[str, Any],
    periods: Sequence[str],
    default_base: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    bases: dict[str, pd.Timestamp] = {}
    raw_meta = manifest.get("_meta", {})
    raw_bases = raw_meta.get("period_bases", {})
    for label in periods:
        value = raw_bases.get(label)
        if value is None:
            continue
        try:
            bases[label] = pd.Timestamp(value)
        except (ValueError, TypeError):  # pragma: no cover - invalid metadata
            continue
    # Fill missing periods with inferred defaults
    missing = [label for label in periods if label not in bases]
    if missing:
        inferred = _infer_period_base_dates(missing, default_base)
        bases.update(inferred)
    return bases


def _infer_period_base_dates(
    periods: Sequence[str],
    base_timestamp: pd.Timestamp,
) -> dict[str, pd.Timestamp]:
    base_dates: dict[str, pd.Timestamp] = {}
    for idx, label in enumerate(periods):
        match = re.search(r"(\d{4})", str(label))
        if match:
            year = int(match.group(1))
            try:
                base_dates[label] = pd.Timestamp(year=year, month=1, day=1)
                continue
            except ValueError:  # pragma: no cover - invalid year safeguard
                pass
        base_dates[label] = base_timestamp + pd.to_timedelta(idx, unit="Y")
    return base_dates


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
) -> tuple[dict[str, list[RepresentativeProfile]], dict[str, pd.Timestamp]]:
    """Aggregate intraday profiles and weights for each investment period."""
    results: dict[str, list[RepresentativeProfile]] = {}
    period_bases: dict[str, pd.Timestamp] = {}
    for period in periods:
        mask = data.index.map(period.contains)
        period_data = data.loc[mask]
        if period_data.empty:
            results[period.label] = []
            period_bases[period.label] = pd.Timestamp(period.start_year, 1, 1)
            continue

        # Record canonical base date (start of first day represented in this period)
        period_start = period_data.index.min().floor("D")
        period_bases[period.label] = period_start

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
            group_mask = period_df["_group"] == group_name
            base_timestamp = (
                period_df.loc[group_mask].index.min().floor("D")
                if group_mask.any()
                else None
            )

            period_profiles.append(
                RepresentativeProfile(
                    name=str(group_name),
                    weight_years=weight_years,
                    offsets=offsets,
                    values=profile,
                    base_timestamp=base_timestamp,
                )
            )

        results[period.label] = period_profiles

    return results, period_bases
