"""Feature detection for auto-building workflows.

This module inspects CSV exports (Data File.csv, Region.csv, Node.csv)
to discover available inputs such as demand, VRE traces, hydro data,
units schedules, and outage information. The goal is to populate a
lightweight ModelFeatures object that can drive an auto-generated
workflow without manual step lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from plexos_to_pypsa_converter.db.csv_readers import (
    load_static_properties,
    load_time_varying_properties,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass
class ModelFeatures:
    """Detected model assets that influence workflow construction."""

    model_dir: Path
    csv_dir: Path
    demand_files: list[Path] = field(default_factory=list)
    vre_profile_files: list[Path] = field(default_factory=list)
    hydro_dispatch_files: list[Path] = field(default_factory=list)
    hydro_inflow_files: list[Path] = field(default_factory=list)
    units_files: list[Path] = field(default_factory=list)
    outage_files: list[Path] = field(default_factory=list)
    region_csv: Path | None = None
    node_csv: Path | None = None
    has_region_factors: bool = False
    has_node_region_map: bool = False
    units_out_dir: Path | None = None
    has_outage_properties: bool = False


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _normalize_path(raw: str, base_dir: Path) -> Path:
    normalized = str(raw).replace("\\", "/")
    candidate = base_dir / normalized
    return candidate


def _collect_filename(row: pd.Series) -> str | None:
    for key in ("Filename(text)", "Filename"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _extract_units_out_dir(candidate: Path) -> Path | None:
    """Return the top-level Units Out directory from a path."""
    if "units out" not in (part.lower() for part in candidate.parts):
        return None
    parts = list(candidate.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "units out":
            return Path(*parts[: idx + 1])
    return None


def _detect_outage_properties(csv_dir: Path) -> bool:
    """Check static and time-varying properties for outage-related fields."""
    outage_properties = {
        "Forced Outage Rate",
        "Maintenance Rate",
        "Mean Time to Repair",
        "Outage Factor",
        "Outage Rating",
    }
    static_df = load_static_properties(csv_dir, "Generator")
    if not static_df.empty and any(
        col in static_df.columns for col in outage_properties
    ):
        return True

    tvp_df = load_time_varying_properties(csv_dir, class_name="Generator")
    return bool(
        not tvp_df.empty
        and tvp_df["property"]
        .str.lower()
        .isin([p.lower() for p in outage_properties])
        .any()
    )


def _maybe_extend(paths: list[Path], candidates: Iterable[str], base_dir: Path) -> None:
    for cand in candidates:
        if not cand:
            continue
        candidate_path = _normalize_path(cand, base_dir)
        paths.append(candidate_path)


def detect_model_features(model_dir: Path, csv_dir: Path) -> ModelFeatures:
    """Inspect CSV exports to discover available inputs for workflow steps."""
    features = ModelFeatures(model_dir=model_dir, csv_dir=csv_dir)
    units_out_candidates: list[Path] = []

    # Data File.csv driven detection
    data_file_path = csv_dir / "Data File.csv"
    data_df = _safe_read_csv(data_file_path)
    if data_df is not None and not data_df.empty:
        for _idx, row in data_df.iterrows():
            filename = _collect_filename(row)
            if not filename:
                continue

            category = str(row.get("category", "")).lower()
            obj = str(row.get("object", "")).lower()

            # Demand / load
            if any(tag in category for tag in ["demand", "load"]) or obj.endswith(
                "demand"
            ):
                _maybe_extend(features.demand_files, [filename], model_dir)
                continue

            # VRE
            if any(tag in obj for tag in ["solar", "wind", "pv"]):
                _maybe_extend(features.vre_profile_files, [filename], model_dir)
                continue

            # Hydro dispatch or inflow
            if "hydro" in category or "hydro" in obj:
                # Use category/object to guess whether this is inflow (storage) vs dispatch
                if "inflow" in obj or "inflow" in category:
                    _maybe_extend(features.hydro_inflow_files, [filename], model_dir)
                else:
                    _maybe_extend(features.hydro_dispatch_files, [filename], model_dir)
                continue

            # Units
            if "units" in obj or "units" in category:
                _maybe_extend(features.units_files, [filename], model_dir)
                continue

            # Outages
            if "outage" in obj or "outage" in category:
                _maybe_extend(features.outage_files, [filename], model_dir)
                units_out_dir = _extract_units_out_dir(
                    _normalize_path(filename, model_dir)
                )
                if units_out_dir:
                    units_out_candidates.append(units_out_dir)
                continue
            if "units out" in filename.lower():
                units_out_dir = _extract_units_out_dir(
                    _normalize_path(filename, model_dir)
                )
                if units_out_dir:
                    units_out_candidates.append(units_out_dir)

    # Region / Node information
    region_path = csv_dir / "Region.csv"
    node_path = csv_dir / "Node.csv"
    region_df = (
        load_static_properties(csv_dir, "Region") if region_path.exists() else None
    )
    node_df = load_static_properties(csv_dir, "Node") if node_path.exists() else None

    if region_df is not None and not region_df.empty:
        features.region_csv = region_path
        if any(col in region_df.columns for col in ["Load", "Load.Variable"]):
            features.has_region_factors = True

    if node_df is not None and not node_df.empty:
        features.node_csv = node_path
        features.has_node_region_map = "Region" in node_df.columns

    # Units Out directory detection (prefer existing paths)
    for cand in units_out_candidates:
        if cand.exists() and cand.is_dir():
            features.units_out_dir = cand
            break
    if features.units_out_dir is None:
        # Fallback: look for a Units Out directory directly under csv_dir or model_dir
        for base in (csv_dir, model_dir):
            maybe_units = base / "Units Out"
            if maybe_units.exists() and maybe_units.is_dir():
                features.units_out_dir = maybe_units
                break

    # Outage property presence (FOR/MR/MTTR/etc.)
    features.has_outage_properties = _detect_outage_properties(csv_dir)

    return features
