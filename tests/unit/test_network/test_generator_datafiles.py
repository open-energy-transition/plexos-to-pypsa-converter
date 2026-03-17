"""Tests for generator Data File parsing and application."""

from pathlib import Path

import pandas as pd
import pypsa

from plexos_to_pypsa_converter.db.csv_readers import read_plexos_input_csv
from plexos_to_pypsa_converter.network.generators_csv import (
    add_generators_csv,
    apply_generator_units_timeseries_csv,
    set_capacity_ratings_csv,
)


def test_read_plexos_input_csv_supports_pattern_and_year_tables(tmp_path: Path):
    """Pattern and Year tables should parse into snapshot-aligned data."""
    snapshots = pd.date_range("2050-01-01 00:00:00", periods=2, freq="h")

    pattern_csv = tmp_path / "pattern.csv"
    pattern_csv.write_text(
        "\n".join(
            [
                "PATTERN,G1,G2",
                '"M01,D01,H01",10,100',
                '"M01,D01,H02",20,200',
            ]
        )
    )

    year_csv = tmp_path / "year.csv"
    year_csv.write_text(
        "\n".join(
            [
                "Year,G1,G2",
                "2049,5,50",
                "2050,7,70",
            ]
        )
    )

    pattern_df = read_plexos_input_csv(pattern_csv, snapshots=snapshots)
    year_df = read_plexos_input_csv(year_csv, snapshots=snapshots)

    assert pattern_df.loc[snapshots[0], "G1"] == 10
    assert pattern_df.loc[snapshots[1], "G1"] == 20
    assert pattern_df.loc[snapshots[0], "G2"] == 100
    assert year_df.loc[snapshots[0], "G1"] == 7
    assert year_df.loc[snapshots[1], "G2"] == 70


def test_generator_data_files_feed_capacity_ratings_and_units(tmp_path: Path):
    """Generator Data Files should set p_nom, p_max_pu, and units-based scaling."""
    model_root = tmp_path / "model"
    csv_dir = model_root / "csvs_from_xml" / "System"
    data_dir = model_root / "datafiles"
    csv_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (csv_dir / "Generator.csv").write_text(
        "\n".join(
            [
                "object,category,Node,Fuel,Max Capacity.Data File,Rating.Data File,Units.Data File",
                "GenA,Hydro,Bus1,,Data File.CapacityA,Data File.RatingA,Data File.UnitsA",
            ]
        )
    )
    (csv_dir / "Data File.csv").write_text(
        "\n".join(
            [
                "object,category,Filename(text)",
                "CapacityA,-,Datafiles/capacity.csv",
                "RatingA,-,Datafiles/rating.csv",
                "UnitsA,-,Datafiles/units.csv",
            ]
        )
    )

    # Put a decoy column first to verify object-specific column selection.
    (data_dir / "capacity.csv").write_text(
        "\n".join(
            [
                "Year,Other,GenA",
                "2050,999,120",
            ]
        )
    )
    (data_dir / "rating.csv").write_text(
        "\n".join(
            [
                "PATTERN,Other,GenA",
                '"M01,D01,H01",999,60',
                '"M01,D01,H02",999,30',
            ]
        )
    )
    (data_dir / "units.csv").write_text(
        "\n".join(
            [
                "Year,Other,GenA",
                "2050,9,2",
            ]
        )
    )

    snapshots = pd.date_range("2050-01-01 00:00:00", periods=2, freq="h")
    network = pypsa.Network()
    network.set_snapshots(snapshots)
    network.add("Bus", "Bus1")

    add_generators_csv(network, csv_dir)
    assert network.generators.at["GenA", "p_nom"] == 120

    set_capacity_ratings_csv(network, csv_dir)
    assert network.generators_t.p_max_pu.loc[snapshots[0], "GenA"] == 0.5
    assert network.generators_t.p_max_pu.loc[snapshots[1], "GenA"] == 0.25

    apply_generator_units_timeseries_csv(network, csv_dir)
    assert network.generators.at["GenA", "p_nom"] == 240
    assert network.generators_t.p_max_pu.loc[snapshots[0], "GenA"] == 0.5
    assert network.generators_t.p_max_pu.loc[snapshots[1], "GenA"] == 0.25
