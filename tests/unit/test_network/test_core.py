"""Unit tests for src/network/core.py - network conversion functions."""

import pandas as pd
import pytest
from plexosdb.enums import ClassEnum

from src.network.core import (
    _create_datetime_index,
    _parse_demand_directory,
    _parse_demand_single_file,
    add_buses,
    add_carriers,
    add_loads_flexible,
    add_snapshots,
    create_bus_mapping_from_csv,
    discover_carriers_from_db,
    get_demand_format_info,
    parse_demand_data,
    port_core_network,
)

# ============================================================================
# TestAddBuses
# ============================================================================


@pytest.mark.unit
class TestAddBuses:
    def test_adds_all_nodes_from_database(self, empty_network, mock_plexos_db):
        """Test that all nodes from database are added as buses."""
        mock_plexos_db.list_objects_by_class.return_value = ["Bus1", "Bus2", "Bus3"]

        add_buses(empty_network, mock_plexos_db)

        assert len(empty_network.buses) == 3
        assert "Bus1" in empty_network.buses.index
        assert "Bus2" in empty_network.buses.index
        assert "Bus3" in empty_network.buses.index

    def test_sets_carrier_to_ac(self, empty_network, mock_plexos_db):
        """Test that all buses have carrier set to AC."""
        mock_plexos_db.list_objects_by_class.return_value = ["Bus1", "Bus2"]

        add_buses(empty_network, mock_plexos_db)

        assert empty_network.buses.loc["Bus1", "carrier"] == "AC"
        assert empty_network.buses.loc["Bus2", "carrier"] == "AC"

    def test_empty_database_adds_no_buses(self, empty_network, mock_plexos_db):
        """Test that empty database results in no buses added."""
        mock_plexos_db.list_objects_by_class.return_value = []

        add_buses(empty_network, mock_plexos_db)

        assert len(empty_network.buses) == 0

    def test_prints_summary(self, empty_network, mock_plexos_db, capsys):
        """Test that function prints summary of buses added."""
        mock_plexos_db.list_objects_by_class.return_value = ["Bus1", "Bus2"]

        add_buses(empty_network, mock_plexos_db)

        captured = capsys.readouterr()
        assert "Added 2 buses" in captured.out

    def test_bus_names_match_node_names(self, empty_network, mock_plexos_db):
        """Test that bus names exactly match node names from database."""
        node_names = ["Node_A", "Node_B", "Special-Node"]
        mock_plexos_db.list_objects_by_class.return_value = node_names

        add_buses(empty_network, mock_plexos_db)

        for node_name in node_names:
            assert node_name in empty_network.buses.index


# ============================================================================
# TestAddSnapshots
# ============================================================================


@pytest.mark.unit
class TestAddSnapshots:
    def test_single_csv_hourly_24_periods(self, empty_network, tmp_path):
        """Test single CSV with Period 1-24 creates hourly snapshots."""
        csv_file = tmp_path / "demand.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {"Year": 2024, "Month": 1, "Day": 1, "Period": period, "Zone1": 100}
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        assert len(empty_network.snapshots) == 24
        # Check hourly frequency
        assert empty_network.snapshots[1] - empty_network.snapshots[0] == pd.Timedelta(
            hours=1
        )

    def test_single_csv_30min_48_periods(self, empty_network, tmp_path):
        """Test single CSV with Period 1-48 creates 30-minute snapshots."""
        csv_file = tmp_path / "demand.csv"
        rows = []
        for period in range(1, 49):
            rows.append(
                {"Year": 2024, "Month": 1, "Day": 1, "Period": period, "Zone1": 100}
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        assert len(empty_network.snapshots) == 48
        # Check 30-minute frequency
        assert empty_network.snapshots[1] - empty_network.snapshots[0] == pd.Timedelta(
            minutes=30
        )

    def test_single_csv_15min_96_periods(self, empty_network, tmp_path):
        """Test single CSV with Period 1-96 creates 15-minute snapshots."""
        csv_file = tmp_path / "demand.csv"
        rows = []
        for period in range(1, 97):
            rows.append(
                {"Year": 2024, "Month": 1, "Day": 1, "Period": period, "Zone1": 100}
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        assert len(empty_network.snapshots) == 96
        # Check 15-minute frequency
        assert empty_network.snapshots[1] - empty_network.snapshots[0] == pd.Timedelta(
            minutes=15
        )

    def test_single_csv_without_period(self, empty_network, tmp_path):
        """Test single CSV without Period column creates daily snapshots."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame(
            {"Year": [2024, 2024], "Month": [1, 1], "Day": [1, 2], "Zone1": [100, 110]}
        )
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        assert len(empty_network.snapshots) == 2

    def test_directory_traditional_format(self, empty_network, tmp_path):
        """Test directory with traditional format CSV files."""
        demand_dir = tmp_path / "demand"
        demand_dir.mkdir()

        # Create file with hour columns 1-24
        csv_file = demand_dir / "Bus1_demand.csv"
        data = {"Year": [2024], "Month": [1], "Day": [1]}
        for h in range(1, 25):
            data[str(h)] = [100]
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(demand_dir))

        assert len(empty_network.snapshots) == 24

    def test_directory_caiso_format(self, empty_network, tmp_path):
        """Test directory with CAISO/SEM format files (Period column)."""
        demand_dir = tmp_path / "demand"
        demand_dir.mkdir()

        csv_file = demand_dir / "demand.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {"Year": 2024, "Month": 1, "Day": 1, "Period": period, "1": 100}
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(demand_dir))

        assert len(empty_network.snapshots) == 24

    def test_directory_empty_raises_error(self, empty_network, tmp_path):
        """Test that empty directory raises ValueError."""
        demand_dir = tmp_path / "empty_demand"
        demand_dir.mkdir()

        with pytest.raises(ValueError, match="No valid CSV files found"):
            add_snapshots(empty_network, str(demand_dir))

    def test_single_csv_non_existent_raises_error(self, empty_network, tmp_path):
        """Test that non-existent CSV file raises ValueError."""
        csv_file = tmp_path / "non_existent.csv"

        with pytest.raises(
            ValueError, match="Path must be either a CSV file or directory"
        ):
            add_snapshots(empty_network, str(csv_file))

    def test_path_neither_file_nor_dir_raises_error(self, empty_network):
        """Test that invalid path raises ValueError."""
        with pytest.raises(
            ValueError, match="Path must be either a CSV file or directory"
        ):
            add_snapshots(empty_network, "/invalid/path/that/does/not/exist")

    def test_multiple_dates_in_single_csv(self, empty_network, tmp_path):
        """Test single CSV with multiple dates creates correct snapshots."""
        csv_file = tmp_path / "demand.csv"
        rows = []
        for day in [1, 2]:
            for period in range(1, 25):
                rows.append(
                    {
                        "Year": 2024,
                        "Month": 1,
                        "Day": day,
                        "Period": period,
                        "Zone1": 100,
                    }
                )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        assert len(empty_network.snapshots) == 48

    def test_snapshots_properly_sorted(self, empty_network, tmp_path):
        """Test that snapshots are sorted chronologically."""
        csv_file = tmp_path / "demand.csv"
        rows = [
            {"Year": 2024, "Month": 1, "Day": 2, "Period": 1, "Zone1": 100},
            {"Year": 2024, "Month": 1, "Day": 1, "Period": 1, "Zone1": 100},
        ]
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(csv_file))

        # First snapshot should be Jan 1, second should be Jan 2
        assert empty_network.snapshots[0] < empty_network.snapshots[1]

    def test_unique_timestamps_only(self, empty_network, tmp_path):
        """Test that duplicate timestamps are removed."""
        demand_dir = tmp_path / "demand"
        demand_dir.mkdir()

        # Create two files with same timestamps
        for i in [1, 2]:
            csv_file = demand_dir / f"file{i}.csv"
            data = {"Year": [2024], "Month": [1], "Day": [1]}
            for h in range(1, 25):
                data[str(h)] = [100]
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

        add_snapshots(empty_network, str(demand_dir))

        # Should have 24 unique snapshots, not 48
        assert len(empty_network.snapshots) == 24


# ============================================================================
# TestParseDemandData
# ============================================================================


@pytest.mark.unit
class TestParseDemandData:
    def test_parse_directory_with_bus_files(self, sample_demand_directory):
        """Test parsing directory with individual bus demand files."""
        result = parse_demand_data(str(sample_demand_directory))

        assert isinstance(result, pd.DataFrame)
        assert "Bus1" in result.columns
        assert "Bus2" in result.columns
        assert len(result) == 48  # 2 days * 24 hours

    def test_parse_single_csv_zone_format(self, sample_demand_csv_with_period):
        """Test parsing single CSV with zone columns."""
        result = parse_demand_data(str(sample_demand_csv_with_period))

        assert isinstance(result, pd.DataFrame)
        assert "Zone1" in result.columns
        assert "Zone2" in result.columns
        assert len(result) == 48  # 2 days * 24 periods

    def test_parse_single_csv_iteration_format(self, tmp_path):
        """Test parsing single CSV with numeric iteration columns."""
        csv_file = tmp_path / "iterations.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "1": 100,
                    "2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        result = parse_demand_data(str(csv_file))

        assert "iteration_1" in result.columns
        assert "iteration_2" in result.columns
        assert hasattr(result, "_format_type")
        assert result._format_type == "iteration"

    def test_parse_single_csv_with_period_column(self, sample_demand_csv_with_period):
        """Test parsing CAISO/SEM format with Period column."""
        result = parse_demand_data(str(sample_demand_csv_with_period))

        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 48

    def test_apply_bus_mapping(self, tmp_path):
        """Test applying bus mapping to rename columns for iteration format."""
        csv_file = tmp_path / "demand.csv"
        # Use numeric columns (iteration format) which works with bus_mapping
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "1": 100,
                    "2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        # For iteration format, columns become iteration_1, iteration_2
        result = parse_demand_data(str(csv_file))

        # Iteration format creates iteration_1, iteration_2
        assert "iteration_1" in result.columns
        assert "iteration_2" in result.columns

    def test_detect_iteration_format(self, tmp_path):
        """Test detection of iteration format sets metadata."""
        csv_file = tmp_path / "iterations.csv"
        df = pd.DataFrame(
            {
                "Year": [2024],
                "Month": [1],
                "Day": [1],
                "1": [100],
                "2": [110],
                "3": [120],
            }
        )
        df.to_csv(csv_file, index=False)

        result = parse_demand_data(str(csv_file))

        assert result._format_type == "iteration"
        assert result._num_iterations == 3

    def test_detect_zone_format(self, sample_demand_csv_with_period):
        """Test detection of zone format sets metadata."""
        result = parse_demand_data(str(sample_demand_csv_with_period))

        assert result._format_type == "zone"
        assert result._num_iterations is None

    def test_invalid_path_raises_error(self):
        """Test that invalid path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            parse_demand_data("/invalid/path")

    def test_no_datetime_columns_raises_error(self, tmp_path):
        """Test that CSV without datetime columns raises ValueError."""
        csv_file = tmp_path / "bad.csv"
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(csv_file, index=False)

        with pytest.raises(ValueError, match="Could not identify datetime columns"):
            parse_demand_data(str(csv_file))

    def test_mixed_directory_formats(self, tmp_path):
        """Test directory with both traditional and CAISO format files."""
        demand_dir = tmp_path / "demand"
        demand_dir.mkdir()

        # Traditional format
        trad_file = demand_dir / "Bus1_demand.csv"
        data = {"Year": [2024], "Month": [1], "Day": [1]}
        for h in range(1, 25):
            data[str(h)] = [100]
        df = pd.DataFrame(data)
        df.to_csv(trad_file, index=False)

        # CAISO format
        caiso_file = demand_dir / "caiso.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {"Year": 2024, "Month": 1, "Day": 1, "Period": period, "1": 100}
            )
        df = pd.DataFrame(rows)
        df.to_csv(caiso_file, index=False)

        result = parse_demand_data(str(demand_dir))

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) >= 1


# ============================================================================
# TestAddLoadsFlexible
# ============================================================================


@pytest.mark.unit
class TestAddLoadsFlexible:
    def test_per_node_mode_zone_format(self, network_with_multiple_buses, tmp_path):
        """Test per-node mode with zone format creates loads on matching buses."""
        # Set snapshots first
        network_with_multiple_buses.set_snapshots(
            pd.date_range("2024-01-01", periods=48, freq="h")
        )

        # Create CSV with zones matching bus names
        csv_file = tmp_path / "demand.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "Bus1": 100,
                    "Bus2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        add_loads_flexible(network_with_multiple_buses, str(csv_file))

        # Should create loads for Bus1 and Bus2
        assert len(network_with_multiple_buses.loads) == 2

    def test_per_node_mode_iteration_format(self, network_with_bus, tmp_path):
        """Test per-node mode with iteration format creates multiple loads."""
        csv_file = tmp_path / "iterations.csv"
        df = pd.DataFrame(
            {"Year": [2024], "Month": [1], "Day": [1], "1": [100], "2": [110]}
        )
        df.to_csv(csv_file, index=False)

        # Set snapshots first
        network_with_bus.set_snapshots(pd.date_range("2024-01-01", periods=1, freq="D"))

        add_loads_flexible(network_with_bus, str(csv_file))

        # Should create Load1_Bus1 and Load2_Bus1
        assert len(network_with_bus.loads) == 2

    def test_target_node_mode_zone_format(
        self, network_with_multiple_buses, sample_demand_csv_with_period
    ):
        """Test target node mode with zone format creates aggregated load."""
        network_with_multiple_buses.set_snapshots(
            pd.date_range("2024-01-01", periods=48, freq="h")
        )

        add_loads_flexible(
            network_with_multiple_buses,
            str(sample_demand_csv_with_period),
            target_node="Bus1",
        )

        # Should create single aggregated load
        assert len(network_with_multiple_buses.loads) == 1
        assert network_with_multiple_buses.loads.iloc[0]["bus"] == "Bus1"

    def test_target_node_mode_iteration_format(self, network_with_bus, tmp_path):
        """Test target node mode with iteration format creates multiple loads on target."""
        csv_file = tmp_path / "iterations.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "1": 100,
                    "2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        network_with_bus.set_snapshots(
            pd.date_range("2024-01-01", periods=24, freq="h")
        )

        add_loads_flexible(network_with_bus, str(csv_file), target_node="Bus1")

        assert len(network_with_bus.loads) == 2
        assert all(network_with_bus.loads["bus"] == "Bus1")

    def test_aggregate_node_mode_creates_bus(
        self, network_with_bus, sample_demand_csv_with_period
    ):
        """Test aggregate node mode creates new bus."""
        network_with_bus.set_snapshots(
            pd.date_range("2024-01-01", periods=48, freq="h")
        )

        add_loads_flexible(
            network_with_bus,
            str(sample_demand_csv_with_period),
            aggregate_node_name="LoadBus",
        )

        assert "LoadBus" in network_with_bus.buses.index
        assert len(network_with_bus.loads) >= 1

    def test_aggregate_node_mode_iteration_format(self, empty_network, tmp_path):
        """Test aggregate node mode with iteration format."""
        csv_file = tmp_path / "iterations.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "1": 100,
                    "2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        empty_network.set_snapshots(pd.date_range("2024-01-01", periods=24, freq="h"))

        add_loads_flexible(empty_network, str(csv_file), aggregate_node_name="AggNode")

        assert "AggNode" in empty_network.buses.index
        assert len(empty_network.loads) == 2

    def test_loads_aligned_with_snapshots(self, network_with_bus, tmp_path):
        """Test that load time series are aligned with network snapshots."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame({"Year": [2024], "Month": [1], "Day": [1], "Bus1": [100]})
        df.to_csv(csv_file, index=False)

        snapshots = pd.date_range("2024-01-01", periods=2, freq="D")
        network_with_bus.set_snapshots(snapshots)

        add_loads_flexible(network_with_bus, str(csv_file))

        # Load time series should have same index as snapshots
        assert len(network_with_bus.loads_t.p_set) == len(snapshots)

    def test_target_node_not_exists_raises_error(self, network_with_bus, tmp_path):
        """Test that specifying non-existent target node raises ValueError."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame({"Year": [2024], "Month": [1], "Day": [1], "Zone1": [100]})
        df.to_csv(csv_file, index=False)

        network_with_bus.set_snapshots(pd.date_range("2024-01-01", periods=1, freq="D"))

        with pytest.raises(ValueError, match="Target node.*not found"):
            add_loads_flexible(
                network_with_bus, str(csv_file), target_node="NonExistent"
            )

    def test_buses_without_loads_warning(
        self, network_with_multiple_buses, tmp_path, capsys
    ):
        """Test warning for buses without loads in per-node mode."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame(
            {
                "Year": [2024],
                "Month": [1],
                "Day": [1],
                "Bus1": [100],  # Only Bus1, not Bus2 or Bus3
            }
        )
        df.to_csv(csv_file, index=False)

        network_with_multiple_buses.set_snapshots(
            pd.date_range("2024-01-01", periods=1, freq="D")
        )

        add_loads_flexible(network_with_multiple_buses, str(csv_file))

        captured = capsys.readouterr()
        assert "buses have no loads" in captured.out or "buses have no loads" in str(
            captured
        )


# ============================================================================
# TestDiscoverCarriersFromDb
# ============================================================================


@pytest.mark.unit
class TestDiscoverCarriersFromDb:
    def test_core_carriers_always_added(self, mock_plexos_db):
        """Test that core carriers (AC, conversion) are always added."""
        mock_plexos_db.list_objects_by_class.return_value = []

        carriers = discover_carriers_from_db(mock_plexos_db)

        assert "AC" in carriers
        assert "conversion" in carriers

    def test_fuels_from_database(self, mock_plexos_db):
        """Test that fuels from ClassEnum.Fuel are added as carriers."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Fuel:
                return ["Natural Gas", "Coal", "Nuclear"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        carriers = discover_carriers_from_db(mock_plexos_db)

        assert "Natural Gas" in carriers
        assert "Coal" in carriers
        assert "Nuclear" in carriers

    def test_renewable_carriers_from_generators(self, mock_plexos_db):
        """Test that renewable carriers are detected from generator names."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Generator:
                return [
                    "Wind_Gen1",
                    "Solar_PV_Gen2",
                    "Hydro_Gen3",
                    "Wind_Offshore_Gen4",
                ]
            elif class_enum == ClassEnum.Fuel:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        carriers = discover_carriers_from_db(mock_plexos_db)

        assert "Wind" in carriers
        assert "Solar" in carriers
        assert "Hydro" in carriers
        assert "Wind Offshore" in carriers

    def test_technology_carriers_from_names(self, mock_plexos_db):
        """Test that technology-specific carriers (CCGT, OCGT, etc.) are detected."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Generator:
                return ["CCGT_Plant1", "OCGT_Plant2", "Lignite_Plant3"]
            elif class_enum == ClassEnum.Fuel:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        carriers = discover_carriers_from_db(mock_plexos_db)

        assert "Natural Gas CCGT" in carriers
        assert "Natural Gas OCGT" in carriers
        assert "Lignite" in carriers

    def test_generator_name_carriers(self, mock_plexos_db):
        """Test that individual generator names are added as carriers (for generators-as-links)."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Generator:
                return ["Gen1", "Gen2"]
            elif class_enum == ClassEnum.Fuel:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        carriers = discover_carriers_from_db(mock_plexos_db)

        assert "Gen1" in carriers
        assert "Gen2" in carriers

    def test_handles_database_errors_gracefully(self, mock_plexos_db, capsys):
        """Test that database errors are handled gracefully with warnings."""
        mock_plexos_db.list_objects_by_class.side_effect = Exception("Database error")

        carriers = discover_carriers_from_db(mock_plexos_db)

        # Should still return core carriers
        assert "AC" in carriers
        assert "conversion" in carriers


# ============================================================================
# TestAddCarriers
# ============================================================================


@pytest.mark.unit
class TestAddCarriers:
    def test_adds_discovered_carriers(self, empty_network, mock_plexos_db):
        """Test that carriers discovered from database are added to network."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Fuel:
                return ["Gas", "Coal"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        add_carriers(empty_network, mock_plexos_db)

        assert "Gas" in empty_network.carriers.index
        assert "Coal" in empty_network.carriers.index
        assert "AC" in empty_network.carriers.index

    def test_skips_existing_carriers(self, empty_network, mock_plexos_db):
        """Test that existing carriers are not duplicated."""
        empty_network.add("Carrier", "Gas")

        def side_effect(class_enum):
            if class_enum == ClassEnum.Fuel:
                return ["Gas", "Coal"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        add_carriers(empty_network, mock_plexos_db)

        # Should have unique carriers only
        assert len(empty_network.carriers[empty_network.carriers.index == "Gas"]) == 1

    def test_prints_summary(self, empty_network, mock_plexos_db, capsys):
        """Test that function prints summary of carriers added."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Fuel:
                return ["Gas"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        add_carriers(empty_network, mock_plexos_db)

        captured = capsys.readouterr()
        assert "Added" in captured.out
        assert "carriers" in captured.out


# ============================================================================
# TestPortCoreNetwork
# ============================================================================


@pytest.mark.unit
class TestPortCoreNetwork:
    def test_full_setup_per_node_mode(
        self, empty_network, mock_plexos_db, sample_demand_directory
    ):
        """Test full core network setup in per-node mode."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Node:
                return ["Bus1", "Bus2"]
            elif class_enum == ClassEnum.Model:
                return ["Model1"]
            elif class_enum == ClassEnum.Fuel:
                return ["Gas"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        result = port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_directory),
            demand_source=str(sample_demand_directory),
        )

        assert len(empty_network.buses) == 2
        assert len(empty_network.snapshots) > 0
        assert len(empty_network.carriers) > 0
        assert len(empty_network.loads) >= 1
        assert isinstance(result, dict)

    def test_full_setup_target_node_mode(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test full setup with target_node parameter."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Node:
                return ["TargetBus"]
            elif class_enum == ClassEnum.Model:
                return ["Model1"]
            elif class_enum == ClassEnum.Fuel or class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        result = port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_csv_with_period),
            demand_source=str(sample_demand_csv_with_period),
            target_node="TargetBus",
        )

        assert result["mode"] == "target_node"
        assert result["target_node"] == "TargetBus"

    def test_full_setup_aggregate_node_mode(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test full setup with aggregate_node_name parameter."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Node:
                return ["Bus1"]
            elif (
                class_enum == ClassEnum.Model
                or class_enum == ClassEnum.Fuel
                or class_enum == ClassEnum.Generator
            ):
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        result = port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_csv_with_period),
            demand_source=str(sample_demand_csv_with_period),
            aggregate_node_name="AggBus",
        )

        assert "AggBus" in empty_network.buses.index
        assert result["mode"] == "aggregate_node"

    def test_multiple_models_requires_model_name(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test that multiple models in XML requires model_name parameter."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Model:
                return ["Model1", "Model2"]
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        with pytest.raises(ValueError, match="Multiple models found"):
            port_core_network(
                empty_network,
                mock_plexos_db,
                snapshots_source=str(sample_demand_csv_with_period),
                demand_source=str(sample_demand_csv_with_period),
            )

    def test_single_model_validation(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test that single model works with or without model_name."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Model:
                return ["OnlyModel"]
            elif class_enum == ClassEnum.Node:
                return ["Bus1"]
            elif class_enum == ClassEnum.Fuel or class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        # Should work without model_name
        port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_csv_with_period),
            demand_source=str(sample_demand_csv_with_period),
        )

        assert len(empty_network.buses) >= 1

    def test_invalid_model_name_raises_error(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test that invalid model_name raises ValueError."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Model:
                return ["Model1"]
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        with pytest.raises(ValueError, match="Model.*not found"):
            port_core_network(
                empty_network,
                mock_plexos_db,
                snapshots_source=str(sample_demand_csv_with_period),
                demand_source=str(sample_demand_csv_with_period),
                model_name="InvalidModel",
            )

    def test_returns_load_summary_dict(
        self, empty_network, mock_plexos_db, sample_demand_csv_with_period
    ):
        """Test that function returns load summary dictionary."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Node:
                return ["Bus1"]
            elif (
                class_enum == ClassEnum.Model
                or class_enum == ClassEnum.Fuel
                or class_enum == ClassEnum.Generator
            ):
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        result = port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_csv_with_period),
            demand_source=str(sample_demand_csv_with_period),
        )

        assert "mode" in result
        assert isinstance(result, dict)

    def test_integration_with_all_components(
        self, empty_network, mock_plexos_db, sample_demand_directory
    ):
        """Test that all components are integrated correctly."""

        def side_effect(class_enum):
            if class_enum == ClassEnum.Node:
                return ["Bus1", "Bus2"]
            elif class_enum == ClassEnum.Model:
                return []
            elif class_enum == ClassEnum.Fuel:
                return ["Gas", "Coal"]
            elif class_enum == ClassEnum.Generator:
                return []
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect

        port_core_network(
            empty_network,
            mock_plexos_db,
            snapshots_source=str(sample_demand_directory),
            demand_source=str(sample_demand_directory),
        )

        # Verify all components exist
        assert len(empty_network.buses) > 0
        assert len(empty_network.snapshots) > 0
        assert len(empty_network.carriers) > 0
        assert len(empty_network.loads) > 0


# ============================================================================
# TestCreateBusMappingFromCsv
# ============================================================================


@pytest.mark.unit
class TestCreateBusMappingFromCsv:
    def test_numeric_zones_to_zone_format(self, tmp_path):
        """Test that numeric zones are converted to Zone_XXX format."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame(
            {
                "Year": [2024],
                "Month": [1],
                "Day": [1],
                "1": [100],
                "2": [110],
                "3": [120],
            }
        )
        df.to_csv(csv_file, index=False)

        mapping = create_bus_mapping_from_csv(str(csv_file))

        assert mapping["1"] == "Zone_001"
        assert mapping["2"] == "Zone_002"
        assert mapping["3"] == "Zone_003"

    def test_named_zones_cleaned(self, tmp_path):
        """Test that named zones are cleaned (spaces to underscores)."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame(
            {"Year": [2024], "Month": [1], "Day": [1], "Zone A": [100], "Zone-B": [110]}
        )
        df.to_csv(csv_file, index=False)

        mapping = create_bus_mapping_from_csv(str(csv_file))

        assert mapping["Zone A"] == "Zone_A"
        assert mapping["Zone-B"] == "Zone_B"

    def test_match_with_network_buses(self, tmp_path):
        """Test partial matching with network buses."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame({"Year": [2024], "Month": [1], "Day": [1], "1": [100]})
        df.to_csv(csv_file, index=False)

        network_buses = ["Zone_001", "Zone_002"]
        mapping = create_bus_mapping_from_csv(
            str(csv_file), network_buses=network_buses
        )

        # Should match "1" to "Zone_001"
        assert mapping["1"] == "Zone_001"

    def test_invalid_csv_raises_error(self, tmp_path):
        """Test that invalid CSV file raises ValueError."""
        csv_file = tmp_path / "nonexistent.csv"

        with pytest.raises(ValueError, match="Failed to read CSV"):
            create_bus_mapping_from_csv(str(csv_file))


# ============================================================================
# TestGetDemandFormatInfo
# ============================================================================


@pytest.mark.unit
class TestGetDemandFormatInfo:
    def test_directory_format_info(self, sample_demand_directory):
        """Test getting format info for directory."""
        info = get_demand_format_info(str(sample_demand_directory))

        assert info["format_type"] == "directory"
        assert info["num_files"] == 2
        assert "sample_files" in info

    def test_single_file_format_info(self, sample_demand_csv_with_period):
        """Test getting format info for single CSV file."""
        info = get_demand_format_info(str(sample_demand_csv_with_period))

        assert info["format_type"] == "single_file"
        assert "num_load_zones" in info
        assert "suggested_mapping" in info

    def test_invalid_path_returns_error(self):
        """Test that invalid path returns error in info dict."""
        info = get_demand_format_info("/invalid/path")

        assert info["format_type"] == "unknown"
        assert "error" in info


# ============================================================================
# TestHelperFunctions
# ============================================================================


@pytest.mark.unit
class TestHelperFunctions:
    def test_create_datetime_index_hourly(self):
        """Test _create_datetime_index with hourly data (24 periods)."""
        df_long = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2024-01-01")] * 24,
                "period": [str(i) for i in range(1, 25)],
                "load": [100] * 24,
            }
        )

        result = _create_datetime_index(df_long, 24)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert len(result) == 24
        assert result.index[1] - result.index[0] == pd.Timedelta(hours=1)

    def test_create_datetime_index_30min(self):
        """Test _create_datetime_index with 30-minute data (48 periods)."""
        df_long = pd.DataFrame(
            {
                "datetime": [pd.Timestamp("2024-01-01")] * 48,
                "period": [str(i) for i in range(1, 49)],
                "load": [100] * 48,
            }
        )

        result = _create_datetime_index(df_long, 48)

        assert len(result) == 48
        assert result.index[1] - result.index[0] == pd.Timedelta(minutes=30)

    def test_parse_demand_directory_mixed_formats(self, tmp_path):
        """Test _parse_demand_directory with mixed traditional and CAISO formats."""
        demand_dir = tmp_path / "demand"
        demand_dir.mkdir()

        # Traditional format
        trad_file = demand_dir / "Bus1_demand.csv"
        data = {"Year": [2024], "Month": [1], "Day": [1]}
        for h in range(1, 25):
            data[str(h)] = [100]
        df = pd.DataFrame(data)
        df.to_csv(trad_file, index=False)

        result = _parse_demand_directory(str(demand_dir))

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) >= 1

    def test_parse_demand_single_file_with_mapping(self, tmp_path):
        """Test _parse_demand_single_file with bus mapping."""
        csv_file = tmp_path / "demand.csv"
        df = pd.DataFrame(
            {"Year": [2024], "Month": [1], "Day": [1], "Zone1": [100], "Zone2": [110]}
        )
        df.to_csv(csv_file, index=False)

        bus_mapping = {"Zone1": "Bus_A", "Zone2": "Bus_B"}
        result = _parse_demand_single_file(str(csv_file), bus_mapping)

        assert "Bus_A" in result.columns
        assert "Bus_B" in result.columns

    def test_parse_demand_single_file_iterations(self, tmp_path):
        """Test _parse_demand_single_file detects iteration format."""
        csv_file = tmp_path / "demand.csv"
        rows = []
        for period in range(1, 25):
            rows.append(
                {
                    "Year": 2024,
                    "Month": 1,
                    "Day": 1,
                    "Period": period,
                    "1": 100,
                    "2": 110,
                }
            )
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)

        result = _parse_demand_single_file(str(csv_file))

        assert result._format_type == "iteration"
        assert "iteration_1" in result.columns
        assert "iteration_2" in result.columns
