"""Unit tests for src.db.parse module.

This module tests the database parsing functions that extract and transform
data from PLEXOS databases, including bus/fuel finding, timeslice parsing,
and property activity masking.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from plexosdb.enums import ClassEnum

from db.parse import (
    find_bus_for_object,
    find_bus_for_storage_via_generators,
    find_fuel_for_generator,
    get_dataid_timeslice_map,
    get_emission_rates,
    get_property_active_mask,
    read_timeslice_activity,
)

# ============================================================================
# Tests for find_bus_for_object
# ============================================================================


@pytest.mark.unit
class TestFindBusForObject:
    """Test suite for find_bus_for_object function."""

    def test_direct_child_node(self, mock_plexos_db):
        """Test finding bus via direct child Node membership."""
        memberships = [
            {
                "child_class_name": "Node",
                "child_object_name": "Bus1",
                "parent_class_name": "Generator",
                "parent_object_name": "Gen1",
                "collection_name": "Node",
                "name": "Bus1",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result == "Bus1"
        mock_plexos_db.get_memberships_system.assert_called_once_with(
            "Gen1", object_class=ClassEnum.Generator
        )

    def test_direct_parent_node(self, mock_plexos_db):
        """Test finding bus when Node is parent (less common pattern)."""
        memberships = [
            {
                "parent_class_name": "Node",
                "parent_object_name": "Bus2",
                "child_class_name": "Storage",
                "child_object_name": "Storage1",
                "collection_name": "Storage",
                "name": "Storage1",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Storage1", ClassEnum.Storage)

        assert result == "Bus2"

    def test_returns_first_match_when_multiple_nodes(self, mock_plexos_db):
        """Test returns first Node when multiple exist in memberships."""
        memberships = [
            {
                "child_class_name": "Node",
                "child_object_name": "Bus1",
                "collection_name": "Node",
                "name": "Bus1",
            },
            {
                "child_class_name": "Node",
                "child_object_name": "Bus2",
                "collection_name": "Node",
                "name": "Bus2",
            },
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result == "Bus1"  # First match

    def test_via_collection_name_with_node(self, mock_plexos_db):
        """Test finding bus via collection name containing 'node'."""
        memberships = [
            {
                "child_class_name": "SomeClass",
                "child_object_name": "SomeObject",
                "parent_class_name": "Storage",
                "parent_object_name": "Storage1",
                "collection_name": "Node Connection",
                "name": "Bus3",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Storage1", ClassEnum.Storage)

        assert result == "Bus3"

    def test_via_collection_name_case_insensitive(self, mock_plexos_db):
        """Test collection name matching is case-insensitive."""
        memberships = [
            {
                "child_class_name": "SomeClass",
                "collection_name": "NODE",
                "name": "Bus4",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result == "Bus4"

    def test_no_memberships_found_returns_none(self, mock_plexos_db):
        """Test returns None when no memberships exist."""
        mock_plexos_db.get_memberships_system.return_value = []

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result is None

    def test_exception_handling_returns_none(self, mock_plexos_db):
        """Test returns None when exception occurs querying memberships."""
        mock_plexos_db.get_memberships_system.side_effect = Exception("DB error")

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result is None

    def test_empty_memberships_list_returns_none(self, mock_plexos_db):
        """Test returns None when memberships list is empty."""
        mock_plexos_db.get_memberships_system.return_value = []

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result is None

    def test_membership_without_node_returns_none(self, mock_plexos_db):
        """Test returns None when memberships don't contain any Node."""
        memberships = [
            {
                "child_class_name": "Fuel",
                "child_object_name": "Gas",
                "collection_name": "Fuels",
                "name": "Gas",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result is None

    def test_for_generator_class(self, mock_plexos_db):
        """Test works correctly with Generator class enum."""
        memberships = [
            {
                "child_class_name": "Node",
                "child_object_name": "GenBus",
                "collection_name": "Node",
                "name": "GenBus",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Gen1", ClassEnum.Generator)

        assert result == "GenBus"
        # Verify correct class enum was passed
        call_args = mock_plexos_db.get_memberships_system.call_args
        assert call_args[1]["object_class"] == ClassEnum.Generator

    def test_for_storage_class(self, mock_plexos_db):
        """Test works correctly with Storage class enum."""
        memberships = [
            {
                "child_class_name": "Node",
                "child_object_name": "StorageBus",
                "collection_name": "Node",
                "name": "StorageBus",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Battery1", ClassEnum.Storage)

        assert result == "StorageBus"

    def test_for_battery_class(self, mock_plexos_db):
        """Test works correctly with Battery class enum."""
        memberships = [
            {
                "child_class_name": "Node",
                "child_object_name": "BatteryBus",
                "collection_name": "Node",
                "name": "BatteryBus",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_bus_for_object(mock_plexos_db, "Battery1", ClassEnum.Battery)

        assert result == "BatteryBus"


# ============================================================================
# Tests for find_bus_for_storage_via_generators
# ============================================================================


@pytest.mark.unit
class TestFindBusForStorageViaGenerators:
    """Test suite for find_bus_for_storage_via_generators function."""

    def test_single_connected_generator(self, mock_plexos_db):
        """Test finding bus via single connected generator."""
        # Mock SQL query returning one generator
        query_result = [("Gen1", "Generator", "Storage1", "Storage", "Head Storage")]
        mock_plexos_db.query.return_value = query_result

        # Mock find_bus_for_object to return bus for generator
        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = "Bus1"

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result == ("Bus1", "Gen1")
            mock_find_bus.assert_called_once_with(
                mock_plexos_db, "Gen1", ClassEnum.Generator
            )

    def test_multiple_connected_generators(self, mock_plexos_db):
        """Test finding bus when multiple generators connected."""
        query_result = [
            ("Gen1", "Generator", "Storage1", "Storage", "Head Storage"),
            ("Gen2", "Generator", "Storage1", "Storage", "Tail Storage"),
        ]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            # First generator returns bus
            mock_find_bus.return_value = "Bus1"

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            # Should return first successful match
            assert result == ("Bus1", "Gen1")

    def test_prioritizes_first_generator_with_bus(self, mock_plexos_db):
        """Test prioritizes first generator that has a bus."""
        query_result = [
            ("Gen1", "Generator", "Storage1", "Storage", "Head Storage"),
            ("Gen2", "Generator", "Storage1", "Storage", "Tail Storage"),
        ]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            # Gen1 has no bus, Gen2 has bus
            mock_find_bus.side_effect = [None, "Bus2"]

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result == ("Bus2", "Gen2")
            # Should have tried Gen1 first, then Gen2
            assert mock_find_bus.call_count == 2

    def test_head_storage_collection(self, mock_plexos_db):
        """Test works with Head Storage collection type."""
        query_result = [("Gen1", "Generator", "Storage1", "Storage", "Head Storage")]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = "Bus1"

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result == ("Bus1", "Gen1")

    def test_tail_storage_collection(self, mock_plexos_db):
        """Test works with Tail Storage collection type."""
        query_result = [("Gen1", "Generator", "Storage1", "Storage", "Tail Storage")]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = "Bus1"

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result == ("Bus1", "Gen1")

    def test_no_connected_generators_returns_none(self, mock_plexos_db):
        """Test returns None when no generators connected."""
        mock_plexos_db.query.return_value = []

        result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

        assert result is None

    def test_query_exception_returns_none(self, mock_plexos_db):
        """Test returns None when SQL query fails."""
        mock_plexos_db.query.side_effect = Exception("SQL error")

        result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

        assert result is None

    def test_generator_has_no_bus_returns_none(self, mock_plexos_db):
        """Test returns None when generator has no bus."""
        query_result = [("Gen1", "Generator", "Storage1", "Storage", "Head Storage")]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = None

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result is None

    def test_all_generators_have_no_bus_returns_none(self, mock_plexos_db):
        """Test returns None when none of the generators have buses."""
        query_result = [
            ("Gen1", "Generator", "Storage1", "Storage", "Head Storage"),
            ("Gen2", "Generator", "Storage1", "Storage", "Tail Storage"),
        ]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = None

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert result is None
            # Should have tried all generators
            assert mock_find_bus.call_count == 2

    def test_return_type_is_tuple(self, mock_plexos_db):
        """Test returns tuple of (bus_name, generator_name)."""
        query_result = [("Gen1", "Generator", "Storage1", "Storage", "Head Storage")]
        mock_plexos_db.query.return_value = query_result

        with patch("db.parse.find_bus_for_object") as mock_find_bus:
            mock_find_bus.return_value = "Bus1"

            result = find_bus_for_storage_via_generators(mock_plexos_db, "Storage1")

            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == "Bus1"  # bus_name
            assert result[1] == "Gen1"  # generator_name


# ============================================================================
# Tests for find_fuel_for_generator
# ============================================================================


@pytest.mark.unit
class TestFindFuelForGenerator:
    """Test suite for find_fuel_for_generator function."""

    def test_single_fuel_found(self, mock_plexos_db):
        """Test finding single fuel for generator."""
        memberships = [
            {
                "class": "Fuel",
                "name": "Natural Gas",
                "collection_name": "Fuels",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        assert result == "Natural Gas"

    def test_no_fuel_found_returns_none(self, mock_plexos_db):
        """Test returns None when no fuel membership exists."""
        memberships = [
            {
                "class": "Node",
                "name": "Bus1",
                "collection_name": "Node",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        assert result is None

    def test_multiple_fuels_returns_first(self, mock_plexos_db):
        """Test returns first fuel when multiple exist."""
        memberships = [
            {
                "class": "Fuel",
                "name": "Natural Gas",
                "collection_name": "Fuels",
            },
            {
                "class": "Fuel",
                "name": "Coal",
                "collection_name": "Fuels",
            },
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        assert result == "Natural Gas"  # First match

    def test_exception_handling_returns_none(self, mock_plexos_db):
        """Test returns None when exception occurs."""
        mock_plexos_db.get_memberships_system.side_effect = Exception("DB error")

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        assert result is None

    def test_empty_memberships_returns_none(self, mock_plexos_db):
        """Test returns None when memberships list is empty."""
        mock_plexos_db.get_memberships_system.return_value = []

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        assert result is None

    def test_class_check_is_exact(self, mock_plexos_db):
        """Test class check is exact match (not case-insensitive)."""
        memberships = [
            {
                "class": "fuel",  # lowercase
                "name": "Natural Gas",
                "collection_name": "Fuels",
            }
        ]
        mock_plexos_db.get_memberships_system.return_value = memberships

        result = find_fuel_for_generator(mock_plexos_db, "Gen1")

        # Should not match lowercase "fuel"
        assert result is None


# ============================================================================
# Tests for read_timeslice_activity
# ============================================================================


@pytest.mark.unit
class TestReadTimesliceActivity:
    """Test suite for read_timeslice_activity function."""

    def test_basic_timeslice_parsing(self, tmp_path):
        """Test parsing basic timeslice CSV with single timeslice."""
        csv_data = "DATETIME,NAME,TIMESLICE\n01/01/2024 00:00:00,TS1,-1\n"
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (24, 1)  # 24 hours, 1 timeslice
        assert "TS1" in result.columns

    def test_datetime_conversion_dayfirst(self, tmp_path):
        """Test datetime parsing with dayfirst=True format."""
        csv_data = "DATETIME,NAME,TIMESLICE\n01/01/2024 00:00:00,TS1,-1\n"
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # Should parse correctly with dayfirst
        assert result.loc[snapshots[0], "TS1"]

    def test_active_periods(self, tmp_path):
        """Test TIMESLICE=-1 marks periods as active (True)."""
        csv_data = "DATETIME,NAME,TIMESLICE\n01/01/2024 00:00:00,TS1,-1\n"
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # All periods should be active
        assert result["TS1"].all()

    def test_inactive_periods(self, tmp_path):
        """Test TIMESLICE=0 marks periods as inactive (False)."""
        csv_data = "DATETIME,NAME,TIMESLICE\n01/01/2024 00:00:00,TS1,0\n"
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # All periods should be inactive
        assert not result["TS1"].any()

    def test_activity_changes_at_datetime(self, tmp_path):
        """Test activity changes at specified datetime."""
        csv_data = """DATETIME,NAME,TIMESLICE
01/01/2024 00:00:00,TS1,-1
01/01/2024 12:00:00,TS1,0
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # First 12 hours should be active
        assert result.loc[snapshots[0:12], "TS1"].all()
        # After 12 hours should be inactive
        assert not result.loc[snapshots[12:], "TS1"].any()

    def test_carries_over_last_setting_before_first_snapshot(self, tmp_path):
        """Test carries over last setting before first snapshot."""
        csv_data = """DATETIME,NAME,TIMESLICE
31/12/2023 00:00:00,TS1,0
31/12/2023 12:00:00,TS1,-1
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        # Snapshots start after the last setting
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # Should carry over the last setting (-1 = active)
        assert result["TS1"].all()

    def test_no_setting_before_first_snapshot(self, tmp_path):
        """Test behavior when no setting exists before first snapshot."""
        csv_data = """DATETIME,NAME,TIMESLICE
01/01/2024 12:00:00,TS1,-1
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # First 12 hours should be False (default)
        assert not result.loc[snapshots[0:12], "TS1"].any()
        # After 12:00 should be True
        assert result.loc[snapshots[12:], "TS1"].all()

    def test_setting_exactly_at_first_snapshot(self, tmp_path):
        """Test when setting occurs exactly at first snapshot."""
        csv_data = """DATETIME,NAME,TIMESLICE
01/01/2024 00:00:00,TS1,-1
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # All periods should be active
        assert result["TS1"].all()

    def test_multiple_timeslices(self, tmp_path):
        """Test parsing multiple timeslices."""
        csv_data = """DATETIME,NAME,TIMESLICE
01/01/2024 00:00:00,TS1,-1
01/01/2024 00:00:00,TS2,0
01/01/2024 12:00:00,TS1,0
01/01/2024 12:00:00,TS2,-1
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        assert result.shape == (24, 2)  # 24 hours, 2 timeslices
        assert "TS1" in result.columns
        assert "TS2" in result.columns
        # TS1 active first half, TS2 active second half
        assert result.loc[snapshots[0], "TS1"]
        assert not result.loc[snapshots[0], "TS2"]
        assert not result.loc[snapshots[12], "TS1"]
        assert result.loc[snapshots[12], "TS2"]

    def test_overlapping_timeslices(self, tmp_path):
        """Test timeslices can overlap (both active simultaneously)."""
        csv_data = """DATETIME,NAME,TIMESLICE
01/01/2024 00:00:00,TS1,-1
01/01/2024 00:00:00,TS2,-1
"""
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        # Both should be active
        assert result["TS1"].all()
        assert result["TS2"].all()

    def test_empty_csv_returns_empty_dataframe(self, tmp_path):
        """Test handling of CSV with only headers."""
        csv_data = "DATETIME,NAME,TIMESLICE\n"
        csv_file = tmp_path / "timeslice.csv"
        csv_file.write_text(csv_data)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        result = read_timeslice_activity(str(csv_file), snapshots)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (24, 0)  # 24 rows, 0 columns (no timeslices)


# ============================================================================
# Tests for get_dataid_timeslice_map
# ============================================================================


@pytest.mark.unit
class TestGetDataidTimesliceMap:
    """Test suite for get_dataid_timeslice_map function."""

    def test_single_mapping(self, mock_plexos_db):
        """Test mapping single data_id to timeslice."""
        query_result = [
            (1, 101, "TS1"),
        ]
        mock_plexos_db.query.return_value = query_result

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert isinstance(result, dict)
        assert result == {1: ["TS1"]}

    def test_multiple_timeslices_per_data_id(self, mock_plexos_db):
        """Test multiple timeslices mapped to same data_id."""
        query_result = [
            (1, 101, "TS1"),
            (1, 102, "TS2"),
        ]
        mock_plexos_db.query.return_value = query_result

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert result == {1: ["TS1", "TS2"]}

    def test_multiple_data_ids(self, mock_plexos_db):
        """Test mapping multiple data_ids."""
        query_result = [
            (1, 101, "TS1"),
            (2, 102, "TS2"),
            (3, 103, "TS3"),
        ]
        mock_plexos_db.query.return_value = query_result

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert result == {1: ["TS1"], 2: ["TS2"], 3: ["TS3"]}

    def test_empty_result_returns_empty_dict(self, mock_plexos_db):
        """Test returns empty dict when no results."""
        mock_plexos_db.query.return_value = []

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert result == {}

    def test_sql_query_filters_class_76(self, mock_plexos_db):
        """Test SQL query filters for class_id=76 (timeslice)."""
        mock_plexos_db.query.return_value = []

        get_dataid_timeslice_map(mock_plexos_db)

        # Verify query was called
        assert mock_plexos_db.query.called
        # Verify query string contains class_id=76
        call_args = mock_plexos_db.query.call_args[0]
        query_sql = call_args[0]
        assert "class_id = 76" in query_sql or "class_id=76" in query_sql

    def test_appends_to_list_correctly(self, mock_plexos_db):
        """Test that timeslices are appended to list (not overwritten)."""
        query_result = [
            (1, 101, "TS1"),
            (1, 102, "TS2"),
            (1, 103, "TS3"),
        ]
        mock_plexos_db.query.return_value = query_result

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert len(result[1]) == 3
        assert "TS1" in result[1]
        assert "TS2" in result[1]
        assert "TS3" in result[1]

    def test_return_type_is_dict_of_lists(self, mock_plexos_db):
        """Test return type is dict with list values."""
        query_result = [
            (1, 101, "TS1"),
        ]
        mock_plexos_db.query.return_value = query_result

        result = get_dataid_timeslice_map(mock_plexos_db)

        assert isinstance(result, dict)
        for key, value in result.items():
            assert isinstance(key, int)
            assert isinstance(value, list)


# ============================================================================
# Tests for get_property_active_mask
# ============================================================================


@pytest.mark.unit
class TestGetPropertyActiveMask:
    """Test suite for get_property_active_mask function."""

    def test_no_date_constraints_all_true(self):
        """Test mask is all True when no date constraints."""
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        assert result.all()
        assert len(result) == 48

    def test_from_date_only(self):
        """Test mask with only 'from' date constraint."""
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        from_date = pd.Timestamp("2024-01-01 12:00:00")
        row = pd.Series({"from": from_date, "to": pd.NaT, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        # Should be False before 12:00, True after
        assert not result.iloc[0:12].any()
        assert result.iloc[12:].all()

    def test_to_date_only(self):
        """Test mask with only 'to' date constraint."""
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        to_date = pd.Timestamp("2024-01-01 12:00:00")
        row = pd.Series({"from": pd.NaT, "to": to_date, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        # Should be True before and at 12:00, False after
        assert result.iloc[0:13].all()
        assert not result.iloc[13:].any()

    def test_from_and_to_dates(self):
        """Test mask with both 'from' and 'to' date constraints."""
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        from_date = pd.Timestamp("2024-01-01 12:00:00")
        to_date = pd.Timestamp("2024-01-01 18:00:00")
        row = pd.Series({"from": from_date, "to": to_date, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        # Should be True only between 12:00 and 18:00
        assert not result.iloc[0:12].any()
        assert result.iloc[12:19].all()
        assert not result.iloc[19:].any()

    def test_outside_date_range_all_false(self):
        """Test mask is all False when snapshots outside date range."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        from_date = pd.Timestamp("2024-02-01")
        to_date = pd.Timestamp("2024-02-28")
        row = pd.Series({"from": from_date, "to": to_date, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        assert not result.any()

    def test_with_timeslice_activity(self):
        """Test mask with timeslice activity."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        timeslice_activity = pd.DataFrame(
            {"TS1": [True] * 12 + [False] * 12}, index=snapshots
        )
        dataid_to_timeslice = {1: ["TS1"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should match timeslice activity
        assert result.iloc[0:12].all()
        assert not result.iloc[12:].any()

    def test_timeslice_and_date_constraints(self):
        """Test mask with both timeslice and date constraints."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        from_date = pd.Timestamp("2024-01-01 06:00:00")
        timeslice_activity = pd.DataFrame({"TS1": [True] * 24}, index=snapshots)
        dataid_to_timeslice = {1: ["TS1"]}
        row = pd.Series({"from": from_date, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be True only after 06:00 (date constraint AND timeslice)
        assert not result.iloc[0:6].any()
        assert result.iloc[6:].all()

    def test_multiple_timeslices_all_must_match(self):
        """Test with multiple timeslices - all must be active (AND logic)."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        timeslice_activity = pd.DataFrame(
            {
                "TS1": [True] * 12 + [False] * 12,
                "TS2": [True] * 18 + [False] * 6,
            },
            index=snapshots,
        )
        dataid_to_timeslice = {1: ["TS1", "TS2"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be True only where BOTH TS1 and TS2 are active (first 12 hours)
        assert result.iloc[0:12].all()
        assert not result.iloc[12:].any()

    def test_timeslice_not_in_activity_dataframe(self):
        """Test when timeslice in map but not in activity DataFrame."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        timeslice_activity = pd.DataFrame({"TS1": [True] * 24}, index=snapshots)
        dataid_to_timeslice = {1: ["TS_NOT_EXIST"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should still be all True (unknown timeslice doesn't restrict)
        assert result.all()

    def test_month_timeslice_M1(self):
        """Test month-based timeslice M1 (January)."""
        snapshots = pd.date_range("2024-01-01", periods=60, freq="D")
        timeslice_activity = pd.DataFrame(index=snapshots)  # Empty
        dataid_to_timeslice = {1: ["M1"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be True only in January (first 31 days)
        assert result.iloc[0:31].all()
        assert not result.iloc[31:].any()

    def test_month_timeslice_M12(self):
        """Test month-based timeslice M12 (December)."""
        snapshots = pd.date_range("2024-12-01", periods=31, freq="D")
        timeslice_activity = pd.DataFrame(index=snapshots)  # Empty
        dataid_to_timeslice = {1: ["M12"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be True for all of December
        assert result.all()

    def test_month_timeslice_with_date_range(self):
        """Test month timeslice combined with date range."""
        snapshots = pd.date_range("2024-01-01", periods=60, freq="D")
        timeslice_activity = pd.DataFrame(index=snapshots)
        dataid_to_timeslice = {1: ["M1"]}
        from_date = pd.Timestamp("2024-01-15")
        row = pd.Series({"from": from_date, "to": pd.NaT, "data_id": 1})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be True only after Jan 15 AND in January
        assert not result.iloc[0:14].any()  # Before Jan 15
        assert result.iloc[14:31].all()  # Jan 15-31
        assert not result.iloc[31:].any()  # February

    def test_null_dates_handled_correctly(self):
        """Test that pd.NaT/null dates don't cause errors."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        row = pd.Series({"from": None, "to": None, "data_id": None})

        result = get_property_active_mask(row, snapshots)

        assert result.all()

    def test_no_data_id_with_timeslice_params(self):
        """Test when data_id is None but timeslice params provided."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        timeslice_activity = pd.DataFrame({"TS1": [True] * 24}, index=snapshots)
        dataid_to_timeslice = {1: ["TS1"]}
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": None})

        result = get_property_active_mask(
            row, snapshots, timeslice_activity, dataid_to_timeslice
        )

        # Should be all True (no timeslice constraint applies)
        assert result.all()

    def test_all_parameters_none(self):
        """Test when all optional parameters are None."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        row = pd.Series({"from": pd.NaT, "to": pd.NaT, "data_id": None})

        result = get_property_active_mask(row, snapshots, None, None)

        assert result.all()


# ============================================================================
# Tests for get_emission_rates
# ============================================================================


@pytest.mark.unit
class TestGetEmissionRates:
    """Test suite for get_emission_rates function."""

    def test_co2_emission_found(self, mock_plexos_db):
        """Test finding CO2 emission rate."""
        properties = [
            {
                "property": "CO2 Emission Rate",
                "value": "0.5",
                "unit": "t/MWh",
                "tag": "emission co2",
            }
        ]
        mock_plexos_db.get_object_properties.return_value = properties

        result = get_emission_rates(mock_plexos_db, "Gen1")

        assert "co2 emission rate" in result
        assert result["co2 emission rate"] == ("0.5", "t/MWh")

    def test_multiple_emissions(self, mock_plexos_db):
        """Test finding multiple emission types."""
        properties = [
            {
                "property": "CO2 Emission Rate",
                "value": "0.5",
                "unit": "t/MWh",
                "tag": "emission co2",
            },
            {
                "property": "NOx Emission Rate",
                "value": "0.001",
                "unit": "t/MWh",
                "tag": "emission nox",
            },
            {
                "property": "SO2 Emission Rate",
                "value": "0.002",
                "unit": "t/MWh",
                "tag": "emission so2",
            },
        ]
        mock_plexos_db.get_object_properties.return_value = properties

        result = get_emission_rates(mock_plexos_db, "Gen1")

        assert len(result) == 3
        assert "co2 emission rate" in result
        assert "nox emission rate" in result
        assert "so2 emission rate" in result

    def test_no_emissions_returns_empty_dict(self, mock_plexos_db):
        """Test returns empty dict when no emissions found."""
        properties = [
            {
                "property": "Max Capacity",
                "value": "100",
                "unit": "MW",
                "tag": "capacity",
            }
        ]
        mock_plexos_db.get_object_properties.return_value = properties

        result = get_emission_rates(mock_plexos_db, "Gen1")

        assert result == {}

    def test_exception_handling_returns_empty_dict(self, mock_plexos_db):
        """Test returns empty dict when exception occurs."""
        mock_plexos_db.get_object_properties.side_effect = Exception("DB error")

        result = get_emission_rates(mock_plexos_db, "Gen1")

        assert result == {}

    def test_tag_matching_case_insensitive(self, mock_plexos_db):
        """Test tag matching is case-insensitive."""
        properties = [
            {
                "property": "CO2 Rate",
                "value": "0.5",
                "unit": "t/MWh",
                "tag": "EMISSION CO2",  # Uppercase
            }
        ]
        mock_plexos_db.get_object_properties.return_value = properties

        result = get_emission_rates(mock_plexos_db, "Gen1")

        assert len(result) > 0

    def test_return_format_is_tuple(self, mock_plexos_db):
        """Test return format is dict of (value, unit) tuples."""
        properties = [
            {
                "property": "CO2 Emission Rate",
                "value": "0.5",
                "unit": "t/MWh",
                "tag": "emission co2",
            }
        ]
        mock_plexos_db.get_object_properties.return_value = properties

        result = get_emission_rates(mock_plexos_db, "Gen1")

        for key, value in result.items():
            assert isinstance(value, tuple)
            assert len(value) == 2


# ============================================================================
# Fixtures specific to test_parse.py
# ============================================================================


@pytest.fixture
def sample_memberships():
    """Return sample membership data for testing."""
    return [
        {
            "name": "Bus1",
            "child_class_name": "Node",
            "child_object_name": "Bus1",
            "parent_class_name": "Generator",
            "parent_object_name": "Gen1",
            "collection_name": "Node",
        }
    ]
