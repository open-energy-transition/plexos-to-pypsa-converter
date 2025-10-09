"""Unit tests for src/network/generators.py - generator conversion functions."""

import pandas as pd
import pytest
from plexosdb.enums import ClassEnum

from src.network.generators import (
    add_generators,
    parse_generator_ratings,
    port_generators,
    reassign_generators_to_node,
    set_capacity_ratings,
    set_capital_costs,
    set_generator_efficiencies,
    set_marginal_costs,
    set_vre_profiles,
)

# ============================================================================
# TestAddGenerators
# ============================================================================


@pytest.mark.unit
class TestAddGenerators:
    def test_adds_all_generators_from_db(self, empty_network, mock_plexos_db, mocker):
        """Test that all generators from database are added to network."""
        # Mock find_bus_for_object to return a bus
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )

        empty_network.add("Bus", "Bus1", carrier="AC")

        def side_effect(class_enum, name=None):
            if class_enum == ClassEnum.Generator:
                return ["Gen1", "Gen2", "Gen3"]
            return []

        mock_plexos_db.list_objects_by_class.side_effect = side_effect
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"}
        ]

        add_generators(empty_network, mock_plexos_db)

        assert len(empty_network.generators) == 3
        assert "Gen1" in empty_network.generators.index
        assert "Gen2" in empty_network.generators.index
        assert "Gen3" in empty_network.generators.index

    def test_sets_p_nom_from_max_capacity(self, empty_network, mock_plexos_db, mocker):
        """Test that p_nom is set from Max Capacity property."""
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value=None
        )

        empty_network.add("Bus", "Bus1", carrier="AC")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "250"}
        ]

        add_generators(empty_network, mock_plexos_db)

        assert empty_network.generators.loc["Gen1", "p_nom"] == 250

    def test_sets_all_generator_attributes(self, empty_network, mock_plexos_db, mocker):
        """Test that all generator attributes are set correctly."""
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value=None
        )

        empty_network.add("Bus", "Bus1", carrier="AC")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"},
            {"property": "Min Capacity", "value": "10"},
            {"property": "Start Cost", "value": "1000"},
            {"property": "Shutdown Cost", "value": "500"},
            {"property": "Min Up Time", "value": "4"},
            {"property": "Min Down Time", "value": "2"},
            {"property": "Max Ramp Up", "value": "50"},
            {"property": "Max Ramp Down", "value": "50"},
        ]

        add_generators(empty_network, mock_plexos_db)

        gen = empty_network.generators.loc["Gen1"]
        assert gen["p_nom_min"] == 10
        assert gen["start_up_cost"] == 1000
        assert gen["shut_down_cost"] == 500
        assert gen["min_up_time"] == 4
        assert gen["min_down_time"] == 2
        assert gen["ramp_limit_up"] == 50
        assert gen["ramp_limit_down"] == 50

    def test_finds_bus_via_find_bus_for_object(
        self, empty_network, mock_plexos_db, mocker
    ):
        """Test that bus is found via find_bus_for_object from parse.py."""
        mock_find_bus = mocker.patch(
            "src.network.generators.find_bus_for_object", return_value="FoundBus"
        )
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value=None
        )

        empty_network.add("Bus", "FoundBus", carrier="AC")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"}
        ]

        add_generators(empty_network, mock_plexos_db)

        mock_find_bus.assert_called()
        assert empty_network.generators.loc["Gen1", "bus"] == "FoundBus"

    def test_finds_carrier_via_find_fuel_for_generator(
        self, empty_network, mock_plexos_db, mocker
    ):
        """Test that carrier is found via find_fuel_for_generator."""
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mock_find_fuel = mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Natural Gas"
        )

        empty_network.add("Bus", "Bus1", carrier="AC")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"}
        ]

        add_generators(empty_network, mock_plexos_db)

        mock_find_fuel.assert_called()
        assert empty_network.generators.loc["Gen1", "carrier"] == "Natural Gas"

    def test_skips_generators_without_bus(
        self, empty_network, mock_plexos_db, mocker, capsys
    ):
        """Test that generators without bus are skipped and reported."""
        mocker.patch("src.network.generators.find_bus_for_object", return_value=None)
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value=None
        )

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1", "Gen2"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"}
        ]

        add_generators(empty_network, mock_plexos_db)

        assert len(empty_network.generators) == 0

        captured = capsys.readouterr()
        assert "Skipped 2 generators with no associated bus" in captured.out

    def test_skips_generators_without_properties(
        self, empty_network, mock_plexos_db, capsys
    ):
        """Test that generators without properties are skipped."""
        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.side_effect = Exception("No properties")

        add_generators(empty_network, mock_plexos_db)

        assert len(empty_network.generators) == 0

        captured = capsys.readouterr()
        assert "Skipped 1 generators with no properties" in captured.out

    def test_generators_as_links_mode(self, empty_network, mock_plexos_db, mocker):
        """Test generators-as-links mode creates fuel buses and links."""
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator",
            return_value="Natural Gas CCGT",
        )

        empty_network.add("Bus", "Bus1", carrier="AC")

        mock_plexos_db.list_objects_by_class.return_value = ["CCGT_Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"}
        ]

        add_generators(empty_network, mock_plexos_db, generators_as_links=True)

        # Should create fuel bus and link
        assert "fuel_Natural_Gas" in empty_network.buses.index
        assert "gen_link_CCGT_Gen1" in empty_network.links.index


# ============================================================================
# TestParseGeneratorRatings
# ============================================================================


@pytest.mark.unit
class TestParseGeneratorRatings:
    def test_rating_factor_used_when_present(
        self, network_with_generator, mock_plexos_db
    ):
        """Test that Rating Factor is used when present (p_max_pu = Rating Factor / 100)."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        # Mock SQL query results
        mock_plexos_db.query.side_effect = [
            # gen_query: generator object_id and name
            [(1, "Gen1")],
            # membership_query
            [(1, 1, 1)],
            # prop_query: Rating Factor property
            [(1, 1, "Rating Factor", "80", None, None, None)],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        assert "Gen1" in result.columns
        assert result["Gen1"].iloc[0] == 0.8  # 80 / 100

    def test_rating_used_when_no_factor(self, network_with_generator, mock_plexos_db):
        """Test that Rating is used when Rating Factor not present (p_max_pu = Rating / p_nom)."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        # Mock SQL query results
        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            # Rating property: 50 MW
            [(1, 1, "Rating", "50", None, None, None)],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        # p_nom for Gen1 is 100, so Rating 50 / 100 = 0.5
        assert result["Gen1"].iloc[0] == 0.5

    def test_max_capacity_fallback(self, network_with_generator, mock_plexos_db):
        """Test fallback to Max Capacity when neither Rating nor Rating Factor present."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            # Max Capacity property
            [(1, 1, "Max Capacity", "100", None, None, None)],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        # Should default to 1.0 when only Max Capacity
        assert result["Gen1"].iloc[0] == 1.0

    def test_default_to_1_when_no_data(self, network_with_generator, mock_plexos_db):
        """Test default to 1.0 when no rating data available."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [],  # No properties
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        assert result["Gen1"].iloc[0] == 1.0

    def test_timeslice_activity_applied(
        self, network_with_generator, mock_plexos_db, tmp_path, mocker
    ):
        """Test that timeslice activity is applied to ratings."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        # Create timeslice CSV
        timeslice_csv = tmp_path / "timeslice.csv"
        timeslice_data = [
            {"DATETIME": "01/01/2024 00:00:00", "NAME": "TS1", "TIMESLICE": -1},
            {"DATETIME": "01/01/2024 12:00:00", "NAME": "TS1", "TIMESLICE": 0},
        ]
        df = pd.DataFrame(timeslice_data)
        df.to_csv(timeslice_csv, index=False)

        # Mock get_dataid_timeslice_map
        mocker.patch(
            "src.network.generators.get_dataid_timeslice_map", return_value={1: ["TS1"]}
        )

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Rating Factor", "50", None, None, "TS1")],
        ]

        result = parse_generator_ratings(
            mock_plexos_db, network, timeslice_csv=str(timeslice_csv)
        )

        # Rating should be active for first 12 hours, then inactive
        assert "Gen1" in result.columns

    def test_date_from_constraint(self, network_with_generator, mock_plexos_db):
        """Test that date_from constraint is applied (effective from date)."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            # Rating Factor effective from Jan 2
            [(1, 1, "Rating Factor", "60", "2024-01-02 00:00:00", None, None)],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        # Should be 1.0 before Jan 2, then 0.6 from Jan 2 onwards
        assert result["Gen1"].iloc[0] == 1.0  # Jan 1
        assert result["Gen1"].iloc[24] == 0.6  # Jan 2

    def test_date_to_constraint(self, network_with_generator, mock_plexos_db):
        """Test that date_to constraint is applied (effective until date)."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            # Rating Factor effective until Jan 2
            [(1, 1, "Rating Factor", "70", None, "2024-01-01 23:59:59", None)],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        # Should be 0.7 until Jan 2, then 1.0
        assert result["Gen1"].iloc[0] == 0.7  # Jan 1
        assert result["Gen1"].iloc[24] == 1.0  # Jan 2

    def test_date_range_constraint(self, network_with_generator, mock_plexos_db):
        """Test that both date_from and date_to constraints are applied."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=72, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            # Rating Factor effective only on Jan 2
            [
                (
                    1,
                    1,
                    "Rating Factor",
                    "50",
                    "2024-01-02 00:00:00",
                    "2024-01-02 23:59:59",
                    None,
                )
            ],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        # Should be 1.0 except Jan 2 when it's 0.5
        assert result["Gen1"].iloc[0] == 1.0  # Jan 1
        assert result["Gen1"].iloc[24] == 0.5  # Jan 2
        assert result["Gen1"].iloc[48] == 1.0  # Jan 3

    def test_multiple_generators_in_dataframe(
        self, network_with_generators, mock_plexos_db
    ):
        """Test that result DataFrame has index=snapshots, columns=generator names."""
        network = network_with_generators
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1"), (2, "Gen2"), (3, "Gen3")],
            [(1, 1, 1), (1, 2, 2), (1, 3, 3)],
            [
                (1, 1, "Rating Factor", "80", None, None, None),
                (2, 2, "Rating Factor", "90", None, None, None),
                (3, 3, "Max Capacity", "150", None, None, None),
            ],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 24
        assert "Gen1" in result.columns
        assert "Gen2" in result.columns
        assert "Gen3" in result.columns

    def test_only_network_generators_included(
        self, network_with_generator, mock_plexos_db
    ):
        """Test that only generators in network are included in result."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        # Database has Gen1 and Gen2, but network only has Gen1
        mock_plexos_db.query.side_effect = [
            [(1, "Gen1"), (2, "Gen2")],
            [(1, 1, 1), (1, 2, 2)],
            [
                (1, 1, "Rating Factor", "80", None, None, None),
                (2, 2, "Rating Factor", "90", None, None, None),
            ],
        ]

        result = parse_generator_ratings(mock_plexos_db, network)

        assert "Gen1" in result.columns
        assert "Gen2" not in result.columns  # Not in network


# ============================================================================
# TestSetCapacityRatings
# ============================================================================


@pytest.mark.unit
class TestSetCapacityRatings:
    def test_sets_p_max_pu_for_all_generators(
        self, network_with_generators, mock_plexos_db
    ):
        """Test that p_max_pu is set for all generators in network."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generators.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1"), (2, "Gen2"), (3, "Gen3")],
            [(1, 1, 1), (1, 2, 2), (1, 3, 3)],
            [
                (1, 1, "Rating Factor", "80", None, None, None),
                (2, 2, "Rating Factor", "90", None, None, None),
                (3, 3, "Rating Factor", "100", None, None, None),
            ],
        ]

        set_capacity_ratings(network_with_generators, mock_plexos_db)

        assert len(network_with_generators.generators_t.p_max_pu.columns) >= 3
        assert "Gen1" in network_with_generators.generators_t.p_max_pu.columns

    def test_skips_missing_generators(
        self, network_with_generators, mock_plexos_db, capsys
    ):
        """Test that generators not in ratings DataFrame print warning."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generators.set_snapshots(snapshots)

        # Query returns Gen1 and Gen2, but not Gen3
        mock_plexos_db.query.side_effect = [
            [(1, "Gen1"), (2, "Gen2")],  # Missing Gen3
            [(1, 1, 1), (1, 2, 2)],
            [
                (1, 1, "Max Capacity", "100", None, None, None),
                (2, 2, "Max Capacity", "200", None, None, None),
            ],
        ]

        set_capacity_ratings(network_with_generators, mock_plexos_db)

        captured = capsys.readouterr()
        assert "Warning: Generator Gen3 not found in ratings DataFrame" in captured.out

    def test_assigns_timeseries_to_network(
        self, network_with_generator, mock_plexos_db
    ):
        """Test that time series is assigned to network.generators_t.p_max_pu."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Rating Factor", "75", None, None, None)],
        ]

        set_capacity_ratings(network_with_generator, mock_plexos_db)

        assert "Gen1" in network_with_generator.generators_t.p_max_pu.columns
        assert network_with_generator.generators_t.p_max_pu["Gen1"].iloc[0] == 0.75

    def test_with_timeslice_csv(
        self, network_with_generator, mock_plexos_db, tmp_path, mocker
    ):
        """Test that timeslice_csv parameter is passed correctly."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        timeslice_csv = tmp_path / "timeslice.csv"
        df = pd.DataFrame(
            [{"DATETIME": "01/01/2024 00:00:00", "NAME": "TS1", "TIMESLICE": -1}]
        )
        df.to_csv(timeslice_csv, index=False)

        mocker.patch("src.network.generators.get_dataid_timeslice_map", return_value={})

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Rating Factor", "80", None, None, None)],
        ]

        set_capacity_ratings(
            network_with_generator, mock_plexos_db, timeslice_csv=str(timeslice_csv)
        )

        assert "Gen1" in network_with_generator.generators_t.p_max_pu.columns

    def test_batch_assignment_no_fragmentation(
        self, network_with_generators, mock_plexos_db
    ):
        """Test that all columns are assigned at once to avoid DataFrame fragmentation."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generators.set_snapshots(snapshots)

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1"), (2, "Gen2"), (3, "Gen3")],
            [(1, 1, 1), (1, 2, 2), (1, 3, 3)],
            [
                (1, 1, "Max Capacity", "100", None, None, None),
                (2, 2, "Max Capacity", "200", None, None, None),
                (3, 3, "Max Capacity", "150", None, None, None),
            ],
        ]

        set_capacity_ratings(network_with_generators, mock_plexos_db)

        # Should have all generators in p_max_pu
        assert len(network_with_generators.generators_t.p_max_pu.columns) >= 3


# ============================================================================
# TestSetGeneratorEfficiencies
# ============================================================================


@pytest.mark.unit
class TestSetGeneratorEfficiencies:
    def test_calculates_efficiency_from_heat_rates(
        self, network_with_generator, mock_plexos_db
    ):
        """Test efficiency calculation: (p_nom / fuel) * 3.6."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "10"},
            {"property": "Heat Rate Incr1", "value": "9.5"},
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=True
        )

        # p_nom = 100, fuel = 10 + 9.5*100 = 960, efficiency = (100/960)*3.6 = 0.375
        efficiency = network_with_generator.generators.loc["Gen1", "efficiency"]
        assert abs(efficiency - 0.375) < 0.001

    def test_heat_rate_base_only(self, network_with_generator, mock_plexos_db):
        """Test efficiency with Heat Rate Base only (no incremental)."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "10"}
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=True
        )

        # fuel = 10, efficiency = (100/10)*3.6 = 36
        efficiency = network_with_generator.generators.loc["Gen1", "efficiency"]
        assert abs(efficiency - 36.0) < 0.001

    def test_heat_rate_base_plus_single_incr(
        self, network_with_generator, mock_plexos_db
    ):
        """Test efficiency with Heat Rate Base + single Incr (fuel = base + incr*p_nom)."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "100"},
            {"property": "Heat Rate Incr1", "value": "0.5"},
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=True
        )

        # fuel = 100 + 0.5*100 = 150, efficiency = (100/150)*3.6 = 2.4
        efficiency = network_with_generator.generators.loc["Gen1", "efficiency"]
        assert abs(efficiency - 2.4) < 0.001

    def test_heat_rate_base_plus_multiple_incr(
        self, network_with_generator, mock_plexos_db
    ):
        """Test efficiency with multiple Heat Rate Incr (segmented calculation)."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "50"},
            {"property": "Heat Rate Incr1", "value": "1.0"},
            {"property": "Heat Rate Incr2", "value": "1.5"},
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=True
        )

        # p_nom=100, n=2, p_seg=50
        # fuel = 50 + (1.0*50) + (1.5*50) = 50 + 50 + 75 = 175
        # efficiency = (100/175)*3.6 = 2.057
        efficiency = network_with_generator.generators.loc["Gen1", "efficiency"]
        assert abs(efficiency - 2.057) < 0.01

    def test_use_incr_false_ignores_incr(
        self, network_with_generator, mock_plexos_db, capsys
    ):
        """Test that use_incr=False ignores Heat Rate Incr with warning."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "10"},
            {"property": "Heat Rate Incr1", "value": "5"},
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=False
        )

        captured = capsys.readouterr()
        assert "Only Heat Rate Base will be used" in captured.out

        # fuel = 10 (incr ignored), efficiency = (100/10)*3.6 = 36
        efficiency = network_with_generator.generators.loc["Gen1", "efficiency"]
        assert abs(efficiency - 36.0) < 0.001

    def test_efficiency_gt_1_prints_warning(
        self, network_with_generator, mock_plexos_db, capsys
    ):
        """Test that efficiency > 1 prints warning."""
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Base", "value": "1"},  # Very low heat rate
        ]

        set_generator_efficiencies(
            network_with_generator, mock_plexos_db, use_incr=True
        )

        captured = capsys.readouterr()
        assert "efficiency > 1" in captured.out


# ============================================================================
# TestSetVreProfiles
# ============================================================================


@pytest.mark.unit
class TestSetVreProfiles:
    def test_sets_solar_profiles(
        self, network_with_generator, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that solar profiles are detected and set."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        # Initialize p_max_pu
        network.generators_t.p_max_pu = pd.DataFrame(
            1.0, index=snapshots, columns=["Gen1"]
        )

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
            }
        ]

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        # Check that p_max_pu was modified (multiplied by CF)
        assert "Gen1" in network.generators_t.p_max_pu.columns

    def test_sets_wind_profiles(
        self, network_with_generator, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that wind profiles are detected and set."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        network.generators_t.p_max_pu = pd.DataFrame(
            1.0, index=snapshots, columns=["Gen1"]
        )

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/wind/Gen_Wind.csv",
            }
        ]

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        assert "Gen1" in network.generators_t.p_max_pu.columns

    def test_sets_carrier_to_solar_or_wind(
        self, network_with_generator, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that generator carrier is updated to Solar or Wind."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        network.generators_t.p_max_pu = pd.DataFrame(
            1.0, index=snapshots, columns=["Gen1"]
        )

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
            }
        ]

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        assert network.generators.loc["Gen1", "carrier"] == "Solar"

    def test_multiplies_by_p_max_pu(
        self, network_with_generator, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that VRE profile is multiplied by existing p_max_pu (dispatch = cf * p_max_pu)."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        # Set initial p_max_pu to 0.5
        network.generators_t.p_max_pu = pd.DataFrame(
            0.5, index=snapshots, columns=["Gen1"]
        )

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
            }
        ]

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        # Profile CF is 0.8 during day, should be multiplied by 0.5
        # Result should be max 0.4 during day
        max_value = network.generators_t.p_max_pu["Gen1"].max()
        assert max_value <= 0.5  # Should not exceed original p_max_pu

    def test_aligns_with_network_snapshots(
        self, network_with_generator, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that VRE profile is aligned with network snapshots using reindex()."""
        network = network_with_generator
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network.set_snapshots(snapshots)

        network.generators_t.p_max_pu = pd.DataFrame(
            1.0, index=snapshots, columns=["Gen1"]
        )

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
            }
        ]

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        # Index should match network snapshots
        assert len(network.generators_t.p_max_pu) == len(snapshots)

    def test_skips_adelaide_desal_ffp(
        self, empty_network, mock_plexos_db, sample_vre_profiles_dir, capsys
    ):
        """Test that Adelaide_Desal_FFP is skipped (hardcoded)."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        empty_network.add("Generator", "Adelaide_Desal_FFP", bus="Bus1", p_nom=100)

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        mock_plexos_db.get_object_properties.return_value = [
            {
                "property": "Load",
                "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
            }
        ]

        set_vre_profiles(empty_network, mock_plexos_db, str(sample_vre_profiles_dir))

        captured = capsys.readouterr()
        assert "Skipping generator Adelaide_Desal_FFP" in captured.out

    def test_batch_assignment_to_p_max_pu(
        self, network_with_generators, mock_plexos_db, sample_vre_profiles_dir
    ):
        """Test that all profiles are assigned at once using concat (single operation)."""
        network = network_with_generators
        snapshots = pd.date_range("2024-01-01", periods=48, freq="h")
        network.set_snapshots(snapshots)

        network.generators_t.p_max_pu = pd.DataFrame(
            1.0, index=snapshots, columns=["Gen1", "Gen2", "Gen3"]
        )

        def side_effect_props(class_enum, name):
            if name == "Gen1":
                return [
                    {
                        "property": "Load",
                        "texts": f"{sample_vre_profiles_dir}/Traces/solar/Gen_Solar.csv",
                    }
                ]
            elif name == "Gen2":
                return [
                    {
                        "property": "Load",
                        "texts": f"{sample_vre_profiles_dir}/Traces/wind/Gen_Wind.csv",
                    }
                ]
            return []

        mock_plexos_db.get_object_properties.side_effect = side_effect_props

        set_vre_profiles(network, mock_plexos_db, str(sample_vre_profiles_dir))

        # Should have profiles for Gen1 and Gen2
        assert "Gen1" in network.generators_t.p_max_pu.columns
        assert "Gen2" in network.generators_t.p_max_pu.columns


# ============================================================================
# TestSetCapitalCosts
# ============================================================================


@pytest.mark.unit
class TestSetCapitalCosts:
    def test_calls_generic_capital_costs(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test that set_capital_costs calls generic function."""
        mock_generic = mocker.patch("src.network.generators.set_capital_costs_generic")

        set_capital_costs(network_with_generator, mock_plexos_db)

        mock_generic.assert_called_once()

    def test_passes_correct_class_enum(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test that correct ClassEnum.Generator is passed to generic function."""
        mock_generic = mocker.patch("src.network.generators.set_capital_costs_generic")

        set_capital_costs(network_with_generator, mock_plexos_db)

        # Check that ClassEnum.Generator was passed
        call_args = mock_generic.call_args
        assert call_args[0][3] == ClassEnum.Generator


# ============================================================================
# TestSetMarginalCosts
# ============================================================================


@pytest.mark.unit
class TestSetMarginalCosts:
    def test_calculates_marginal_cost_timeseries(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test marginal cost calculation: fuel_price * heat_rate + vo_m."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        # Mock fuel prices
        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        # Marginal cost = 10 * 9.5 + 5 = 100
        assert hasattr(network_with_generator.generators_t, "marginal_cost")
        assert (
            network_with_generator.generators_t.marginal_cost["Gen1"].iloc[0] == 100.0
        )

    def test_uses_carrier_from_network(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test that carrier is read from network.generators.carrier."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        # Gen1 already has carrier="Gas"
        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        # Should use carrier from network
        assert "Gen1" in network_with_generator.generators_t.marginal_cost.columns

    def test_falls_back_to_find_fuel_for_generator(
        self, empty_network, mock_plexos_db, mocker
    ):
        """Test fallback to find_fuel_for_generator if carrier not set in network."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        empty_network.add(
            "Generator", "Gen1", bus="Bus1", p_nom=100
        )  # No carrier (None)
        # Explicitly set carrier to None or NaN
        empty_network.generators.loc["Gen1", "carrier"] = None

        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"Coal": [8.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )
        mock_find_fuel = mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Coal"
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "10"},
            {"property": "Heat Rate Inc", "value": "10"},
            {"property": "VO&M Charge", "value": "3"},
        ]

        set_marginal_costs(empty_network, mock_plexos_db)

        # Should have called find_fuel since carrier not set
        assert mock_find_fuel.called

    def test_skips_generators_without_carrier(
        self, network_with_generator, mock_plexos_db, mocker, capsys
    ):
        """Test that generators without carrier are skipped with message."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"UnknownFuel": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        # Gen1 has carrier="Gas" but fuel_prices doesn't have "Gas"
        mock_plexos_db.get_object_properties.return_value = []

        set_marginal_costs(network_with_generator, mock_plexos_db)

        captured = capsys.readouterr()
        assert "Skipped" in captured.out

    def test_skips_generators_without_heat_rate_inc(
        self, network_with_generator, mock_plexos_db, mocker, capsys
    ):
        """Test that generators without Heat Rate Inc are skipped."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "VO&M Charge", "value": "5"},  # No Heat Rate Inc
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        captured = capsys.readouterr()
        # Logger.warning won't show in capsys, check the summary message
        assert "Skipped" in captured.out

    def test_skips_generators_without_vo_m_charge(
        self, network_with_generator, mock_plexos_db, mocker, capsys
    ):
        """Test that generators without VO&M Charge are skipped."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "Heat Rate Inc", "value": "9.5"},  # No VO&M Charge
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        captured = capsys.readouterr()
        # Logger.warning won't show in capsys, check the summary message
        assert "Skipped" in captured.out

    def test_average_multiple_heat_rate_inc(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test that multiple Heat Rate Inc values are averaged."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "8"},
            {"property": "Heat Rate Incr2", "value": "10"},
            {"property": "Heat Rate Incr3", "value": "12"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        # Average heat rate = (8+10+12)/3 = 10
        # Marginal cost = 10 * 10 + 5 = 105
        assert (
            network_with_generator.generators_t.marginal_cost["Gen1"].iloc[0] == 105.0
        )

    def test_assigns_to_generators_t_marginal_cost(
        self, network_with_generator, mock_plexos_db, mocker
    ):
        """Test that marginal cost time series is assigned to network.generators_t.marginal_cost."""
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        network_with_generator.set_snapshots(snapshots)

        fuel_prices = pd.DataFrame({"Gas": [10.0] * 24}, index=snapshots)
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=fuel_prices
        )

        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        set_marginal_costs(network_with_generator, mock_plexos_db)

        assert hasattr(network_with_generator.generators_t, "marginal_cost")
        assert isinstance(
            network_with_generator.generators_t.marginal_cost, pd.DataFrame
        )
        assert len(network_with_generator.generators_t.marginal_cost) == 24


# ============================================================================
# TestReassignGeneratorsToNode
# ============================================================================


@pytest.mark.unit
class TestReassignGeneratorsToNode:
    def test_reassigns_all_generators_to_target(self, network_with_generators):
        """Test that all generators are reassigned to target node."""
        # Gen1, Gen2, Gen3 all on Bus1, reassign to Bus2
        network_with_generators.add("Bus", "Bus2", carrier="AC")

        reassign_generators_to_node(network_with_generators, "Bus2")

        assert all(network_with_generators.generators["bus"] == "Bus2")

    def test_target_node_not_exists_raises_error(self, network_with_generators):
        """Test that non-existent target node raises ValueError."""
        with pytest.raises(ValueError, match="not found in network buses"):
            reassign_generators_to_node(network_with_generators, "NonExistent")

    def test_returns_summary_dict(self, network_with_generators):
        """Test that function returns summary dict with original_buses, count."""
        network_with_generators.add("Bus", "Bus2", carrier="AC")

        result = reassign_generators_to_node(network_with_generators, "Bus2")

        assert "reassigned_count" in result
        assert "target_node" in result
        assert "original_buses" in result
        assert result["reassigned_count"] == 3

    def test_prints_reassignment_summary(self, network_with_generators, capsys):
        """Test that function prints reassignment summary."""
        network_with_generators.add("Bus", "Bus2", carrier="AC")

        reassign_generators_to_node(network_with_generators, "Bus2")

        captured = capsys.readouterr()
        assert "Reassigned 3 generators" in captured.out


# ============================================================================
# TestPortGenerators
# ============================================================================


@pytest.mark.unit
class TestPortGenerators:
    def test_full_workflow_all_steps(
        self, empty_network, mock_plexos_db, mocker, capsys
    ):
        """Test full port_generators workflow with all steps."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        # Mock all dependencies
        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )
        mocker.patch(
            "src.network.generators.parse_fuel_prices",
            return_value=pd.DataFrame({"Gas": [10] * 24}, index=snapshots),
        )
        mocker.patch("src.network.generators.set_capital_costs_generic")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"},
            {"property": "Heat Rate Base", "value": "10"},
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Max Capacity", "100", None, None, None)],
        ]

        port_generators(empty_network, mock_plexos_db)

        captured = capsys.readouterr()
        # Should print all 7 steps
        assert "1. Adding generators" in captured.out
        assert "2. Setting capacity ratings" in captured.out
        assert "3. Setting generator efficiencies" in captured.out
        assert "4. Setting capital costs" in captured.out
        assert "5. Setting marginal costs" in captured.out

    def test_with_target_node_reassignment(self, empty_network, mock_plexos_db, mocker):
        """Test that target_node parameter triggers reassignment."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        empty_network.add("Bus", "TargetBus", carrier="AC")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )
        mocker.patch(
            "src.network.generators.parse_fuel_prices",
            return_value=pd.DataFrame({"Gas": [10] * 24}, index=snapshots),
        )
        mocker.patch("src.network.generators.set_capital_costs_generic")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"},
            {"property": "Heat Rate Base", "value": "10"},
        ]

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Max Capacity", "100", None, None, None)],
        ]

        result = port_generators(empty_network, mock_plexos_db, target_node="TargetBus")

        assert result is not None
        assert "target_node" in result
        assert empty_network.generators.loc["Gen1", "bus"] == "TargetBus"

    def test_with_generators_as_links(self, empty_network, mock_plexos_db, mocker):
        """Test that generators_as_links parameter is passed to add_generators."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator",
            return_value="Natural Gas CCGT",
        )
        mocker.patch(
            "src.network.generators.parse_fuel_prices", return_value=pd.DataFrame()
        )
        mocker.patch("src.network.generators.set_capital_costs_generic")
        # Mock parse_generator_ratings to avoid empty concatenation error
        mocker.patch(
            "src.network.generators.parse_generator_ratings",
            return_value=pd.DataFrame(),
        )

        mock_plexos_db.list_objects_by_class.return_value = ["CCGT_Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100", "texts": None}
        ]

        port_generators(empty_network, mock_plexos_db, generators_as_links=True)

        # Should create fuel bus
        assert "fuel_Natural_Gas" in empty_network.buses.index

    def test_with_all_optional_params(
        self, empty_network, mock_plexos_db, mocker, tmp_path
    ):
        """Test with all optional parameters (timeslice_csv, vre_profiles_path)."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        timeslice_csv = tmp_path / "timeslice.csv"
        df = pd.DataFrame(
            [{"DATETIME": "01/01/2024 00:00:00", "NAME": "TS1", "TIMESLICE": -1}]
        )
        df.to_csv(timeslice_csv, index=False)

        vre_path = tmp_path / "vre"
        vre_path.mkdir()

        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )
        mocker.patch(
            "src.network.generators.parse_fuel_prices",
            return_value=pd.DataFrame({"Gas": [10] * 24}, index=snapshots),
        )
        mocker.patch("src.network.generators.set_capital_costs_generic")
        mocker.patch("src.network.generators.get_dataid_timeslice_map", return_value={})

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100", "texts": None},
            {"property": "Heat Rate Base", "value": "10", "texts": None},
        ]

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Max Capacity", "100", None, None, None)],
        ]

        port_generators(
            empty_network,
            mock_plexos_db,
            timeslice_csv=str(timeslice_csv),
            vre_profiles_path=str(vre_path),
        )

        # Should complete without error
        assert len(empty_network.generators) >= 1

    def test_prints_step_summaries(self, empty_network, mock_plexos_db, mocker, capsys):
        """Test that each of the 7 steps prints a summary."""
        empty_network.add("Bus", "Bus1", carrier="AC")
        snapshots = pd.date_range("2024-01-01", periods=24, freq="h")
        empty_network.set_snapshots(snapshots)

        mocker.patch("src.network.generators.find_bus_for_object", return_value="Bus1")
        mocker.patch(
            "src.network.generators.find_fuel_for_generator", return_value="Gas"
        )
        mocker.patch(
            "src.network.generators.parse_fuel_prices",
            return_value=pd.DataFrame({"Gas": [10] * 24}, index=snapshots),
        )
        mocker.patch("src.network.generators.set_capital_costs_generic")

        mock_plexos_db.list_objects_by_class.return_value = ["Gen1"]
        mock_plexos_db.get_object_properties.return_value = [
            {"property": "Max Capacity", "value": "100"},
            {"property": "Heat Rate Base", "value": "10"},
            {"property": "Heat Rate Incr1", "value": "9.5"},
            {"property": "VO&M Charge", "value": "5"},
        ]

        mock_plexos_db.query.side_effect = [
            [(1, "Gen1")],
            [(1, 1, 1)],
            [(1, 1, "Max Capacity", "100", None, None, None)],
        ]

        port_generators(empty_network, mock_plexos_db)

        captured = capsys.readouterr()
        assert "1. Adding generators" in captured.out
        assert "2. Setting capacity ratings" in captured.out
        assert "3. Setting generator efficiencies" in captured.out
        assert "4. Setting capital costs" in captured.out
        assert "5. Setting marginal costs" in captured.out
        assert "6." in captured.out  # VRE profiles step
        assert "7." in captured.out  # Reassignment step
