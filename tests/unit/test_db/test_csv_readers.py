"""Unit tests for CSV reading utilities."""

from plexos_to_pypsa_converter.db.csv_readers import parse_numeric_value


class TestParseNumericValue:
    """Test numeric value extraction from CSV data."""

    def test_simple_integer(self):
        """Test extracting simple integer."""
        assert parse_numeric_value("100") == 100
        assert parse_numeric_value("0") == 0
        assert parse_numeric_value("12345") == 12345

    def test_simple_float(self):
        """Test extracting simple float."""
        assert parse_numeric_value("100.5") == 100.5
        assert parse_numeric_value("0.42") == 0.42
        assert parse_numeric_value("3.14159") == 3.14159

    def test_array_string_single_quotes(self):
        """Test extracting first value from array with single quotes."""
        assert parse_numeric_value("['100', '200', '300']") == 100
        assert parse_numeric_value("['5.5', '6.5']") == 5.5

    def test_array_string_double_quotes(self):
        """Test extracting first value from array with double quotes."""
        assert parse_numeric_value('["100", "200", "300"]') == 100
        assert parse_numeric_value('["5.5", "6.5"]') == 5.5

    def test_timeslice_array(self):
        """Test extracting first value from timeslice array."""
        # 12-month timeslice pattern
        timeslice = "['100', '120', '140', '160', '180', '200', '220', '200', '170', '140', '110', '90']"
        assert parse_numeric_value(timeslice) == 100

    def test_empty_string(self):
        """Test that empty string returns None."""
        assert parse_numeric_value("") is None
        assert parse_numeric_value("   ") is None

    def test_none_value(self):
        """Test that None returns None."""
        assert parse_numeric_value(None) is None

    def test_already_numeric(self):
        """Test that numeric values pass through."""
        assert parse_numeric_value(100) == 100
        assert parse_numeric_value(5.5) == 5.5
        assert parse_numeric_value(0) == 0

    def test_invalid_string(self):
        """Test that invalid strings return None or raise error."""
        # Depending on implementation, might return None or raise
        try:
            result = parse_numeric_value("not_a_number")
            assert result is None
        except (ValueError, TypeError):
            pass  # Either behavior is acceptable

    def test_numeric_with_whitespace(self):
        """Test numeric values with surrounding whitespace."""
        assert parse_numeric_value("  100  ") == 100
        assert parse_numeric_value("  5.5  ") == 5.5


# Note: Additional CSV reader tests (read_properties_csv, read_data_files_csv, etc.)
# would require actual CSV files or mocking, which is better suited for integration
# tests or more complex fixtures.
