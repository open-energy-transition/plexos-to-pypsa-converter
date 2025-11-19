"""Unit tests for workflow filter presets."""

import pytest

from plexos_to_pypsa_converter.workflow.filters import (
    FILTER_PRESETS,
    resolve_filter_preset,
)


class TestFilterPresets:
    """Test filter preset definitions."""

    def test_all_presets_defined(self):
        """Test that all expected presets are defined."""
        expected_presets = [
            "all",
            "vre_only",
            "exclude_vre",
            "thermal_only",
            "has_carrier",
            "exclude_vre_and_low_ramp_limits",
        ]
        for preset in expected_presets:
            assert preset in FILTER_PRESETS, f"Missing preset: {preset}"

    def test_preset_structure(self):
        """Test that all presets have required structure."""
        for name, preset in FILTER_PRESETS.items():
            assert "filter" in preset, f"Preset {name} missing 'filter' key"
            assert "requires_network" in preset, (
                f"Preset {name} missing 'requires_network' key"
            )
            assert "description" in preset, f"Preset {name} missing 'description' key"


class TestResolveFilterPreset:
    """Test filter preset resolution."""

    def test_resolve_all_returns_none(self):
        """Test that 'all' preset returns None (no filtering)."""
        filter_fn = resolve_filter_preset("all", network=None)
        assert filter_fn is None

    def test_resolve_none_returns_none(self):
        """Test that None returns None (no filtering)."""
        filter_fn = resolve_filter_preset(None, network=None)
        assert filter_fn is None

    def test_resolve_vre_only(self):
        """Test VRE-only filter logic."""
        filter_fn = resolve_filter_preset("vre_only", network=None)
        assert filter_fn is not None

        # Test positive cases
        assert filter_fn("Wind Offshore") is True
        assert filter_fn("Solar PV") is True
        assert filter_fn("Wind Farm") is True
        assert filter_fn("Solar Farm") is True

        # Test negative cases
        assert filter_fn("CCGT") is False
        assert filter_fn("Coal Plant") is False
        assert filter_fn("Gas Turbine") is False

    def test_resolve_exclude_vre(self, network_with_carriers):
        """Test exclude VRE filter with network."""
        network = network_with_carriers
        filter_fn = resolve_filter_preset("exclude_vre", network=network)
        assert filter_fn is not None

        # Generators with empty carrier should be excluded
        assert filter_fn("CCGT") is True  # Has carrier = "gas"
        assert filter_fn("Coal Plant") is True  # Has carrier = "coal"
        assert filter_fn("Wind Farm") is False  # Empty carrier = VRE
        assert filter_fn("Solar Farm") is False  # Empty carrier = VRE

    def test_resolve_thermal_only(self, network_with_carriers):
        """Test thermal-only filter with network."""
        network = network_with_carriers
        filter_fn = resolve_filter_preset("thermal_only", network=network)
        assert filter_fn is not None

        # Only thermal (non-VRE) generators
        assert filter_fn("CCGT") is True
        assert filter_fn("Coal Plant") is True
        assert filter_fn("Wind Farm") is False
        assert filter_fn("Solar Farm") is False

    def test_resolve_has_carrier(self, network_with_carriers):
        """Test has-carrier filter with network."""
        network = network_with_carriers
        filter_fn = resolve_filter_preset("has_carrier", network=network)
        assert filter_fn is not None

        # Generators with non-empty carrier
        assert filter_fn("CCGT") is True
        assert filter_fn("Coal Plant") is True
        assert filter_fn("Wind Farm") is False
        assert filter_fn("Solar Farm") is False

    def test_resolve_unknown_preset_raises_error(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown filter preset"):
            resolve_filter_preset("nonexistent", network=None)

    def test_resolve_network_required_preset_without_network_raises_error(self):
        """Test that network-requiring preset without network raises error."""
        with pytest.raises(ValueError, match="requires a network"):
            resolve_filter_preset("exclude_vre", network=None)
