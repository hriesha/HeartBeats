"""
Acceptance tests for Pace → Step BPM (pace-only).

Run: pytest recs/test_pace_to_step_bpm.py -v
"""

import math
import pytest

from recs.pace_to_step_bpm import (
    InvalidPaceError,
    InvalidUnitError,
    InvalidInputError,
    pace_to_speed_mph,
    pace_to_step_bpm,
    DEFAULT_BASE_SPM,
    DEFAULT_SPM_PER_MPH,
)


# -----------------------------------------------------------------------------
# Basic pace conversion (min/mile)
# -----------------------------------------------------------------------------


def test_pace_min_per_mile_conversion():
    """Input: pace=8.0, unit=min/mile → speed_mph = 7.5"""
    speed = pace_to_speed_mph(8.0, "min/mile")
    assert speed == 7.5


def test_step_bpm_raw_min_per_mile():
    """Input: pace=8.0, unit=min/mile → step_bpm_raw ≈ BASE + SPM_PER_MPH * 7.5 (using defaults)"""
    out = pace_to_step_bpm(8.0, "min/mile", include_debug=False)
    assert out["speed_mph"] == 7.5
    expected = DEFAULT_BASE_SPM + DEFAULT_SPM_PER_MPH * 7.5
    assert math.isclose(out["step_bpm_raw"], expected, rel_tol=1e-9)
    assert out["model_info"]["model_name"] == "linear_v1"
    assert out["model_info"]["constants_used"]["BASE_SPM"] == DEFAULT_BASE_SPM
    assert out["model_info"]["constants_used"]["SPM_PER_MPH"] == DEFAULT_SPM_PER_MPH


# -----------------------------------------------------------------------------
# Basic pace conversion (min/km)
# -----------------------------------------------------------------------------


def test_pace_min_per_km_conversion():
    """Input: pace=5.0, unit=min/km → speed_kmh = 12, speed_mph ≈ 7.456"""
    speed = pace_to_speed_mph(5.0, "min/km")
    speed_kmh = 60.0 / 5.0
    assert speed_kmh == 12.0
    assert math.isclose(speed, 12.0 / 1.609344, rel_tol=1e-5)


def test_step_bpm_raw_min_per_km():
    """Input: pace=5.0, unit=min/km → step_bpm_raw ≈ BASE + SPM_PER_MPH * speed_mph"""
    out = pace_to_step_bpm(5.0, "min/km", include_debug=False)
    expected_speed_mph = (60.0 / 5.0) / 1.609344
    assert math.isclose(out["speed_mph"], expected_speed_mph, rel_tol=1e-5)
    expected_raw = DEFAULT_BASE_SPM + DEFAULT_SPM_PER_MPH * expected_speed_mph
    assert math.isclose(out["step_bpm_raw"], expected_raw, rel_tol=1e-5)


# -----------------------------------------------------------------------------
# Clamp behavior
# -----------------------------------------------------------------------------


def test_clamp_very_slow_pace():
    """Very slow pace (e.g. 20 min/mile) should not return < 140 after default clamp."""
    out = pace_to_step_bpm(20.0, "min/mile", include_debug=False)
    assert out["step_bpm_final"] >= 140.0
    assert out["step_bpm_final"] <= 200.0


def test_clamp_extreme_slow():
    """Extreme slow pace produces raw < 140; clamp brings it to 140."""
    out2 = pace_to_step_bpm(60.0, "min/mile", include_debug=False)
    assert out2["step_bpm_raw"] < 140.0
    assert out2["step_bpm_final"] == 140.0


# -----------------------------------------------------------------------------
# Snap behavior
# -----------------------------------------------------------------------------


def test_snap_grid_2_even_integer():
    """With snap_grid=2, output cadence should be an even integer."""
    out = pace_to_step_bpm(
        8.0,
        "min/mile",
        snap_grid_spm=2,
        smoothing={"enabled": False},
        include_debug=False,
    )
    assert int(out["step_bpm_final"]) % 2 == 0


# -----------------------------------------------------------------------------
# Smoothing behavior
# -----------------------------------------------------------------------------


def test_smoothing_ema_partial_move():
    """Given previous cadence and new estimate, EMA output moves partially (not full jump)."""
    # Second call: previous 160, new estimate from 8 min/mile (raw = BASE + 5.5*7.5)
    out2 = pace_to_step_bpm(
        8.0,
        "min/mile",
        previous_step_bpm_final=160.0,
        include_debug=False,
    )
    raw = DEFAULT_BASE_SPM + DEFAULT_SPM_PER_MPH * 7.5
    assert math.isclose(out2["step_bpm_raw"], raw, rel_tol=1e-9)
    # step_bpm_final = 0.2 * raw + 0.8 * 160 (between 160 and raw)
    assert 160.0 < out2["step_bpm_final"] < raw


def test_smoothing_no_previous_equals_snapped():
    """If no previous value, step_bpm_final equals step_bpm_snapped (after clamp/snap)."""
    out = pace_to_step_bpm(8.0, "min/mile", include_debug=False)
    expected = round(DEFAULT_BASE_SPM + DEFAULT_SPM_PER_MPH * 7.5)
    assert out["step_bpm_final"] == float(expected)


# -----------------------------------------------------------------------------
# Error handling
# -----------------------------------------------------------------------------


def test_invalid_pace_zero():
    with pytest.raises(InvalidPaceError):
        pace_to_speed_mph(0.0, "min/mile")


def test_invalid_pace_negative():
    with pytest.raises(InvalidPaceError):
        pace_to_speed_mph(-5.0, "min/km")


def test_invalid_unit():
    with pytest.raises(InvalidUnitError):
        pace_to_speed_mph(8.0, "min/yd")


def test_invalid_input_nan():
    with pytest.raises(InvalidInputError):
        pace_to_speed_mph(float("nan"), "min/mile")


def test_invalid_input_inf():
    with pytest.raises(InvalidInputError):
        pace_to_speed_mph(float("inf"), "min/mile")


# -----------------------------------------------------------------------------
# Output structure
# -----------------------------------------------------------------------------


def test_output_has_required_fields():
    out = pace_to_step_bpm(8.0, "min/mile")
    assert "speed_mph" in out
    assert "step_bpm_raw" in out
    assert "step_bpm_final" in out
    assert "model_info" in out
    assert out["model_info"]["model_name"] == "linear_v1"
    assert "constants_used" in out["model_info"]
    assert "debug" in out
    assert out["debug"]["pace_value"] == 8.0
    assert out["debug"]["pace_unit"] == "min/mile"
