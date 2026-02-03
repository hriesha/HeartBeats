"""
Pace → Step BPM (pace-only).

Estimates runner's step cadence (steps per minute) from pace (min/mile or min/km)
for tempo-matching music. No height, stride, accelerometer, or heart-rate data.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class InvalidPaceError(ValueError):
    """pace_value <= 0 or invalid."""


class InvalidUnitError(ValueError):
    """Unknown pace_unit."""


class InvalidInputError(ValueError):
    """Non-finite numbers (NaN/inf)."""


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

KM_PER_MILE = 1.609344

# linear_v1 cadence model (tuned slightly lower for more typical cadence range)
DEFAULT_BASE_SPM = 125.0
DEFAULT_SPM_PER_MPH = 5.5

DEFAULT_CLAMP_RANGE_SPM = (140.0, 200.0)
DEFAULT_SNAP_GRID_SPM = 1
DEFAULT_SMOOTHING_ALPHA = 0.2


# -----------------------------------------------------------------------------
# Unit conversion: pace → speed (mph)
# -----------------------------------------------------------------------------


def pace_to_speed_mph(
    pace_value: float,
    pace_unit: Literal["min/mile", "min/km"],
) -> float:
    """
    Convert pace (minutes per unit distance) to speed in miles per hour.

    - pace_unit "min/mile": speed_mph = 60 / pace_value
    - pace_unit "min/km": speed_kmh = 60 / pace_value, speed_mph = speed_kmh / 1.609344

    Raises InvalidPaceError if pace_value <= 0.
    Raises InvalidUnitError if pace_unit is unknown.
    Raises InvalidInputError if pace_value is non-finite.
    """
    if not math.isfinite(pace_value):
        raise InvalidInputError(f"pace_value must be finite, got {pace_value}")

    if pace_value <= 0:
        raise InvalidPaceError(f"pace_value must be positive, got {pace_value}")

    if pace_unit == "min/mile":
        return 60.0 / pace_value
    if pace_unit == "min/km":
        speed_kmh = 60.0 / pace_value
        return speed_kmh / KM_PER_MILE
    raise InvalidUnitError(f"Unknown pace_unit: {pace_unit!r}")


# -----------------------------------------------------------------------------
# Cadence model: linear_v1
# -----------------------------------------------------------------------------


def cadence_linear_v1(
    speed_mph: float,
    base_spm: float = DEFAULT_BASE_SPM,
    spm_per_mph: float = DEFAULT_SPM_PER_MPH,
) -> float:
    """
    step_bpm_raw = base_spm + spm_per_mph * speed_mph
    """
    if not math.isfinite(speed_mph):
        raise InvalidInputError(f"speed_mph must be finite, got {speed_mph}")
    return base_spm + spm_per_mph * speed_mph


# -----------------------------------------------------------------------------
# Post-processing: clamp → snap → smoothing (EMA)
# -----------------------------------------------------------------------------


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def snap_to_grid(value: float, grid: float) -> float:
    if grid <= 0:
        return value
    return round(value / grid) * grid


def apply_smoothing(
    current: float,
    previous: Optional[float],
    alpha: float,
) -> float:
    if previous is None:
        return current
    return alpha * current + (1.0 - alpha) * previous


# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------


def pace_to_step_bpm(
    pace_value: float,
    pace_unit: Literal["min/mile", "min/km"],
    *,
    cadence_model: Literal["linear_v1"] = "linear_v1",
    clamp_range_spm: tuple[float, float] = DEFAULT_CLAMP_RANGE_SPM,
    snap_grid_spm: Optional[int | float] = DEFAULT_SNAP_GRID_SPM,
    smoothing: Optional[dict[str, Any]] = None,
    previous_step_bpm_final: Optional[float] = None,
    # Extensibility: user overrides for linear_v1
    base_spm: Optional[float] = None,
    spm_per_mph: Optional[float] = None,
    include_debug: bool = True,
) -> dict[str, Any]:
    """
    Estimate step cadence (Step BPM) from pace only.

    Post-processing order: clamp → snap → smoothing (EMA).

    Returns structured dict with speed_mph, step_bpm_raw, step_bpm_final,
    model_info, and optional debug.
    """
    # Default smoothing enabled (EMA)
    if smoothing is None:
        smoothing = {"enabled": True, "alpha": DEFAULT_SMOOTHING_ALPHA}
    smoothing_enabled = smoothing.get("enabled", True)
    smoothing_alpha = float(smoothing.get("alpha", DEFAULT_SMOOTHING_ALPHA))

    # Validate inputs
    if not math.isfinite(pace_value):
        raise InvalidInputError(f"pace_value must be finite, got {pace_value}")

    low_clamp, high_clamp = clamp_range_spm
    if not (math.isfinite(low_clamp) and math.isfinite(high_clamp)):
        raise InvalidInputError("clamp_range_spm must be finite")

    # 1) Pace → speed (mph)
    speed_mph = pace_to_speed_mph(pace_value, pace_unit)

    # 2) Cadence model
    if cadence_model == "linear_v1":
        base = base_spm if base_spm is not None else DEFAULT_BASE_SPM
        rate = spm_per_mph if spm_per_mph is not None else DEFAULT_SPM_PER_MPH
        step_bpm_raw = cadence_linear_v1(speed_mph, base_spm=base, spm_per_mph=rate)
        constants_used = {"BASE_SPM": base, "SPM_PER_MPH": rate}
    else:
        raise InvalidInputError(f"Unknown cadence_model: {cadence_model!r}")

    # 3) Clamp
    step_bpm_clamped = clamp(step_bpm_raw, low_clamp, high_clamp)

    # 4) Snap
    if snap_grid_spm is not None:
        grid = float(snap_grid_spm)
        if not math.isfinite(grid) or grid <= 0:
            raise InvalidInputError(f"snap_grid_spm must be positive finite, got {snap_grid_spm}")
        step_bpm_snapped = snap_to_grid(step_bpm_clamped, grid)
    else:
        step_bpm_snapped = step_bpm_clamped

    # 5) Smoothing (EMA)
    if smoothing_enabled:
        step_bpm_final = apply_smoothing(
            step_bpm_snapped,
            previous_step_bpm_final,
            smoothing_alpha,
        )
    else:
        step_bpm_final = step_bpm_snapped

    # Build result
    result: dict[str, Any] = {
        "speed_mph": speed_mph,
        "step_bpm_raw": step_bpm_raw,
        "step_bpm_final": step_bpm_final,
        "model_info": {
            "model_name": cadence_model,
            "constants_used": constants_used,
        },
    }

    if include_debug:
        result["debug"] = {
            "pace_value": pace_value,
            "pace_unit": pace_unit,
            "clamp_range_spm": clamp_range_spm,
            "snap_grid_spm": snap_grid_spm,
            "smoothing_params": smoothing,
        }

    return result
