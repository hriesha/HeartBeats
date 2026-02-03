"""
recs: music recommendation / pace → step BPM + trained cluster/KNN model.
"""

from recs.pace_to_step_bpm import (
    InvalidPaceError,
    InvalidUnitError,
    InvalidInputError,
    pace_to_speed_mph,
    pace_to_step_bpm,
    DEFAULT_BASE_SPM,
    DEFAULT_SPM_PER_MPH,
    DEFAULT_CLAMP_RANGE_SPM,
    DEFAULT_SNAP_GRID_SPM,
    DEFAULT_SMOOTHING_ALPHA,
)
from recs.inference import (
    get_cluster_only,
    get_cluster_and_neighbors,
    predict_cluster_from_features,
    predict_audio_features_from_metadata,
    get_cluster_for_track,
)

__all__ = [
    "InvalidPaceError",
    "InvalidUnitError",
    "InvalidInputError",
    "pace_to_speed_mph",
    "pace_to_step_bpm",
    "DEFAULT_BASE_SPM",
    "DEFAULT_SPM_PER_MPH",
    "DEFAULT_CLAMP_RANGE_SPM",
    "DEFAULT_SNAP_GRID_SPM",
    "DEFAULT_SMOOTHING_ALPHA",
    "get_cluster_only",
    "get_cluster_and_neighbors",
    "predict_cluster_from_features",
    "predict_audio_features_from_metadata",
    "get_cluster_for_track",
]
