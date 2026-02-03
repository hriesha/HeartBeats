# recs — Pace → Step BPM (pace-only)

Estimate a runner’s step cadence (steps per minute), treated as **Step BPM**, using only pace (min/mile or min/km). No height, stride length, accelerometer, or heart-rate data. Intended for tempo-matching music playback.

## Usage

```python
from recs import pace_to_step_bpm, InvalidPaceError

# Basic: 8:00 min/mile → step BPM
out = pace_to_step_bpm(8.0, "min/mile")
# out["speed_mph"] == 7.5
# out["step_bpm_raw"] == 175.0  (130 + 6*7.5)
# out["step_bpm_final"] == 175.0 (after clamp/snap/smoothing)

# Min/km
out = pace_to_step_bpm(5.0, "min/km")

# With smoothing (e.g. live updates): pass previous final for EMA
out = pace_to_step_bpm(8.0, "min/mile", previous_step_bpm_final=170.0)

# Custom clamp, snap, or smoothing
out = pace_to_step_bpm(
    8.0, "min/mile",
    clamp_range_spm=(140.0, 200.0),
    snap_grid_spm=2,
    smoothing={"enabled": True, "alpha": 0.2},
)
```

## Inputs

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `pace_value` | float | required | Pace as minutes per unit (e.g. 8.0 for 8:00) |
| `pace_unit` | `"min/mile"` \| `"min/km"` | required | Unit of pace |
| `cadence_model` | str | `"linear_v1"` | Cadence model |
| `clamp_range_spm` | (float, float) | (140, 200) | Clamp step BPM to this range |
| `snap_grid_spm` | int/float or None | 1 | Round to nearest multiple (None = no snap) |
| `smoothing` | dict or None | EMA alpha=0.2 | `{"enabled": True, "alpha": 0.2}` |
| `previous_step_bpm_final` | float or None | None | For EMA smoothing on live updates |

## Output

- `speed_mph` — speed in miles per hour
- `step_bpm_raw` — cadence before clamp/snap/smoothing
- `step_bpm_final` — cadence after all post-processing
- `model_info` — `model_name`, `constants_used`
- `debug` (optional) — inputs and params

## Model: linear_v1

`step_bpm_raw = BASE_SPM + SPM_PER_MPH * speed_mph`
Defaults: `BASE_SPM = 130`, `SPM_PER_MPH = 6`. Override with `base_spm` and `spm_per_mph` for personalization.

## Errors

- `InvalidPaceError` — pace_value <= 0
- `InvalidUnitError` — unknown pace_unit
- `InvalidInputError` — non-finite numbers

## Tests

```bash
pytest recs/test_pace_to_step_bpm.py -v
```

---

## Training the baseline model (scaler + KMeans)

Train once on Anna's Archive (or fallback CSV); saves to `recs/model/` for inference (no DB at request time).

**From project root:**

```bash
# Default: 200k rows, 6 clusters, output recs/model/
python3 -m recs.train

# Larger sample (beta)
python3 -m recs.train --sample 500000 --clusters 6

# Custom DB path / output
python3 -m recs.train --sample 200000 --clusters 6 --db /path/to/spotify_clean_audio_features.sqlite3 --out-dir recs/model

# Random sample (slower on huge DBs)
python3 -m recs.train --sample 200000 --random-sample
```

**Output:** `recs/model/scaler.joblib`, `recs/model/centroids.npy`, `recs/model/config.json`, `recs/model/track_lookup.db`. The **track lookup** lets inference use **track_id only** (no audio features at request time).

---

## Inference: track_id → cluster + KNN (no audio features)

After training, you can classify a track and get KNN neighbors using **only track_id** (no Spotify audio-features call at inference).

```python
from recs import get_cluster_only, get_cluster_and_neighbors

# Classify into cluster
cluster = get_cluster_only("6zo3dEK35yf8aOTkQBe1I9")  # returns 0..n_clusters-1 or None

# Cluster + KNN neighbor track_ids (no audio features needed)
out = get_cluster_and_neighbors("6zo3dEK35yf8aOTkQBe1I9", top_k=10)
# out = (cluster_id, [neighbor_track_id, ...]) or None if track not in lookup
```

**Limitation:** Only tracks that were in the training corpus are in the lookup. For tracks not in the lookup, return `None` and fall back to fetching audio features (e.g. Spotify API) and assigning to nearest centroid + KNN in your runtime data.
