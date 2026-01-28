#!/usr/bin/env python3
"""
Dynamic Cluster Naming System

Generates personalized cluster names and tags based on:
- Cluster audio feature statistics
- User's library characteristics
- BPM context
- Feature weighting patterns
"""

import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

log = logging.getLogger("cluster_naming")

# Cluster name templates based on feature combinations
CLUSTER_NAME_TEMPLATES = {
    # High tempo + High energy combinations
    (True, True, True, False): ["Power Up", "Energy Rush", "Intense Beats", "Turbo Charge"],
    (True, True, False, True): ["Energetic Flow", "Upbeat Vibes", "Active Pulse", "Dynamic Drive"],
    (True, True, False, False): ["Steady Burn", "Consistent Power", "Sustained Energy"],

    # High tempo + Low energy (rare but interesting)
    (True, False, True, False): ["Focused Tempo", "Precision Beats", "Calm Rhythm"],
    (True, False, False, True): ["Gentle Pace", "Easy Flow", "Relaxed Tempo"],

    # Low tempo + High energy (also interesting)
    (False, True, True, False): ["Deep Intensity", "Powerful Chill", "Strong Calm"],
    (False, True, False, True): ["Warm Energy", "Cozy Power", "Comfortable Drive"],

    # Low tempo + Low energy
    (False, False, True, False): ["Chill Flow", "Peaceful Vibes", "Calm Waves", "Serene Space"],
    (False, False, False, True): ["Ambient Mood", "Soft Atmosphere", "Gentle Drift"],
}


def analyze_cluster_characteristics(
    cluster_df: pd.DataFrame,
    overall_stats: Optional[Dict] = None
) -> Dict[str, any]:
    """
    Analyze cluster characteristics to understand its musical profile.

    Returns dict with:
    - tempo_profile: "slow", "moderate", "fast"
    - energy_profile: "low", "moderate", "high"
    - mood_profile: "calm", "balanced", "intense"
    - danceability_profile: "low", "moderate", "high"
    - distinctiveness: what makes this cluster unique
    """
    stats = {}

    # Tempo analysis
    mean_tempo = cluster_df["tempo"].mean()
    if mean_tempo < 90:
        stats["tempo_profile"] = "slow"
        stats["tempo_level"] = 1
    elif mean_tempo < 130:
        stats["tempo_profile"] = "moderate"
        stats["tempo_level"] = 2
    else:
        stats["tempo_profile"] = "fast"
        stats["tempo_level"] = 3
    stats["mean_tempo"] = mean_tempo

    # Energy analysis
    mean_energy = cluster_df["energy"].mean()
    if mean_energy < 0.4:
        stats["energy_profile"] = "low"
        stats["energy_level"] = 1
    elif mean_energy < 0.6:
        stats["energy_profile"] = "moderate"
        stats["energy_level"] = 2
    else:
        stats["energy_profile"] = "high"
        stats["energy_level"] = 3
    stats["mean_energy"] = mean_energy

    # Danceability analysis
    mean_danceability = cluster_df["danceability"].mean()
    if mean_danceability < 0.5:
        stats["danceability_profile"] = "low"
        stats["danceability_level"] = 1
    elif mean_danceability < 0.7:
        stats["danceability_profile"] = "moderate"
        stats["danceability_level"] = 2
    else:
        stats["danceability_profile"] = "high"
        stats["danceability_level"] = 3
    stats["mean_danceability"] = mean_danceability

    # Valence (mood/positivity)
    mean_valence = cluster_df["valence"].mean()
    if mean_valence < 0.4:
        stats["mood_profile"] = "melancholic"
        stats["mood_level"] = 1
    elif mean_valence < 0.6:
        stats["mood_profile"] = "balanced"
        stats["mood_level"] = 2
    else:
        stats["mood_profile"] = "uplifting"
        stats["mood_level"] = 3
    stats["mean_valence"] = mean_valence

    # Loudness analysis
    mean_loudness = cluster_df["loudness"].mean()
    stats["mean_loudness"] = mean_loudness
    stats["is_loud"] = mean_loudness > -10

    # Acousticness (if available)
    if "acousticness" in cluster_df.columns:
        mean_acousticness = cluster_df["acousticness"].mean()
        stats["mean_acousticness"] = mean_acousticness
        stats["is_acoustic"] = mean_acousticness > 0.5

    # Instrumentalness (if available)
    if "instrumentalness" in cluster_df.columns:
        mean_instrumentalness = cluster_df["instrumentalness"].mean()
        stats["mean_instrumentalness"] = mean_instrumentalness
        stats["is_instrumental"] = mean_instrumentalness > 0.5

    # Distinctiveness: compare to overall stats if available
    if overall_stats:
        stats["distinctiveness"] = {
            "tempo_delta": abs(mean_tempo - overall_stats.get("mean_tempo", 120)),
            "energy_delta": abs(mean_energy - overall_stats.get("mean_energy", 0.5)),
            "unique_features": []
        }

        # Identify what makes this cluster unique
        d = stats["distinctiveness"]
        if d["tempo_delta"] > 20:
            d["unique_features"].append("tempo")
        if d["energy_delta"] > 0.2:
            d["unique_features"].append("energy")

    return stats


def generate_cluster_name(
    cluster_stats: Dict,
    cluster_index: int,
    user_library_size: int
) -> Tuple[str, List[str]]:
    """
    Generate a personalized cluster name and tags based on characteristics.

    Returns:
        (name, tags)
    """
    tempo_level = cluster_stats["tempo_level"]
    energy_level = cluster_stats["energy_level"]
    danceability_level = cluster_stats["danceability_level"]
    mood_level = cluster_stats["mood_level"]

    # Key boolean flags
    is_fast = tempo_level >= 3
    is_high_energy = energy_level >= 3
    is_danceable = danceability_level >= 2
    is_positive = mood_level >= 2

    # Generate name based on combinations
    key = (
        is_fast,
        is_high_energy,
        is_danceable,
        is_positive
    )

    # Get template names for this combination
    template_names = CLUSTER_NAME_TEMPLATES.get(key, ["Vibe Cluster"])

    # Select name based on cluster index (for variety)
    name = template_names[cluster_index % len(template_names)]

    # Refine name based on specific characteristics
    name = refine_name(name, cluster_stats)

    # Generate tags
    tags = generate_tags(cluster_stats, user_library_size)

    return name, tags


def refine_name(base_name: str, stats: Dict) -> str:
    """Refine cluster name based on specific characteristics."""
    name = base_name

    # Add tempo qualifier if extreme
    if stats["mean_tempo"] < 80:
        name = f"Ultra Slow {name}" if "Slow" not in name else name
    elif stats["mean_tempo"] > 160:
        name = f"Hyper {name}" if "Fast" not in name else name

    # Add energy qualifier
    if stats["mean_energy"] > 0.8:
        name = f"Intense {name}" if "Intense" not in name and "Power" not in name else name

    # Add acoustic/instrumental qualifier
    if stats.get("is_acoustic", False):
        name = f"Acoustic {name}"
    elif stats.get("is_instrumental", False):
        name = f"Instrumental {name}"

    # Add mood qualifier for extreme cases
    if stats["mean_valence"] < 0.3:
        name = f"Deep {name}"
    elif stats["mean_valence"] > 0.8:
        name = f"Joyful {name}"

    return name


def generate_tags(stats: Dict, user_library_size: int) -> List[str]:
    """Generate descriptive tags for the cluster."""
    tags = []

    # Tempo-based tags
    if stats["mean_tempo"] < 90:
        tags.append("slow-burn")
        tags.append("chill")
    elif stats["mean_tempo"] > 140:
        tags.append("high-energy")
        tags.append("intense")
    else:
        tags.append("steady")

    # Energy-based tags
    if stats["mean_energy"] > 0.7:
        tags.append("powerful")
    elif stats["mean_energy"] < 0.4:
        tags.append("calm")

    # Mood-based tags
    if stats["mean_valence"] > 0.6:
        tags.append("uplifting")
    elif stats["mean_valence"] < 0.4:
        tags.append("introspective")

    # Danceability tags
    if stats["mean_danceability"] > 0.7:
        tags.append("danceable")

    # Special characteristics
    if stats.get("is_acoustic", False):
        tags.append("acoustic")
    if stats.get("is_instrumental", False):
        tags.append("instrumental")

    # Remove duplicates and limit to 4-5 tags
    tags = list(dict.fromkeys(tags))[:5]

    return tags


def generate_cluster_names(
    clustered_df: pd.DataFrame,
    user_library_size: Optional[int] = None
) -> List[Dict[str, any]]:
    """
    Generate names and metadata for all clusters in a clustered dataset.

    Returns list of dicts with:
    - cluster_id
    - name
    - tags
    - color (suggested)
    - mean_tempo
    - track_count
    - characteristics (stats dict)
    """
    if user_library_size is None:
        user_library_size = len(clustered_df)

    # Calculate overall library statistics for comparison
    overall_stats = {
        "mean_tempo": clustered_df["tempo"].mean(),
        "mean_energy": clustered_df["energy"].mean(),
        "mean_danceability": clustered_df["danceability"].mean(),
        "mean_valence": clustered_df["valence"].mean(),
    }

    cluster_info = []
    unique_clusters = sorted(clustered_df["cluster"].unique())

    # Color palette that works well
    color_palette = [
        "#EAE2B7",  # Soft yellow
        "#FCBF49",  # Gold
        "#F77F00",  # Orange
        "#D62828",  # Red
        "#003049",  # Dark blue
        "#9B59B6",  # Purple
        "#2ECC71",  # Green
        "#3498DB",  # Blue
    ]

    for i, cluster_id in enumerate(unique_clusters):
        cluster_df = clustered_df[clustered_df["cluster"] == cluster_id]

        # Analyze cluster characteristics
        stats = analyze_cluster_characteristics(cluster_df, overall_stats)

        # Generate name and tags
        name, tags = generate_cluster_name(stats, i, user_library_size)

        cluster_info.append({
            "id": int(cluster_id),
            "name": name,
            "tags": tags,
            "color": color_palette[i % len(color_palette)],
            "mean_tempo": float(stats["mean_tempo"]),
            "mean_energy": float(stats["mean_energy"]),
            "mean_danceability": float(stats["mean_danceability"]),
            "mean_valence": float(stats["mean_valence"]),
            "track_count": len(cluster_df),
            "characteristics": stats,
        })

        log.info(f"Generated cluster {cluster_id}: '{name}' with tags {tags}")

    return cluster_info


# Future: LLM-powered naming
def generate_cluster_name_with_llm(
    cluster_stats: Dict,
    sample_tracks: List[Dict],
    user_library_context: Dict
) -> Tuple[str, List[str], str]:
    """
    Future: Use LLM to generate personalized cluster names based on:
    - Cluster audio features
    - Sample track names/artists from cluster
    - User's overall library characteristics

    This would provide more contextual, creative names.
    """
    # TODO: Implement LLM integration
    # - OpenAI API
    # - Anthropic Claude
    # - Or local model

    pass
