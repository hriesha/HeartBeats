"""
Vibe configuration: maps Deezer genres to HeartBeats vibes.

Each vibe has a display name, Deezer genre_id, search keywords,
color, and tags for the frontend.
"""

from typing import List, Dict, Any

# Deezer genre IDs
GENRE_POP = 132
GENRE_HIPHOP = 116
GENRE_DANCE = 113
GENRE_ROCK = 152
GENRE_ELECTRO = 106
GENRE_RNB = 165
GENRE_REGGAETON = 122

VIBE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "vibe_id": 0,
        "name": "Pop Hits",
        "genre_id": GENRE_POP,
        "search_keywords": ["pop hits", "top pop", "pop workout"],
        "color": "#FF2D55",
        "tags": ["upbeat", "familiar", "feel-good"],
    },
    {
        "vibe_id": 1,
        "name": "Hip-Hop Flow",
        "genre_id": GENRE_HIPHOP,
        "search_keywords": ["hip hop", "rap workout", "hip hop beats"],
        "color": "#FCBF49",
        "tags": ["bass-heavy", "rhythmic", "confident"],
    },
    {
        "vibe_id": 2,
        "name": "Electronic Energy",
        "genre_id": GENRE_ELECTRO,
        "search_keywords": ["electronic", "edm workout", "electro dance"],
        "color": "#64D2FF",
        "tags": ["high-energy", "driving", "euphoric"],
    },
    {
        "vibe_id": 3,
        "name": "Rock Power",
        "genre_id": GENRE_ROCK,
        "search_keywords": ["rock", "rock workout", "power rock"],
        "color": "#D62828",
        "tags": ["powerful", "intense", "adrenaline"],
    },
    {
        "vibe_id": 4,
        "name": "Dance Floor",
        "genre_id": GENRE_DANCE,
        "search_keywords": ["dance", "dance workout", "dance hits"],
        "color": "#30D158",
        "tags": ["danceable", "fun", "groovy"],
    },
    {
        "vibe_id": 5,
        "name": "R&B Groove",
        "genre_id": GENRE_RNB,
        "search_keywords": ["rnb", "r&b", "r&b groove"],
        "color": "#BF5AF2",
        "tags": ["smooth", "soulful", "groove"],
    },
    {
        "vibe_id": 6,
        "name": "Latin Heat",
        "genre_id": GENRE_REGGAETON,
        "search_keywords": ["reggaeton", "latin workout", "latin pop"],
        "color": "#F77F00",
        "tags": ["latin", "hot", "rhythmic"],
    },
]


def get_vibes_for_bpm(target_bpm: float) -> List[Dict[str, Any]]:
    """Return up to 5 vibes prioritized by BPM fitness."""
    if target_bpm >= 160:
        order = [2, 4, 0, 3, 1]  # Electronic, Dance, Pop, Rock, Hip-Hop
    elif target_bpm >= 130:
        order = [0, 1, 4, 2, 3]  # Pop, Hip-Hop, Dance, Electronic, Rock
    elif target_bpm >= 100:
        order = [0, 5, 1, 4, 6]  # Pop, R&B, Hip-Hop, Dance, Latin
    else:
        order = [5, 0, 1, 6, 4]  # R&B, Pop, Hip-Hop, Latin, Dance

    vibes = []
    for idx in order:
        if idx < len(VIBE_DEFINITIONS):
            vibes.append(VIBE_DEFINITIONS[idx])
    return vibes[:5]
