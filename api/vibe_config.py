"""
Vibe configuration: maps music genres to HeartBeats vibes.

Each vibe represents a real music genre with curated artists,
a Deezer genre_id for chart fetching, and BPM-relevant search keywords.
"""

from typing import List, Dict, Any

# Deezer genre IDs
GENRE_POP       = 132
GENRE_HIPHOP    = 116
GENRE_DANCE     = 113   # covers House, Dance
GENRE_ROCK      = 152
GENRE_ELECTRO   = 106   # covers EDM, Electronic
GENRE_RNB       = 165
GENRE_REGGAETON = 122

VIBE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "vibe_id": 0,
        "name": "Pop",
        "genre_id": GENRE_POP,
        "search_keywords": ["pop hits", "top pop", "pop workout", "pop bangers"],
        "color": "#FF2D55",
        "tags": ["catchy", "feel-good", "familiar"],
        "curated_artists": [
            "Taylor Swift", "The Weeknd", "Dua Lipa",
            "Harry Styles", "Ariana Grande", "Billie Eilish",
            "Charlie XCX", "Sabrina Carpenter", "Olivia Rodrigo",
        ],
    },
    {
        "vibe_id": 1,
        "name": "Hip-Hop",
        "genre_id": GENRE_HIPHOP,
        "search_keywords": ["hip hop", "rap hits", "rap workout", "hip hop beats"],
        "color": "#FCBF49",
        "tags": ["bass-heavy", "rhythmic", "confident"],
        "curated_artists": [
            "Drake", "Travis Scott", "Kendrick Lamar",
            "21 Savage", "Future", "J. Cole",
            "Tyler the Creator", "Metro Boomin", "Don Toliver",
        ],
    },
    {
        "vibe_id": 2,
        "name": "House",
        "genre_id": GENRE_DANCE,
        "search_keywords": ["house music", "deep house", "tech house", "house workout"],
        "color": "#64D2FF",
        "tags": ["driving", "hypnotic", "euphoric"],
        "curated_artists": [
            "Fred Again", "Kaytranada", "Dom Dolla",
            "Fisher", "Disclosure", "Chris Lake",
            "Four Tet", "Jamie xx", "Peggy Gou",
        ],
    },
    {
        "vibe_id": 3,
        "name": "Rock",
        "genre_id": GENRE_ROCK,
        "search_keywords": ["rock", "rock hits", "rock workout", "rock anthems"],
        "color": "#D62828",
        "tags": ["powerful", "intense", "adrenaline"],
        "curated_artists": [
            "Foo Fighters", "Imagine Dragons", "Arctic Monkeys",
            "Twenty One Pilots", "Linkin Park", "Queens of the Stone Age",
            "The Killers", "Muse", "Green Day",
        ],
    },
    {
        "vibe_id": 4,
        "name": "Electronic",
        "genre_id": GENRE_ELECTRO,
        "search_keywords": ["edm hits", "electronic dance", "edm workout", "electro dance"],
        "color": "#30D158",
        "tags": ["high-energy", "build-drop", "euphoric"],
        "curated_artists": [
            "Calvin Harris", "David Guetta", "Martin Garrix",
            "Zedd", "Kygo", "Marshmello",
            "Illenium", "Alesso", "Tiësto",
        ],
    },
    {
        "vibe_id": 5,
        "name": "R&B",
        "genre_id": GENRE_RNB,
        "search_keywords": ["rnb", "r&b hits", "r&b groove", "smooth r&b"],
        "color": "#BF5AF2",
        "tags": ["smooth", "soulful", "groove"],
        "curated_artists": [
            "SZA", "Frank Ocean", "Daniel Caesar",
            "Brent Faiyaz", "Summer Walker", "H.E.R.",
            "Giveon", "Khalid", "Bryson Tiller",
        ],
    },
    {
        "vibe_id": 6,
        "name": "Latin",
        "genre_id": GENRE_REGGAETON,
        "search_keywords": ["reggaeton", "latin hits", "latin workout", "latin pop"],
        "color": "#F77F00",
        "tags": ["latin", "hot", "rhythmic"],
        "curated_artists": [
            "Bad Bunny", "J Balvin", "Maluma",
            "Ozuna", "Karol G", "Daddy Yankee",
            "Rauw Alejandro", "Myke Towers", "Feid",
        ],
    },
]


def get_vibes_for_bpm(target_bpm: float) -> List[Dict[str, Any]]:
    """Return up to 5 vibes prioritised by how well they fit the target BPM."""
    if target_bpm >= 160:
        # Fast run — high energy genres first
        order = [2, 4, 0, 3, 1]   # House, Electronic, Pop, Rock, Hip-Hop
    elif target_bpm >= 130:
        # Medium-fast — pop/hip-hop range
        order = [0, 1, 4, 2, 3]   # Pop, Hip-Hop, Electronic, House, Rock
    elif target_bpm >= 100:
        # Moderate — versatile
        order = [0, 5, 1, 2, 6]   # Pop, R&B, Hip-Hop, House, Latin
    else:
        # Slow — groovy/laid-back
        order = [5, 0, 1, 6, 2]   # R&B, Pop, Hip-Hop, Latin, House

    vibes = []
    for idx in order:
        if idx < len(VIBE_DEFINITIONS):
            vibes.append(VIBE_DEFINITIONS[idx])
    return vibes[:5]
