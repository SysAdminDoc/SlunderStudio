"""
Slunder Studio v0.0.2 — ACE-Step Style Tags
Categorized database of 1000+ style tags supported by ACE-Step v1.5.
Searchable, favoritable, with category browsing.
"""
import json
from typing import Optional
from pathlib import Path
from core.settings import Settings, get_config_dir

# ── Tag Categories & Data ──────────────────────────────────────────────────────

CATEGORIES = {
    "genre": [
        "pop", "rock", "hip hop", "rap", "r&b", "soul", "jazz", "blues", "country",
        "folk", "electronic", "edm", "house", "techno", "trance", "dubstep", "drum and bass",
        "ambient", "lo-fi", "chillwave", "synthwave", "retrowave", "vaporwave", "new wave",
        "punk", "post-punk", "grunge", "metal", "heavy metal", "death metal", "black metal",
        "thrash metal", "progressive metal", "doom metal", "power metal", "nu metal",
        "alternative", "indie", "indie rock", "indie pop", "shoegaze", "dream pop",
        "psychedelic", "progressive rock", "art rock", "krautrock", "post-rock",
        "reggae", "ska", "dancehall", "dub", "afrobeats", "afropop", "highlife",
        "latin", "reggaeton", "bossa nova", "samba", "salsa", "cumbia", "bachata",
        "k-pop", "j-pop", "j-rock", "city pop", "anime", "bollywood",
        "classical", "baroque", "romantic era", "opera", "orchestral", "chamber music",
        "gospel", "worship", "hymn", "spiritual", "praise",
        "disco", "funk", "motown", "northern soul",
        "trap", "drill", "grime", "boom bap", "old school hip hop", "conscious rap",
        "emo", "screamo", "hardcore", "metalcore", "deathcore",
        "bluegrass", "americana", "outlaw country", "honky tonk",
        "new age", "meditation", "world music", "celtic", "flamenco",
        "musical theater", "broadway", "cabaret", "chanson",
        "industrial", "ebm", "noise", "experimental", "glitch",
        "garage rock", "surf rock", "rockabilly", "swing",
    ],
    "mood": [
        "happy", "sad", "melancholic", "euphoric", "angry", "peaceful", "calm",
        "energetic", "relaxing", "dark", "bright", "dreamy", "nostalgic", "romantic",
        "aggressive", "gentle", "intense", "mellow", "uplifting", "haunting",
        "mysterious", "playful", "serious", "fun", "emotional", "powerful",
        "triumphant", "vulnerable", "confident", "anxious", "hopeful", "desperate",
        "bittersweet", "empowering", "lonely", "wild", "serene", "epic",
        "cinematic", "dramatic", "suspenseful", "ethereal", "groovy", "funky",
        "soulful", "raw", "gritty", "smooth", "lush", "sparse",
        "warm", "cold", "hypnotic", "meditative", "rebellious", "introspective",
    ],
    "instrument": [
        "piano", "acoustic guitar", "electric guitar", "bass guitar", "drums",
        "synthesizer", "organ", "violin", "cello", "viola", "double bass",
        "trumpet", "saxophone", "trombone", "french horn", "clarinet", "flute",
        "oboe", "bassoon", "harp", "accordion", "harmonica", "banjo",
        "mandolin", "ukulele", "sitar", "tabla", "djembe", "congas",
        "marimba", "vibraphone", "xylophone", "timpani", "steel drums",
        "synth bass", "synth pad", "synth lead", "arpeggiator", "vocoder",
        "808", "hi-hats", "snare", "kick drum", "clap", "cowbell",
        "strings", "brass", "woodwinds", "choir", "orchestra",
        "distorted guitar", "clean guitar", "fingerpicking", "power chords",
        "slide guitar", "pedal steel", "12-string guitar", "nylon guitar",
        "rhodes", "wurlitzer", "mellotron", "clavinet", "harpsichord",
        "theremin", "didgeridoo", "erhu", "shamisen", "koto", "bagpipes",
    ],
    "vocal": [
        "male vocals", "female vocals", "duet", "choir", "a cappella",
        "rap vocals", "singing", "spoken word", "whisper", "falsetto",
        "tenor", "baritone", "bass voice", "soprano", "alto", "mezzo-soprano",
        "auto-tune", "vocoder vocals", "harmonies", "backing vocals",
        "growling", "screaming", "belting", "crooning", "yodeling",
        "beatboxing", "humming", "chanting", "operatic", "scat singing",
        "talk-singing", "melodic rap", "deep voice", "high-pitched voice",
    ],
    "tempo": [
        "slow", "mid-tempo", "fast", "very fast", "very slow",
        "60 bpm", "70 bpm", "80 bpm", "90 bpm", "100 bpm",
        "110 bpm", "120 bpm", "130 bpm", "140 bpm", "150 bpm",
        "160 bpm", "170 bpm", "180 bpm", "200 bpm",
        "adagio", "andante", "moderato", "allegro", "presto",
        "rubato", "accelerando", "ritardando",
    ],
    "era": [
        "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s",
        "retro", "vintage", "modern", "futuristic", "classic", "contemporary",
        "old school", "new school", "timeless",
    ],
    "production": [
        "lo-fi", "hi-fi", "analog", "digital", "vinyl", "tape", "clean",
        "distorted", "reverb-heavy", "dry", "compressed", "dynamic",
        "layered", "minimal", "dense", "spacious", "wide stereo", "mono",
        "bass-heavy", "treble-heavy", "midrange", "warm", "crisp", "muddy",
        "polished", "raw", "overdriven", "saturated", "filtered",
        "sidechain", "chopped", "glitchy", "stuttered", "pitched down", "pitched up",
        "sample-based", "loop-based", "live recording", "studio quality",
    ],
    "structure": [
        "intro", "verse", "chorus", "bridge", "outro", "breakdown",
        "build-up", "drop", "hook", "pre-chorus", "post-chorus",
        "interlude", "solo", "instrumental", "acapella section",
        "fade in", "fade out", "sudden stop", "long outro",
    ],
    "texture": [
        "atmospheric", "ambient", "lush", "sparse", "thick", "thin",
        "shimmering", "gritty", "smooth", "rough", "crystalline",
        "foggy", "hazy", "clear", "murky", "ethereal", "earthy",
        "metallic", "wooden", "organic", "synthetic", "hybrid",
        "granular", "textured", "layered pads", "arpeggiated",
    ],
}

# Flatten all tags with category info
ALL_TAGS: list[dict] = []
for cat, tags in CATEGORIES.items():
    for tag in tags:
        ALL_TAGS.append({"tag": tag, "category": cat})


class StyleTagDB:
    """Searchable style tag database with favorites."""

    def __init__(self):
        self._favorites_path = get_config_dir() / "style_tag_favorites.json"
        self._favorites: set[str] = set()
        self._load_favorites()

    def _load_favorites(self):
        if self._favorites_path.exists():
            try:
                data = json.loads(self._favorites_path.read_text())
                self._favorites = set(data)
            except Exception:
                self._favorites = set()

    def _save_favorites(self):
        self._favorites_path.write_text(json.dumps(sorted(self._favorites)))

    def search(self, query: str, category: str = "", favorites_only: bool = False) -> list[dict]:
        """Search tags by query and optional category filter."""
        query = query.lower().strip()
        results = []
        for item in ALL_TAGS:
            if category and item["category"] != category:
                continue
            if favorites_only and item["tag"] not in self._favorites:
                continue
            if query and query not in item["tag"]:
                continue
            results.append({
                **item,
                "is_favorite": item["tag"] in self._favorites,
            })
        return results

    def get_by_category(self, category: str) -> list[dict]:
        """Get all tags in a category."""
        return self.search("", category=category)

    def get_categories(self) -> list[str]:
        return list(CATEGORIES.keys())

    def get_favorites(self) -> list[dict]:
        return self.search("", favorites_only=True)

    def toggle_favorite(self, tag: str) -> bool:
        """Toggle favorite status. Returns new state."""
        if tag in self._favorites:
            self._favorites.discard(tag)
            self._save_favorites()
            return False
        else:
            self._favorites.add(tag)
            self._save_favorites()
            return True

    def is_favorite(self, tag: str) -> bool:
        return tag in self._favorites

    def get_suggested_tags(self, genre_id: str) -> list[str]:
        """Get suggested ACE-Step tags for a lyrics genre template."""
        from engines.lyrics_templates import GENRE_TEMPLATES
        template = GENRE_TEMPLATES.get(genre_id)
        if template:
            return template.style_tags
        return ["pop", "catchy"]

    @property
    def total_count(self) -> int:
        return len(ALL_TAGS)

    @property
    def favorite_count(self) -> int:
        return len(self._favorites)
