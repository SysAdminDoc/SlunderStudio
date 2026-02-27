"""
Slunder Studio v0.0.2 — Lyrics Templates & Prompt Engineering
30+ genre-specific prompt templates with structure tags matching ACE-Step v1.5 input format.
Two-stage prompting: plan → generate.
"""
import json
import random
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from core.settings import Settings


# ── Structure Tags (ACE-Step compatible) ───────────────────────────────────────

STRUCTURE_TAGS = [
    "[Intro]", "[Verse 1]", "[Verse 2]", "[Verse 3]",
    "[Pre-Chorus]", "[Chorus]", "[Post-Chorus]",
    "[Bridge]", "[Outro]", "[Hook]", "[Breakdown]",
    "[Instrumental]", "[Interlude]", "[Spoken]", "[Ad-lib]",
    "[Refrain]", "[Drop]",
]

STANDARD_STRUCTURES = {
    "pop": "[Verse 1]\n[Pre-Chorus]\n[Chorus]\n[Verse 2]\n[Pre-Chorus]\n[Chorus]\n[Bridge]\n[Chorus]\n[Outro]",
    "verse_chorus": "[Verse 1]\n[Chorus]\n[Verse 2]\n[Chorus]\n[Bridge]\n[Chorus]",
    "verse_hook": "[Intro]\n[Verse 1]\n[Hook]\n[Verse 2]\n[Hook]\n[Verse 3]\n[Hook]\n[Outro]",
    "hip_hop": "[Intro]\n[Verse 1]\n[Chorus]\n[Verse 2]\n[Chorus]\n[Verse 3]\n[Chorus]\n[Outro]",
    "ballad": "[Verse 1]\n[Verse 2]\n[Chorus]\n[Verse 3]\n[Chorus]\n[Bridge]\n[Chorus]\n[Outro]",
    "edm": "[Intro]\n[Verse 1]\n[Chorus]\n[Drop]\n[Verse 2]\n[Chorus]\n[Drop]\n[Breakdown]\n[Drop]\n[Outro]",
    "aab": "[Verse 1]\n[Verse 2]\n[Chorus]\n[Verse 3]\n[Verse 4]\n[Chorus]",
    "minimal": "[Verse 1]\n[Chorus]\n[Verse 2]\n[Chorus]",
    "extended": "[Intro]\n[Verse 1]\n[Pre-Chorus]\n[Chorus]\n[Post-Chorus]\n[Verse 2]\n[Pre-Chorus]\n[Chorus]\n[Post-Chorus]\n[Bridge]\n[Chorus]\n[Chorus]\n[Outro]",
}

# ── Moods ──────────────────────────────────────────────────────────────────────

MOODS = [
    "happy", "sad", "melancholic", "euphoric", "angry", "peaceful",
    "nostalgic", "romantic", "energetic", "dark", "dreamy", "hopeful",
    "rebellious", "introspective", "playful", "haunting", "triumphant",
    "vulnerable", "confident", "mysterious", "bittersweet", "uplifting",
    "aggressive", "serene", "anxious", "empowering", "lonely", "wild",
]


# ── Genre Templates ────────────────────────────────────────────────────────────

@dataclass
class GenreTemplate:
    """Complete prompt template for a specific genre."""
    id: str
    name: str
    description: str
    category: str  # pop, rock, electronic, hip-hop, etc.
    structure: str  # Default structure key from STANDARD_STRUCTURES
    rhyme_scheme: str  # e.g., "ABAB", "AABB", "ABCB", "free"
    syllable_density: str  # "low", "medium", "high"
    vocabulary_style: str  # description of typical word choices
    emotional_register: str  # typical emotional range
    example_themes: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)  # ACE-Step style tags
    system_prompt_extra: str = ""  # Additional system prompt instructions


GENRE_TEMPLATES: dict[str, GenreTemplate] = {
    "pop": GenreTemplate(
        id="pop", name="Pop", description="Catchy, radio-friendly with strong hooks",
        category="pop", structure="pop", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Accessible, everyday language with memorable phrases. Short, punchy lines.",
        emotional_register="Upbeat to bittersweet. Relatable emotions, universal themes.",
        example_themes=["love", "heartbreak", "dancing", "self-empowerment", "summer nights"],
        style_tags=["pop", "catchy", "radio"],
        system_prompt_extra="Focus on a singable, memorable chorus hook. Keep language simple and universal.",
    ),
    "rock": GenreTemplate(
        id="rock", name="Rock", description="Raw energy, guitar-driven anthems",
        category="rock", structure="verse_chorus", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Visceral, physical imagery. Strong verbs. Occasional metaphor.",
        emotional_register="Intense, passionate. From quiet verses to explosive choruses.",
        example_themes=["rebellion", "freedom", "the road", "heartbreak", "fighting back"],
        style_tags=["rock", "electric guitar", "drums", "powerful"],
    ),
    "hip_hop": GenreTemplate(
        id="hip_hop", name="Hip-Hop", description="Rhythmic flow with wordplay and storytelling",
        category="hip-hop", structure="hip_hop", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Dense, rhythmic wordplay. Internal rhymes, multisyllabic rhymes, slang. Vivid storytelling.",
        emotional_register="Confident, street-smart, sometimes vulnerable. Braggadocio to introspection.",
        example_themes=["hustle", "success", "streets", "legacy", "real talk"],
        style_tags=["hip hop", "rap", "trap beat"],
        system_prompt_extra="Pack lines with internal rhymes and wordplay. Maintain rhythmic flow. Use vivid imagery.",
    ),
    "rnb": GenreTemplate(
        id="rnb", name="R&B", description="Smooth, soulful with vocal runs and emotional depth",
        category="rnb", structure="pop", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Smooth, sensual language. Vocal embellishments implied. Emotional vulnerability.",
        emotional_register="Intimate, romantic, sultry. Deep feeling delivered smoothly.",
        example_themes=["desire", "late nights", "devotion", "heartache", "intimacy"],
        style_tags=["r&b", "soul", "smooth", "groovy"],
    ),
    "country": GenreTemplate(
        id="country", name="Country", description="Storytelling with heart, twang, and imagery",
        category="country", structure="verse_chorus", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Conversational, narrative. Rural imagery, pickup trucks, small towns, whiskey. Down-to-earth.",
        emotional_register="Heartfelt, earnest. From rowdy to tearjerker.",
        example_themes=["home", "lost love", "dirt roads", "family", "friday nights"],
        style_tags=["country", "acoustic guitar", "steel guitar"],
    ),
    "edm": GenreTemplate(
        id="edm", name="EDM / Electronic", description="Repetitive hooks for drops and builds",
        category="electronic", structure="edm", rhyme_scheme="AABB",
        syllable_density="low",
        vocabulary_style="Short, punchy phrases. Repetitive hooks. Focus on rhythm over narrative.",
        emotional_register="Euphoric, hypnotic, high-energy. Build-and-release dynamics.",
        example_themes=["the night", "feeling alive", "lights", "losing yourself", "together"],
        style_tags=["edm", "electronic", "synth", "dance"],
        system_prompt_extra="Keep chorus extremely short and repetitive — designed for electronic drops. Verses are sparse.",
    ),
    "jazz": GenreTemplate(
        id="jazz", name="Jazz", description="Sophisticated, poetic with complex imagery",
        category="jazz", structure="aab", rhyme_scheme="AABB",
        syllable_density="medium",
        vocabulary_style="Sophisticated, poetic. Urban imagery, nightlife, smoke and mirrors. Metaphor-rich.",
        emotional_register="Cool, world-weary, romantic. Understated emotion.",
        example_themes=["midnight", "smoke-filled rooms", "love gone wrong", "city lights", "solitude"],
        style_tags=["jazz", "piano", "saxophone", "smooth"],
    ),
    "blues": GenreTemplate(
        id="blues", name="Blues", description="AAB form with gritty, soulful lament",
        category="blues", structure="aab", rhyme_scheme="AAB",
        syllable_density="low",
        vocabulary_style="Gritty, earthy. Repetition is key (AAB form). Delta imagery. Hardship and humor.",
        emotional_register="Lamenting but resilient. Pain expressed with groove.",
        example_themes=["hard times", "woman done left", "crossroads", "rain", "whiskey"],
        style_tags=["blues", "electric guitar", "harmonica"],
        system_prompt_extra="Use AAB blues form: first line stated, repeated with variation, then resolved. Keep it raw.",
    ),
    "metal": GenreTemplate(
        id="metal", name="Metal", description="Aggressive, powerful with dark imagery",
        category="rock", structure="verse_chorus", rhyme_scheme="ABAB",
        syllable_density="high",
        vocabulary_style="Dark, aggressive, epic. War imagery, fire, destruction, mythology. Power words.",
        emotional_register="Furious, epic, cathartic. From controlled anger to unleashed chaos.",
        example_themes=["war", "destruction", "inner demons", "rising from ashes", "chaos"],
        style_tags=["metal", "heavy", "distorted guitar", "double bass drums"],
    ),
    "folk": GenreTemplate(
        id="folk", name="Folk", description="Storytelling with acoustic warmth and imagery",
        category="folk", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Narrative, pastoral. Nature imagery, journeys, seasons. Traditional but fresh.",
        emotional_register="Warm, wistful, contemplative. Stories told by the fire.",
        example_themes=["the road", "rivers", "old love", "changing seasons", "homecoming"],
        style_tags=["folk", "acoustic", "guitar", "banjo"],
    ),
    "indie": GenreTemplate(
        id="indie", name="Indie / Alternative", description="Artistic, introspective with unexpected imagery",
        category="indie", structure="verse_chorus", rhyme_scheme="free",
        syllable_density="medium",
        vocabulary_style="Artistic, offbeat imagery. Unexpected metaphors. Self-aware, sometimes ironic.",
        emotional_register="Introspective, melancholic with hope. Layered emotions.",
        example_themes=["growing up", "existential doubt", "quiet moments", "suburban ennui", "weird love"],
        style_tags=["indie", "alternative", "lo-fi guitar"],
    ),
    "latin": GenreTemplate(
        id="latin", name="Latin Pop / Reggaeton", description="Rhythmic, passionate with Latin flavor",
        category="latin", structure="pop", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Rhythmic, passionate. Mix of romance and party. Can be bilingual. Danceable.",
        emotional_register="Fiery, sensual, celebratory. Heat and passion.",
        example_themes=["dancing", "passion", "the night", "mi amor", "fiesta"],
        style_tags=["reggaeton", "latin pop", "tropical"],
    ),
    "kpop": GenreTemplate(
        id="kpop", name="K-Pop", description="High-energy with catchy hooks and breakdowns",
        category="pop", structure="extended", rhyme_scheme="ABAB",
        syllable_density="high",
        vocabulary_style="High-energy, trendy. Mix of sweet and fierce. Catchy English hooks in chorus.",
        emotional_register="Energetic, confident, sweet-to-fierce transitions.",
        example_themes=["love at first sight", "confidence", "dancing all night", "fighting for dreams"],
        style_tags=["k-pop", "synth", "dance pop", "energetic"],
    ),
    "gospel": GenreTemplate(
        id="gospel", name="Gospel", description="Uplifting spiritual with call-and-response",
        category="gospel", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Spiritual, uplifting. Biblical imagery, testimony, praise. Call-and-response.",
        emotional_register="Joyful, grateful, transcendent. From struggle to glory.",
        example_themes=["grace", "salvation", "praise", "overcoming", "faith"],
        style_tags=["gospel", "choir", "piano", "uplifting"],
    ),
    "punk": GenreTemplate(
        id="punk", name="Punk", description="Fast, aggressive, anti-establishment",
        category="rock", structure="minimal", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Direct, aggressive, political. Short sharp lines. Anti-authority. Sarcasm.",
        emotional_register="Angry, defiant, sarcastic. Raw and unfiltered.",
        example_themes=["the system", "boredom", "riot", "sell-outs", "suburban hell"],
        style_tags=["punk", "fast", "distorted", "aggressive"],
        system_prompt_extra="Keep it short, fast, angry. No filler. Every line should hit hard.",
    ),
    "synthwave": GenreTemplate(
        id="synthwave", name="Synthwave / Retrowave", description="80s-inspired neon-soaked nostalgia",
        category="electronic", structure="pop", rhyme_scheme="ABAB",
        syllable_density="low",
        vocabulary_style="Retro-futuristic. Neon lights, chrome, night drives, VHS static. 80s nostalgia.",
        emotional_register="Nostalgic, dreamy, melancholic with hope. Neon-lit longing.",
        example_themes=["night drive", "neon city", "lost signal", "retrograde", "electric dreams"],
        style_tags=["synthwave", "retro", "80s synth", "electronic"],
    ),
    "lofi": GenreTemplate(
        id="lofi", name="Lo-Fi / Chill", description="Laid-back, atmospheric with minimal lyrics",
        category="electronic", structure="minimal", rhyme_scheme="free",
        syllable_density="low",
        vocabulary_style="Sparse, atmospheric. Stream of consciousness. Gentle, murmured quality.",
        emotional_register="Calm, reflective, slightly melancholic. Rainy window vibes.",
        example_themes=["rainy days", "coffee", "memories", "quiet rooms", "drifting off"],
        style_tags=["lo-fi", "chill", "jazz hop", "mellow"],
        system_prompt_extra="Very sparse lyrics. Short phrases, lots of space. Impressionistic rather than narrative.",
    ),
    "drill": GenreTemplate(
        id="drill", name="Drill", description="Dark, aggressive with sliding bass and hard-hitting bars",
        category="hip-hop", structure="hip_hop", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Hard, street-level. Slang-heavy, aggressive posturing. Dark humor.",
        emotional_register="Cold, menacing, unflinching. Controlled aggression.",
        example_themes=["the block", "opps", "no lacking", "come up", "staying solid"],
        style_tags=["drill", "dark trap", "808", "aggressive"],
    ),
    "afrobeats": GenreTemplate(
        id="afrobeats", name="Afrobeats", description="Rhythmic, danceable with African influence",
        category="world", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Rhythmic, celebratory. Pidgin English phrases welcome. Dance-oriented.",
        emotional_register="Joyful, celebratory, romantic. Sunshine and dance.",
        example_themes=["dancing", "beautiful woman", "celebration", "afro rhythm", "vibe"],
        style_tags=["afrobeats", "afro pop", "dancehall", "percussion"],
    ),
    "classical_art": GenreTemplate(
        id="classical_art", name="Art Song / Classical", description="Poetic, literary with complex structure",
        category="classical", structure="verse_chorus", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Literary, poetic. Rich metaphor, formal diction. Classical poetry influence.",
        emotional_register="Profound, layered. Beauty and sorrow intertwined.",
        example_themes=["nature", "mortality", "love eternal", "seasons", "the sublime"],
        style_tags=["classical", "orchestral", "piano", "strings"],
    ),
    "reggae": GenreTemplate(
        id="reggae", name="Reggae", description="Laid-back with conscious lyrics and offbeat rhythm",
        category="world", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Conscious, spiritual. Patois welcome. Nature imagery, unity, resistance.",
        emotional_register="Peaceful, conscious, resilient. One love vibration.",
        example_themes=["unity", "Jah", "redemption", "sunshine", "resistance"],
        style_tags=["reggae", "ska", "offbeat guitar", "bass heavy"],
    ),
    "soul": GenreTemplate(
        id="soul", name="Soul", description="Deep emotional expression with gospel roots",
        category="rnb", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Deep, emotive. Gospel-influenced testimony. Raw honesty.",
        emotional_register="Deep feeling, raw vulnerability. From whisper to wail.",
        example_themes=["real love", "struggle", "rising up", "healing", "truth"],
        style_tags=["soul", "motown", "gospel", "organ"],
    ),
    "trap": GenreTemplate(
        id="trap", name="Trap", description="808-heavy with melodic hooks and ad-libs",
        category="hip-hop", structure="hip_hop", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Melodic flow, auto-tune friendly. Ad-libs (yeah, what, skrrt). Flex and feels.",
        emotional_register="Confident exterior, vulnerable moments. Flex meets feelings.",
        example_themes=["drip", "late nights", "trust issues", "counting up", "no sleep"],
        style_tags=["trap", "808", "hi-hats", "auto-tune"],
    ),
    "ambient": GenreTemplate(
        id="ambient", name="Ambient / Experimental", description="Atmospheric textures with abstract lyrics",
        category="electronic", structure="minimal", rhyme_scheme="free",
        syllable_density="low",
        vocabulary_style="Abstract, fragmented. Sound-poetry quality. Words as texture.",
        emotional_register="Ethereal, detached, transcendent. Beyond conventional emotion.",
        example_themes=["dissolving", "space", "echo", "void", "light particles"],
        style_tags=["ambient", "atmospheric", "experimental", "ethereal"],
        system_prompt_extra="Extremely sparse. Fragmented phrases. Words chosen for sound quality as much as meaning.",
    ),
    "musical_theater": GenreTemplate(
        id="musical_theater", name="Musical Theater", description="Dramatic, narrative with character voice",
        category="theater", structure="extended", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Theatrical, dramatic. Character-driven. Internal monologue turned outward. Wit.",
        emotional_register="High drama, humor to heartbreak. Every line advances the story.",
        example_themes=["wanting more", "the big moment", "impossible love", "defiance", "transformation"],
        style_tags=["broadway", "musical", "theatrical", "dramatic"],
    ),
    "grunge": GenreTemplate(
        id="grunge", name="Grunge", description="Raw, disaffected with distorted emotional honesty",
        category="rock", structure="verse_chorus", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Raw, disaffected. Mumbled clarity. Self-deprecating. Anti-pretension.",
        emotional_register="Apathetic exterior, intense pain underneath. Gen-X angst.",
        example_themes=["apathy", "self-destruction", "faking it", "teen spirit", "rain"],
        style_tags=["grunge", "distorted guitar", "90s rock", "raw"],
    ),
    "dancehall": GenreTemplate(
        id="dancehall", name="Dancehall", description="Caribbean rhythm with energetic toasting",
        category="world", structure="verse_hook", rhyme_scheme="AABB",
        syllable_density="high",
        vocabulary_style="Patois, energetic chanting. Dance instructions. Riddim-driven.",
        emotional_register="High energy, party vibes. Boss attitude.",
        example_themes=["wine up", "bad gyal", "party", "nuff respect", "tek it easy"],
        style_tags=["dancehall", "caribbean", "riddim", "bass"],
    ),
    "acoustic": GenreTemplate(
        id="acoustic", name="Acoustic / Singer-Songwriter", description="Intimate, personal with raw honesty",
        category="folk", structure="verse_chorus", rhyme_scheme="ABCB",
        syllable_density="medium",
        vocabulary_style="Personal, confessional. Diary-like honesty. Small details that feel enormous.",
        emotional_register="Intimate, vulnerable. Whispering secrets to a room.",
        example_themes=["unrequited love", "growing apart", "small towns", "3am thoughts", "letters unsent"],
        style_tags=["acoustic", "singer-songwriter", "guitar", "intimate"],
    ),
    "disco": GenreTemplate(
        id="disco", name="Disco / Funk", description="Groovy, feel-good with dancefloor energy",
        category="pop", structure="pop", rhyme_scheme="AABB",
        syllable_density="medium",
        vocabulary_style="Groovy, playful. Dancefloor commands. Feel-good catchphrases.",
        emotional_register="Joyful, liberated, sexy. Saturday night fever.",
        example_themes=["get on the floor", "boogie", "last dance", "staying alive", "groove tonight"],
        style_tags=["disco", "funk", "bass guitar", "strings"],
    ),
    "cinematic": GenreTemplate(
        id="cinematic", name="Cinematic / Orchestral", description="Epic, sweeping with dramatic arc",
        category="cinematic", structure="ballad", rhyme_scheme="ABAB",
        syllable_density="medium",
        vocabulary_style="Epic, sweeping imagery. Cinematic language. Grand metaphors. Anthemic.",
        emotional_register="Overwhelmingly emotional. Goosebumps moments. Epic highs and devastating lows.",
        example_themes=["destiny", "the final stand", "beyond the horizon", "legends never die", "into the storm"],
        style_tags=["cinematic", "orchestral", "epic", "dramatic"],
    ),
}


# ── System Prompts ─────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """You are a professional songwriter and lyricist. You write original, creative song lyrics with proper structure.

RULES:
1. ALWAYS use structure tags on their own line: [Verse 1], [Chorus], [Bridge], etc.
2. Each section should have 4-8 lines unless specified otherwise.
3. Maintain consistent rhyme scheme within sections.
4. The chorus must be the catchiest, most memorable part.
5. Lyrics should flow naturally when sung — consider syllable count and rhythm.
6. Do NOT include notes, explanations, or commentary — ONLY output the lyrics.
7. Do NOT use quotation marks around the lyrics.
8. Start directly with the first structure tag."""

PLAN_SYSTEM_PROMPT = """You are a professional songwriter planning a song. Given a theme or description, output a brief song plan in this exact JSON format:
{
  "title": "Song Title",
  "genre": "genre name",
  "mood": "primary mood",
  "structure": ["Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Pre-Chorus", "Chorus", "Bridge", "Chorus", "Outro"],
  "rhyme_scheme": "ABAB",
  "key_imagery": ["image1", "image2", "image3"],
  "chorus_hook": "the central hook idea"
}
Output ONLY the JSON, no other text."""


def build_generation_prompt(
    user_prompt: str,
    genre_id: str = "pop",
    mood: str = "",
    language: str = "en",
    structure_override: str = "",
    custom_system_prompt: str = "",
) -> tuple[str, str]:
    """
    Build the system prompt and user prompt for lyrics generation.
    Returns (system_prompt, user_prompt).
    """
    template = GENRE_TEMPLATES.get(genre_id, GENRE_TEMPLATES["pop"])

    # Build system prompt
    system = custom_system_prompt if custom_system_prompt else BASE_SYSTEM_PROMPT

    if not custom_system_prompt:
        system += f"\n\nGENRE: {template.name}"
        system += f"\nSTYLE: {template.vocabulary_style}"
        system += f"\nEMOTIONAL REGISTER: {template.emotional_register}"
        system += f"\nRHYME SCHEME: {template.rhyme_scheme}"
        system += f"\nSYLLABLE DENSITY: {template.syllable_density}"

        if template.system_prompt_extra:
            system += f"\nADDITIONAL: {template.system_prompt_extra}"

        if language != "en":
            system += f"\n\nWRITE THE LYRICS IN: {language} (use natural phrasing, not translation)"

    # Build structure
    if structure_override:
        structure = structure_override
    else:
        struct_key = template.structure
        structure = STANDARD_STRUCTURES.get(struct_key, STANDARD_STRUCTURES["verse_chorus"])

    # Build user message
    user_msg = f"Write song lyrics about: {user_prompt}"
    if mood:
        user_msg += f"\nMood: {mood}"
    user_msg += f"\n\nUse this structure:\n{structure}"

    return system, user_msg


def build_quick_prompt(description: str) -> tuple[str, str]:
    """
    Build prompts for Quick Mode — auto-detect genre and structure from a simple description.
    """
    system = BASE_SYSTEM_PROMPT + """

The user will give a brief description. Infer the best genre, mood, and structure.
Write complete, polished lyrics with structure tags. Be creative and original."""

    user_msg = f"Write a song: {description}"
    return system, user_msg


def build_plan_prompt(description: str) -> tuple[str, str]:
    """Build prompts for the planning stage (used by AI Producer)."""
    return PLAN_SYSTEM_PROMPT, f"Plan a song about: {description}"


def get_genre_list() -> list[dict]:
    """Get list of genres for UI dropdowns."""
    return [
        {"id": t.id, "name": t.name, "description": t.description, "category": t.category}
        for t in GENRE_TEMPLATES.values()
    ]


def get_genre_categories() -> list[str]:
    """Get unique genre categories."""
    return sorted(set(t.category for t in GENRE_TEMPLATES.values()))


def get_random_theme(genre_id: str = "pop") -> str:
    """Get a random example theme for a genre."""
    template = GENRE_TEMPLATES.get(genre_id, GENRE_TEMPLATES["pop"])
    return random.choice(template.example_themes) if template.example_themes else "love"


def get_style_tags(genre_id: str) -> list[str]:
    """Get ACE-Step style tags for a genre (used in Song Forge integration)."""
    template = GENRE_TEMPLATES.get(genre_id, GENRE_TEMPLATES["pop"])
    return template.style_tags
