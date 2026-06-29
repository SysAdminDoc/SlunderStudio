"""
Slunder Studio v0.1.22 - Locale catalog and language helpers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable


DEFAULT_LOCALE = "en"
LOCALE_DIR = Path(__file__).resolve().parents[1] / "assets" / "locales"

LANGUAGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("pt", "Portuguese"),
    ("de", "German"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh", "Chinese (Mandarin)"),
    ("ar", "Arabic"),
    ("hi", "Hindi"),
    ("ru", "Russian"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("tr", "Turkish"),
    ("pl", "Polish"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("id", "Indonesian"),
)

GPT_SOVITS_LANGUAGE_CODES = ("en", "zh", "ja")

REQUIRED_I18N_KEYS = (
    "app.window_title",
    "app.accessible_name",
    "app.accessible_description",
    "nav.lyrics",
    "nav.song_forge",
    "nav.midi_studio",
    "nav.vocals",
    "nav.sfx",
    "nav.mixer",
    "nav.ai_producer",
    "nav.projects",
    "nav.model_hub",
    "nav.settings",
    "nav.open",
    "nav.switches",
    "placeholder.coming_soon",
    "status.gpu_detecting",
    "status.gpu_accessible_name",
    "status.gpu_accessible_description",
    "status.vram_accessible_name",
    "status.vram_accessible_description",
    "settings.title",
    "settings.tabs.simple",
    "settings.tabs.advanced",
    "settings.output.group",
    "settings.output.directory",
    "settings.output.placeholder",
    "settings.output.browse",
    "settings.output.format",
    "settings.output.sample_rate",
    "settings.gpu.group",
    "settings.gpu.device_index",
    "settings.gpu.offline_mode",
    "settings.gpu.disable_internet",
    "settings.gpu.hf_token",
    "settings.appearance.group",
    "settings.appearance.experience_level",
    "settings.appearance.default_lyrics_language",
    "settings.lyrics.group",
    "settings.lyrics.model",
    "settings.lyrics.temperature",
    "settings.lyrics.top_p",
    "settings.lyrics.max_tokens",
    "settings.actions.reset_defaults",
    "settings.actions.include_private_inputs",
    "settings.actions.export_health",
    "settings.actions.open_config",
    "settings.dialogs.export_health",
    "lyrics.title",
    "lyrics.quick.tab",
    "lyrics.quick.label",
    "lyrics.quick.placeholder",
    "lyrics.guided.tab",
    "lyrics.guided.theme",
    "lyrics.guided.theme_placeholder",
    "lyrics.guided.genre",
    "lyrics.guided.mood",
    "lyrics.guided.structure",
    "lyrics.guided.language",
    "lyrics.pro.tab",
    "lyrics.pro.system_prompt",
    "lyrics.pro.user_prompt",
    "lyrics.pro.parameters",
    "lyrics.actions.generate",
    "lyrics.actions.cancel",
    "lyrics.actions.regenerate",
    "lyrics.history.title",
    "lyrics.history.search_placeholder",
    "lyrics.history.all",
    "lyrics.history.favorites",
    "lyrics.history.entries_count",
    "lyrics.messages.describe_song",
    "lyrics.messages.enter_theme",
    "lyrics.messages.enter_user_prompt",
    "lyrics.messages.cancelled",
    "lyrics.messages.empty_output",
    "lyrics.messages.complete",
    "lyrics.messages.generated",
    "vocal.tabs.singing",
    "vocal.tabs.lyric_melody",
    "vocal.tabs.conversion",
    "vocal.tabs.cloning",
    "vocal.tabs.autotune",
    "vocal.tabs.stems",
    "vocal.autotune.input_short",
    "vocal.autotune.no_file",
    "vocal.autotune.browse",
    "vocal.autotune.strength",
    "vocal.autotune.apply",
    "vocal.autotune.corrected",
    "vocal.melody.input_short",
    "vocal.melody.no_file",
    "vocal.melody.browse",
    "vocal.melody.lyrics_placeholder",
    "vocal.melody.tempo",
    "vocal.melody.render_diffsinger",
    "vocal.melody.generate",
    "vocal.melody.preview",
    "vocal.clone.language_short",
    "vocal.actions.send_to_forge",
    "vocal.actions.send_to_mixer",
    "vocal.actions.export_wav",
    "vocal.status.select_tab",
)


def normalize_locale(locale: str | None) -> str:
    raw = (locale or DEFAULT_LOCALE).strip().lower().replace("-", "_")
    return raw.split(".")[0] or DEFAULT_LOCALE


def available_locales() -> list[str]:
    if not LOCALE_DIR.exists():
        return [DEFAULT_LOCALE]
    locales = sorted(path.stem for path in LOCALE_DIR.glob("*.json") if path.is_file())
    return locales or [DEFAULT_LOCALE]


@lru_cache(maxsize=16)
def load_catalog(locale: str = DEFAULT_LOCALE) -> dict:
    catalog_locale = normalize_locale(locale)
    path = LOCALE_DIR / f"{catalog_locale}.json"
    if not path.exists() and catalog_locale != DEFAULT_LOCALE:
        path = LOCALE_DIR / f"{DEFAULT_LOCALE}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_catalog(catalog: dict, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in catalog.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_catalog(value, dotted))
        else:
            flat[dotted] = str(value)
    return flat


def catalog_keys(locale: str = DEFAULT_LOCALE) -> set[str]:
    return set(flatten_catalog(load_catalog(locale)).keys())


def missing_keys(required: Iterable[str], locale: str = DEFAULT_LOCALE) -> list[str]:
    keys = catalog_keys(locale)
    return sorted(key for key in required if key not in keys)


def tr(key: str, locale: str = DEFAULT_LOCALE, **params) -> str:
    value = _lookup(load_catalog(locale), key)
    if value is None and normalize_locale(locale) != DEFAULT_LOCALE:
        value = _lookup(load_catalog(DEFAULT_LOCALE), key)
    if value is None:
        return key
    text = str(value)
    return text.format(**params) if params else text


def language_label(code: str | None) -> str:
    normalized = normalize_language_code(code)
    return dict(LANGUAGE_OPTIONS).get(normalized, dict(LANGUAGE_OPTIONS)[DEFAULT_LOCALE])


def language_combo_items(codes: Iterable[str] | None = None) -> list[str]:
    allowed = set(codes) if codes is not None else None
    return [label for code, label in LANGUAGE_OPTIONS if allowed is None or code in allowed]


def language_code_from_label(label: str | None) -> str:
    return normalize_language_code(label)


def normalize_language_code(language: str | None) -> str:
    raw = (language or "").strip().lower().replace("-", "_")
    if not raw:
        return DEFAULT_LOCALE

    labels = {label.lower(): code for code, label in LANGUAGE_OPTIONS}
    labels.update({label.split(" (", 1)[0].lower(): code for code, label in LANGUAGE_OPTIONS})
    aliases = {
        "chinese": "zh",
        "mandarin": "zh",
        "jp": "ja",
        "japanese": "ja",
        "cn": "zh",
        "zh_cn": "zh",
        "zh_tw": "zh",
        "pt_br": "pt",
    }
    if raw in labels:
        return labels[raw]
    if raw in aliases:
        return aliases[raw]
    if raw in dict(LANGUAGE_OPTIONS):
        return raw
    if "(" in raw and ")" in raw:
        inside = raw.rsplit("(", 1)[1].split(")", 1)[0].strip()
        if inside in dict(LANGUAGE_OPTIONS):
            return inside
    prefix = raw.split("_", 1)[0]
    return prefix if prefix in dict(LANGUAGE_OPTIONS) else DEFAULT_LOCALE


def _lookup(catalog: dict, dotted_key: str):
    node = catalog
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node
