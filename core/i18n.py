"""
Slunder Studio - Locale catalog and language helpers.
"""
from __future__ import annotations

import json
import re
import ast
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from core.settings import get_config_dir

DEFAULT_LOCALE = "en"
PSEUDO_LOCALE = "qps-ploc"
RTL_LOCALES = frozenset({"ar", "fa", "he", "ur"})
LOCALE_DIR = Path(__file__).resolve().parents[1] / "assets" / "locales"

_missing_key_log: list[str] = []
_active_locale: str | None = None

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

UI_LOCALE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("ar", "العربية"),
    (PSEUDO_LOCALE, "Pseudo-locale (layout QA)"),
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
    "nav.sections.create",
    "nav.sections.finish",
    "nav.sections.library",
    "nav.sections.system",
    "page.lyrics.title",
    "page.lyrics.subtitle",
    "page.song_forge.title",
    "page.song_forge.subtitle",
    "page.midi_studio.title",
    "page.midi_studio.subtitle",
    "page.vocals.title",
    "page.vocals.subtitle",
    "page.sfx.title",
    "page.sfx.subtitle",
    "page.mixer.title",
    "page.mixer.subtitle",
    "page.ai_producer.title",
    "page.ai_producer.subtitle",
    "page.projects.title",
    "page.projects.subtitle",
    "page.model_hub.title",
    "page.model_hub.subtitle",
    "page.settings.title",
    "page.settings.subtitle",
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
    "settings.osc.group",
    "settings.osc.control",
    "settings.osc.enabled",
    "settings.osc.control_help",
    "settings.osc.port",
    "settings.osc.port_help",
    "settings.osc.lan_access",
    "settings.osc.allow_lan",
    "settings.osc.lan_access_help",
    "settings.osc.allowed_hosts",
    "settings.osc.allowed_hosts_placeholder",
    "settings.osc.allowed_hosts_help",
    "settings.osc.packet_limit",
    "settings.osc.packet_limit_help",
    "settings.osc.rate_limit",
    "settings.osc.rate_limit_help",
    "settings.osc.note",
    "settings.gpu.group",
    "settings.gpu.device_index",
    "settings.gpu.offline_mode",
    "settings.gpu.disable_internet",
    "settings.gpu.hf_token",
    "settings.appearance.group",
    "settings.appearance.experience_level",
    "settings.appearance.default_lyrics_language",
    "settings.appearance.ui_language",
    "settings.appearance.ui_language_help",
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
    "settings.messages.locale_changed",
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
    normalized = raw.split(".")[0] or DEFAULT_LOCALE
    if normalized == "qps_ploc":
        return PSEUDO_LOCALE
    return normalized


def _external_locale_dir() -> Path:
    return get_config_dir() / "locales"


def available_locales() -> list[str]:
    locales: set[str] = set()
    for directory in (LOCALE_DIR, _external_locale_dir()):
        if directory.exists():
            locales.update(path.stem for path in directory.glob("*.json") if path.is_file())
    locales.add(PSEUDO_LOCALE)
    return sorted(locales) or [DEFAULT_LOCALE]


def _read_catalog(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _transform_catalog(catalog: dict, transform) -> dict:
    transformed: dict = {}
    for key, value in catalog.items():
        transformed[key] = (
            _transform_catalog(value, transform)
            if isinstance(value, dict)
            else transform(str(value))
        )
    return transformed


@lru_cache(maxsize=16)
def load_catalog(locale: str = DEFAULT_LOCALE) -> dict:
    catalog_locale = normalize_locale(locale)
    if catalog_locale == PSEUDO_LOCALE:
        return _transform_catalog(load_catalog(DEFAULT_LOCALE), pseudolocalize)

    builtin_base = _read_catalog(LOCALE_DIR / f"{DEFAULT_LOCALE}.json")
    builtin = LOCALE_DIR / f"{catalog_locale}.json"
    external = _external_locale_dir() / f"{catalog_locale}.json"
    catalog = _deep_merge({}, builtin_base)
    if builtin.exists() and catalog_locale != DEFAULT_LOCALE:
        catalog = _deep_merge(catalog, _read_catalog(builtin))
    if external.exists():
        catalog = _deep_merge(catalog, _read_catalog(external))
    return catalog


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


def extract_i18n_keys(paths: Iterable[Path]) -> set[str]:
    """Extract literal ``tr()`` keys for catalog completeness checks."""
    keys: set[str] = set()
    for path in paths:
        source_path = Path(path)
        if not source_path.is_file() or source_path.suffix != ".py":
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            name = (
                function.id
                if isinstance(function, ast.Name)
                else function.attr
                if isinstance(function, ast.Attribute)
                else ""
            )
            if name == "tr" and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    keys.add(node.args[0].value)
    return keys


def tr(key: str, locale: str | None = None, **params) -> str:
    catalog_locale = current_locale() if locale is None else normalize_locale(locale)
    value = _lookup(load_catalog(catalog_locale), key)
    if value is None and catalog_locale != DEFAULT_LOCALE:
        value = _lookup(load_catalog(DEFAULT_LOCALE), key)
    if value is None:
        if key not in _missing_key_log:
            _missing_key_log.append(key)
        return f"[{key}]"
    text = str(value)
    return text.format(**params) if params else text


def user_facing_readiness(readiness, *, model_name: str = "") -> str:
    """Translate engine readiness details into concise UI guidance.

    Engine contracts intentionally use precise internal remedies.  Keep those
    details in the contract and translate them at the presentation boundary so
    views do not leak implementation vocabulary or raw state strings.
    """
    remedy = str(getattr(readiness, "remedy", "") or "").strip()
    capability = getattr(readiness, "capability", None)
    feature = str(getattr(capability, "label", "this feature") or "this feature")
    model = str(
        model_name
        or getattr(readiness, "model_id", "")
        or feature
    ).strip()
    missing = tuple(
        str(package).strip()
        for package in (getattr(readiness, "missing_packages", ()) or ())
        if str(package).strip()
    )
    if missing:
        return tr(
            "runtime.readiness.install_packages",
            packages=", ".join(missing),
            model=model,
        )

    lower = remedy.casefold()
    if "wait for" in lower:
        return tr("runtime.readiness.wait", model=model)
    if "re-download" in lower or "redownload" in lower:
        return tr("runtime.readiness.redownload", model=model)
    if "review and approve" in lower:
        return tr("runtime.readiness.approve", model=model)
    if "install" in lower and "activate" in lower:
        return tr("runtime.readiness.install_activate", model=model)
    if lower.startswith("download"):
        return tr("runtime.readiness.download", model=model)
    if lower.startswith("retry"):
        return tr("runtime.readiness.retry", model=model)
    if lower.startswith("activate"):
        return tr("runtime.readiness.activate", model=model)
    if lower.startswith("select "):
        return tr("runtime.readiness.select", item=remedy[7:].rstrip("."))
    if "demo option" in lower:
        return tr("runtime.readiness.demo", feature=feature)
    if "no model is registered" in lower:
        return tr("runtime.readiness.no_model", feature=feature)
    return tr("runtime.readiness.unavailable", feature=feature)


def get_missing_key_log() -> list[str]:
    return list(_missing_key_log)


def clear_missing_key_log() -> None:
    _missing_key_log.clear()


def current_locale() -> str:
    """Return the active UI locale, loading the persisted setting lazily."""
    global _active_locale
    if _active_locale is None:
        try:
            from core.settings import Settings

            _active_locale = normalize_locale(
                Settings().get("general.ui_locale", DEFAULT_LOCALE)
            )
        except Exception:
            _active_locale = DEFAULT_LOCALE
    return _active_locale


def locale_direction(locale: str | None = None) -> str:
    """Return ``rtl`` for right-to-left UI locales, otherwise ``ltr``."""
    return "rtl" if normalize_locale(locale or current_locale()) in RTL_LOCALES else "ltr"


def is_rtl(locale: str | None = None) -> bool:
    return locale_direction(locale) == "rtl"


def set_locale(
    locale: str | None,
    *,
    persist: bool = True,
    app=None,
) -> str:
    """Set the UI locale, persist it when requested, and apply Qt direction."""
    global _active_locale
    requested = normalize_locale(locale)
    if requested not in available_locales():
        requested = DEFAULT_LOCALE
    _active_locale = requested
    if persist:
        from core.settings import Settings

        Settings().set("general.ui_locale", requested)

    if app is None:
        try:
            from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
        except ImportError:
            app = None
    if app is not None:
        try:
            from PySide6.QtCore import Qt

            app.setLayoutDirection(
                Qt.LayoutDirection.RightToLeft
                if is_rtl(requested)
                else Qt.LayoutDirection.LeftToRight
            )
        except (AttributeError, ImportError):
            pass
    return requested


def ui_locale_options() -> tuple[tuple[str, str], ...]:
    """Return selectable UI locales with stable codes in combo-box data."""
    known = dict(UI_LOCALE_OPTIONS)
    for code in available_locales():
        known.setdefault(code, language_label(code))
    return tuple((code, known[code]) for code in known)


def pseudolocalize(text: str) -> str:
    """Expand English copy while preserving format placeholders for layout QA."""
    source = str(text)
    parts = re.split(r"(\{[^{}]+\})", source)
    transformed: list[str] = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            transformed.append(part)
            continue
        converted = part.translate(str.maketrans({
            "a": "à", "A": "À", "e": "ë", "E": "Ë",
            "i": "ï", "I": "Ï", "o": "õ", "O": "Õ",
            "u": "ü", "U": "Ü", "c": "ç", "C": "Ç",
            "n": "ñ", "N": "Ñ",
        }))
        transformed.append(converted)
    body = "".join(transformed)
    return f"［{body} ···］"


def pseudolocale_overflow(
    text: str,
    measured_width: float,
    available_width: float,
    *,
    margin: float = 4.0,
) -> bool:
    """Return whether a non-empty pseudo-localized label exceeds its slot."""
    if not str(text).strip() or float(available_width) <= 0:
        return False
    return float(measured_width) + float(margin) > float(available_width)


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
