"""
Slunder Studio - WCAG contrast utilities and the approved token pairs.
The theme is gated against these tables in tests, so a palette edit that drops a
text/background pair below the WCAG 2.2 threshold fails the suite.
"""
from __future__ import annotations

from ui.theme import Palette

# WCAG 2.2 SC 1.4.3 Contrast (Minimum).
NORMAL_TEXT_RATIO = 4.5
LARGE_TEXT_RATIO = 3.0
# WCAG 2.2 SC 1.4.11 Non-text Contrast, used for focus rings and borders.
NON_TEXT_RATIO = 3.0

# Backgrounds any text may be drawn on.
SURFACE_TOKENS = ("CRUST", "MANTLE", "BASE", "SURFACE0", "SURFACE1")

# Foregrounds used for normal-size body text.
TEXT_TOKENS = ("TEXT", "SUBTEXT1", "SUBTEXT0", "OVERLAY1", "OVERLAY0")

# Accent and status colors used as text (labels, links, warnings, errors).
SIGNAL_TOKENS = (
    "BLUE", "TEAL", "GREEN", "YELLOW", "PEACH", "RED",
    "MAUVE", "PINK", "SAPPHIRE", "SKY", "LAVENDER",
)


def _channel(value: int) -> float:
    c = value / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of an #rrggbb color."""
    text = color.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"Not a hex color: {color!r}")
    r, g, b = (int(text[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two colors, 1.0 to 21.0."""
    a = relative_luminance(foreground)
    b = relative_luminance(background)
    lighter, darker = (a, b) if a >= b else (b, a)
    return (lighter + 0.05) / (darker + 0.05)


def token(name: str) -> str:
    return getattr(Palette, name)


def failing_text_pairs(minimum: float = NORMAL_TEXT_RATIO) -> list[tuple[str, str, float]]:
    """Every (foreground, background, ratio) below the normal-text threshold."""
    failures: list[tuple[str, str, float]] = []
    for fg in TEXT_TOKENS + SIGNAL_TOKENS:
        for bg in SURFACE_TOKENS:
            ratio = contrast_ratio(token(fg), token(bg))
            if ratio < minimum:
                failures.append((fg, bg, round(ratio, 2)))
    return failures
