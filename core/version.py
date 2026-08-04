"""
Slunder Studio — Version
The single source of truth for the application version. Everything that shows,
stamps, or packages a version reads it from here: the UI, provenance sidecars,
settings and project schemas, the build script, and the release artifact names.

Deliberately dependency-free so the build script and packaging metadata can
import it without pulling in Qt or numpy.
"""
from __future__ import annotations

APP_NAME = "SlunderStudio"
DISPLAY_NAME = "Slunder Studio"
__version__ = "0.1.32"
APP_VERSION = __version__


def version_tuple(parts: int = 4) -> tuple[int, ...]:
    """Return the version as an integer tuple padded to `parts` components."""
    numbers = []
    for chunk in __version__.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < parts:
        numbers.append(0)
    return tuple(numbers[:parts])
