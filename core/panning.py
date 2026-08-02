"""
Slunder Studio - Constant-power pan law.
One tested implementation shared by every mixer, so the two surfaces cannot
drift apart or disagree about what "constant power" means.
"""
from __future__ import annotations

import math

# -3.01 dB at centre is what constant power means: each side carries half the
# power, and the two sides always sum to the full power.
CENTER_GAIN = math.sqrt(0.5)
CENTER_GAIN_DB = -3.0102999566398116


def constant_power_pan(pan: float) -> tuple[float, float]:
    """Return (left_gain, right_gain) for a pan position in [-1.0, 1.0].

    Sine/cosine law: the pan position maps onto a quarter circle, so
    left^2 + right^2 == 1 at every position. Centre gives both sides
    -3.01 dB; the endpoints fully attenuate the opposite channel.
    """
    position = min(max(float(pan), -1.0), 1.0)
    angle = (position + 1.0) * (math.pi / 4.0)
    return math.cos(angle), math.sin(angle)


def pan_gains(pan: float, volume: float = 1.0) -> tuple[float, float]:
    """Constant-power pan gains scaled by a track volume."""
    left, right = constant_power_pan(pan)
    return left * volume, right * volume
