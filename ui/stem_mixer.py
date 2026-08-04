"""
Slunder Studio — Stem Mixer Widget
Visual stem mixer for Demucs separation results.
Per-stem volume, pan, mute/solo, waveform preview, and remix export.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QFrame, QScrollArea, QProgressBar,
)
from PySide6.QtCore import Qt, Signal

import numpy as np

from ui.theme import Palette, ThemeEngine, rgba
from ui.accessibility import install_accessibility
from ui.widgets import ElidedLabel, EmptyStateWidget
from ui.waveform_widget import MiniWaveform
from core.audio_buffers import mixdown_audio
from core.panning import pan_gains


# ── Stem Colors ────────────────────────────────────────────────────────────────

STEM_COLORS = {
    "vocals": Palette.RED,
    "drums": Palette.PEACH,
    "bass": Palette.GREEN,
    "other": Palette.SKY,
    "piano": Palette.MAUVE,
    "guitar": Palette.YELLOW,
}


# ── Stem Strip ─────────────────────────────────────────────────────────────────

class StemStrip(QFrame):
    """Single stem mixer strip with waveform and controls."""

    mute_changed = Signal(str, bool)
    solo_changed = Signal(str, bool)
    volume_changed = Signal(str, float)
    pan_changed = Signal(str, float)
    play_requested = Signal(str)

    def __init__(self, stem_name: str, audio: Optional[np.ndarray] = None,
                 sample_rate: int = 44100, parent=None):
        super().__init__(parent)
        self.stem_name = stem_name
        self.audio = audio
        self.sample_rate = sample_rate
        self._volume = 1.0
        self._pan = 0.0
        self._muted = False
        self._soloed = False

        t = ThemeEngine.get_colors()
        color = STEM_COLORS.get(stem_name, t["accent"])

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            StemStrip {{
                background: {t['surface']};
                border: 1px solid {rgba(color, 68)};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
        """)
        self.setMinimumHeight(100)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        # Left: Name + controls
        left = QVBoxLayout()
        left.setSpacing(3)

        # Stem name
        name_label = QLabel(stem_name.upper())
        name_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 9pt;")
        left.addWidget(name_label)

        # Mute/Solo buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._mute_btn = QPushButton("M")
        self._mute_btn.setMinimumSize(26, 20)
        self._mute_btn.setCheckable(True)
        self._mute_btn.clicked.connect(self._on_mute)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setMinimumSize(26, 20)
        self._solo_btn.setCheckable(True)
        self._solo_btn.clicked.connect(self._on_solo)

        self._play_btn = QPushButton("Play")
        self._play_btn.setMinimumSize(38, 20)
        self._play_btn.clicked.connect(lambda: self.play_requested.emit(self.stem_name))

        for btn in [self._mute_btn, self._solo_btn, self._play_btn]:
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {t['background']};
                    color: {t['text_secondary']};
                    border: 1px solid {t['border']};
                    border-radius: 3px;
                    font-size: 6.75pt; font-weight: bold;
                }}
                QPushButton:hover {{ background: {t['surface_hover']}; }}
                QPushButton:checked {{ background: {color}; color: {Palette.CRUST}; border: none; }}
            """)

        btn_row.addWidget(self._mute_btn)
        btn_row.addWidget(self._solo_btn)
        btn_row.addWidget(self._play_btn)
        left.addLayout(btn_row)

        # Volume slider
        vol_row = QHBoxLayout()
        vol_row.setSpacing(3)
        vol_l = QLabel("Vol")
        vol_l.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        vol_l.setMinimumWidth(18)
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 150)
        self._vol_slider.setValue(100)
        self._vol_slider.setMinimumHeight(14)
        self._vol_slider.valueChanged.connect(self._on_volume)
        self._vol_label = ElidedLabel("100%", minimum_width=32)
        self._vol_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        vol_row.addWidget(vol_l)
        vol_row.addWidget(self._vol_slider)
        vol_row.addWidget(self._vol_label)
        left.addLayout(vol_row)

        # Pan slider
        pan_row = QHBoxLayout()
        pan_row.setSpacing(3)
        pan_l = QLabel("Pan")
        pan_l.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        pan_l.setMinimumWidth(18)
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(-100, 100)
        self._pan_slider.setValue(0)
        self._pan_slider.setMinimumHeight(14)
        self._pan_slider.valueChanged.connect(self._on_pan)
        self._pan_label = ElidedLabel("C", minimum_width=32)
        self._pan_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        pan_row.addWidget(pan_l)
        pan_row.addWidget(self._pan_slider)
        pan_row.addWidget(self._pan_label)
        left.addLayout(pan_row)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(160)
        layout.addWidget(left_widget)

        # Right: Mini waveform
        self._waveform = MiniWaveform()
        if audio is not None:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            self._waveform.load_audio(mono, sample_rate)
        layout.addWidget(self._waveform, 1)

        install_accessibility(
            self,
            f"{stem_name.title()} stem",
            named_controls=[
                (self._mute_btn, f"Mute {stem_name}", "Mutes this stem in the remix."),
                (self._solo_btn, f"Solo {stem_name}", "Solos this stem in the remix."),
                (self._play_btn, f"Play {stem_name}", "Plays this stem preview."),
                (self._vol_slider, f"{stem_name.title()} volume", "Adjusts the volume of this stem."),
                (self._pan_slider, f"{stem_name.title()} pan", "Positions this stem in the stereo field."),
            ],
        )

    def _on_mute(self):
        self._muted = self._mute_btn.isChecked()
        self.mute_changed.emit(self.stem_name, self._muted)

    def _on_solo(self):
        self._soloed = self._solo_btn.isChecked()
        self.solo_changed.emit(self.stem_name, self._soloed)

    def _on_volume(self, val):
        self._volume = val / 100.0
        self._vol_label.setText(f"{val}%")
        self.volume_changed.emit(self.stem_name, self._volume)

    def _on_pan(self, val):
        self._pan = val / 100.0
        if val == 0:
            self._pan_label.setText("C")
        elif val < 0:
            self._pan_label.setText(f"L{abs(val)}")
        else:
            self._pan_label.setText(f"R{val}")
        self.pan_changed.emit(self.stem_name, self._pan)

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def pan(self) -> float:
        return self._pan

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def is_soloed(self) -> bool:
        return self._soloed


# ── Stem Mixer ─────────────────────────────────────────────────────────────────

class StemMixer(QWidget):
    """Multi-stem mixer with remix export."""

    remix_requested = Signal()
    stem_play = Signal(str)  # stem name
    empty_action_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: dict[str, StemStrip] = {}
        self._sample_rate = 44100

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        title = QLabel("Stem Mixer")
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9.75pt;")

        self._remix_btn = QPushButton("Export Remix")
        self._remix_btn.setProperty("class", "success")
        self._remix_btn.setEnabled(False)
        self._remix_btn.clicked.connect(self.remix_requested.emit)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._remix_btn)
        layout.addLayout(header)

        # Scroll area for stems
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setContentsMargins(0, 0, 0, 0)
        self._container_layout.setSpacing(4)
        self._empty = EmptyStateWidget(
            "No stems separated yet",
            "Choose an audio file above and separate it to mix vocals, drums, bass, and other stems.",
            "Choose audio",
        )
        self._empty.action_requested.connect(self.empty_action_requested.emit)
        self._container_layout.addWidget(self._empty)
        self._container_layout.addStretch()

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        install_accessibility(
            self,
            "Stem Mixer",
            named_controls=[
                (self._remix_btn, "Export stem remix", "Exports a mix of the loaded stems."),
            ],
        )

    def load_stems(self, stems: list, sample_rate: int = 44100):
        """
        Load stems from SeparationResult.stems.
        Each stem should have .name and .audio attributes.
        """
        self.clear()
        self._sample_rate = sample_rate

        for stem in stems:
            strip = StemStrip(stem.name, stem.audio, sample_rate)
            strip.play_requested.connect(self.stem_play.emit)
            self._strips[stem.name] = strip
            self._container_layout.insertWidget(
                self._container_layout.count() - 1, strip
            )

        self._remix_btn.setEnabled(len(self._strips) > 0)
        self._empty.setVisible(not self._strips)

    def clear(self):
        for strip in self._strips.values():
            self._container_layout.removeWidget(strip)
            strip.deleteLater()
        self._strips.clear()
        self._remix_btn.setEnabled(False)
        self._empty.show()

    def get_remix_audio(self) -> Optional[np.ndarray]:
        """Mix stems according to current volume/pan/mute/solo settings."""
        if not self._strips:
            return None
        return mixdown_audio(
            [
                (
                    strip.audio,
                    strip.volume,
                    strip.pan,
                    strip.is_muted,
                    strip.is_soloed,
                )
                for strip in self._strips.values()
            ]
        )

    def get_stem_names(self) -> list[str]:
        return list(self._strips.keys())
