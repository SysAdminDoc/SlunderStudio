"""
Slunder Studio — Stem Mixer Widget
Visual stem mixer for Demucs separation results.
Per-stem volume, pan, mute/solo, waveform preview, and remix export.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QFrame, QScrollArea, QProgressBar, QComboBox,
)
from PySide6.QtCore import Qt, Signal

import numpy as np

from ui.theme import Palette, ThemeEngine, rgba
from ui.accessibility import install_accessibility
from ui.widgets import ElidedLabel, EmptyStateWidget
from ui.waveform_widget import MiniWaveform
from core.audio_buffers import mixdown_audio
from core.i18n import tr
from core.panning import pan_gains
from core.settings import Settings
from core.stem_export import STEM_EXPORT_TEMPLATES


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
                 sample_rate: int = 44100, parent=None, *, source_path: str = ""):
        super().__init__(parent)
        self.stem_name = stem_name
        self.audio = audio
        self.sample_rate = sample_rate
        self.source_path = source_path
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

        self._mute_btn = QPushButton(tr("mixer.stem.mute_short"))
        self._mute_btn.setMinimumSize(26, 20)
        self._mute_btn.setCheckable(True)
        self._mute_btn.clicked.connect(self._on_mute)

        self._solo_btn = QPushButton(tr("mixer.stem.solo_short"))
        self._solo_btn.setMinimumSize(26, 20)
        self._solo_btn.setCheckable(True)
        self._solo_btn.clicked.connect(self._on_solo)

        self._play_btn = QPushButton(tr("mixer.stem.play"))
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
        vol_l = QLabel(tr("mixer.stem.volume_short"))
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
        pan_l = QLabel(tr("mixer.stem.pan_short"))
        pan_l.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        pan_l.setMinimumWidth(18)
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(-100, 100)
        self._pan_slider.setValue(0)
        self._pan_slider.setMinimumHeight(14)
        self._pan_slider.valueChanged.connect(self._on_pan)
        self._pan_label = ElidedLabel(tr("mixer.stem.pan_center"), minimum_width=32)
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
            tr("mixer.stem.accessibility.stem_name", stem=stem_name.title()),
            named_controls=[
                (self._mute_btn, tr("mixer.stem.accessibility.mute_name", stem=stem_name), tr("mixer.stem.accessibility.mute_description")),
                (self._solo_btn, tr("mixer.stem.accessibility.solo_name", stem=stem_name), tr("mixer.stem.accessibility.solo_description")),
                (self._play_btn, tr("mixer.stem.accessibility.play_name", stem=stem_name), tr("mixer.stem.accessibility.play_description")),
                (self._vol_slider, tr("mixer.stem.accessibility.volume_name", stem=stem_name.title()), tr("mixer.stem.accessibility.volume_description")),
                (self._pan_slider, tr("mixer.stem.accessibility.pan_name", stem=stem_name.title()), tr("mixer.stem.accessibility.pan_description")),
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
            self._pan_label.setText(tr("mixer.stem.pan_center"))
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
    stems_export_requested = Signal()
    stem_play = Signal(str)  # stem name
    empty_action_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: dict[str, StemStrip] = {}
        self._sample_rate = 44100
        self._settings = Settings()

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header
        header = QHBoxLayout()
        title = QLabel(tr("mixer.stem.title"))
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9.75pt;")

        naming_label = QLabel(tr("mixer.stem.naming_label"))
        naming_label.setToolTip(tr("mixer.stem.naming_tooltip"))
        self._export_template_combo = QComboBox()
        for template in STEM_EXPORT_TEMPLATES:
            self._export_template_combo.addItem(
                tr(f"mixer.stem.templates.{template.id}"), template.id
            )
        saved_template = str(
            self._settings.get("general.stem_export_template", "generic") or "generic"
        )
        saved_index = self._export_template_combo.findData(saved_template)
        self._export_template_combo.setCurrentIndex(saved_index if saved_index >= 0 else 0)
        self._export_template_combo.setMinimumWidth(150)
        self._export_template_combo.currentIndexChanged.connect(
            self._on_export_template_changed
        )

        self._export_stems_btn = QPushButton(tr("mixer.stem.export_stems"))
        self._export_stems_btn.setProperty("class", "success")
        self._export_stems_btn.setEnabled(False)
        self._export_stems_btn.clicked.connect(self.stems_export_requested.emit)

        self._remix_btn = QPushButton(tr("mixer.stem.export_remix"))
        self._remix_btn.setProperty("class", "success")
        self._remix_btn.setEnabled(False)
        self._remix_btn.clicked.connect(self.remix_requested.emit)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(naming_label)
        header.addWidget(self._export_template_combo)
        header.addWidget(self._export_stems_btn)
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
            tr("mixer.stem.empty_title"),
            tr("mixer.stem.empty_description"),
            tr("mixer.stem.empty_action"),
        )
        self._empty.action_requested.connect(self.empty_action_requested.emit)
        self._container_layout.addWidget(self._empty)
        self._container_layout.addStretch()

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        install_accessibility(
            self,
            tr("mixer.stem.accessibility.name"),
            named_controls=[
                (self._export_template_combo, tr("mixer.stem.accessibility.template_name"), tr("mixer.stem.accessibility.template_description")),
                (self._export_stems_btn, tr("mixer.stem.accessibility.export_name"), tr("mixer.stem.accessibility.export_description")),
                (self._remix_btn, tr("mixer.stem.accessibility.remix_name"), tr("mixer.stem.accessibility.remix_description")),
            ],
        )

    def _on_export_template_changed(self, index: int):
        template_id = self._export_template_combo.itemData(index)
        if template_id:
            self._settings.set("general.stem_export_template", str(template_id))

    @property
    def stem_export_template_id(self) -> str:
        return str(self._export_template_combo.currentData() or "generic")

    def set_stem_export_busy(self, busy: bool):
        """Keep the multi-file export action cancellable from the mixer header."""
        self._export_stems_btn.setText(
            tr("mixer.stem.cancel_export") if busy else tr("mixer.stem.export_stems")
        )
        self._export_stems_btn.setEnabled(bool(busy) or bool(self._strips))
        self._export_template_combo.setEnabled(not busy)

    def load_stems(self, stems: list, sample_rate: int = 44100):
        """
        Load stems from SeparationResult.stems.
        Each stem should have .name and .audio attributes.
        """
        self.clear()
        self._sample_rate = sample_rate

        for stem in stems:
            strip = StemStrip(
                stem.name,
                stem.audio,
                sample_rate,
                source_path=str(getattr(stem, "file_path", "") or ""),
            )
            strip.play_requested.connect(self.stem_play.emit)
            self._strips[stem.name] = strip
            self._container_layout.insertWidget(
                self._container_layout.count() - 1, strip
            )

        self._remix_btn.setEnabled(len(self._strips) > 0)
        self._export_stems_btn.setEnabled(len(self._strips) > 0)
        self._empty.setVisible(not self._strips)

    @property
    def sample_rate(self) -> int:
        """Sample rate shared by the currently loaded stem set."""
        return int(self._sample_rate)

    def clear(self):
        for strip in self._strips.values():
            self._container_layout.removeWidget(strip)
            strip.deleteLater()
        self._strips.clear()
        self._remix_btn.setEnabled(False)
        self._export_stems_btn.setEnabled(False)
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

    def get_stem_export_snapshots(self) -> list[dict]:
        """Return immutable stem buffers and source metadata for a worker export."""
        return [
            {
                "name": strip.stem_name,
                "audio": np.asarray(strip.audio, dtype=np.float32).copy(),
                "sample_rate": int(strip.sample_rate),
                "source_path": strip.source_path,
            }
            for strip in self._strips.values()
            if strip.audio is not None
        ]
