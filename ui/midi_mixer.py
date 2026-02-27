"""
Slunder Studio v0.0.2 — MIDI Mixer Widget
Per-track mixer with volume, pan, mute/solo, program selector, and master controls.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QComboBox, QScrollArea, QFrame, QSpinBox,
)
from PySide6.QtCore import Qt, Signal

from core.midi_utils import MidiData, TrackData, GM_PROGRAMS, get_program_name
from ui.theme import ThemeEngine


# ── Track Strip ────────────────────────────────────────────────────────────────

class TrackStrip(QFrame):
    """Single track mixer strip with controls."""

    mute_changed = Signal(int, bool)    # track_idx, muted
    solo_changed = Signal(int, bool)    # track_idx, soloed
    volume_changed = Signal(int, int)   # track_idx, volume 0-127
    pan_changed = Signal(int, int)      # track_idx, pan -64..63
    program_changed = Signal(int, int)  # track_idx, program
    select_requested = Signal(int)      # track_idx

    def __init__(self, track: TrackData, track_idx: int, parent=None):
        super().__init__(parent)
        self.track = track
        self.track_idx = track_idx
        self._muted = False
        self._soloed = False
        self._selected = False

        t = ThemeEngine.get_colors()
        self.setFrameShape(QFrame.StyledPanel)
        self._base_style = f"""
            TrackStrip {{
                background: {t.get('surface', '#161b22')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 6px;
            }}
        """
        self._selected_style = f"""
            TrackStrip {{
                background: {t.get('surface_hover', '#1c2333')};
                border: 1px solid {t.get('accent', '#58a6ff')};
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(self._base_style)
        self.setFixedHeight(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Track name + note count
        header = QHBoxLayout()
        name_label = QLabel(track.name)
        name_label.setStyleSheet(f"color: {t.get('text', '#e6edf3')}; font-weight: bold; font-size: 11px;")
        count_label = QLabel(f"{track.note_count} notes")
        count_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px;")
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(count_label)
        layout.addLayout(header)

        # Instrument selector
        self._program_combo = QComboBox()
        self._program_combo.setFixedHeight(24)
        self._program_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 10px;
            }}
        """)
        # Populate with common GM programs
        common_programs = [0, 4, 5, 24, 25, 26, 29, 30, 32, 33, 38, 40, 42,
                           48, 56, 57, 60, 61, 65, 66, 73, 80, 88, 128]
        for prog in common_programs:
            name = get_program_name(prog, prog == 128)
            self._program_combo.addItem(name, prog)

        # Set current
        idx = self._program_combo.findData(track.program)
        if idx >= 0:
            self._program_combo.setCurrentIndex(idx)
        self._program_combo.currentIndexChanged.connect(self._on_program_changed)
        layout.addWidget(self._program_combo)

        # Volume slider
        vol_row = QHBoxLayout()
        vol_row.setSpacing(4)
        vol_label = QLabel("Vol")
        vol_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px;")
        vol_label.setFixedWidth(22)
        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 127)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedHeight(16)
        self._volume_slider.valueChanged.connect(
            lambda v: self.volume_changed.emit(self.track_idx, v))
        self._vol_value = QLabel("100")
        self._vol_value.setFixedWidth(24)
        self._vol_value.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px;")
        self._volume_slider.valueChanged.connect(lambda v: self._vol_value.setText(str(v)))

        vol_row.addWidget(vol_label)
        vol_row.addWidget(self._volume_slider)
        vol_row.addWidget(self._vol_value)
        layout.addLayout(vol_row)

        # Pan slider
        pan_row = QHBoxLayout()
        pan_row.setSpacing(4)
        pan_label = QLabel("Pan")
        pan_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px;")
        pan_label.setFixedWidth(22)
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(-64, 63)
        self._pan_slider.setValue(0)
        self._pan_slider.setFixedHeight(16)
        self._pan_slider.valueChanged.connect(
            lambda v: self.pan_changed.emit(self.track_idx, v))
        self._pan_value = QLabel("C")
        self._pan_value.setFixedWidth(24)
        self._pan_value.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')}; font-size: 10px;")
        self._pan_slider.valueChanged.connect(self._update_pan_label)

        pan_row.addWidget(pan_label)
        pan_row.addWidget(self._pan_slider)
        pan_row.addWidget(self._pan_value)
        layout.addLayout(pan_row)

        # Mute / Solo buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._mute_btn = QPushButton("M")
        self._mute_btn.setFixedSize(28, 22)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setStyleSheet(self._mute_style(False))
        self._mute_btn.clicked.connect(self._on_mute)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setFixedSize(28, 22)
        self._solo_btn.setCheckable(True)
        self._solo_btn.setStyleSheet(self._solo_style(False))
        self._solo_btn.clicked.connect(self._on_solo)

        btn_row.addWidget(self._mute_btn)
        btn_row.addWidget(self._solo_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Click to select
        self.mousePressEvent = lambda e: self.select_requested.emit(self.track_idx)

    def set_selected(self, selected: bool):
        self._selected = selected
        self.setStyleSheet(self._selected_style if selected else self._base_style)

    def _update_pan_label(self, val):
        if val == 0:
            self._pan_value.setText("C")
        elif val < 0:
            self._pan_value.setText(f"L{abs(val)}")
        else:
            self._pan_value.setText(f"R{val}")

    def _on_mute(self):
        self._muted = self._mute_btn.isChecked()
        self._mute_btn.setStyleSheet(self._mute_style(self._muted))
        self.mute_changed.emit(self.track_idx, self._muted)

    def _on_solo(self):
        self._soloed = self._solo_btn.isChecked()
        self._solo_btn.setStyleSheet(self._solo_style(self._soloed))
        self.solo_changed.emit(self.track_idx, self._soloed)

    def _mute_style(self, active: bool) -> str:
        t = ThemeEngine.get_colors()
        if active:
            return f"""
                QPushButton {{ background: #da3633; color: white; border: none;
                    border-radius: 3px; font-weight: bold; font-size: 10px; }}
            """
        return f"""
            QPushButton {{ background: {t.get('surface', '#161b22')};
                color: {t.get('text_secondary', '#8b949e')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 3px; font-weight: bold; font-size: 10px; }}
            QPushButton:hover {{ background: {t.get('surface_hover', '#1c2333')}; }}
        """

    def _solo_style(self, active: bool) -> str:
        t = ThemeEngine.get_colors()
        if active:
            return f"""
                QPushButton {{ background: #d29922; color: white; border: none;
                    border-radius: 3px; font-weight: bold; font-size: 10px; }}
            """
        return self._mute_style(False)

    def _on_program_changed(self, idx):
        prog = self._program_combo.itemData(idx)
        if prog is not None:
            self.track.program = prog
            self.program_changed.emit(self.track_idx, prog)


# ── MIDI Mixer ─────────────────────────────────────────────────────────────────

class MidiMixer(QWidget):
    """Multi-track MIDI mixer panel."""

    track_selected = Signal(int)
    mix_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: list[TrackStrip] = []
        self._muted: set[int] = set()
        self._soloed: set[int] = set()
        self._selected_track: int = 0

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header
        header = QHBoxLayout()
        title = QLabel("Mixer")
        title.setStyleSheet(f"color: {t.get('text', '#e6edf3')}; font-weight: bold; font-size: 12px;")
        self._add_track_btn = QPushButton("+ Track")
        self._add_track_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t.get('accent', '#58a6ff')};
                color: white; border: none; border-radius: 4px;
                padding: 4px 10px; font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t.get('accent_hover', '#79c0ff')}; }}
        """)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._add_track_btn)
        layout.addLayout(header)

        # Scroll area for strips
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
        """)

        self._strip_container = QWidget()
        self._strip_layout = QVBoxLayout(self._strip_container)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch()

        self._scroll.setWidget(self._strip_container)
        layout.addWidget(self._scroll, 1)

    def load_midi(self, midi_data: MidiData):
        """Load all tracks from MidiData into mixer."""
        self.clear()
        for i, track in enumerate(midi_data.tracks):
            self._add_strip(track, i)
        if self._strips:
            self._strips[0].set_selected(True)
            self._selected_track = 0

    def clear(self):
        """Remove all track strips."""
        for strip in self._strips:
            self._strip_layout.removeWidget(strip)
            strip.deleteLater()
        self._strips.clear()
        self._muted.clear()
        self._soloed.clear()

    def _add_strip(self, track: TrackData, idx: int):
        strip = TrackStrip(track, idx)
        strip.mute_changed.connect(self._on_mute)
        strip.solo_changed.connect(self._on_solo)
        strip.volume_changed.connect(lambda *_: self.mix_changed.emit())
        strip.pan_changed.connect(lambda *_: self.mix_changed.emit())
        strip.program_changed.connect(lambda *_: self.mix_changed.emit())
        strip.select_requested.connect(self._on_track_select)

        # Insert before stretch
        self._strip_layout.insertWidget(self._strip_layout.count() - 1, strip)
        self._strips.append(strip)

    def _on_mute(self, idx: int, muted: bool):
        if muted:
            self._muted.add(idx)
        else:
            self._muted.discard(idx)
        self.mix_changed.emit()

    def _on_solo(self, idx: int, soloed: bool):
        if soloed:
            self._soloed.add(idx)
        else:
            self._soloed.discard(idx)
        self.mix_changed.emit()

    def _on_track_select(self, idx: int):
        for strip in self._strips:
            strip.set_selected(strip.track_idx == idx)
        self._selected_track = idx
        self.track_selected.emit(idx)

    def get_muted_tracks(self) -> set[int]:
        return self._muted.copy()

    def get_solo_track(self) -> Optional[int]:
        """Return soloed track index, or None if no solo."""
        if self._soloed:
            return min(self._soloed)  # First soloed track
        return None

    @property
    def selected_track(self) -> int:
        return self._selected_track
