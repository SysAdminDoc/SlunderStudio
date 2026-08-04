"""
Slunder Studio — Mixer View
Multi-track mixer timeline with per-track volume/pan/effects,
smart mastering presets, waveform overview, and final export.
"""
import os
from dataclasses import replace
from typing import Callable, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QScrollArea, QSlider, QFileDialog, QDoubleSpinBox,
    QProgressBar, QTabWidget,
)
from PySide6.QtCore import Qt, Signal, QTimer

import numpy as np

from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget, OperationProgressWidget
from ui.theme import Palette, ThemeEngine
from ui.widgets import ElidedLabel
from ui.waveform_widget import WaveformWidget, MiniWaveform
from core.workers import CancelledJobError, InferenceWorker
from core.panning import pan_gains
from core.audio_export import (
    ExportSettings,
    export_from_numpy,
)
from core.audio_buffers import (
    decode_audio_file,
    mixdown_audio,
    normalize_channel_layout,
    prepare_audio_buffer,
    validate_audio_buffer,
    validate_sample_rate,
)
from core.mastering import (
    LUFS_TARGETS,
    PRESETS,
    DynamicEQSuggestion,
    LoudnessMatchResult,
    MasteringPreset,
    apply_dynamic_eq,
    match_loudness_to_reference,
    master_audio,
    measure_loudness_range,
    measure_lufs,
    measure_true_peak_db,
    measure_momentary_lufs,
    measure_short_term_max_lufs,
    measure_short_term_lufs,
    suggest_dynamic_eq_curve,
)
from ui.file_dialogs import (
    ensure_extension,
    open_audio_file,
    open_audio_files,
    save_audio_file,
)


def _mix_track_snapshots(
    track_snapshots,
    project_sample_rate: int,
    progress_cb=None,
    cancel_event=None,
) -> Optional[np.ndarray]:
    """Mix immutable track snapshots away from the GUI thread."""
    if not track_snapshots:
        return None

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    prepared_tracks = []
    total = max(len(track_snapshots), 1)
    for position, (audio, sample_rate, *_settings) in enumerate(track_snapshots, 1):
        if _cancelled():
            raise CancelledJobError("Mixer operation cancelled")
        frames = validate_audio_buffer(audio)
        if sample_rate != project_sample_rate or frames.ndim != 2 or frames.shape[1] != 2:
            frames = prepare_audio_buffer(
                frames,
                sample_rate,
                project_sample_rate,
                target_channels=2,
            )
        prepared_tracks.append(frames)
        if progress_cb:
            progress_cb(int(position * 45 / total))

    mixed = mixdown_audio(
        [
            (
                audio,
                snapshot[2],
                snapshot[3],
                snapshot[4],
                snapshot[5],
            )
            for audio, snapshot in zip(prepared_tracks, track_snapshots)
        ],
        progress_cb=(
            (lambda value: progress_cb(45 + int(value * 0.5)))
            if progress_cb else None
        ),
        cancel_event=cancel_event,
    )
    if progress_cb:
        progress_cb(95)
    return mixed


def _master_audio_task(
    track_snapshots,
    sample_rate: int,
    preset: MasteringPreset,
    reference_audio: Optional[np.ndarray] = None,
    reference_sample_rate: int = 44100,
    progress_cb=None,
    step_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Run mastering and reference matching off the Qt GUI thread."""
    mixed = _mix_track_snapshots(
        track_snapshots,
        sample_rate,
        progress_cb=(
            (lambda value: progress_cb(int(value * 0.25)))
            if progress_cb else None
        ),
        cancel_event=cancel_event,
    )
    if mixed is None:
        raise ValueError("No active mixer audio is available")

    def _progress(value: float, message: str):
        if progress_cb:
            normalized = float(value) * 100 if value <= 1 else float(value)
            progress_cb(25 + int(normalized * 0.75))
        if step_cb:
            step_cb(message)

    result = master_audio(
        mixed,
        sample_rate,
        preset,
        progress_callback=_progress,
    )
    if getattr(result, "error", None) or getattr(result, "audio", None) is None:
        return result, None

    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Mastering cancelled")

    if reference_audio is not None:
        match = match_loudness_to_reference(
            result.audio,
            sample_rate,
            reference_audio,
            reference_sample_rate,
            ceiling_db=preset.limiter_ceiling,
        )
        result.audio = match.audio
        result.output_lufs = match.output_lufs
        result.peak_db = match.peak_db
        result.output_lra_lu = measure_loudness_range(match.audio, sample_rate)
        result.true_peak_dbtp = measure_true_peak_db(match.audio, sample_rate)
        result.short_term_max_lufs = measure_short_term_max_lufs(match.audio, sample_rate)
        result.momentary_max_lufs = measure_momentary_lufs(match.audio, sample_rate)
        result.target_lufs = match.reference_lufs
        return result, match
    return result, None


def _dynamic_eq_analysis_task(
    track_snapshots,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Analyze copied track buffers without touching widgets or live state."""
    suggestions = {}
    total = max(len(track_snapshots), 1)
    for position, (index, audio, sample_rate, name) in enumerate(track_snapshots, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("Dynamic EQ analysis cancelled")
        suggestions[index] = suggest_dynamic_eq_curve(audio, sample_rate, name)
        if progress_cb:
            progress_cb(int(position * 100 / total))
    return suggestions


def _gain_matched_dynamic_eq(
    audio: np.ndarray,
    sample_rate: int,
    suggestion: DynamicEQSuggestion,
    strength: float,
) -> Optional[np.ndarray]:
    if suggestion is None or not suggestion.bands:
        return None
    processed = apply_dynamic_eq(audio, sample_rate, suggestion.bands, strength=strength)
    source_rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    output_rms = float(np.sqrt(np.mean(np.square(processed)))) if processed.size else 0.0
    if source_rms > 1e-9 and output_rms > 1e-9:
        processed = processed * (source_rms / output_rms)
    return processed


def _dynamic_eq_operation_task(
    operation_snapshots,
    strength,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Apply copied EQ buffers off-thread and return replacements by track index."""
    processed = {}
    total = max(len(operation_snapshots), 1)
    for position, (index, audio, sample_rate, suggestion) in enumerate(
        operation_snapshots, 1
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("Dynamic EQ operation cancelled")
        output = _gain_matched_dynamic_eq(audio, sample_rate, suggestion, strength)
        if output is not None:
            processed[index] = output
        if progress_cb:
            progress_cb(int(position * 100 / total))
    return processed


def _decode_mixer_track_task(
    file_path: str,
    project_sample_rate: int,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Decode and resample a track before handing it back to the GUI."""
    source_audio, source_sample_rate = decode_audio_file(
        file_path,
        target_channels=2,
        progress_cb=(
            (lambda value: progress_cb(int(value * 0.7)))
            if progress_cb else None
        ),
        cancel_event=cancel_event,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Mixer import cancelled")
    if progress_cb:
        progress_cb(75)
    audio = prepare_audio_buffer(
        source_audio,
        source_sample_rate,
        project_sample_rate,
        target_channels=2,
    )
    if progress_cb:
        progress_cb(100)
    return {
        "audio": audio,
        "sample_rate": project_sample_rate,
        "source_sample_rate": source_sample_rate,
        "source_frames": len(source_audio),
    }


def _reference_track_task(
    file_path: str,
    name: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Decode and analyze a loudness reference without blocking the mixer."""
    audio, sample_rate = decode_audio_file(
        file_path,
        target_channels=2,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Reference loading cancelled")
    if progress_cb:
        progress_cb(80)
    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "name": name,
        "lufs": measure_lufs(audio, sample_rate),
        "profile": measure_short_term_lufs(audio, sample_rate),
    }


# ── Mixer Track Strip ─────────────────────────────────────────────────────────

class MixerTrackStrip(QFrame):
    """Single track in the mixer with waveform, volume, pan, mute/solo."""

    volume_changed = Signal(int, float)
    pan_changed = Signal(int, float)
    mute_changed = Signal(int, bool)
    solo_changed = Signal(int, bool)
    remove_requested = Signal(int)

    def __init__(self, track_idx: int, name: str,
                 audio: Optional[np.ndarray] = None,
                 sr: int = 44100, parent=None):
        super().__init__(parent)
        self.track_idx = track_idx
        self.name = name
        self.audio = audio
        self.sr = sr
        self._volume = 1.0
        self._pan = 0.0
        self._muted = False
        self._soloed = False

        t = ThemeEngine.get_colors()
        self.setStyleSheet(f"""
            MixerTrackStrip {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
        """)
        self.setMinimumHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Track name
        name_label = ElidedLabel(name, minimum_width=80)
        self._name_label = name_label
        name_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 8.25pt;")
        layout.addWidget(name_label)

        # Mini waveform
        self._waveform = MiniWaveform()
        if audio is not None:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            self._waveform.load_audio(mono, sr)
        self._waveform.setMinimumWidth(160)
        layout.addWidget(self._waveform)

        # Volume
        vol_col = QVBoxLayout()
        vol_col.setSpacing(1)
        vl = QLabel("Vol")
        vl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        vl.setAlignment(Qt.AlignCenter)
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 150)
        self._vol_slider.setValue(100)
        self._vol_slider.setMinimumWidth(80)
        self._vol_slider.setMinimumHeight(14)
        self._vol_val = QLabel("100%")
        self._vol_val.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        self._vol_val.setAlignment(Qt.AlignCenter)
        self._vol_slider.valueChanged.connect(self._on_vol)
        vol_col.addWidget(vl)
        vol_col.addWidget(self._vol_slider)
        vol_col.addWidget(self._vol_val)
        layout.addLayout(vol_col)

        # Pan
        pan_col = QVBoxLayout()
        pan_col.setSpacing(1)
        pl = QLabel("Pan")
        pl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        pl.setAlignment(Qt.AlignCenter)
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(-100, 100)
        self._pan_slider.setValue(0)
        self._pan_slider.setMinimumWidth(80)
        self._pan_slider.setMinimumHeight(14)
        self._pan_val = QLabel("C")
        self._pan_val.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        self._pan_val.setAlignment(Qt.AlignCenter)
        self._pan_slider.valueChanged.connect(self._on_pan)
        pan_col.addWidget(pl)
        pan_col.addWidget(self._pan_slider)
        pan_col.addWidget(self._pan_val)
        layout.addLayout(pan_col)

        # M/S buttons
        btn_style = f"""
            QPushButton {{
                background: {t['background']}; color: {t['text_secondary']};
                border: 1px solid {t['border']}; border-radius: 3px;
                font-size: 6.75pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
                QPushButton:checked {{ color: {Palette.CRUST}; border: none; }}
        """
        self._mute_btn = QPushButton("M")
        self._mute_btn.setMinimumSize(24, 20)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setStyleSheet(
            btn_style
            + f"QPushButton:checked {{ background: {Palette.RED}; color: {Palette.CRUST}; }}"
        )
        self._mute_btn.clicked.connect(self._on_mute)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setMinimumSize(24, 20)
        self._solo_btn.setCheckable(True)
        self._solo_btn.setStyleSheet(
            btn_style
            + f"QPushButton:checked {{ background: {Palette.YELLOW}; color: {Palette.CRUST}; }}"
        )
        self._solo_btn.clicked.connect(self._on_solo)

        self._remove_btn = QPushButton("X")
        self._remove_btn.setMinimumSize(24, 20)
        self._remove_btn.setStyleSheet(btn_style)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.track_idx))

        layout.addWidget(self._mute_btn)
        layout.addWidget(self._solo_btn)
        layout.addWidget(self._remove_btn)

        install_accessibility(
            self,
            f"Track {name}",
            named_controls=[
                (self._vol_slider, f"Track {name} volume", "Adjusts track volume from 0% to 150%."),
                (self._pan_slider, f"Track {name} pan", "Adjusts stereo pan from full left to full right."),
                (self._mute_btn, f"Mute track {name}", "Toggles muting of this track."),
                (self._solo_btn, f"Solo track {name}", "Solos this track, silencing all others."),
                (self._remove_btn, f"Remove track {name}", "Removes this track from the mixer."),
            ],
            tab_order=[self._vol_slider, self._pan_slider, self._mute_btn, self._solo_btn, self._remove_btn],
        )

    def _on_vol(self, val):
        self._volume = val / 100.0
        self._vol_val.setText(f"{val}%")
        self.volume_changed.emit(self.track_idx, self._volume)

    def _on_pan(self, val):
        self._pan = val / 100.0
        self._pan_val.setText("C" if val == 0 else f"L{abs(val)}" if val < 0 else f"R{val}")
        self.pan_changed.emit(self.track_idx, self._pan)

    def _on_mute(self):
        self._muted = self._mute_btn.isChecked()
        self.mute_changed.emit(self.track_idx, self._muted)

    def _on_solo(self):
        self._soloed = self._solo_btn.isChecked()
        self.solo_changed.emit(self.track_idx, self._soloed)

    def set_mix_state(
        self,
        *,
        volume: float = 1.0,
        pan: float = 0.0,
        muted: bool = False,
        soloed: bool = False,
    ):
        """Restore strip controls without emitting user-change signals."""
        self._volume = max(0.0, min(1.5, float(volume)))
        self._pan = max(-1.0, min(1.0, float(pan)))
        self._muted = bool(muted)
        self._soloed = bool(soloed)
        self._vol_slider.blockSignals(True)
        self._pan_slider.blockSignals(True)
        self._mute_btn.blockSignals(True)
        self._solo_btn.blockSignals(True)
        try:
            self._vol_slider.setValue(round(self._volume * 100))
            self._pan_slider.setValue(round(self._pan * 100))
            self._mute_btn.setChecked(self._muted)
            self._solo_btn.setChecked(self._soloed)
        finally:
            self._vol_slider.blockSignals(False)
            self._pan_slider.blockSignals(False)
            self._mute_btn.blockSignals(False)
            self._solo_btn.blockSignals(False)
        vol_value = round(self._volume * 100)
        pan_value = round(self._pan * 100)
        self._vol_val.setText(f"{vol_value}%")
        self._pan_val.setText(
            "C" if pan_value == 0
            else f"L{abs(pan_value)}" if pan_value < 0
            else f"R{pan_value}"
        )

    @property
    def volume(self): return self._volume
    @property
    def pan(self): return self._pan
    @property
    def is_muted(self): return self._muted
    @property
    def is_soloed(self): return self._soloed


# ── Mixer View ─────────────────────────────────────────────────────────────────

class MixerView(QWidget):
    """Multi-track mixer with mastering and export."""

    def __init__(
        self,
        parent=None,
        project_sample_rate: Optional[int] = None,
        toast_mgr=None,
    ):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        from core.settings import Settings

        self._settings = Settings()
        if project_sample_rate is None:
            project_sample_rate = self._settings.get("general.sample_rate", 48000)
        try:
            self._project_sample_rate = validate_sample_rate(project_sample_rate)
        except ValueError:
            self._project_sample_rate = 48000
        self._strips: list[MixerTrackStrip] = []
        self._tracks: list[dict] = []  # normalized project-rate buffers
        self._dynamic_eq_suggestions: dict[int, DynamicEQSuggestion] = {}
        # Pre-EQ audio, kept so Preview and Apply are both reversible.
        self._dynamic_eq_originals: dict[int, np.ndarray] = {}
        self._dynamic_eq_previewing = False
        self._dynamic_eq_applied = False
        self._master_worker: Optional[InferenceWorker] = None
        self._export_worker: Optional[InferenceWorker] = None
        self._dynamic_eq_worker: Optional[InferenceWorker] = None
        self._dynamic_eq_operation_worker: Optional[InferenceWorker] = None
        self._import_worker: Optional[InferenceWorker] = None
        self._import_queue: list[str] = []
        self._reference_worker: Optional[InferenceWorker] = None
        self._worker_references: set[InferenceWorker] = set()
        self._dynamic_eq_analysis_token = 0
        self._dynamic_eq_operation_token = 0
        self._dynamic_eq_operation_mode = ""
        self._master_preset_name = "Balanced"
        self._master_sample_rate = self._project_sample_rate
        self._reference_audio: Optional[np.ndarray] = None
        self._reference_sr: int = 44100
        self._reference_name: str = ""
        self._last_loudness_match: Optional[LoudnessMatchResult] = None
        self._syncing_lufs_target = False

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Top: Track list ────────────────────────────────────────────────
        tracks_header = QHBoxLayout()
        tl = QLabel("Tracks")
        tl.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9.75pt;")
        self._project_rate_label = QLabel(
            f"Project: {self._project_sample_rate / 1000:g} kHz stereo"
        )
        self._project_rate_label.setStyleSheet(
            f"color: {t['text_secondary']}; font-size: 7.5pt;"
        )

        self._add_btn = QPushButton("+ Import Track")
        self._add_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: {t['background']}; border: none;
                border-radius: 4px; padding: 5px 12px;
                font-size: 8.25pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._add_btn.clicked.connect(self._on_import_track)

        eq_btn_style = f"""
            QPushButton {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 5px 12px; font-size: 8.25pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
            QPushButton:disabled {{ color: {Palette.OVERLAY0}; border-color: {t['border']}; }}
        """

        # Analyze, Preview, Apply and Revert are separate: analysis never
        # mutates a track, preview is gain-matched and reversible, and the
        # originals are always recoverable.
        self._dynamic_eq_btn = QPushButton("Analyze Dynamic EQ")
        self._dynamic_eq_btn.setStyleSheet(eq_btn_style)
        self._dynamic_eq_btn.setEnabled(False)
        self._dynamic_eq_btn.clicked.connect(self._on_suggest_dynamic_eq)

        self._dynamic_eq_preview_btn = QPushButton("Preview EQ")
        self._dynamic_eq_preview_btn.setStyleSheet(eq_btn_style)
        self._dynamic_eq_preview_btn.setCheckable(True)
        self._dynamic_eq_preview_btn.setEnabled(False)
        self._dynamic_eq_preview_btn.toggled.connect(self._on_preview_dynamic_eq)

        self._dynamic_eq_apply_btn = QPushButton("Apply EQ")
        self._dynamic_eq_apply_btn.setStyleSheet(eq_btn_style)
        self._dynamic_eq_apply_btn.setEnabled(False)
        self._dynamic_eq_apply_btn.clicked.connect(self._on_apply_dynamic_eq)

        self._dynamic_eq_revert_btn = QPushButton("Revert EQ")
        self._dynamic_eq_revert_btn.setStyleSheet(eq_btn_style)
        self._dynamic_eq_revert_btn.setEnabled(False)
        self._dynamic_eq_revert_btn.clicked.connect(self._on_revert_dynamic_eq)

        tracks_header.addWidget(tl)
        tracks_header.addWidget(self._project_rate_label)
        tracks_header.addStretch()
        tracks_header.addWidget(self._dynamic_eq_btn)
        tracks_header.addWidget(self._dynamic_eq_preview_btn)
        tracks_header.addWidget(self._dynamic_eq_apply_btn)
        tracks_header.addWidget(self._dynamic_eq_revert_btn)
        tracks_header.addWidget(self._add_btn)
        layout.addLayout(tracks_header)

        # Track strips scroll area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMaximumHeight(350)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._strips_container = QWidget()
        self._strips_layout = QVBoxLayout(self._strips_container)
        self._strips_layout.setContentsMargins(0, 0, 0, 0)
        self._strips_layout.setSpacing(4)
        self._tracks_empty = EmptyStateWidget(
            "No mixer tracks yet",
            "Import an audio track to start balancing levels, pan, and mastering.",
            "Import track",
        )
        self._tracks_empty.action_requested.connect(self._add_btn.click)
        self._strips_layout.addWidget(self._tracks_empty)
        self._strips_layout.addStretch()

        self._scroll.setWidget(self._strips_container)
        layout.addWidget(self._scroll)

        # ── Middle: Mastering ──────────────────────────────────────────────
        master_frame = QFrame()
        master_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        master_layout = QHBoxLayout(master_frame)
        master_layout.setContentsMargins(12, 8, 12, 8)
        master_layout.setSpacing(12)

        ml = QLabel("Mastering:")
        ml.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt; border: none;")
        master_layout.addWidget(ml)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESETS.keys())
        self._preset_combo.setCurrentText("Balanced")
        self._preset_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 10px; font-size: 8.25pt; min-width: 140px;
            }}
        """)
        master_layout.addWidget(self._preset_combo)

        # Target LUFS
        tlufs = QLabel("Target:")
        tlufs.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
        self._target_combo = QComboBox()
        for target in LUFS_TARGETS.values():
            self._target_combo.addItem(target.label, target.key)
        self._target_combo.addItem("Custom", "custom")
        self._target_combo.setCurrentIndex(0)
        self._target_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 10px; font-size: 8.25pt; min-width: 220px;
            }}
        """)
        self._target_combo.currentIndexChanged.connect(self._on_lufs_target_changed)
        target_key = self._settings.get("production.mastering_target", "streaming")
        target_key = "streaming" if target_key == "spotify" else target_key
        target_index = self._target_combo.findData(target_key)
        if target_index >= 0:
            self._target_combo.blockSignals(True)
            self._target_combo.setCurrentIndex(target_index)
            self._target_combo.blockSignals(False)

        ll = QLabel("LUFS:")
        ll.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
        self._lufs_spin = QDoubleSpinBox()
        self._lufs_spin.setRange(-30.0, -6.0)
        self._lufs_spin.setValue(-14.0)
        self._lufs_spin.setSuffix(" LUFS")
        self._lufs_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 8.25pt;
            }}
        """)
        self._lufs_spin.valueChanged.connect(self._on_lufs_spin_changed)
        master_layout.addWidget(tlufs)
        master_layout.addWidget(self._target_combo)
        master_layout.addWidget(ll)
        master_layout.addWidget(self._lufs_spin)

        ms_style = f"""
            QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 8.25pt;
            }}
        """
        mid_label = QLabel("Mid:")
        mid_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
        self._mid_gain_spin = QDoubleSpinBox()
        self._mid_gain_spin.setRange(-6.0, 6.0)
        self._mid_gain_spin.setSingleStep(0.5)
        self._mid_gain_spin.setSuffix(" dB")
        self._mid_gain_spin.setMinimumWidth(78)
        self._mid_gain_spin.setStyleSheet(ms_style)

        side_label = QLabel("Side:")
        side_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
        self._side_gain_spin = QDoubleSpinBox()
        self._side_gain_spin.setRange(-6.0, 6.0)
        self._side_gain_spin.setSingleStep(0.5)
        self._side_gain_spin.setSuffix(" dB")
        self._side_gain_spin.setMinimumWidth(78)
        self._side_gain_spin.setStyleSheet(ms_style)

        master_layout.addWidget(mid_label)
        master_layout.addWidget(self._mid_gain_spin)
        master_layout.addWidget(side_label)
        master_layout.addWidget(self._side_gain_spin)

        self._ref_btn = QPushButton("Load Ref")
        self._ref_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 5px 10px; font-size: 8.25pt; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """)
        self._ref_btn.clicked.connect(self._on_load_reference)
        master_layout.addWidget(self._ref_btn)

        self._master_btn = QPushButton("Master + Export")
        self._master_btn.setProperty("class", "success")
        self._master_btn.setEnabled(False)
        self._master_btn.clicked.connect(self._on_master_export)
        master_layout.addWidget(self._master_btn)

        master_layout.addStretch()

        # LUFS meter display
        self._reference_label = QLabel("Ref: none")
        self._reference_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border: none;")
        master_layout.addWidget(self._reference_label)

        self._lufs_label = QLabel("")
        self._lufs_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border: none;")
        master_layout.addWidget(self._lufs_label)

        layout.addWidget(master_frame)

        # ── Bottom: Master output waveform ─────────────────────────────────
        self._master_waveform = WaveformWidget()
        self._master_waveform.empty_action_requested.connect(self._add_btn.click)
        layout.addWidget(self._master_waveform, 1)

        # Status
        self._status = QLabel("Import audio tracks to begin mixing")
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(self._status)

        self._operation_progress = OperationProgressWidget()
        self._operation_progress.cancel_requested.connect(
            self._cancel_active_operation
        )
        layout.addWidget(self._operation_progress)
        if target_index >= 0:
            self._on_lufs_target_changed()

        install_accessibility(
            self,
            "Mixer",
            named_controls=[
                (self._add_btn, "Import track", "Opens file dialog to import an audio track."),
                (self._dynamic_eq_btn, "Analyze dynamic EQ", "Analyzes tracks and suggests per-band dynamic EQ curves without changing them."),
                (self._dynamic_eq_preview_btn, "Preview dynamic EQ", "Toggles a gain-matched preview of the suggested curves."),
                (self._dynamic_eq_apply_btn, "Apply dynamic EQ", "Commits the suggested curves; originals stay recoverable."),
                (self._dynamic_eq_revert_btn, "Revert dynamic EQ", "Restores the original audio for every affected track."),
                (self._preset_combo, "Mastering preset", "Selects a mastering preset profile."),
                (self._target_combo, "LUFS delivery target", "Selects a loudness target standard."),
                (self._lufs_spin, "Target LUFS", "Sets the target loudness in LUFS."),
                (self._mid_gain_spin, "Mid gain trim", "Adjusts mid-channel gain in dB."),
                (self._side_gain_spin, "Side gain trim", "Adjusts side-channel gain in dB."),
                (self._ref_btn, "Load reference track", "Loads a loudness reference track for mastering comparison."),
                (self._master_btn, "Master and export", "Masters the mix and opens export dialog."),
                (self._operation_progress.cancel_button, "Cancel mixer operation", "Cancels the running import, EQ, reference, mastering, or export operation."),
            ],
            tab_order=[
                self._add_btn, self._dynamic_eq_btn,
                self._dynamic_eq_preview_btn, self._dynamic_eq_apply_btn,
                self._dynamic_eq_revert_btn,
                self._preset_combo, self._target_combo, self._lufs_spin,
                self._mid_gain_spin, self._side_gain_spin,
                self._ref_btn, self._master_btn,
            ],
        )
        self._settings.on_change(self._on_settings_change)

    def _on_settings_change(self, key: str, value, _old_value):
        """Apply production defaults to the live mastering controls."""
        if key == "production.mastering_target":
            target_key = "streaming" if value == "spotify" else value
            index = self._target_combo.findData(target_key)
            if index >= 0:
                self._target_combo.setCurrentIndex(index)

    # ── Track Management ───────────────────────────────────────────────────────

    def _report_error(self, message: str):
        """Keep inline mixer state and shared notification history aligned."""
        self._status.setText(message)
        if self.toast_mgr is not None:
            self.toast_mgr.error(message)

    def _active_operation(self):
        """Return the current worker, label, and related action button."""
        operations = (
            (self._master_worker, "Mastering", self._master_btn),
            (self._export_worker, "Master export", self._master_btn),
            (self._dynamic_eq_operation_worker, "Dynamic EQ", self._dynamic_eq_apply_btn),
            (self._dynamic_eq_worker, "Dynamic EQ analysis", self._dynamic_eq_btn),
            (self._import_worker, "Audio import", self._add_btn),
            (self._reference_worker, "Reference load", self._ref_btn),
        )
        return next((item for item in operations if item[0] is not None), None)

    def _cancel_active_operation(self):
        """Request cancellation without waiting on the worker thread."""
        active = self._active_operation()
        if active is None:
            self._operation_progress.finish()
            return
        worker, label, button = active
        if worker is self._import_worker:
            self._import_queue.clear()
        self._operation_progress.mark_cancelling()
        worker.cancel()
        button.setEnabled(False)
        self._status.setText(f"Cancelling {label.lower()}...")

    def _start_operation_progress(self, label: str):
        self._operation_progress.start(label, determinate=True)

    def _on_operation_progress(self, label: str, percent: int):
        self._operation_progress.set_progress(percent, label)
        self._status.setText(f"{label}... {percent}%")

    def _on_operation_step(self, message: str):
        self._operation_progress.set_step(message)
        self._status.setText(message)

    def _finish_operation_progress(self):
        self._operation_progress.finish()

    def add_track(self, name: str, audio: np.ndarray, sr: int = 44100):
        """Add an audio track to the mixer."""
        source_sr = validate_sample_rate(sr)
        source_frames = len(audio)
        prepared = prepare_audio_buffer(
            audio,
            source_sr,
            self._project_sample_rate,
            target_channels=2,
        )
        return self._append_prepared_track(
            name,
            prepared,
            source_sr=source_sr,
            source_frames=source_frames,
        )

    def _append_prepared_track(
        self,
        name: str,
        audio: np.ndarray,
        *,
        source_sr: int,
        source_frames: int,
    ) -> int:
        """Append a project-rate stereo buffer after background preparation."""
        self._dynamic_eq_analysis_token += 1
        if self._dynamic_eq_worker is not None and self._dynamic_eq_worker.isRunning():
            self._dynamic_eq_worker.cancel()
        self._invalidate_dynamic_eq_operation()
        return self._insert_prepared_track(
            len(self._strips),
            name,
            audio,
            source_sr=source_sr,
            source_frames=source_frames,
        )

    def _insert_prepared_track(
        self,
        index: int,
        name: str,
        audio: np.ndarray,
        *,
        source_sr: int,
        source_frames: int,
        strip_state: Optional[dict] = None,
    ) -> int:
        """Insert a validated project-rate track at a specific position."""
        prepared = validate_audio_buffer(audio)
        if prepared.ndim != 2 or prepared.shape[1] != 2:
            prepared = normalize_channel_layout(prepared, target_channels=2)
        prepared = np.ascontiguousarray(prepared, dtype=np.float32)
        source_sr = validate_sample_rate(source_sr)
        idx = max(0, min(int(index), len(self._strips)))
        self._tracks.insert(idx, {
            "name": name,
            "audio": prepared,
            "sr": self._project_sample_rate,
            "source_sr": source_sr,
            "source_frames": int(source_frames),
        })

        strip = MixerTrackStrip(
            idx,
            name,
            prepared,
            self._project_sample_rate,
        )
        if strip_state:
            strip.set_mix_state(**strip_state)
        strip.remove_requested.connect(self._on_remove_track)
        strip.volume_changed.connect(lambda *_: self._update_mix_state())
        strip.pan_changed.connect(lambda *_: self._update_mix_state())
        strip.mute_changed.connect(lambda *_: self._update_mix_state())
        strip.solo_changed.connect(lambda *_: self._update_mix_state())

        self._strips.insert(idx, strip)
        self._strips_layout.insertWidget(idx, strip)
        for track_idx, current_strip in enumerate(self._strips):
            current_strip.track_idx = track_idx
        self._master_btn.setEnabled(True)
        self._tracks_empty.hide()
        self._update_mix_state()
        return idx

    def select_track(self, index: int) -> bool:
        """Focus one track strip so a routed artifact lands somewhere visible."""
        if not (0 <= index < len(self._strips)):
            return False
        strip = self._strips[index]
        self._scroll.ensureWidgetVisible(strip)
        strip.setFocus()
        self._selected_track_index = index
        return True

    @property
    def selected_track_index(self) -> int:
        return getattr(self, "_selected_track_index", -1)

    def add_track_from_file(
        self,
        file_path: str,
        on_complete: Optional[Callable[[bool, int], None]] = None,
    ):
        """Import and prepare an audio file on an inference worker."""
        if self._import_worker is not None and self._import_worker.isRunning():
            self._status.setText("An audio import is already in progress")
            if on_complete:
                on_complete(False, -1)
            return None

        path = str(file_path)
        name = os.path.splitext(os.path.basename(path))[0]
        worker = InferenceWorker(
            _decode_mixer_track_task,
            path,
            self._project_sample_rate,
            job_kind="mixer_import",
            job_label=f"Mixer import: {name}",
            job_inputs={"path": path, "project_sample_rate": self._project_sample_rate},
        )
        worker.progress.connect(
            lambda pct, n=name: self._on_operation_progress(f"Importing {n}", pct)
        )
        worker.step_info.connect(self._on_operation_step)
        worker.finished.connect(
            lambda payload, n=name, callback=on_complete:
            self._on_import_finished(n, payload, callback)
        )
        worker.error.connect(
            lambda message, callback=on_complete:
            self._on_import_error(message, callback)
        )
        worker.cancelled.connect(
            lambda callback=on_complete: self._on_import_cancelled(callback)
        )
        self._import_worker = worker
        self._add_btn.setEnabled(False)
        self._status.setText(f"Importing {name}...")
        self._start_operation_progress(f"Importing {name}")
        worker.start()
        return worker

    def _on_import_finished(
        self,
        name: str,
        payload: dict,
        on_complete: Optional[Callable[[bool, int], None]],
    ):
        worker = self._import_worker
        self._settle_worker(worker)
        self._import_worker = None
        self._finish_operation_progress()
        try:
            index = self._append_prepared_track(
                name,
                payload["audio"],
                source_sr=payload["source_sample_rate"],
                source_frames=payload["source_frames"],
            )
            source_sr = int(payload["source_sample_rate"])
            self._status.setText(
                f"Added track: {name} ({len(payload['audio']) / self._project_sample_rate:.1f}s, "
                f"{source_sr / 1000:g}->{self._project_sample_rate / 1000:g} kHz)"
            )
            if on_complete:
                on_complete(True, index)
        except Exception as exc:
            self._report_error(f"Import error: {exc}")
            if on_complete:
                on_complete(False, -1)
        finally:
            self._update_mix_state()

    def _on_import_error(
        self,
        message: str,
        on_complete: Optional[Callable[[bool, int], None]],
    ):
        worker = self._import_worker
        self._settle_worker(worker)
        self._import_worker = None
        self._finish_operation_progress()
        self._report_error(f"Import error: {message}")
        if on_complete:
            on_complete(False, -1)
        self._update_mix_state()

    def _on_import_cancelled(self, on_complete: Optional[Callable[[bool, int], None]]):
        worker = self._import_worker
        self._settle_worker(worker)
        self._import_worker = None
        self._finish_operation_progress()
        self._status.setText("Audio import cancelled")
        if on_complete:
            on_complete(False, -1)
        self._update_mix_state()

    def set_reference_track(self, name: str, audio: np.ndarray, sr: int = 44100,
                            path: str = ""):
        """Set a loudness reference track for mastering."""
        reference_sr = validate_sample_rate(sr)
        reference_audio = normalize_channel_layout(
            audio,
            target_channels=2,
        )
        self._apply_reference_analysis(
            name or os.path.basename(path) or "Reference",
            reference_audio,
            reference_sr,
            measure_lufs(reference_audio, reference_sr),
            measure_short_term_lufs(reference_audio, reference_sr),
        )

    def _apply_reference_analysis(
        self,
        name: str,
        audio: np.ndarray,
        sample_rate: int,
        ref_lufs: float,
        profile,
    ):
        """Apply already-computed reference analysis on the Qt thread."""
        self._reference_sr = validate_sample_rate(sample_rate)
        self._reference_audio = normalize_channel_layout(
            audio,
            target_channels=2,
        )
        self._reference_name = name or "Reference"
        if ref_lufs > -60:
            self._lufs_spin.setValue(max(self._lufs_spin.minimum(), min(self._lufs_spin.maximum(), ref_lufs)))
            self._set_target_combo_key("custom")

        if profile:
            low = min(point.lufs for point in profile)
            high = max(point.lufs for point in profile)
            self._reference_label.setText(
                f"Ref: {self._reference_name} {ref_lufs:.1f} LUFS | ST {low:.1f}..{high:.1f}"
            )
        else:
            self._reference_label.setText(f"Ref: {self._reference_name} {ref_lufs:.1f} LUFS")

    def _on_load_reference(self):
        path, _ = open_audio_file(
            self,
            "Load Loudness Reference",
            operation_kind="mixer_reference",
            dialog=QFileDialog,
        )
        if not path:
            return

        name = os.path.splitext(os.path.basename(path))[0]
        if self._reference_worker is not None and self._reference_worker.isRunning():
            self._status.setText("A reference load is already in progress")
            return
        worker = InferenceWorker(
            _reference_track_task,
            str(path),
            name,
            job_kind="mixer_reference_import",
            job_label=f"Mixer reference: {name}",
            job_inputs={"path": str(path)},
        )
        worker.progress.connect(
            lambda pct, n=name: self._on_operation_progress(
                f"Loading reference {n}", pct
            )
        )
        worker.step_info.connect(self._on_operation_step)
        worker.finished.connect(
            lambda payload, n=name: self._on_reference_finished(n, payload)
        )
        worker.error.connect(self._on_reference_error)
        worker.cancelled.connect(self._on_reference_cancelled)
        self._reference_worker = worker
        self._ref_btn.setEnabled(False)
        self._status.setText(f"Loading loudness reference: {name}...")
        self._start_operation_progress(f"Loading reference {name}")
        worker.start()

    def _on_reference_finished(self, name: str, payload: dict):
        worker = self._reference_worker
        self._settle_worker(worker)
        self._reference_worker = None
        self._finish_operation_progress()
        try:
            self._apply_reference_analysis(
                name,
                payload["audio"],
                payload["sample_rate"],
                payload["lufs"],
                payload["profile"],
            )
            self._status.setText(f"Loaded loudness reference: {name}")
        except Exception as exc:
            self._report_error(f"Reference load error: {exc}")
        finally:
            self._ref_btn.setEnabled(True)

    def _on_reference_error(self, message: str):
        worker = self._reference_worker
        self._settle_worker(worker)
        self._reference_worker = None
        self._finish_operation_progress()
        self._ref_btn.setEnabled(True)
        self._report_error(f"Reference load error: {message}")

    def _on_reference_cancelled(self):
        worker = self._reference_worker
        self._settle_worker(worker)
        self._reference_worker = None
        self._finish_operation_progress()
        self._ref_btn.setEnabled(True)
        self._status.setText("Reference load cancelled")

    def _on_import_track(self):
        paths, _ = open_audio_files(
            self,
            "Import Audio Tracks",
            operation_kind="mixer_audio_import",
            dialog=QFileDialog,
        )
        if not paths:
            return
        self._import_queue = paths
        self._start_next_queued_import()

    def _start_next_queued_import(self):
        """Import a multi-selection serially so each worker owns the decoder."""
        if not self._import_queue:
            self._update_mix_state()
            return
        path = self._import_queue.pop(0)
        self.add_track_from_file(
            path,
            on_complete=lambda _success, _index: self._start_next_queued_import(),
        )

    def _on_remove_track(self, idx: int):
        if 0 <= idx < len(self._strips):
            self._dynamic_eq_analysis_token += 1
            if self._dynamic_eq_worker is not None and self._dynamic_eq_worker.isRunning():
                self._dynamic_eq_worker.cancel()
            self._invalidate_dynamic_eq_operation()
            removed_idx = idx
            old_suggestions = dict(self._dynamic_eq_suggestions)
            strip = self._strips[idx]
            track = self._tracks[idx]
            snapshot = {
                "index": removed_idx,
                "track": {
                    **track,
                    "audio": np.asarray(track["audio"], dtype=np.float32).copy(),
                },
                "strip_state": {
                    "volume": strip.volume,
                    "pan": strip.pan,
                    "muted": strip.is_muted,
                    "soloed": strip.is_soloed,
                },
                "suggestions": old_suggestions,
                "originals": {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in self._dynamic_eq_originals.items()
                },
                "restored": False,
            }
            self._strips_layout.removeWidget(strip)
            strip.deleteLater()
            self._strips.pop(idx)
            self._tracks.pop(idx)

            # Re-index remaining strips
            for i, s in enumerate(self._strips):
                s.track_idx = i

            old_track_count = len(self._tracks) + 1
            old_originals = dict(self._dynamic_eq_originals)
            self._dynamic_eq_suggestions = {}
            self._dynamic_eq_originals = {}
            for new_idx, old_idx in enumerate(
                old_idx for old_idx in range(old_track_count) if old_idx != removed_idx
            ):
                if old_idx in old_suggestions:
                    self._dynamic_eq_suggestions[new_idx] = old_suggestions[old_idx]
                if old_idx in old_originals:
                    self._dynamic_eq_originals[new_idx] = old_originals[old_idx]
            self._master_btn.setEnabled(len(self._strips) > 0)
            self._tracks_empty.setVisible(not self._strips)
            self._update_mix_state()
            if self.toast_mgr:
                self.toast_mgr.info(
                    "Mixer track removed.",
                    duration_ms=8000,
                    action_label="Undo",
                    action_callback=lambda item=snapshot: self._restore_removed_track(item),
                )

    def _restore_removed_track(self, snapshot: dict):
        """Restore a removed track and its mixer/EQ state from memory."""
        if snapshot.get("restored"):
            return
        snapshot["restored"] = True
        track = snapshot["track"]
        self._dynamic_eq_analysis_token += 1
        if self._dynamic_eq_worker is not None and self._dynamic_eq_worker.isRunning():
            self._dynamic_eq_worker.cancel()
        self._invalidate_dynamic_eq_operation()
        self._insert_prepared_track(
            snapshot["index"],
            track["name"],
            track["audio"],
            source_sr=track["source_sr"],
            source_frames=track["source_frames"],
            strip_state=snapshot["strip_state"],
        )
        self._dynamic_eq_suggestions = dict(snapshot["suggestions"])
        self._dynamic_eq_originals = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in snapshot["originals"].items()
        }
        self._update_mix_state()
        if self.toast_mgr:
            self.toast_mgr.success("Mixer track restored.")

    def _update_mix_state(self):
        """Update master button state."""
        has_tracks = len(self._strips) > 0
        has_suggestions = bool(self._dynamic_eq_suggestions)
        master_busy = self._master_worker is not None and self._master_worker.isRunning()
        export_busy = self._export_worker is not None and self._export_worker.isRunning()
        analysis_busy = self._dynamic_eq_worker is not None and self._dynamic_eq_worker.isRunning()
        operation_busy = (
            self._dynamic_eq_operation_worker is not None
            and self._dynamic_eq_operation_worker.isRunning()
        )
        import_busy = self._import_worker is not None and self._import_worker.isRunning()
        if master_busy:
            self._master_btn.setEnabled(True)
            self._master_btn.setText("Cancel Mastering")
        elif export_busy:
            self._master_btn.setEnabled(True)
            self._master_btn.setText("Cancel Export")
        else:
            self._master_btn.setEnabled(
                has_tracks and not operation_busy and not import_busy
            )
            self._master_btn.setText("Master + Export")
        self._dynamic_eq_btn.setEnabled(
            has_tracks and not analysis_busy and not operation_busy and not import_busy
        )
        eq_controls_ready = not analysis_busy and not operation_busy and not import_busy
        self._dynamic_eq_preview_btn.setEnabled(
            has_tracks and has_suggestions and eq_controls_ready
        )
        self._dynamic_eq_apply_btn.setEnabled(
            has_tracks
            and has_suggestions
            and not self._dynamic_eq_previewing
            and eq_controls_ready
        )
        self._dynamic_eq_revert_btn.setEnabled(bool(self._dynamic_eq_originals))
        self._add_btn.setEnabled(not import_busy)

    def _settle_worker(self, worker: Optional[InferenceWorker]):
        """Keep a worker alive until its QThread exits without blocking Qt."""
        if worker is None:
            return
        self._worker_references.add(worker)
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._settle_worker(worker))
            return
        self._worker_references.discard(worker)

    def _set_target_combo_key(self, key: str):
        for idx in range(self._target_combo.count()):
            if self._target_combo.itemData(idx) == key:
                self._target_combo.setCurrentIndex(idx)
                return

    def _on_lufs_target_changed(self):
        key = self._target_combo.currentData()
        target = LUFS_TARGETS.get(key)
        if target is None:
            return
        self._syncing_lufs_target = True
        self._lufs_spin.setValue(target.lufs)
        self._syncing_lufs_target = False
        self._status.setText(f"Target loudness: {target.label}")

    def _on_lufs_spin_changed(self, value: float):
        if self._syncing_lufs_target:
            return
        for key, target in LUFS_TARGETS.items():
            if abs(value - target.lufs) < 0.05:
                self._set_target_combo_key(key)
                return
        self._set_target_combo_key("custom")

    DYNAMIC_EQ_STRENGTH = 0.75

    def _on_suggest_dynamic_eq(self):
        """Analyze only. Never mutates a track."""
        if not self._tracks:
            self._status.setText("Import audio tracks before dynamic EQ")
            return

        if self._dynamic_eq_worker is not None and self._dynamic_eq_worker.isRunning():
            return

        self._dynamic_eq_analysis_token += 1
        token = self._dynamic_eq_analysis_token
        snapshots = [
            (idx, track["audio"].copy(), int(track["sr"]), track["name"])
            for idx, track in enumerate(self._tracks)
        ]
        self._dynamic_eq_suggestions = {}
        self._status.setText("Analyzing dynamic EQ curves...")

        worker = InferenceWorker(
            _dynamic_eq_analysis_task,
            snapshots,
            job_kind="mixer_dynamic_eq_analysis",
            job_label="Mixer dynamic EQ analysis",
            job_inputs={"track_count": len(snapshots)},
        )
        worker.progress.connect(
            lambda pct, t=token: self._on_dynamic_eq_analysis_progress(t, pct)
        )
        worker.step_info.connect(self._on_operation_step)
        worker.finished.connect(
            lambda suggestions, t=token:
            self._on_dynamic_eq_analysis_finished(t, suggestions)
        )
        worker.error.connect(
            lambda message, t=token: self._on_dynamic_eq_analysis_error(t, message)
        )
        worker.cancelled.connect(
            lambda t=token: self._on_dynamic_eq_analysis_cancelled(t)
        )
        self._dynamic_eq_worker = worker
        self._start_operation_progress("Dynamic EQ analysis")
        self._update_mix_state()
        worker.start()

    def _on_dynamic_eq_analysis_progress(self, token: int, percent: int):
        if token == self._dynamic_eq_analysis_token:
            self._on_operation_progress("Dynamic EQ analysis", percent)

    def _on_dynamic_eq_analysis_finished(self, token: int, suggestions: dict):
        worker = self._dynamic_eq_worker
        self._settle_worker(worker)
        self._dynamic_eq_worker = None
        self._finish_operation_progress()
        if token != self._dynamic_eq_analysis_token:
            self._update_mix_state()
            return

        self._dynamic_eq_suggestions = dict(suggestions)
        summaries: list[str] = []
        for idx, suggestion in self._dynamic_eq_suggestions.items():
            if idx >= len(self._tracks):
                continue
            if suggestion.bands:
                moves = ", ".join(
                    f"{band.frequency_hz:.0f}Hz {band.gain_db:+.1f}dB"
                    for band in suggestion.bands[:3]
                )
                summaries.append(f"{self._tracks[idx]['name']}: {moves}")
            else:
                summaries.append(f"{self._tracks[idx]['name']}: balanced")
        self._status.setText(
            "Dynamic EQ suggested (not applied) - " + " | ".join(summaries[:3])
        )
        self._update_mix_state()

    def _on_dynamic_eq_analysis_error(self, token: int, message: str):
        worker = self._dynamic_eq_worker
        self._settle_worker(worker)
        self._dynamic_eq_worker = None
        self._finish_operation_progress()
        if token == self._dynamic_eq_analysis_token:
            self._dynamic_eq_suggestions = {}
            self._report_error(f"Dynamic EQ analysis error: {message}")
        self._update_mix_state()

    def _on_dynamic_eq_analysis_cancelled(self, token: int):
        worker = self._dynamic_eq_worker
        self._settle_worker(worker)
        self._dynamic_eq_worker = None
        self._finish_operation_progress()
        if token == self._dynamic_eq_analysis_token:
            self._status.setText("Dynamic EQ analysis cancelled")
        self._update_mix_state()

    def _dynamic_eq_processed(self, idx: int) -> Optional[np.ndarray]:
        """Gain-matched EQ result for one track, or None when there is nothing to do."""
        suggestion = self._dynamic_eq_suggestions.get(idx)
        if suggestion is None or not suggestion.bands:
            return None
        track = self._tracks[idx]
        source = self._dynamic_eq_originals.get(idx, track["audio"])
        return _gain_matched_dynamic_eq(
            source,
            track["sr"],
            suggestion,
            self.DYNAMIC_EQ_STRENGTH,
        )

    def _set_track_audio(self, idx: int, audio: np.ndarray):
        self._tracks[idx]["audio"] = audio
        if idx < len(self._strips):
            strip = self._strips[idx]
            strip.audio = audio
            mono = audio[:, 0] if audio.ndim == 2 else audio
            strip._waveform.load_audio(mono, self._tracks[idx]["sr"])

    def _on_preview_dynamic_eq(self, enabled: bool):
        """Toggle a reversible, gain-matched preview of the suggested curves."""
        if enabled:
            if not self._dynamic_eq_suggestions:
                self._dynamic_eq_preview_btn.setChecked(False)
                self._status.setText("Analyze dynamic EQ before previewing")
                return
            self._start_dynamic_eq_operation("preview")
        else:
            self._invalidate_dynamic_eq_operation()
            self._restore_dynamic_eq_originals()
            self._dynamic_eq_previewing = False
            self._status.setText("Preview off - original audio restored")
        self._update_mix_state()

    def _on_apply_dynamic_eq(self):
        """Commit the suggested curves. Originals stay recoverable via Revert."""
        if not self._dynamic_eq_suggestions:
            self._status.setText("Analyze dynamic EQ before applying")
            return
        self._start_dynamic_eq_operation("apply")
        self._update_mix_state()

    def _start_dynamic_eq_operation(self, mode: str):
        if (
            self._dynamic_eq_operation_worker is not None
            and self._dynamic_eq_operation_worker.isRunning()
        ):
            return

        snapshots = []
        for idx, track in enumerate(self._tracks):
            suggestion = self._dynamic_eq_suggestions.get(idx)
            if suggestion is None:
                continue
            source = self._dynamic_eq_originals.get(idx, track["audio"])
            if suggestion.bands:
                self._dynamic_eq_originals.setdefault(idx, track["audio"])
            snapshots.append((idx, source.copy(), int(track["sr"]), suggestion))

        self._dynamic_eq_operation_token += 1
        token = self._dynamic_eq_operation_token
        self._dynamic_eq_operation_mode = mode
        label = "preview" if mode == "preview" else "apply"
        self._status.setText(f"Dynamic EQ {label} in progress...")
        worker = InferenceWorker(
            _dynamic_eq_operation_task,
            snapshots,
            self.DYNAMIC_EQ_STRENGTH,
            job_kind="mixer_dynamic_eq",
            job_label=f"Mixer dynamic EQ {label}",
            job_inputs={"track_count": len(snapshots), "mode": mode},
        )
        worker.progress.connect(
            lambda pct, t=token, m=mode: self._on_dynamic_eq_operation_progress(t, m, pct)
        )
        worker.step_info.connect(self._on_operation_step)
        worker.finished.connect(
            lambda result, t=token: self._on_dynamic_eq_operation_finished(t, result)
        )
        worker.error.connect(
            lambda message, t=token: self._on_dynamic_eq_operation_error(t, message)
        )
        worker.cancelled.connect(
            lambda t=token: self._on_dynamic_eq_operation_cancelled(t)
        )
        self._dynamic_eq_operation_worker = worker
        self._start_operation_progress(
            f"Dynamic EQ {'preview' if mode == 'preview' else 'apply'}"
        )
        self._update_mix_state()
        worker.start()

    def _on_dynamic_eq_operation_progress(self, token: int, mode: str, percent: int):
        if token == self._dynamic_eq_operation_token:
            self._on_operation_progress(
                f"Dynamic EQ {'preview' if mode == 'preview' else 'apply'}",
                percent,
            )

    def _on_dynamic_eq_operation_finished(self, token: int, processed: dict):
        worker = self._dynamic_eq_operation_worker
        self._settle_worker(worker)
        self._dynamic_eq_operation_worker = None
        self._finish_operation_progress()
        if token != self._dynamic_eq_operation_token:
            self._update_mix_state()
            return

        mode = self._dynamic_eq_operation_mode
        applied = 0
        for idx, audio in processed.items():
            if 0 <= idx < len(self._tracks):
                self._set_track_audio(idx, audio)
                applied += 1

        if mode == "preview":
            self._dynamic_eq_previewing = applied > 0
            if not applied:
                self._dynamic_eq_preview_btn.blockSignals(True)
                self._dynamic_eq_preview_btn.setChecked(False)
                self._dynamic_eq_preview_btn.blockSignals(False)
            self._status.setText(
                f"Previewing gain-matched dynamic EQ on {applied} track(s)"
                if applied else "No EQ moves were suggested"
            )
        else:
            self._dynamic_eq_applied = applied > 0
            self._dynamic_eq_previewing = False
            self._dynamic_eq_preview_btn.blockSignals(True)
            self._dynamic_eq_preview_btn.setChecked(False)
            self._dynamic_eq_preview_btn.blockSignals(False)
            self._status.setText(
                f"Dynamic EQ applied to {applied} track(s) - Revert restores the originals"
                if applied else "No EQ moves were suggested"
            )
        self._update_mix_state()

    def _on_dynamic_eq_operation_error(self, token: int, message: str):
        worker = self._dynamic_eq_operation_worker
        self._settle_worker(worker)
        self._dynamic_eq_operation_worker = None
        self._finish_operation_progress()
        if token == self._dynamic_eq_operation_token:
            self._report_error(f"Dynamic EQ error: {message}")
        self._update_mix_state()

    def _on_dynamic_eq_operation_cancelled(self, token: int):
        worker = self._dynamic_eq_operation_worker
        self._settle_worker(worker)
        self._dynamic_eq_operation_worker = None
        self._finish_operation_progress()
        if token == self._dynamic_eq_operation_token:
            self._status.setText("Dynamic EQ cancelled")
        self._update_mix_state()

    def _invalidate_dynamic_eq_operation(self):
        self._dynamic_eq_operation_token += 1
        worker = self._dynamic_eq_operation_worker
        if worker is not None and worker.isRunning():
            worker.cancel()

    def _on_revert_dynamic_eq(self):
        """Restore every stored original."""
        self._invalidate_dynamic_eq_operation()
        restored = self._restore_dynamic_eq_originals()
        self._dynamic_eq_applied = False
        self._dynamic_eq_previewing = False
        self._dynamic_eq_preview_btn.blockSignals(True)
        self._dynamic_eq_preview_btn.setChecked(False)
        self._dynamic_eq_preview_btn.blockSignals(False)
        self._status.setText(
            f"Reverted dynamic EQ on {restored} track(s)" if restored
            else "Nothing to revert"
        )
        self._update_mix_state()

    def _restore_dynamic_eq_originals(self) -> int:
        restored = 0
        for idx, original in list(self._dynamic_eq_originals.items()):
            if idx < len(self._tracks):
                self._set_track_audio(idx, original)
                restored += 1
        self._dynamic_eq_originals = {}
        return restored

    # ── Mixing ─────────────────────────────────────────────────────────────────

    def _get_mixed_audio(self) -> Optional[np.ndarray]:
        """Mix all tracks according to current settings."""
        if not self._strips:
            return None

        prepared_tracks = []
        for track in self._tracks:
            audio = validate_audio_buffer(track["audio"])
            if (
                track["sr"] != self._project_sample_rate
                or audio.ndim != 2
                or audio.shape[1] != 2
            ):
                audio = prepare_audio_buffer(
                    audio,
                    track["sr"],
                    self._project_sample_rate,
                    target_channels=2,
                )
            prepared_tracks.append(audio)
        return mixdown_audio(
            [
                (
                    prepared_tracks[strip.track_idx]
                    if 0 <= strip.track_idx < len(prepared_tracks)
                    else None,
                    strip.volume,
                    strip.pan,
                    strip.is_muted,
                    strip.is_soloed,
                )
                for strip in self._strips
            ]
        )

    # ── Mastering + Export ─────────────────────────────────────────────────────

    def _on_master_export(self):
        if self._master_worker is not None and self._master_worker.isRunning():
            self._cancel_active_operation()
            return
        if self._export_worker is not None and self._export_worker.isRunning():
            self._cancel_active_operation()
            return
        if not self._tracks:
            self._status.setText("No audio to master")
            return

        track_snapshots = [
            (
                track["audio"].copy(),
                int(track["sr"]),
                float(strip.volume),
                float(strip.pan),
                bool(strip.is_muted),
                bool(strip.is_soloed),
            )
            for track, strip in zip(self._tracks, self._strips)
        ]

        sr = self._project_sample_rate

        # Get preset
        preset_name = self._preset_combo.currentText()
        preset = replace(PRESETS.get(preset_name, PRESETS["Balanced"]))
        preset.auto_eq = bool(self._settings.get("production.mastering_auto_eq", True))
        preset.auto_compress = bool(
            self._settings.get("production.mastering_auto_compress", True)
        )

        # Override target LUFS
        preset.target_lufs = self._lufs_spin.value()
        target = LUFS_TARGETS.get(self._target_combo.currentData())
        if target is not None and target.true_peak_dbtp is not None:
            preset.limiter_ceiling = target.true_peak_dbtp
        preset.ms_mid_gain_db = self._mid_gain_spin.value()
        preset.ms_side_gain_db = self._side_gain_spin.value()

        self._master_preset_name = preset_name
        self._master_sample_rate = sr
        reference_audio = (
            self._reference_audio.copy()
            if self._reference_audio is not None
            else None
        )
        reference_sr = self._reference_sr
        worker = InferenceWorker(
            _master_audio_task,
            track_snapshots,
            sr,
            preset,
            reference_audio,
            reference_sr,
            job_kind="mixer_mastering",
            job_label="Mixer mastering",
            job_inputs={
                "preset": preset_name,
                "sample_rate": sr,
                "has_reference": reference_audio is not None,
                "track_count": len(track_snapshots),
                "auto_eq": preset.auto_eq,
                "auto_compress": preset.auto_compress,
            },
        )
        worker.progress.connect(self._on_master_progress)
        worker.step_info.connect(self._on_master_step)
        worker.finished.connect(self._on_master_finished)
        worker.error.connect(self._on_master_error)
        worker.cancelled.connect(self._on_master_cancelled)
        self._master_worker = worker
        self._status.setText("Mastering...")
        self._start_operation_progress("Mastering")
        self._update_mix_state()
        worker.start()

    def _on_master_progress(self, percent: int):
        self._on_operation_progress("Mastering", percent)

    def _on_master_step(self, message: str):
        self._operation_progress.set_step(f"Mastering: {message}")
        self._status.setText(f"Mastering: {message}")

    def _on_master_finished(self, payload):
        worker = self._master_worker
        self._settle_worker(worker)
        self._master_worker = None
        self._finish_operation_progress()
        try:
            if isinstance(payload, tuple) and len(payload) == 2:
                result, match = payload
            else:
                result, match = payload, None

            if result.error:
                self._last_loudness_match = None
                self._report_error(f"Mastering error: {result.error}")
                return

            self._last_loudness_match = match
            sr = self._master_sample_rate
            preset_name = self._master_preset_name

            # Show in waveform
            if result.audio is not None:
                mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
                self._master_waveform.load_audio(mono, sr)

            if self._last_loudness_match:
                self._lufs_label.setText(
                    f"EBU Mode — Integrated: {result.output_lufs:.1f} LUFS | "
                    f"Ref: {match.reference_lufs:.1f} | "
                    f"Short-term (3 s) avg delta {match.average_short_term_delta_db:.1f} dB | "
                    f"Momentary (400 ms) max {result.momentary_max_lufs:.1f} LUFS | "
                    f"True peak {result.true_peak_dbtp:.2f} dBTP"
                )
            else:
                self._lufs_label.setText(
                    f"EBU Mode — Integrated: {result.output_lufs:.1f} LUFS | "
                    f"Short-term (3 s) max: {result.short_term_max_lufs:.1f} LUFS | "
                    f"Momentary (400 ms) max: {result.momentary_max_lufs:.1f} LUFS | "
                    f"LRA: {result.output_lra_lu:.1f} LU | "
                    f"True peak: {result.true_peak_dbtp:.2f} dBTP"
                )

            path, selected_filter = save_audio_file(
                self,
                "Export Mastered Audio",
                "master.wav",
                operation_kind="mixer_master_export",
                dialog=QFileDialog,
            )
            if path and result.audio is not None:
                path = ensure_extension(path, selected_filter)
                fmt = os.path.splitext(path)[1].lower().lstrip(".") or "wav"
                settings = ExportSettings(
                    format=fmt,
                    sample_rate=sr,
                    bit_depth=24,
                    title=os.path.splitext(os.path.basename(path))[0],
                    comment=" | ".join(result.report_lines()),
                )
                self._start_master_export(
                    result.audio.copy(), sr, path, settings,
                    {
                        "preset": result.preset_name,
                        "input_lufs": result.input_lufs,
                        "output_lufs": result.output_lufs,
                        "output_lra_lu": result.output_lra_lu,
                        "true_peak_dbtp": result.true_peak_dbtp,
                        "target_lufs": result.target_lufs,
                        "meets_target": result.meets_target,
                    },
                )
                return
            elif self._last_loudness_match:
                self._status.setText(
                    f"Mastered ({preset_name}) matched to {self._reference_name} | "
                    f"{result.output_lufs:.1f} LUFS | "
                    f"ST avg delta {match.average_short_term_delta_db:.1f} dB | "
                    f"{result.processing_time:.1f}s"
                )
            else:
                self._status.setText(
                    f"Mastered ({preset_name}) | "
                    f"{result.output_lufs:.1f} LUFS | "
                    f"{result.processing_time:.1f}s"
                )
        except Exception as exc:
            self._report_error(f"Error: {exc}")
        finally:
            self._update_mix_state()

    def _start_master_export(
        self,
        audio: np.ndarray,
        sample_rate: int,
        path: str,
        settings: ExportSettings,
        mastering_metadata: dict,
    ):
        if self._export_worker is not None and self._export_worker.isRunning():
            return
        worker = InferenceWorker(
            export_from_numpy,
            audio,
            sample_rate,
            path,
            settings,
            module="mixer",
            operation="master_export",
            provenance_extra={"mastering": mastering_metadata},
        )
        worker.progress.connect(
            lambda pct: self._on_operation_progress("Exporting master", pct)
        )
        worker.step_info.connect(self._on_operation_step)
        worker.finished.connect(self._on_master_export_finished)
        worker.error.connect(self._on_master_export_error)
        worker.cancelled.connect(self._on_master_export_cancelled)
        self._export_worker = worker
        self._start_operation_progress("Exporting master")
        self._update_mix_state()
        worker.start()

    def _on_master_export_finished(self, written: str):
        worker = self._export_worker
        self._settle_worker(worker)
        self._export_worker = None
        self._finish_operation_progress()
        self._status.setText(f"Exported master: {written}")
        self._update_mix_state()

    def _on_master_export_error(self, message: str):
        worker = self._export_worker
        self._settle_worker(worker)
        self._export_worker = None
        self._finish_operation_progress()
        self._report_error(f"Master export error: {message}")
        self._update_mix_state()

    def _on_master_export_cancelled(self):
        worker = self._export_worker
        self._settle_worker(worker)
        self._export_worker = None
        self._finish_operation_progress()
        self._status.setText("Master export cancelled")
        self._update_mix_state()

    def _on_master_error(self, message: str):
        worker = self._master_worker
        self._settle_worker(worker)
        self._master_worker = None
        self._finish_operation_progress()
        self._report_error(f"Mastering error: {message}")
        self._update_mix_state()

    def _on_master_cancelled(self):
        worker = self._master_worker
        self._settle_worker(worker)
        self._master_worker = None
        self._finish_operation_progress()
        self._status.setText("Mastering cancelled")
        self._update_mix_state()
