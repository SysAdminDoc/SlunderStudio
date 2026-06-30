"""
Slunder Studio v0.1.27 — Mixer View
Multi-track mixer timeline with per-track volume/pan/effects,
smart mastering presets, waveform overview, and final export.
"""
import os
import time
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QScrollArea, QSlider, QFileDialog, QDoubleSpinBox,
    QProgressBar, QTabWidget,
)
from PySide6.QtCore import Qt, Signal

import numpy as np

from ui.theme import ThemeEngine
from ui.waveform_widget import WaveformWidget, MiniWaveform
from core.mastering import (
    PRESETS,
    DynamicEQSuggestion,
    LoudnessMatchResult,
    MasteringPreset,
    apply_dynamic_eq,
    match_loudness_to_reference,
    master_audio,
    measure_lufs,
    measure_short_term_lufs,
    suggest_dynamic_eq_curve,
)


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
        self.setFixedHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Track name
        name_label = QLabel(name)
        name_label.setFixedWidth(80)
        name_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 11px;")
        layout.addWidget(name_label)

        # Mini waveform
        self._waveform = MiniWaveform()
        if audio is not None:
            mono = audio[:, 0] if audio.ndim == 2 else audio
            self._waveform.set_audio(mono, sr)
        self._waveform.setFixedWidth(160)
        layout.addWidget(self._waveform)

        # Volume
        vol_col = QVBoxLayout()
        vol_col.setSpacing(1)
        vl = QLabel("Vol")
        vl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
        vl.setAlignment(Qt.AlignCenter)
        self._vol_slider = QSlider(Qt.Horizontal)
        self._vol_slider.setRange(0, 150)
        self._vol_slider.setValue(100)
        self._vol_slider.setFixedWidth(80)
        self._vol_slider.setFixedHeight(14)
        self._vol_val = QLabel("100%")
        self._vol_val.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
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
        pl.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
        pl.setAlignment(Qt.AlignCenter)
        self._pan_slider = QSlider(Qt.Horizontal)
        self._pan_slider.setRange(-100, 100)
        self._pan_slider.setValue(0)
        self._pan_slider.setFixedWidth(80)
        self._pan_slider.setFixedHeight(14)
        self._pan_val = QLabel("C")
        self._pan_val.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
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
                font-size: 9px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
            QPushButton:checked {{ color: white; border: none; }}
        """
        self._mute_btn = QPushButton("M")
        self._mute_btn.setFixedSize(24, 20)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setStyleSheet(btn_style + "QPushButton:checked { background: #da3633; }")
        self._mute_btn.clicked.connect(self._on_mute)

        self._solo_btn = QPushButton("S")
        self._solo_btn.setFixedSize(24, 20)
        self._solo_btn.setCheckable(True)
        self._solo_btn.setStyleSheet(btn_style + "QPushButton:checked { background: #d29922; }")
        self._solo_btn.clicked.connect(self._on_solo)

        self._remove_btn = QPushButton("X")
        self._remove_btn.setFixedSize(24, 20)
        self._remove_btn.setStyleSheet(btn_style)
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.track_idx))

        layout.addWidget(self._mute_btn)
        layout.addWidget(self._solo_btn)
        layout.addWidget(self._remove_btn)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._strips: list[MixerTrackStrip] = []
        self._tracks: list[dict] = []  # {name, audio, sr}
        self._dynamic_eq_suggestions: dict[int, DynamicEQSuggestion] = {}
        self._reference_audio: Optional[np.ndarray] = None
        self._reference_sr: int = 44100
        self._reference_name: str = ""
        self._last_loudness_match: Optional[LoudnessMatchResult] = None

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Top: Track list ────────────────────────────────────────────────
        tracks_header = QHBoxLayout()
        tl = QLabel("Tracks")
        tl.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 13px;")

        self._add_btn = QPushButton("+ Import Track")
        self._add_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: white; border: none;
                border-radius: 4px; padding: 5px 12px;
                font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._add_btn.clicked.connect(self._on_import_track)

        self._dynamic_eq_btn = QPushButton("Suggest Dynamic EQ")
        self._dynamic_eq_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 5px 12px; font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
            QPushButton:disabled {{ color: #555; border-color: {t['border']}; }}
        """)
        self._dynamic_eq_btn.setEnabled(False)
        self._dynamic_eq_btn.clicked.connect(self._on_suggest_dynamic_eq)

        tracks_header.addWidget(tl)
        tracks_header.addStretch()
        tracks_header.addWidget(self._dynamic_eq_btn)
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
        ml.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px; border: none;")
        master_layout.addWidget(ml)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(PRESETS.keys())
        self._preset_combo.setCurrentText("Balanced")
        self._preset_combo.setStyleSheet(f"""
            QComboBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 10px; font-size: 11px; min-width: 140px;
            }}
        """)
        master_layout.addWidget(self._preset_combo)

        # Target LUFS
        ll = QLabel("Target LUFS:")
        ll.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px; border: none;")
        self._lufs_spin = QDoubleSpinBox()
        self._lufs_spin.setRange(-24.0, -6.0)
        self._lufs_spin.setValue(-14.0)
        self._lufs_spin.setSuffix(" LUFS")
        self._lufs_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
        """)
        master_layout.addWidget(ll)
        master_layout.addWidget(self._lufs_spin)

        self._ref_btn = QPushButton("Load Ref")
        self._ref_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 5px 10px; font-size: 11px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """)
        self._ref_btn.clicked.connect(self._on_load_reference)
        master_layout.addWidget(self._ref_btn)

        self._master_btn = QPushButton("Master + Export")
        self._master_btn.setStyleSheet(f"""
            QPushButton {{
                background: #238636; color: white; border: none;
                border-radius: 5px; padding: 6px 16px;
                font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ background: #2ea043; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._master_btn.setEnabled(False)
        self._master_btn.clicked.connect(self._on_master_export)
        master_layout.addWidget(self._master_btn)

        master_layout.addStretch()

        # LUFS meter display
        self._reference_label = QLabel("Ref: none")
        self._reference_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
        master_layout.addWidget(self._reference_label)

        self._lufs_label = QLabel("")
        self._lufs_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
        master_layout.addWidget(self._lufs_label)

        layout.addWidget(master_frame)

        # ── Bottom: Master output waveform ─────────────────────────────────
        self._master_waveform = WaveformWidget()
        layout.addWidget(self._master_waveform, 1)

        # Status
        self._status = QLabel("Import audio tracks to begin mixing")
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        layout.addWidget(self._status)

    # ── Track Management ───────────────────────────────────────────────────────

    def add_track(self, name: str, audio: np.ndarray, sr: int = 44100):
        """Add an audio track to the mixer."""
        idx = len(self._strips)
        self._tracks.append({"name": name, "audio": audio, "sr": sr})

        strip = MixerTrackStrip(idx, name, audio, sr)
        strip.remove_requested.connect(self._on_remove_track)
        strip.volume_changed.connect(lambda *_: self._update_mix_state())
        strip.pan_changed.connect(lambda *_: self._update_mix_state())
        strip.mute_changed.connect(lambda *_: self._update_mix_state())
        strip.solo_changed.connect(lambda *_: self._update_mix_state())

        self._strips.append(strip)
        self._strips_layout.insertWidget(self._strips_layout.count() - 1, strip)
        self._master_btn.setEnabled(True)
        self._update_mix_state()

    def _read_audio_file(self, file_path: str) -> tuple[np.ndarray, int]:
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path, dtype="float32", always_2d=True)
        except Exception:
            import wave
            with wave.open(file_path, "r") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1:
                    audio = audio.reshape(-1, channels)

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        return audio.astype(np.float32), int(sr)

    def add_track_from_file(self, file_path: str):
        """Import an audio file as a track."""
        try:
            name = os.path.splitext(os.path.basename(file_path))[0]
            audio, sr = self._read_audio_file(file_path)
            self.add_track(name, audio, sr)
            self._status.setText(f"Added track: {name} ({len(audio) / sr:.1f}s)")
        except Exception as e:
            self._status.setText(f"Import error: {e}")

    def set_reference_track(self, name: str, audio: np.ndarray, sr: int = 44100,
                            path: str = ""):
        """Set a loudness reference track for mastering."""
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        elif audio.ndim == 2 and audio.shape[1] == 1:
            audio = np.column_stack([audio[:, 0], audio[:, 0]])
        elif audio.ndim == 2 and audio.shape[1] > 2:
            audio = audio[:, :2]

        self._reference_audio = audio.astype(np.float32)
        self._reference_sr = int(sr)
        self._reference_name = name or os.path.basename(path) or "Reference"
        ref_lufs = measure_lufs(self._reference_audio, self._reference_sr)
        profile = measure_short_term_lufs(self._reference_audio, self._reference_sr)
        if ref_lufs > -60:
            self._lufs_spin.setValue(max(self._lufs_spin.minimum(), min(self._lufs_spin.maximum(), ref_lufs)))

        if profile:
            low = min(point.lufs for point in profile)
            high = max(point.lufs for point in profile)
            self._reference_label.setText(
                f"Ref: {self._reference_name} {ref_lufs:.1f} LUFS | ST {low:.1f}..{high:.1f}"
            )
        else:
            self._reference_label.setText(f"Ref: {self._reference_name} {ref_lufs:.1f} LUFS")

    def _on_load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Loudness Reference", "",
            "Audio (*.wav *.flac *.mp3 *.ogg)"
        )
        if not path:
            return

        try:
            audio, sr = self._read_audio_file(path)
            name = os.path.splitext(os.path.basename(path))[0]
            self.set_reference_track(name, audio, sr, path)
            self._status.setText(f"Loaded loudness reference: {name}")
        except Exception as e:
            self._status.setText(f"Reference load error: {e}")

    def _on_import_track(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Audio Track", "",
            "Audio (*.wav *.flac *.mp3 *.ogg)"
        )
        if path:
            self.add_track_from_file(path)

    def _on_remove_track(self, idx: int):
        if 0 <= idx < len(self._strips):
            removed_idx = idx
            old_suggestions = dict(self._dynamic_eq_suggestions)
            strip = self._strips[idx]
            self._strips_layout.removeWidget(strip)
            strip.deleteLater()
            self._strips.pop(idx)
            self._tracks.pop(idx)

            # Re-index remaining strips
            for i, s in enumerate(self._strips):
                s.track_idx = i

            old_track_count = len(self._tracks) + 1
            self._dynamic_eq_suggestions = {}
            for new_idx, old_idx in enumerate(
                old_idx for old_idx in range(old_track_count) if old_idx != removed_idx
            ):
                if old_idx in old_suggestions:
                    self._dynamic_eq_suggestions[new_idx] = old_suggestions[old_idx]
            self._master_btn.setEnabled(len(self._strips) > 0)
            self._update_mix_state()

    def _update_mix_state(self):
        """Update master button state."""
        has_tracks = len(self._strips) > 0
        self._master_btn.setEnabled(has_tracks)
        self._dynamic_eq_btn.setEnabled(has_tracks)

    def _on_suggest_dynamic_eq(self):
        if not self._tracks:
            self._status.setText("Import audio tracks before dynamic EQ")
            return

        self._dynamic_eq_btn.setEnabled(False)
        self._status.setText("Suggesting dynamic EQ curves...")

        summaries: list[str] = []
        try:
            for idx, track in enumerate(self._tracks):
                audio = track["audio"]
                sr = track["sr"]
                name = track["name"]
                suggestion = suggest_dynamic_eq_curve(audio, sr, name)
                processed = apply_dynamic_eq(audio, sr, suggestion.bands, strength=0.75)
                self._tracks[idx]["audio"] = processed
                self._dynamic_eq_suggestions[idx] = suggestion

                if idx < len(self._strips):
                    strip = self._strips[idx]
                    strip.audio = processed
                    mono = processed[:, 0] if processed.ndim == 2 else processed
                    strip._waveform.set_audio(mono, sr)

                if suggestion.bands:
                    first_moves = ", ".join(
                        f"{band.frequency_hz:.0f}Hz {band.gain_db:+.1f}dB"
                        for band in suggestion.bands[:3]
                    )
                    summaries.append(f"{name}: {first_moves}")
                else:
                    summaries.append(f"{name}: balanced")

            self._status.setText("Dynamic EQ applied - " + " | ".join(summaries[:3]))
            self._update_mix_state()
        except Exception as e:
            self._status.setText(f"Dynamic EQ error: {e}")
        finally:
            self._dynamic_eq_btn.setEnabled(len(self._strips) > 0)

    # ── Mixing ─────────────────────────────────────────────────────────────────

    def _get_mixed_audio(self) -> Optional[np.ndarray]:
        """Mix all tracks according to current settings."""
        if not self._strips:
            return None

        # Find max length and target SR
        sr = self._tracks[0]["sr"] if self._tracks else 44100
        max_len = max(len(t["audio"]) for t in self._tracks) if self._tracks else 0
        if max_len == 0:
            return None

        soloed = [s for s in self._strips if s.is_soloed]
        active = soloed if soloed else [s for s in self._strips if not s.is_muted]

        output = np.zeros((max_len, 2), dtype=np.float32)

        for strip in active:
            if strip.track_idx >= len(self._tracks):
                continue
            audio = self._tracks[strip.track_idx]["audio"]
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])

            length = min(len(audio), max_len)
            vol = strip.volume
            pan = strip.pan

            left_gain = vol * np.cos(max(0, pan) * np.pi / 2)
            right_gain = vol * np.cos(max(0, -pan) * np.pi / 2)

            output[:length, 0] += audio[:length, 0] * left_gain
            output[:length, 1] += audio[:length, 1] * right_gain

        peak = np.max(np.abs(output))
        if peak > 1.0:
            output /= peak

        return output

    # ── Mastering + Export ─────────────────────────────────────────────────────

    def _on_master_export(self):
        mixed = self._get_mixed_audio()
        if mixed is None:
            self._status.setText("No audio to master")
            return

        sr = self._tracks[0]["sr"] if self._tracks else 44100

        # Get preset
        preset_name = self._preset_combo.currentText()
        preset = PRESETS.get(preset_name, PRESETS["Balanced"])

        # Override target LUFS
        preset.target_lufs = self._lufs_spin.value()

        self._master_btn.setEnabled(False)
        self._status.setText("Mastering...")

        try:
            result = master_audio(mixed, sr, preset)

            if result.error:
                self._status.setText(f"Mastering error: {result.error}")
                return

            if self._reference_audio is not None and result.audio is not None:
                match = match_loudness_to_reference(
                    result.audio,
                    sr,
                    self._reference_audio,
                    self._reference_sr,
                )
                self._last_loudness_match = match
                result.audio = match.audio
                result.output_lufs = match.output_lufs
                result.peak_db = match.peak_db
            else:
                self._last_loudness_match = None

            # Show in waveform
            if result.audio is not None:
                mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
                self._master_waveform.set_audio(mono, sr)

            if self._last_loudness_match:
                match = self._last_loudness_match
                self._lufs_label.setText(
                    f"In: {result.input_lufs:.1f} LUFS | "
                    f"Ref: {match.reference_lufs:.1f} | "
                    f"Out: {result.output_lufs:.1f} | "
                    f"ST avg delta {match.average_short_term_delta_db:.1f} dB"
                )
            else:
                self._lufs_label.setText(
                    f"In: {result.input_lufs:.1f} LUFS | "
                    f"Out: {result.output_lufs:.1f} LUFS | "
                    f"Peak: {result.peak_db:.1f} dB"
                )

            # Export dialog
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Mastered Audio", "master.wav",
                "WAV (*.wav);;FLAC (*.flac)"
            )
            if path and result.audio is not None:
                import wave
                int_audio = (result.audio * 32767).clip(-32768, 32767).astype(np.int16)
                with wave.open(path, "w") as wf:
                    wf.setnchannels(2)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(int_audio.tobytes())

                self._status.setText(
                    f"Exported: {path} | "
                    f"{result.output_lufs:.1f} LUFS | "
                    f"{result.processing_time:.1f}s"
                )
            else:
                if self._last_loudness_match:
                    match = self._last_loudness_match
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

        except Exception as e:
            self._status.setText(f"Error: {e}")
        finally:
            self._master_btn.setEnabled(True)
