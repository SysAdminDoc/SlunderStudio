"""
Slunder Studio — Waveform Widget
pyqtgraph-based waveform and spectrogram display with playback cursor,
selection regions, and zoom/pan.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QColor, QTransform

import numpy as np
from pathlib import Path

from ui.theme import Palette

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    # Auto-install and retry
    try:
        from core.deps import ensure
        ensure("pyqtgraph")
        import pyqtgraph as pg
        HAS_PYQTGRAPH = True
    except Exception as _e:
        print(f"[Slunder Studio] pyqtgraph unavailable: {_e}")
        HAS_PYQTGRAPH = False
        pg = None


class WaveformWidget(QWidget):
    """
    Waveform + spectrogram display with playback cursor overlay.
    Supports: waveform view, spectrogram view, selection region.
    """
    position_clicked = Signal(float)  # normalized 0-1 position
    region_selected = Signal(float, float)  # start, end in seconds

    def __init__(self, parent=None, show_controls: bool = True):
        super().__init__(parent)
        self._audio_data = None
        self._sample_rate = 48000
        self._duration = 0.0
        self._playback_pos = 0.0  # seconds
        self._show_controls = show_controls
        self._mode = "waveform"  # waveform or spectrogram
        self._last_error = ""

        # Operable by keyboard: Left/Right seek, PageUp/Down scrub, Home/End
        # jump, M switches waveform and spectrogram.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Audio waveform")
        self.setAccessibleDescription(
            "Waveform and spectrogram view. Left and Right seek one second, "
            "Page Up and Page Down seek ten seconds, Home and End jump to the "
            "start or end, M switches between waveform and spectrogram."
        )

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if not HAS_PYQTGRAPH:
            lbl = QLabel("Waveform display unavailable — restart to retry")
            lbl.setStyleSheet(f"color: {Palette.RED}; padding: 20px;")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            return

        pg.setConfigOptions(antialias=True)

        # Controls bar
        if self._show_controls:
            ctrl = QHBoxLayout()
            ctrl.setSpacing(4)

            self._waveform_btn = QPushButton("Waveform")
            self._waveform_btn.setFixedHeight(24)
            self._waveform_btn.setProperty("class", "secondary")
            self._waveform_btn.clicked.connect(lambda: self._set_mode("waveform"))

            self._spectro_btn = QPushButton("Spectrogram")
            self._spectro_btn.setFixedHeight(24)
            self._spectro_btn.setProperty("class", "secondary")
            self._spectro_btn.clicked.connect(lambda: self._set_mode("spectrogram"))

            self._info_label = QLabel("")
            self._info_label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 11px;")

            ctrl.addWidget(self._waveform_btn)
            ctrl.addWidget(self._spectro_btn)
            ctrl.addStretch()
            ctrl.addWidget(self._info_label)
            layout.addLayout(ctrl)

        # Stacked views
        self._stack = QStackedWidget()

        # Waveform view
        self._waveform_plot = pg.PlotWidget()
        self._waveform_plot.setBackground(Palette.MANTLE)
        self._waveform_plot.setMouseEnabled(x=True, y=False)
        self._waveform_plot.showGrid(x=True, y=False, alpha=0.15)
        self._waveform_plot.getAxis("bottom").setPen(pg.mkPen(Palette.OVERLAY0))
        self._waveform_plot.getAxis("left").setPen(pg.mkPen(Palette.OVERLAY0))
        self._waveform_plot.getAxis("bottom").setTextPen(pg.mkPen(Palette.SUBTEXT0))
        self._waveform_plot.getAxis("left").setTextPen(pg.mkPen(Palette.SUBTEXT0))
        self._waveform_curve = self._waveform_plot.plot(pen=pg.mkPen(Palette.BLUE, width=1))

        # Playback cursor line
        self._cursor_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(Palette.RED, width=2),
            movable=False,
        )
        self._waveform_plot.addItem(self._cursor_line)
        self._cursor_line.hide()

        # Click handler
        self._waveform_plot.scene().sigMouseClicked.connect(self._on_waveform_click)

        self._stack.addWidget(self._waveform_plot)

        # Spectrogram view
        self._spectro_plot = pg.PlotWidget()
        self._spectro_plot.setBackground(Palette.MANTLE)
        self._spectro_plot.getAxis("bottom").setPen(pg.mkPen(Palette.OVERLAY0))
        self._spectro_plot.getAxis("left").setPen(pg.mkPen(Palette.OVERLAY0))
        self._spectro_plot.getAxis("bottom").setTextPen(pg.mkPen(Palette.SUBTEXT0))
        self._spectro_plot.getAxis("left").setTextPen(pg.mkPen(Palette.SUBTEXT0))
        self._spectro_item = pg.ImageItem()
        self._spectro_plot.addItem(self._spectro_item)

        self._spectro_cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen(Palette.RED, width=2),
            movable=False,
        )
        self._spectro_plot.addItem(self._spectro_cursor)
        self._spectro_cursor.hide()

        self._stack.addWidget(self._spectro_plot)

        layout.addWidget(self._stack)

    def _set_mode(self, mode: str):
        self._mode = mode
        if HAS_PYQTGRAPH:
            self._stack.setCurrentIndex(0 if mode == "waveform" else 1)
        if self._show_controls and hasattr(self, "_waveform_btn"):
            self._waveform_btn.setEnabled(mode != "waveform")
            self._spectro_btn.setEnabled(mode != "spectrogram")
        self._announce_position()

    def load_audio(
        self,
        source: str | Path | np.ndarray,
        sample_rate: int | None = None,
    ) -> bool:
        """Load a file or mono/stereo array and replace the displayed audio.

        File sources keep their encoded sample rate and therefore must not pass
        ``sample_rate``. Array sources require a positive ``sample_rate``.
        Arrays may be mono, frames-by-channels, or channels-by-frames. Invalid
        input clears any previous waveform, records ``last_error``, and returns
        ``False`` instead of leaving a stale preview visible.
        """
        self.clear()
        try:
            if isinstance(source, (str, Path)):
                if sample_rate is not None:
                    raise ValueError(
                        "sample_rate is only valid when loading an audio array"
                    )
                path = Path(source)
                if not path.is_file():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                import soundfile as sf

                audio, resolved_rate = sf.read(str(path), dtype="float32")
            else:
                if sample_rate is None:
                    raise ValueError("sample_rate is required for audio arrays")
                audio = source
                resolved_rate = sample_rate

            resolved_rate = self._validate_sample_rate(resolved_rate)
            normalized = self._normalize_audio(audio)
            if not HAS_PYQTGRAPH:
                raise RuntimeError("Waveform display is unavailable")
            self._display_audio(normalized, resolved_rate)
            return True
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            self._set_info(f"Error: {exc}")
            return False

    def load_file(self, file_path: str | Path) -> bool:
        """Compatibility wrapper; use :meth:`load_audio` for new callers."""
        return self.load_audio(file_path)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 48000) -> bool:
        """Compatibility wrapper; use :meth:`load_audio` for new callers."""
        return self.load_audio(audio, sample_rate)

    @staticmethod
    def _validate_sample_rate(sample_rate: object) -> int:
        if isinstance(sample_rate, bool):
            raise ValueError("sample_rate must be a positive integer")
        try:
            numeric = float(sample_rate)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("sample_rate must be a positive integer") from exc
        if not np.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
            raise ValueError("sample_rate must be a positive integer")
        return int(numeric)

    @staticmethod
    def _normalize_audio(audio: object) -> np.ndarray:
        data = np.asarray(audio)
        if data.ndim not in {1, 2}:
            raise ValueError("audio must be a mono or multi-channel array")
        if data.size == 0:
            raise ValueError("audio array is empty")
        if np.issubdtype(data.dtype, np.complexfloating):
            raise ValueError("complex audio arrays are not supported")
        if np.issubdtype(data.dtype, np.bool_):
            raise ValueError("boolean audio arrays are not supported")

        if data.ndim == 2:
            rows, columns = data.shape
            if columns in {1, 2}:
                pass
            elif rows in {1, 2}:
                data = data.T
            elif columns <= 8:
                pass
            elif rows <= 8:
                data = data.T
            else:
                raise ValueError(
                    "audio channel layout is ambiguous; use frames-by-channels "
                    "or channels-by-frames with at most 8 channels"
                )

        if np.issubdtype(data.dtype, np.unsignedinteger):
            info = np.iinfo(data.dtype)
            midpoint = (info.max + 1) / 2.0
            data = (data.astype(np.float32) - midpoint) / midpoint
        elif np.issubdtype(data.dtype, np.signedinteger):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), info.max))
            data = data.astype(np.float32) / scale
        else:
            data = data.astype(np.float32, copy=False)

        if not np.isfinite(data).all():
            raise ValueError("audio contains non-finite samples")
        return np.ascontiguousarray(data)

    def _display_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        self._audio_data = audio
        self._sample_rate = sample_rate
        self._last_error = ""

        # Convert to mono for display
        if audio.ndim == 2:
            mono = audio.mean(axis=1, dtype=np.float32)
        else:
            mono = audio

        self._duration = len(mono) / sample_rate

        # Downsample for display (max 10000 points)
        max_points = 10000
        if len(mono) > max_points:
            step = int(np.ceil(len(mono) / max_points))
            display = mono[::step]
        else:
            step = 1
            display = mono

        time_axis = np.arange(len(display), dtype=np.float64) * step / sample_rate

        self._waveform_curve.setData(time_axis, display)
        self._waveform_plot.setXRange(0, self._duration)
        self._waveform_plot.setYRange(-1, 1)

        # Update spectrogram
        self._update_spectrogram(mono, sample_rate)

        if self._show_controls:
            dur_str = f"{self._duration:.1f}s"
            sr_str = f"{sample_rate/1000:.1f}kHz"
            channels = audio.shape[1] if audio.ndim == 2 else 1
            ch_str = (
                "mono" if channels == 1
                else "stereo" if channels == 2
                else f"{channels} ch"
            )
            self._info_label.setText(f"{dur_str} | {sr_str} | {ch_str}")

    def _update_spectrogram(self, mono: np.ndarray, sr: int):
        """Compute and display mel spectrogram."""
        try:
            if len(mono) < 32:
                self._spectro_item.clear()
                return
            import librosa

            n_fft = 1 << min(11, len(mono).bit_length() - 1)
            S = librosa.feature.melspectrogram(
                y=mono,
                sr=sr,
                n_fft=n_fft,
                hop_length=max(1, n_fft // 4),
                n_mels=min(64, n_fft // 2),
                fmax=min(8000, sr / 2),
            )
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Custom colormap: dark blue -> blue -> cyan -> yellow
            cmap = pg.ColorMap(
                pos=[0.0, 0.33, 0.66, 1.0],
                color=[
                    QColor(Palette.CRUST),
                    QColor("#1E1E6E"),
                    QColor(Palette.BLUE),
                    QColor(Palette.YELLOW),
                ],
            )
            lut = cmap.getLookupTable(nPts=256)
            self._spectro_item.setImage(S_dB.T, autoLevels=True)
            self._spectro_item.setLookupTable(lut)

            # Scale to time axis
            self._spectro_item.setTransform(
                QTransform().scale(
                    self._duration / S_dB.shape[1],
                    sr / (2 * S_dB.shape[0]),
                )
            )
        except Exception:
            self._spectro_item.clear()

    def set_playback_position(self, seconds: float):
        """Update playback cursor position."""
        if not HAS_PYQTGRAPH:
            return
        self._playback_pos = seconds
        self._cursor_line.setValue(seconds)
        self._cursor_line.show()
        self._spectro_cursor.setValue(seconds)
        self._spectro_cursor.show()
        self._announce_position()

    def clear_cursor(self):
        if HAS_PYQTGRAPH:
            self._cursor_line.hide()
            self._spectro_cursor.hide()

    # ── Keyboard operation and screen-reader state ─────────────────────────────

    SEEK_STEP_SECONDS = 1.0
    SEEK_PAGE_SECONDS = 10.0

    @property
    def playback_position(self) -> float:
        return self._playback_pos

    def accessible_state(self) -> str:
        """One sentence a screen reader can read for the current view."""
        if not self.has_audio or self._duration <= 0:
            return "Waveform, no audio loaded"
        percent = int(round(100 * self._playback_pos / self._duration))
        return (
            f"{self._mode.capitalize()} view, position "
            f"{self._playback_pos:.1f} of {self._duration:.1f} seconds ({percent}%)"
        )

    def _announce_position(self):
        """Publish the current value so assistive tech sees the change."""
        state = self.accessible_state()
        self.setAccessibleDescription(state)
        try:
            from PySide6.QtGui import QAccessible, QAccessibleValueChangeEvent

            if QAccessible.isActive():
                QAccessible.updateAccessibility(
                    QAccessibleValueChangeEvent(self, f"{self._playback_pos:.3f}")
                )
        except Exception:
            # Announcements are best-effort; never break playback over them.
            pass

    def _seek_by(self, delta: float):
        if self._duration <= 0:
            return
        target = min(max(self._playback_pos + delta, 0.0), self._duration)
        self.set_playback_position(target)
        self.position_clicked.emit(target / self._duration)

    def _seek_to(self, seconds: float):
        if self._duration <= 0:
            return
        target = min(max(seconds, 0.0), self._duration)
        self.set_playback_position(target)
        self.position_clicked.emit(target / self._duration)

    def keyPressEvent(self, event):
        """Seek and switch views without a mouse."""
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self._seek_by(
                self.SEEK_STEP_SECONDS if key == Qt.Key.Key_Right
                else -self.SEEK_STEP_SECONDS
            )
        elif key in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self._seek_by(
                self.SEEK_PAGE_SECONDS if key == Qt.Key.Key_PageDown
                else -self.SEEK_PAGE_SECONDS
            )
        elif key == Qt.Key.Key_Home:
            self._seek_to(0.0)
        elif key == Qt.Key.Key_End:
            self._seek_to(self._duration)
        elif key == Qt.Key.Key_M:
            self._set_mode(
                "spectrogram" if self._mode == "waveform" else "waveform"
            )
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _on_waveform_click(self, event):
        """Handle click on waveform to seek."""
        if self._duration <= 0:
            return
        try:
            pos = self._waveform_plot.plotItem.vb.mapSceneToView(event.scenePos())
            t = pos.x()
            if 0 <= t <= self._duration:
                normalized = t / self._duration
                self.position_clicked.emit(normalized)
        except Exception:
            pass

    def clear(self):
        """Clear display."""
        if HAS_PYQTGRAPH:
            self._waveform_curve.setData([], [])
            self._spectro_item.clear()
            self._spectro_item.setTransform(QTransform())
            self._cursor_line.hide()
            self._spectro_cursor.hide()
        self._audio_data = None
        self._duration = 0.0
        self._last_error = ""
        self._set_info("")

    def _set_info(self, text: str) -> None:
        if self._show_controls and hasattr(self, "_info_label"):
            self._info_label.setText(text)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def has_audio(self) -> bool:
        return self._audio_data is not None

    @property
    def last_error(self) -> str:
        return self._last_error


class MiniWaveform(QWidget):
    """Compact waveform thumbnail for batch view cards."""
    clicked = Signal()
    double_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self._waveform = WaveformWidget(self, show_controls=False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._waveform)

    def set_audio(self, audio: np.ndarray, sample_rate: int = 48000):
        return self._waveform.load_audio(audio, sample_rate)

    def load_file(self, file_path: str):
        return self._waveform.load_audio(file_path)

    def load_audio(
        self,
        source: str | Path | np.ndarray,
        sample_rate: int | None = None,
    ) -> bool:
        return self._waveform.load_audio(source, sample_rate)

    def set_playback_position(self, seconds: float):
        self._waveform.set_playback_position(seconds)

    def clear_cursor(self):
        self._waveform.clear_cursor()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
