"""
Slunder Studio v0.0.2 — Waveform Widget
pyqtgraph-based waveform and spectrogram display with playback cursor,
selection regions, and zoom/pan.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QColor

import numpy as np

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

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if not HAS_PYQTGRAPH:
            lbl = QLabel("Waveform display unavailable — restart to retry")
            lbl.setStyleSheet("color: #F38BA8; padding: 20px;")
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
            self._info_label.setStyleSheet("color: #A6ADC8; font-size: 11px;")

            ctrl.addWidget(self._waveform_btn)
            ctrl.addWidget(self._spectro_btn)
            ctrl.addStretch()
            ctrl.addWidget(self._info_label)
            layout.addLayout(ctrl)

        # Stacked views
        self._stack = QStackedWidget()

        # Waveform view
        self._waveform_plot = pg.PlotWidget()
        self._waveform_plot.setBackground("#181825")
        self._waveform_plot.setMouseEnabled(x=True, y=False)
        self._waveform_plot.showGrid(x=True, y=False, alpha=0.15)
        self._waveform_plot.getAxis("bottom").setPen(pg.mkPen("#6C7086"))
        self._waveform_plot.getAxis("left").setPen(pg.mkPen("#6C7086"))
        self._waveform_plot.getAxis("bottom").setTextPen(pg.mkPen("#A6ADC8"))
        self._waveform_plot.getAxis("left").setTextPen(pg.mkPen("#A6ADC8"))
        self._waveform_curve = self._waveform_plot.plot(pen=pg.mkPen("#89B4FA", width=1))

        # Playback cursor line
        self._cursor_line = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#F38BA8", width=2),
            movable=False,
        )
        self._waveform_plot.addItem(self._cursor_line)
        self._cursor_line.hide()

        # Click handler
        self._waveform_plot.scene().sigMouseClicked.connect(self._on_waveform_click)

        self._stack.addWidget(self._waveform_plot)

        # Spectrogram view
        self._spectro_plot = pg.PlotWidget()
        self._spectro_plot.setBackground("#181825")
        self._spectro_plot.getAxis("bottom").setPen(pg.mkPen("#6C7086"))
        self._spectro_plot.getAxis("left").setPen(pg.mkPen("#6C7086"))
        self._spectro_plot.getAxis("bottom").setTextPen(pg.mkPen("#A6ADC8"))
        self._spectro_plot.getAxis("left").setTextPen(pg.mkPen("#A6ADC8"))
        self._spectro_item = pg.ImageItem()
        self._spectro_plot.addItem(self._spectro_item)

        self._spectro_cursor = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#F38BA8", width=2),
            movable=False,
        )
        self._spectro_plot.addItem(self._spectro_cursor)
        self._spectro_cursor.hide()

        self._stack.addWidget(self._spectro_plot)

        layout.addWidget(self._stack)

    def _set_mode(self, mode: str):
        self._mode = mode
        self._stack.setCurrentIndex(0 if mode == "waveform" else 1)
        if self._show_controls:
            self._waveform_btn.setEnabled(mode != "waveform")
            self._spectro_btn.setEnabled(mode != "spectrogram")

    def load_file(self, file_path: str):
        """Load audio from file and display waveform."""
        try:
            import soundfile as sf
            data, sr = sf.read(file_path, dtype="float32")
            self.set_audio(data, sr)
        except Exception as e:
            if self._show_controls:
                self._info_label.setText(f"Error: {e}")

    def set_audio(self, audio: np.ndarray, sample_rate: int = 48000):
        """Set audio data and update display."""
        if not HAS_PYQTGRAPH:
            return

        self._audio_data = audio
        self._sample_rate = sample_rate

        # Convert to mono for display
        if audio.ndim == 2:
            mono = audio.mean(axis=1)
        else:
            mono = audio

        self._duration = len(mono) / sample_rate

        # Downsample for display (max 10000 points)
        max_points = 10000
        if len(mono) > max_points:
            step = len(mono) // max_points
            display = mono[::step]
        else:
            display = mono

        time_axis = np.linspace(0, self._duration, len(display))

        self._waveform_curve.setData(time_axis, display)
        self._waveform_plot.setXRange(0, self._duration)
        self._waveform_plot.setYRange(-1, 1)

        # Update spectrogram
        self._update_spectrogram(mono, sample_rate)

        if self._show_controls:
            dur_str = f"{self._duration:.1f}s"
            sr_str = f"{sample_rate/1000:.1f}kHz"
            ch_str = "stereo" if audio.ndim == 2 else "mono"
            self._info_label.setText(f"{dur_str} | {sr_str} | {ch_str}")

    def _update_spectrogram(self, mono: np.ndarray, sr: int):
        """Compute and display mel spectrogram."""
        try:
            import librosa
            S = librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=64, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Custom colormap: dark blue -> blue -> cyan -> yellow
            cmap = pg.ColorMap(
                pos=[0.0, 0.33, 0.66, 1.0],
                color=[
                    QColor("#11111B"),
                    QColor("#1E1E6E"),
                    QColor("#89B4FA"),
                    QColor("#F9E2AF"),
                ],
            )
            lut = cmap.getLookupTable(nPts=256)
            self._spectro_item.setImage(S_dB.T, autoLevels=True)
            self._spectro_item.setLookupTable(lut)

            # Scale to time axis
            tr = self._spectro_item.transform()
            self._spectro_item.setTransform(
                tr.scale(self._duration / S_dB.shape[1], sr / (2 * S_dB.shape[0]))
            )
        except ImportError:
            pass  # librosa not available

    def set_playback_position(self, seconds: float):
        """Update playback cursor position."""
        if not HAS_PYQTGRAPH:
            return
        self._playback_pos = seconds
        self._cursor_line.setValue(seconds)
        self._cursor_line.show()
        self._spectro_cursor.setValue(seconds)
        self._spectro_cursor.show()

    def clear_cursor(self):
        if HAS_PYQTGRAPH:
            self._cursor_line.hide()
            self._spectro_cursor.hide()

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
        if not HAS_PYQTGRAPH:
            return
        self._waveform_curve.setData([], [])
        self._spectro_item.clear()
        self._cursor_line.hide()
        self._spectro_cursor.hide()
        self._audio_data = None
        self._duration = 0.0
        if self._show_controls:
            self._info_label.setText("")

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def has_audio(self) -> bool:
        return self._audio_data is not None


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
        self._waveform.set_audio(audio, sample_rate)

    def load_file(self, file_path: str):
        self._waveform.load_file(file_path)

    def set_playback_position(self, seconds: float):
        self._waveform.set_playback_position(seconds)

    def clear_cursor(self):
        self._waveform.clear_cursor()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()
