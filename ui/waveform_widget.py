"""
Slunder Studio — Waveform Widget
pyqtgraph-based waveform and spectrogram display with playback cursor,
selection regions, and zoom/pan.
"""
import logging

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QColor, QTransform

import numpy as np
from pathlib import Path

from ui.theme import Palette
from ui.widgets import EmptyStateWidget
from core.i18n import tr
from core.workers import CancelledJobError, InferenceWorker


logger = logging.getLogger(__name__)


def _qt_object_alive(obj) -> bool:
    """Return whether the wrapped QWidget still has a native Qt object."""
    try:
        import shiboken6

        return bool(shiboken6.isValid(obj))
    except (ImportError, RuntimeError):
        return True

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    # Optional UI dependencies fail closed with diagnostics; supported setup profiles install them.
    try:
        from core.deps import ensure
        ensure("pyqtgraph")
        import pyqtgraph as pg
        HAS_PYQTGRAPH = True
    except Exception as _e:
        logger.warning("pyqtgraph unavailable", exc_info=True)
        HAS_PYQTGRAPH = False
        pg = None


def _decode_waveform_file(
    file_path: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
) -> tuple[np.ndarray, int]:
    """Decode a waveform in chunks so the GUI never owns file I/O."""
    import soundfile as sf

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    chunks = []
    with sf.SoundFile(str(path), mode="r") as source:
        sample_rate = int(source.samplerate)
        total_frames = max(0, int(len(source)))
        frames_read = 0
        if progress_cb:
            progress_cb(0)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledJobError("Audio loading cancelled")
            chunk = source.read(
                frames=1_048_576,
                dtype="float32",
                always_2d=True,
            )
            if len(chunk) == 0:
                break
            chunks.append(chunk)
            frames_read += len(chunk)
            if progress_cb and total_frames:
                progress_cb(min(99, int(frames_read * 100 / total_frames)))

    if not chunks:
        raise ValueError("Audio file contains no frames")
    audio = np.concatenate(chunks, axis=0)
    if progress_cb:
        progress_cb(100)
    return np.ascontiguousarray(audio), sample_rate


def _compute_spectrogram(
    mono: np.ndarray,
    sample_rate: int,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
) -> np.ndarray:
    """Compute a mel image away from the Qt GUI thread."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Spectrogram computation cancelled")
    if len(mono) < 32:
        return np.empty((0, 0), dtype=np.float32)

    import librosa

    if progress_cb:
        progress_cb(10)
    n_fft = 1 << min(11, len(mono).bit_length() - 1)
    spectrogram = librosa.feature.melspectrogram(
        y=mono,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=max(1, n_fft // 4),
        n_mels=min(64, n_fft // 2),
        fmax=min(8000, sample_rate / 2),
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Spectrogram computation cancelled")
    if progress_cb:
        progress_cb(75)
    result = librosa.power_to_db(spectrogram, ref=np.max).astype(
        np.float32,
        copy=False,
    )
    if progress_cb:
        progress_cb(100)
    return result


class WaveformWidget(QWidget):
    """
    Waveform + spectrogram display with playback cursor overlay.
    Supports: waveform view, spectrogram view, selection region.
    """
    position_clicked = Signal(float)  # normalized 0-1 position
    region_selected = Signal(float, float)  # start, end in seconds
    audio_load_started = Signal(str)
    audio_load_progress = Signal(int)
    audio_load_finished = Signal(bool)
    spectrogram_progress = Signal(int)
    empty_action_requested = Signal()

    def __init__(self, parent=None, show_controls: bool = True):
        super().__init__(parent)
        self._audio_data = None
        self._has_audio = False
        self._spectrogram_ready = False
        self._sample_rate = 48000
        self._duration = 0.0
        self._playback_pos = 0.0  # seconds
        self._show_controls = show_controls
        self._mode = "waveform"  # waveform or spectrogram
        self._last_error = ""
        self._audio_load_token = 0
        self._audio_load_worker = None
        self._audio_load_workers = set()
        self._pending_audio_load_events = {}
        self._spectrogram_token = 0
        self._spectrogram_worker = None
        self._spectrogram_workers = set()
        self._closed = False
        self._selection_enabled = False
        self._selected_region: tuple[float, float] | None = None
        self._selection_region = None

        # Operable by keyboard: Left/Right seek, PageUp/Down scrub, Home/End
        # jump, M switches waveform and spectrogram.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(tr("runtime.waveform.accessibility_name"))
        self.setAccessibleDescription(
            tr("runtime.waveform.accessibility_description")
        )

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if not HAS_PYQTGRAPH:
            lbl = QLabel(tr("runtime.waveform.unavailable"))
            lbl.setStyleSheet(f"color: {Palette.RED}; padding: 20px;")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            return

        pg.setConfigOptions(antialias=True)

        # Controls bar
        if self._show_controls:
            ctrl = QHBoxLayout()
            ctrl.setSpacing(4)

            self._waveform_btn = QPushButton(tr("runtime.waveform.waveform"))
            self._waveform_btn.setMinimumHeight(24)
            self._waveform_btn.setProperty("class", "secondary")
            self._waveform_btn.clicked.connect(lambda: self._set_mode("waveform"))

            self._spectro_btn = QPushButton(tr("runtime.waveform.spectrogram"))
            self._spectro_btn.setMinimumHeight(24)
            self._spectro_btn.setProperty("class", "secondary")
            self._spectro_btn.clicked.connect(lambda: self._set_mode("spectrogram"))

            self._info_label = QLabel("")
            self._info_label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;")

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

        self._selection_region = pg.LinearRegionItem(
            [0.0, 0.0],
            movable=True,
            brush=pg.mkBrush(137, 180, 250, 45),
            pen=pg.mkPen(Palette.BLUE, width=1),
        )
        self._selection_region.setZValue(5)
        self._selection_region.hide()
        self._selection_region.sigRegionChanged.connect(self._on_selection_region_changed)
        self._waveform_plot.addItem(self._selection_region)

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

        # Keep the empty page at index 2 so the waveform/spectrogram indexes
        # remain stable for existing callers.
        self._empty_state = EmptyStateWidget(
            tr("runtime.waveform.empty_title"),
            tr("runtime.waveform.empty_description"),
            tr("runtime.waveform.empty_action") if self._show_controls else "",
        )
        self._empty_state.action_requested.connect(self.empty_action_requested.emit)
        self._stack.addWidget(self._empty_state)
        self._stack.setCurrentWidget(self._empty_state)

        layout.addWidget(self._stack)

    def _set_mode(self, mode: str):
        self._mode = mode
        if mode == "spectrogram":
            self._ensure_spectrogram()
        if HAS_PYQTGRAPH:
            if self._has_audio:
                self._stack.setCurrentIndex(0 if mode == "waveform" else 1)
            elif hasattr(self, "_empty_state"):
                self._stack.setCurrentWidget(self._empty_state)
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
                self._start_audio_load(path)
                return True
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
            self._set_info(tr("runtime.waveform.error", error=exc))
            return False

    def _start_audio_load(self, path: Path) -> None:
        """Queue a file decode and keep the latest selection authoritative."""
        self._audio_load_token += 1
        token = self._audio_load_token
        worker = InferenceWorker(_decode_waveform_file, str(path))
        self._audio_load_workers.add(worker)
        self._audio_load_worker = worker
        self._set_info(tr("runtime.waveform.loading_file", name=path.name))
        self.audio_load_started.emit(str(path))
        worker.progress.connect(
            lambda percent, t=token: self._on_audio_load_progress(t, percent)
        )
        worker.finished.connect(
            lambda payload, t=token, w=worker: self._on_audio_load_finished(t, w, payload)
        )
        worker.error.connect(
            lambda message, t=token, w=worker: self._on_audio_load_error(t, w, message)
        )
        worker.cancelled.connect(
            lambda t=token, w=worker: self._on_audio_load_cancelled(t, w)
        )
        try:
            worker.start()
        except Exception as exc:
            self._on_audio_load_error(token, worker, f"{type(exc).__name__}: {exc}")

    def _forget_worker_later(self, worker, registry: set) -> None:
        """Release a worker only after its QThread has actually stopped."""
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._forget_worker_later(worker, registry))
            return
        registry.discard(worker)
        if registry is self._audio_load_workers:
            event = self._pending_audio_load_events.pop(worker, None)
            if event is not None:
                self._finalize_audio_load_event(worker, event)

    @staticmethod
    def _disconnect_worker(worker) -> None:
        """Remove queued callbacks before a widget is destroyed."""
        for signal in (
            worker.progress,
            worker.finished,
            worker.error,
            worker.cancelled,
        ):
            try:
                signal.disconnect()
            except (RuntimeError, TypeError):
                pass

    def _on_audio_load_progress(self, token: int, percent: int) -> None:
        if not _qt_object_alive(self) or self._closed or token != self._audio_load_token:
            return
        self.audio_load_progress.emit(percent)
        self._set_info(tr("runtime.waveform.loading", percent=percent))

    def _on_audio_load_finished(self, token: int, worker, payload) -> None:
        self._pending_audio_load_events[worker] = ("finished", token, payload)
        self._forget_worker_later(worker, self._audio_load_workers)

    def _on_audio_load_error(self, token: int, worker, message: str) -> None:
        self._pending_audio_load_events[worker] = ("error", token, message)
        self._forget_worker_later(worker, self._audio_load_workers)

    def _on_audio_load_cancelled(self, token: int, worker) -> None:
        self._pending_audio_load_events[worker] = ("cancelled", token)
        self._forget_worker_later(worker, self._audio_load_workers)

    def _finalize_audio_load_event(self, worker, event) -> None:
        """Apply a terminal file-load event after the decoder thread stops."""
        kind, token, *payload = event
        if not _qt_object_alive(self) or self._closed or token != self._audio_load_token:
            return
        if self._audio_load_worker is worker:
            self._audio_load_worker = None
        if kind == "finished":
            try:
                audio, sample_rate = payload[0]
                normalized = self._normalize_audio(audio)
                if not HAS_PYQTGRAPH:
                    raise RuntimeError("Waveform display is unavailable")
                self._display_audio(normalized, self._validate_sample_rate(sample_rate))
                self._set_info("")
                self.audio_load_finished.emit(True)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._last_error = str(exc)
                self._set_info(tr("runtime.waveform.error", error=exc))
                self.audio_load_finished.emit(False)
        elif kind == "error":
            self._last_error = str(payload[0])
            self._set_info(tr("runtime.waveform.error", error=payload[0]))
            self.audio_load_finished.emit(False)
        elif kind == "cancelled":
            self._last_error = tr("runtime.waveform.audio_cancelled")
            self._set_info(self._last_error)
            self.audio_load_finished.emit(False)

    def cancel_audio_load(self, *, notify: bool = True) -> None:
        """Request cancellation without blocking the GUI thread."""
        worker = self._audio_load_worker
        if worker is None or not worker.isRunning():
            return
        self._audio_load_token += 1
        self._audio_load_worker = None
        worker.cancel()
        if notify:
            self._last_error = tr("runtime.waveform.audio_cancelled")
            self._set_info(self._last_error)
            self.audio_load_finished.emit(False)

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
        self._audio_data = audio if self._show_controls else None
        self._has_audio = True
        self._spectrogram_ready = False
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
            display = mono[::step].copy()
        else:
            step = 1
            display = mono.copy()

        time_axis = np.arange(len(display), dtype=np.float64) * step / sample_rate

        self._waveform_curve.setData(time_axis, display)
        self._waveform_plot.setXRange(0, self._duration)
        self._waveform_plot.setYRange(-1, 1)

        if self._selection_enabled:
            if self._selected_region is None:
                self.set_selection(0.0, self._duration, emit=False)
            else:
                self.set_selection(*self._selected_region, emit=False)

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

        if self._mode == "spectrogram":
            self._ensure_spectrogram()
        self._stack.setCurrentIndex(0 if self._mode == "waveform" else 1)

    def _ensure_spectrogram(self):
        """Build the mel image only when the full waveform enters that view."""
        if self._spectrogram_ready or not self._show_controls:
            return
        if self._audio_data is None:
            return
        audio = self._audio_data
        mono = audio.mean(axis=1, dtype=np.float32) if audio.ndim == 2 else audio
        if self._spectrogram_worker is not None and self._spectrogram_worker.isRunning():
            return
        self._spectrogram_token += 1
        token = self._spectrogram_token
        worker = InferenceWorker(_compute_spectrogram, mono.copy(), self._sample_rate)
        self._spectrogram_workers.add(worker)
        self._spectrogram_worker = worker
        self._set_info(tr("runtime.waveform.computing_start"))
        worker.progress.connect(
            lambda percent, t=token: self._on_spectrogram_progress(t, percent)
        )
        worker.finished.connect(
            lambda data, t=token, w=worker: self._on_spectrogram_finished(t, w, data)
        )
        worker.error.connect(
            lambda message, t=token, w=worker: self._on_spectrogram_error(t, w, message)
        )
        worker.cancelled.connect(
            lambda t=token, w=worker: self._on_spectrogram_cancelled(t, w)
        )
        try:
            worker.start()
        except Exception as exc:
            self._on_spectrogram_error(token, worker, f"{type(exc).__name__}: {exc}")

    def _on_spectrogram_progress(self, token: int, percent: int) -> None:
        if not _qt_object_alive(self) or self._closed or token != self._spectrogram_token:
            return
        self.spectrogram_progress.emit(percent)
        self._set_info(tr("runtime.waveform.computing", percent=percent))

    def _on_spectrogram_finished(self, token: int, worker, data: np.ndarray) -> None:
        self._forget_worker_later(worker, self._spectrogram_workers)
        if not _qt_object_alive(self) or self._closed or token != self._spectrogram_token:
            return
        self._spectrogram_worker = None
        self._apply_spectrogram(data, self._sample_rate)
        self._spectrogram_ready = True
        self._set_info("")

    def _on_spectrogram_error(self, token: int, worker, message: str) -> None:
        self._forget_worker_later(worker, self._spectrogram_workers)
        if not _qt_object_alive(self) or self._closed or token != self._spectrogram_token:
            return
        self._spectrogram_worker = None
        self._spectro_item.clear()
        self._set_info(tr("runtime.waveform.spectrogram_unavailable", error=message))

    def _on_spectrogram_cancelled(self, token: int, worker) -> None:
        self._forget_worker_later(worker, self._spectrogram_workers)
        if not _qt_object_alive(self) or self._closed or token != self._spectrogram_token:
            return
        self._spectrogram_worker = None
        self._set_info(tr("runtime.waveform.spectrogram_cancelled"))

    def _cancel_spectrogram(self) -> None:
        worker = self._spectrogram_worker
        if worker is None or not worker.isRunning():
            return
        self._spectrogram_token += 1
        self._spectrogram_worker = None
        worker.cancel()

    def _apply_spectrogram(self, S_dB: np.ndarray, sr: int) -> None:
        """Apply an already-computed mel image on the GUI thread."""
        if S_dB.size == 0:
            self._spectro_item.clear()
            return
        cmap = pg.ColorMap(
            pos=[0.0, 0.33, 0.66, 1.0],
            color=[
                QColor(Palette.CRUST),
                QColor(Palette.MANTLE),
                QColor(Palette.BLUE),
                QColor(Palette.YELLOW),
            ],
        )
        lut = cmap.getLookupTable(nPts=256)
        self._spectro_item.setImage(S_dB.T, autoLevels=True)
        self._spectro_item.setLookupTable(lut)
        self._spectro_item.setTransform(
            QTransform().scale(
                self._duration / S_dB.shape[1],
                sr / (2 * S_dB.shape[0]),
            )
        )

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

    def set_selection_enabled(self, enabled: bool = True) -> None:
        """Show draggable start/end handles for an editable time region."""
        self._selection_enabled = bool(enabled)
        if not HAS_PYQTGRAPH or self._selection_region is None:
            return
        if not self._selection_enabled:
            self._selection_region.hide()
            self._selected_region = None
            return
        if self._duration > 0:
            start, end = self._selected_region or (0.0, self._duration)
            self.set_selection(start, end, emit=False)

    def set_selection(self, start: float, end: float, *, emit: bool = True) -> None:
        """Set and optionally announce the current region in seconds."""
        if self._duration <= 0:
            return
        start_value = max(0.0, min(float(start), self._duration))
        end_value = max(0.0, min(float(end), self._duration))
        if end_value < start_value:
            start_value, end_value = end_value, start_value
        if end_value <= start_value:
            end_value = min(self._duration, start_value + 1.0 / self._sample_rate)
        self._selected_region = (start_value, end_value)
        if HAS_PYQTGRAPH and self._selection_region is not None:
            self._selection_region.blockSignals(True)
            self._selection_region.setRegion(self._selected_region)
            self._selection_region.show()
            self._selection_region.blockSignals(False)
        if emit:
            self.region_selected.emit(start_value, end_value)

    @property
    def selected_region(self) -> tuple[float, float] | None:
        """Return the selected start/end seconds, if region editing is enabled."""
        return self._selected_region if self._selection_enabled else None

    def _on_selection_region_changed(self):
        if not self._selection_enabled or self._selection_region is None or self._duration <= 0:
            return
        start, end = self._selection_region.getRegion()
        self.set_selection(start, end)
        self._set_info(tr("runtime.waveform.selected", start=start, end=end))

    # ── Keyboard operation and screen-reader state ─────────────────────────────

    SEEK_STEP_SECONDS = 1.0
    SEEK_PAGE_SECONDS = 10.0

    @property
    def playback_position(self) -> float:
        return self._playback_pos

    def accessible_state(self) -> str:
        """One sentence a screen reader can read for the current view."""
        if not self.has_audio or self._duration <= 0:
            return tr("runtime.waveform.accessible_no_audio")
        percent = int(round(100 * self._playback_pos / self._duration))
        state = tr(
            "runtime.waveform.accessible_state",
            mode=tr(f"runtime.waveform.{self._mode}"),
            position=self._playback_pos,
            duration=self._duration,
            percent=percent,
        )
        if self.selected_region:
            start, end = self.selected_region
            state += tr(
                "runtime.waveform.accessible_selection",
                start=start,
                end=end,
            )
        return state

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
        self.cancel_audio_load(notify=False)
        self._cancel_spectrogram()
        self._audio_load_token += 1
        self._spectrogram_token += 1
        if HAS_PYQTGRAPH:
            self._waveform_curve.setData([], [])
            self._spectro_item.clear()
            self._spectro_item.setTransform(QTransform())
            self._cursor_line.hide()
            self._spectro_cursor.hide()
            if self._selection_region is not None:
                self._selection_region.hide()
            if hasattr(self, "_empty_state"):
                self._stack.setCurrentWidget(self._empty_state)
        self._audio_data = None
        self._has_audio = False
        self._spectrogram_ready = False
        self._duration = 0.0
        self._selected_region = None
        self._last_error = ""
        self._set_info("")

    def closeEvent(self, event):
        self._closed = True
        workers = set(self._audio_load_workers) | set(self._spectrogram_workers)
        for worker in workers:
            self._disconnect_worker(worker)
            if worker.isRunning():
                worker.cancel()
        self._audio_load_worker = None
        self._spectrogram_worker = None
        super().closeEvent(event)

    def _set_info(self, text: str) -> None:
        if self._show_controls and hasattr(self, "_info_label"):
            self._info_label.setText(text)

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def has_audio(self) -> bool:
        # The thumbnail intentionally releases its source buffer after the
        # waveform display arrays are built, so retain a separate state bit.
        return self._has_audio or self._audio_data is not None

    @property
    def empty_state(self) -> EmptyStateWidget | None:
        """Return the first-use card so parent views can tailor its action."""
        return getattr(self, "_empty_state", None)

    def set_empty_state(
        self,
        title: str,
        message: str,
        action_text: str = "",
    ) -> None:
        """Customize the first-use copy without changing audio behavior."""
        if self.empty_state is not None:
            self.empty_state.set_state(title, message, action_text)

    @property
    def last_error(self) -> str:
        return self._last_error


class MiniWaveform(QWidget):
    """Compact waveform thumbnail for batch view cards."""
    clicked = Signal()
    double_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
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
