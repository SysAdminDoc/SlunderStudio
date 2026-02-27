"""
Slunder Studio v0.0.2 — Audio Engine
sounddevice + soundfile playback with transport controls,
seek, loop, and waveform data extraction for mini-display.
"""
import os
import threading
import numpy as np
from typing import Optional, Callable
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QTimer

# Lazy imports for audio libraries
_sd = None
_sf = None


def _ensure_audio_libs():
    """Lazy-import sounddevice and soundfile."""
    global _sd, _sf
    if _sd is None:
        import sounddevice as _sd
    if _sf is None:
        import soundfile as _sf


class AudioEngine(QObject):
    """
    Central audio playback engine.
    Supports play/pause/stop/seek/loop for NumPy arrays and audio files.

    Signals:
        position_changed(float)   - current playback position in seconds
        playback_started()
        playback_paused()
        playback_stopped()
        playback_finished()       - reached end of audio
        duration_changed(float)   - total duration in seconds
        waveform_ready(ndarray)   - downsampled waveform data for visualization
    """
    position_changed = Signal(float)
    playback_started = Signal()
    playback_paused = Signal()
    playback_stopped = Signal()
    playback_finished = Signal()
    duration_changed = Signal(float)
    waveform_ready = Signal(object)

    _instance: Optional["AudioEngine"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True

        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 48000
        self._position: int = 0  # current sample position
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._loop_enabled: bool = False
        self._loop_start: int = 0
        self._loop_end: int = 0
        self._volume: float = 1.0
        self._stream = None
        self._lock = threading.Lock()
        self._source_path: Optional[str] = None

        # Position update timer (fires ~30 times/sec during playback)
        self._pos_timer = QTimer(self)
        self._pos_timer.setInterval(33)
        self._pos_timer.timeout.connect(self._emit_position)

    @property
    def is_playing(self) -> bool:
        return self._is_playing and not self._is_paused

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self._audio_data is None:
            return 0.0
        return len(self._audio_data) / self._sample_rate

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        return self._position / self._sample_rate if self._sample_rate > 0 else 0.0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def volume(self) -> float:
        return self._volume

    @volume.setter
    def volume(self, value: float):
        self._volume = max(0.0, min(1.0, value))

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_file(self, file_path: str) -> bool:
        """Load an audio file for playback."""
        _ensure_audio_libs()
        try:
            data, sr = _sf.read(file_path, dtype="float32", always_2d=True)
            self._source_path = file_path
            return self._load_data(data, sr)
        except Exception as e:
            print(f"AudioEngine: Failed to load {file_path}: {e}")
            return False

    def load_array(self, data: np.ndarray, sample_rate: int) -> bool:
        """Load a NumPy array for playback. Shape: (samples,) or (samples, channels)."""
        self._source_path = None
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return self._load_data(data.astype(np.float32), sample_rate)

    def _load_data(self, data: np.ndarray, sample_rate: int) -> bool:
        """Internal: set audio data and emit signals."""
        self.stop()
        with self._lock:
            self._audio_data = data
            self._sample_rate = sample_rate
            self._position = 0
            self._loop_start = 0
            self._loop_end = len(data)

        self.duration_changed.emit(self.duration)
        self._generate_waveform()
        return True

    def _generate_waveform(self):
        """Generate downsampled waveform data for the mini-display."""
        if self._audio_data is None:
            return
        # Downsample to ~2000 points for visualization
        target_points = 2000
        data = self._audio_data
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono mix
        step = max(1, len(data) // target_points)
        # Compute envelope (max of absolute values per chunk)
        chunks = len(data) // step
        if chunks == 0:
            return
        trimmed = data[:chunks * step].reshape(chunks, step)
        envelope = np.max(np.abs(trimmed), axis=1)
        self.waveform_ready.emit(envelope)

    # ── Transport Controls ─────────────────────────────────────────────────────

    def play(self):
        """Start or resume playback."""
        if self._audio_data is None:
            return

        _ensure_audio_libs()

        if self._is_paused:
            self._is_paused = False
            self.playback_started.emit()
            self._pos_timer.start()
            return

        self.stop()
        self._is_playing = True
        self._is_paused = False

        channels = self._audio_data.shape[1] if self._audio_data.ndim > 1 else 1

        def _callback(outdata, frames, time_info, status):
            with self._lock:
                if self._is_paused or not self._is_playing:
                    outdata[:] = 0
                    return

                end = self._position + frames
                remaining = len(self._audio_data) - self._position

                if remaining <= 0:
                    if self._loop_enabled:
                        self._position = self._loop_start
                        end = self._position + frames
                    else:
                        outdata[:] = 0
                        self._is_playing = False
                        return

                if end > len(self._audio_data):
                    # Partial fill at end
                    available = len(self._audio_data) - self._position
                    chunk = self._audio_data[self._position:self._position + available]
                    outdata[:available] = chunk * self._volume
                    outdata[available:] = 0

                    if self._loop_enabled:
                        self._position = self._loop_start
                    else:
                        self._position = len(self._audio_data)
                else:
                    outdata[:] = self._audio_data[self._position:end] * self._volume
                    self._position = end

        try:
            self._stream = _sd.OutputStream(
                samplerate=self._sample_rate,
                channels=channels,
                dtype="float32",
                callback=_callback,
                blocksize=1024,
            )
            self._stream.start()
            self.playback_started.emit()
            self._pos_timer.start()
        except Exception as e:
            print(f"AudioEngine: Playback error: {e}")
            self._is_playing = False

    def pause(self):
        """Pause playback."""
        if self._is_playing and not self._is_paused:
            self._is_paused = True
            self._pos_timer.stop()
            self.playback_paused.emit()

    def stop(self):
        """Stop playback and reset position."""
        self._is_playing = False
        self._is_paused = False
        self._pos_timer.stop()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        with self._lock:
            self._position = 0

        self.playback_stopped.emit()
        self.position_changed.emit(0.0)

    def toggle_play(self):
        """Toggle between play and pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float):
        """Seek to a specific time in seconds."""
        if self._audio_data is None:
            return
        with self._lock:
            sample = int(seconds * self._sample_rate)
            self._position = max(0, min(sample, len(self._audio_data)))
        self.position_changed.emit(self.position)

    def seek_relative(self, delta_seconds: float):
        """Seek relative to current position."""
        self.seek(self.position + delta_seconds)

    # ── Loop ───────────────────────────────────────────────────────────────────

    def set_loop(self, enabled: bool, start_sec: float = 0.0, end_sec: float = -1.0):
        """Enable/disable loop with optional region."""
        self._loop_enabled = enabled
        if self._audio_data is not None:
            self._loop_start = int(start_sec * self._sample_rate)
            if end_sec < 0:
                self._loop_end = len(self._audio_data)
            else:
                self._loop_end = int(end_sec * self._sample_rate)

    @property
    def loop_enabled(self) -> bool:
        return self._loop_enabled

    # ── Export ─────────────────────────────────────────────────────────────────

    def save_to_file(self, file_path: str, audio_data: Optional[np.ndarray] = None,
                     sample_rate: Optional[int] = None) -> bool:
        """Save audio data to a file. Uses current loaded audio if none provided."""
        _ensure_audio_libs()
        data = audio_data if audio_data is not None else self._audio_data
        sr = sample_rate or self._sample_rate

        if data is None:
            return False

        try:
            ext = Path(file_path).suffix.lower()
            if ext in (".wav", ".flac", ".ogg"):
                _sf.write(file_path, data, sr)
            elif ext == ".mp3":
                # MP3 requires pydub + ffmpeg
                from pydub import AudioSegment
                # Convert to 16-bit PCM for pydub
                pcm = (data * 32767).astype(np.int16)
                channels = pcm.shape[1] if pcm.ndim > 1 else 1
                seg = AudioSegment(
                    pcm.tobytes(),
                    frame_rate=sr,
                    sample_width=2,
                    channels=channels,
                )
                seg.export(file_path, format="mp3", bitrate="320k")
            else:
                _sf.write(file_path, data, sr)
            return True
        except Exception as e:
            print(f"AudioEngine: Save error: {e}")
            return False

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit_position(self):
        """Emit current position for UI updates."""
        if self._is_playing and not self._is_paused:
            pos = self.position
            self.position_changed.emit(pos)
            # Check if playback finished
            if self._audio_data is not None and self._position >= len(self._audio_data):
                if not self._loop_enabled:
                    self.stop()
                    self.playback_finished.emit()

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self._audio_data = None


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        seconds = 0
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
