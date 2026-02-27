"""
Slunder Studio v0.0.2 — FluidSynth Engine
MIDI-to-audio rendering via FluidSynth with SoundFont support.
Renders MidiData to WAV/numpy arrays for playback and export.
"""
import os
import time
import struct
import wave
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.settings import get_config_dir
from core.midi_utils import MidiData, save_midi


@dataclass
class RenderSettings:
    """Settings for FluidSynth rendering."""
    sample_rate: int = 44100
    channels: int = 2  # stereo
    gain: float = 0.8  # 0.0-1.0
    reverb: bool = True
    reverb_room: float = 0.6
    reverb_damp: float = 0.4
    reverb_level: float = 0.7
    chorus: bool = True
    chorus_depth: float = 8.0
    chorus_level: float = 2.0


class FluidSynthEngine:
    """
    FluidSynth wrapper for MIDI-to-audio rendering.
    Supports multiple SoundFonts, per-track mute/solo, and real-time preview.
    """

    def __init__(self):
        self._synth = None
        self._soundfont_id: Optional[int] = None
        self._soundfont_path: Optional[str] = None
        self._settings = RenderSettings()
        self._render_dir = os.path.join(get_config_dir(), "generations", "midi_renders")
        os.makedirs(self._render_dir, exist_ok=True)

    @property
    def is_ready(self) -> bool:
        return self._synth is not None and self._soundfont_id is not None

    def initialize(self, soundfont_path: str, settings: Optional[RenderSettings] = None):
        """Initialize FluidSynth with a SoundFont."""
        from core.deps import ensure
        ensure("fluidsynth", pip_name="pyfluidsynth")
        import fluidsynth

        if settings:
            self._settings = settings

        s = self._settings
        self._synth = fluidsynth.Synth(samplerate=float(s.sample_rate), gain=s.gain)

        # Effects
        if s.reverb:
            self._synth.set_reverb(s.reverb_room, s.reverb_damp, 1.0, s.reverb_level)
        if s.chorus:
            self._synth.set_chorus(3, s.chorus_depth, 0.3, s.chorus_level, 0)

        # Load SoundFont
        sf_id = self._synth.sfload(soundfont_path)
        if sf_id < 0:
            raise RuntimeError(f"Failed to load SoundFont: {soundfont_path}")

        self._soundfont_id = sf_id
        self._soundfont_path = soundfont_path

    def shutdown(self):
        """Clean up FluidSynth resources."""
        if self._synth is not None:
            self._synth.delete()
            self._synth = None
            self._soundfont_id = None

    def set_soundfont(self, soundfont_path: str):
        """Switch to a different SoundFont."""
        if self._synth is None:
            self.initialize(soundfont_path)
            return

        # Unload current
        if self._soundfont_id is not None:
            self._synth.sfunload(self._soundfont_id)

        sf_id = self._synth.sfload(soundfont_path)
        if sf_id < 0:
            raise RuntimeError(f"Failed to load SoundFont: {soundfont_path}")
        self._soundfont_id = sf_id
        self._soundfont_path = soundfont_path

    def render_to_numpy(self, midi_data: MidiData,
                        mute_tracks: Optional[set[int]] = None,
                        solo_track: Optional[int] = None,
                        progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Render MidiData to numpy float32 array.
        Returns shape (samples, channels).
        """
        if not self.is_ready:
            raise RuntimeError("FluidSynth not initialized. Call initialize() first.")

        s = self._settings
        duration = midi_data.duration + 1.0  # extra second for release
        total_samples = int(duration * s.sample_rate)

        if progress_callback:
            progress_callback(0.05, "Preparing MIDI events...")

        # Build event list: (time, type, channel, param1, param2)
        events = []

        for track_idx, track in enumerate(midi_data.tracks):
            # Handle mute/solo
            if solo_track is not None and track_idx != solo_track:
                continue
            if mute_tracks and track_idx in mute_tracks:
                continue

            ch = track.channel if track.channel < 16 else track_idx % 16
            if track.is_drum:
                ch = 9

            # Program change at time 0
            events.append((0.0, "prog", ch, track.program, 0))

            for note in track.notes:
                events.append((note.start, "on", ch, note.pitch, note.velocity))
                events.append((note.end, "off", ch, note.pitch, 0))

        events.sort(key=lambda e: e[0])

        if progress_callback:
            progress_callback(0.1, "Rendering audio...")

        # Render in chunks
        chunk_size = 1024
        output = np.zeros((total_samples, s.channels), dtype=np.float32)
        event_idx = 0
        sample_pos = 0

        while sample_pos < total_samples:
            current_time = sample_pos / s.sample_rate

            # Process events up to current time
            while event_idx < len(events) and events[event_idx][0] <= current_time:
                evt = events[event_idx]
                t, etype, ch, p1, p2 = evt
                if etype == "prog":
                    self._synth.program_select(ch, self._soundfont_id, 0, p1)
                elif etype == "on":
                    self._synth.noteon(ch, p1, p2)
                elif etype == "off":
                    self._synth.noteoff(ch, p1)
                event_idx += 1

            # Synthesize chunk
            remaining = min(chunk_size, total_samples - sample_pos)
            samples = self._synth.get_samples(remaining)

            # FluidSynth returns interleaved stereo int16
            if s.channels == 2:
                chunk = np.frombuffer(samples, dtype=np.int16).reshape(-1, 2)
            else:
                chunk = np.frombuffer(samples, dtype=np.int16).reshape(-1, 1)

            chunk_f = chunk.astype(np.float32) / 32768.0
            end_pos = sample_pos + chunk_f.shape[0]
            output[sample_pos:end_pos] = chunk_f
            sample_pos = end_pos

            if progress_callback and sample_pos % (chunk_size * 100) == 0:
                prog = 0.1 + 0.85 * (sample_pos / total_samples)
                progress_callback(prog, f"Rendering... {int(prog * 100)}%")

        # All notes off
        for ch in range(16):
            self._synth.cc(ch, 123, 0)  # All notes off

        if progress_callback:
            progress_callback(1.0, "Render complete")

        return output

    def render_to_wav(self, midi_data: MidiData, output_path: str,
                      mute_tracks: Optional[set[int]] = None,
                      solo_track: Optional[int] = None,
                      progress_callback: Optional[Callable] = None) -> str:
        """Render MidiData to WAV file."""
        audio = self.render_to_numpy(midi_data, mute_tracks, solo_track, progress_callback)

        s = self._settings
        int_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(s.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(s.sample_rate)
            wf.writeframes(int_audio.tobytes())

        return output_path

    def render_track(self, midi_data: MidiData, track_idx: int,
                     progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Render a single track in isolation."""
        return self.render_to_numpy(midi_data, solo_track=track_idx,
                                    progress_callback=progress_callback)


# ── Fallback Renderer (no FluidSynth) ─────────────────────────────────────────

def render_midi_simple(midi_data: MidiData, sample_rate: int = 44100) -> np.ndarray:
    """
    Simple sine-wave MIDI renderer for preview when FluidSynth is unavailable.
    No SoundFont needed — just maps pitches to sine waves.
    """
    duration = midi_data.duration + 0.5
    total_samples = int(duration * sample_rate)
    output = np.zeros(total_samples, dtype=np.float32)

    for track in midi_data.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            freq = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            start_sample = int(note.start * sample_rate)
            end_sample = int(note.end * sample_rate)
            n_samples = end_sample - start_sample
            if n_samples <= 0 or start_sample >= total_samples:
                continue

            t = np.arange(n_samples) / sample_rate
            # Simple ADSR envelope
            attack = min(0.01, note.duration * 0.1)
            release = min(0.05, note.duration * 0.2)
            env = np.ones(n_samples)
            attack_samples = int(attack * sample_rate)
            release_samples = int(release * sample_rate)
            if attack_samples > 0:
                env[:attack_samples] = np.linspace(0, 1, attack_samples)
            if release_samples > 0 and release_samples < n_samples:
                env[-release_samples:] = np.linspace(1, 0, release_samples)

            amp = (note.velocity / 127.0) * 0.15
            wave_data = amp * env * np.sin(2 * np.pi * freq * t)

            # Mix into output
            end_idx = min(start_sample + n_samples, total_samples)
            output[start_sample:end_idx] += wave_data[:end_idx - start_sample]

    # Clip and convert to stereo
    output = np.clip(output, -1.0, 1.0)
    return np.column_stack([output, output])


# ── SoundFont Discovery ───────────────────────────────────────────────────────

SOUNDFONT_SEARCH_PATHS = [
    # Linux
    "/usr/share/sounds/sf2",
    "/usr/share/soundfonts",
    "/usr/local/share/soundfonts",
    # Windows
    os.path.expandvars(r"%PROGRAMFILES%\soundfonts"),
    os.path.expandvars(r"%LOCALAPPDATA%\soundfonts"),
    # macOS
    os.path.expanduser("~/Library/Audio/Sounds/Banks"),
    # App-local
    os.path.join(get_config_dir(), "soundfonts"),
]


def find_soundfonts() -> list[dict]:
    """Discover available SoundFont files."""
    results = []
    seen = set()

    for search_dir in SOUNDFONT_SEARCH_PATHS:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if f.lower().endswith((".sf2", ".sf3")):
                    path = os.path.join(root, f)
                    if path not in seen:
                        seen.add(path)
                        size_mb = os.path.getsize(path) / (1024 * 1024)
                        results.append({
                            "name": f,
                            "path": path,
                            "size_mb": round(size_mb, 1),
                        })

    return sorted(results, key=lambda x: x["name"])


def get_default_soundfont() -> Optional[str]:
    """Find the best default SoundFont."""
    # Check app-local first
    local_sf = os.path.join(get_config_dir(), "soundfonts")
    if os.path.isdir(local_sf):
        for f in sorted(os.listdir(local_sf)):
            if f.lower().endswith((".sf2", ".sf3")):
                return os.path.join(local_sf, f)

    # Search system
    fonts = find_soundfonts()
    if fonts:
        # Prefer GeneralUser GS or FluidR3
        for sf in fonts:
            name_lower = sf["name"].lower()
            if "generaluser" in name_lower or "fluidr3" in name_lower:
                return sf["path"]
        return fonts[0]["path"]

    return None


# ── High-Level Functions ───────────────────────────────────────────────────────

_engine: Optional[FluidSynthEngine] = None


def get_fluidsynth() -> FluidSynthEngine:
    global _engine
    if _engine is None:
        _engine = FluidSynthEngine()
    return _engine


def render_midi_to_audio(midi_data: MidiData,
                         soundfont_path: Optional[str] = None,
                         output_path: Optional[str] = None,
                         progress_callback: Optional[Callable] = None) -> np.ndarray:
    """
    Render MIDI to audio. Uses FluidSynth if available, falls back to sine waves.
    Called by InferenceWorker.
    """
    # Try FluidSynth
    try:
        engine = get_fluidsynth()
        sf_path = soundfont_path or get_default_soundfont()

        if sf_path and os.path.isfile(sf_path):
            if not engine.is_ready:
                engine.initialize(sf_path)

            if output_path:
                engine.render_to_wav(midi_data, output_path,
                                     progress_callback=progress_callback)
                # Also return numpy
                import wave as wave_mod
                with wave_mod.open(output_path, "r") as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, 2)
                    return audio.astype(np.float32) / 32768.0

            return engine.render_to_numpy(midi_data, progress_callback=progress_callback)
    except Exception:
        pass  # Fall through to simple renderer

    # Fallback
    if progress_callback:
        progress_callback(0.1, "Rendering preview (sine waves)...")

    audio = render_midi_simple(midi_data)

    if progress_callback:
        progress_callback(1.0, "Done")

    if output_path:
        int_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(output_path, "w") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(int_audio.tobytes())

    return audio
