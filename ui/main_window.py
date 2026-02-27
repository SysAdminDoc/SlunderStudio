"""
Slunder Studio v0.0.2 — Main Window
QMainWindow shell with animated sidebar navigation, stacked module views,
global audio transport bar, VRAM status indicator, and drag-and-drop support.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QPushButton, QFrame, QSlider, QSizePolicy,
    QStatusBar, QApplication,
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent

from core.settings import Settings, APP_VERSION
from core.audio_engine import AudioEngine, format_time
from core.model_manager import ModelManager
from ui.theme import Palette, build_stylesheet
from ui.toast import ToastManager
from ui.model_hub import ModelHubView
from ui.settings_view import SettingsView
from ui.lyrics_view import LyricsView
from ui.song_forge_view import SongForgeView
from ui.midi_studio_view import MidiStudioView
from ui.vocal_suite_view import VocalSuiteView
from ui.sfx_view import SFXView
from ui.mixer_view import MixerView
from ui.project_manager import ProjectManagerView
from ui.ai_producer_view import AIProducerView


# ── Sidebar Navigation ────────────────────────────────────────────────────────

class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and label."""

    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setText(f"  {icon_text}  {label}")
        self.setFixedHeight(44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 12))


class Sidebar(QWidget):
    """Left sidebar with navigation buttons."""

    page_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(4)

        # Logo/title
        logo = QLabel("  \U0001f3b5  SLUNDER")
        logo.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 800;
            color: {Palette.BLUE};
            padding: 8px 4px 16px 4px;
            letter-spacing: 2px;
        """)
        layout.addWidget(logo)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {Palette.SURFACE1}; max-height: 1px;")
        layout.addWidget(sep)
        layout.addSpacing(8)

        # Navigation buttons
        self._buttons: list[SidebarButton] = []
        nav_items = [
            ("\U0001f3a4", "Lyrics"),
            ("\U0001f3b6", "Song Forge"),
            ("\U0001f3b9", "MIDI Studio"),
            ("\U0001f399", "Vocals"),
            ("\U0001f4a5", "SFX"),
            ("\U0001f39b", "Mixer"),
            ("\U0001f916", "AI Producer"),
            ("\U0001f4c1", "Projects"),
        ]

        for i, (icon, label) in enumerate(nav_items):
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._on_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Bottom section — separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"background-color: {Palette.SURFACE1}; max-height: 1px;")
        layout.addWidget(sep2)
        layout.addSpacing(4)

        # Bottom nav
        bottom_items = [
            ("\U0001f4e6", "Model Hub"),
            ("\u2699", "Settings"),
        ]
        for i, (icon, label) in enumerate(bottom_items):
            btn = SidebarButton(icon, label)
            idx = len(nav_items) + i
            btn.clicked.connect(lambda checked, idx=idx: self._on_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        # Select first button
        if self._buttons:
            self._buttons[0].setChecked(True)

    def _on_clicked(self, index: int):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)
        self.page_selected.emit(index)

    def select_page(self, index: int):
        """Programmatically select a page."""
        self._on_clicked(index)


# ── Transport Bar ──────────────────────────────────────────────────────────────

class TransportBar(QWidget):
    """Global audio transport bar pinned to bottom of window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transportBar")
        self.setFixedHeight(64)
        self._audio = AudioEngine()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # Transport buttons
        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setObjectName("transportBtn")
        self._play_btn.setFixedSize(40, 40)
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u25A0")
        self._stop_btn.setObjectName("transportBtn")
        self._stop_btn.setFixedSize(40, 40)
        self._stop_btn.clicked.connect(self._audio.stop)
        layout.addWidget(self._stop_btn)

        # Time display
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setStyleSheet(f"""
            font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
            font-size: 13px;
            color: {Palette.SUBTEXT0};
            min-width: 100px;
        """)
        layout.addWidget(self._time_label)

        # Seek slider
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        self._seek_slider.setFixedHeight(20)
        layout.addWidget(self._seek_slider, 1)

        # Loop toggle
        self._loop_btn = QPushButton("\U0001f501")
        self._loop_btn.setObjectName("transportBtn")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setFixedSize(40, 40)
        self._loop_btn.toggled.connect(lambda v: self._audio.set_loop(v))
        layout.addWidget(self._loop_btn)

        # Volume
        vol_icon = QLabel("\U0001f50a")
        vol_icon.setStyleSheet(f"font-size: 14px; color: {Palette.SUBTEXT0};")
        layout.addWidget(vol_icon)

        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(100)
        self._vol_slider.setFixedWidth(100)
        self._vol_slider.valueChanged.connect(lambda v: setattr(self._audio, 'volume', v / 100))
        layout.addWidget(self._vol_slider)

        # Connect audio signals
        self._audio.position_changed.connect(self._on_position)
        self._audio.duration_changed.connect(self._on_duration)
        self._audio.playback_started.connect(lambda: self._play_btn.setText("\u23F8"))
        self._audio.playback_paused.connect(lambda: self._play_btn.setText("\u25B6"))
        self._audio.playback_stopped.connect(self._on_stopped)
        self._audio.playback_finished.connect(self._on_stopped)

        self._duration = 0.0

    def _toggle_play(self):
        self._audio.toggle_play()

    def _on_seek(self, value):
        if self._duration > 0:
            self._audio.seek(value / 1000 * self._duration)

    def _on_position(self, pos: float):
        self._time_label.setText(f"{format_time(pos)} / {format_time(self._duration)}")
        if self._duration > 0 and not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(int(pos / self._duration * 1000))

    def _on_duration(self, dur: float):
        self._duration = dur
        self._time_label.setText(f"0:00 / {format_time(dur)}")

    def _on_stopped(self):
        self._play_btn.setText("\u25B6")
        self._seek_slider.setValue(0)
        self._time_label.setText(f"0:00 / {format_time(self._duration)}")


# ── Placeholder Pages ──────────────────────────────────────────────────────────

class PlaceholderPage(QWidget):
    """Placeholder for modules not yet built."""

    def __init__(self, title: str, description: str, phase: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 60, 40, 40)
        layout.setSpacing(16)

        icon = QLabel("\U0001f6a7")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        title_label = QLabel(title)
        title_label.setObjectName("heading")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("subheading")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        phase_label = QLabel(f"Coming in {phase}")
        phase_label.setObjectName("caption")
        phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        phase_label.setStyleSheet(f"font-size: 14px; color: {Palette.BLUE}; font-weight: 600;")
        layout.addWidget(phase_label)

        layout.addStretch()


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Slunder Studio main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Slunder Studio v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self._settings = Settings()
        self._model_mgr = ModelManager()

        # Toast manager
        self.toast_mgr = ToastManager(self)

        self._build_ui()
        self._start_gpu_monitor()

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Content area (sidebar + stacked pages)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        self._sidebar.page_selected.connect(self._on_page_selected)
        content.addWidget(self._sidebar)

        # Stacked pages
        self._pages = QStackedWidget()
        self._create_pages()
        content.addWidget(self._pages, 1)

        main_layout.addLayout(content, 1)

        # Transport bar
        self._transport = TransportBar()
        main_layout.addWidget(self._transport)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._gpu_status_label = QLabel("GPU: Detecting...")
        self._gpu_status_label.setStyleSheet(f"font-size: 11px; color: {Palette.OVERLAY0};")
        self._status_bar.addPermanentWidget(self._gpu_status_label)

        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet(f"font-size: 11px; color: {Palette.BLUE};")
        self._status_bar.addPermanentWidget(self._vram_label)

    def _create_pages(self):
        """Create all module pages (placeholders for future phases)."""
        # Page 0: Lyrics (Phase 2 — LIVE)
        self._lyrics_view = LyricsView(toast_mgr=self.toast_mgr)
        self._lyrics_view.send_to_forge.connect(self._on_send_to_forge)
        self._pages.addWidget(self._lyrics_view)

        # Page 1: Song Forge (Phase 3)
        self._song_forge_view = SongForgeView(toast_mgr=self.toast_mgr)
        self._song_forge_view.send_to_vocals.connect(self._on_send_to_vocals)
        self._pages.addWidget(self._song_forge_view)

        # Page 2: MIDI Studio (Phase 4)
        self._midi_studio_view = MidiStudioView()
        self._midi_studio_view.send_to_forge.connect(self._on_midi_to_forge)
        self._midi_studio_view.send_to_vocals.connect(self._on_send_to_vocals)
        self._pages.addWidget(self._midi_studio_view)

        # Page 3: Vocals (Phase 5)
        self._vocal_suite_view = VocalSuiteView()
        self._vocal_suite_view.send_to_forge.connect(self._on_vocal_to_forge)
        self._vocal_suite_view.send_to_mixer.connect(self._on_vocal_to_mixer)
        self._pages.addWidget(self._vocal_suite_view)

        # Page 4: SFX (Phase 6)
        self._sfx_view = SFXView()
        self._sfx_view.send_to_mixer.connect(self._on_sfx_to_mixer)
        self._pages.addWidget(self._sfx_view)

        # Page 5: Mixer (Phase 6)
        self._mixer_view = MixerView()
        self._pages.addWidget(self._mixer_view)

        # Page 6: AI Producer (Phase 7)
        self._ai_producer_view = AIProducerView()
        self._pages.addWidget(self._ai_producer_view)

        # Page 7: Projects (Phase 6)
        self._project_mgr_view = ProjectManagerView()
        self._pages.addWidget(self._project_mgr_view)

        # Page 8: Model Hub (built now)
        self._model_hub = ModelHubView(toast_mgr=self.toast_mgr)
        self._pages.addWidget(self._model_hub)

        # Page 9: Settings (built now)
        self._settings_view = SettingsView(toast_mgr=self.toast_mgr)
        self._pages.addWidget(self._settings_view)

    def _on_page_selected(self, index: int):
        """Switch to the selected page."""
        if 0 <= index < self._pages.count():
            self._pages.setCurrentIndex(index)

    # ── GPU Monitoring ─────────────────────────────────────────────────────────

    def _start_gpu_monitor(self):
        """Start periodic GPU status updates."""
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_status)
        self._gpu_timer.start(2000)
        self._update_gpu_status()

    def _update_gpu_status(self):
        """Update GPU status in the status bar."""
        gpu = self._model_mgr.get_gpu_status()
        if gpu.get("available"):
            self._gpu_status_label.setText(f"\U0001f4bb {gpu['name']}")
            used = gpu["used_gb"]
            total = gpu["total_gb"]
            pct = (used / total * 100) if total > 0 else 0
            color = Palette.GREEN if pct < 60 else (Palette.YELLOW if pct < 85 else Palette.RED)
            self._vram_label.setText(f"VRAM: {used:.1f} / {total:.1f} GB ({pct:.0f}%)")
            self._vram_label.setStyleSheet(f"font-size: 11px; color: {color};")

            current = gpu.get("current_model_name")
            if current:
                self._status_bar.showMessage(f"Active model: {current}", 0)
            else:
                self._status_bar.showMessage("No model loaded", 0)
        else:
            self._gpu_status_label.setText("\u26a0 No GPU")
            self._vram_label.setText("")
            self._status_bar.showMessage("CUDA not available — running on CPU", 0)

    # ── Drag and Drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
                if ext in ("wav", "flac", "mp3", "ogg", "aiff"):
                    audio = AudioEngine()
                    if audio.load_file(path):
                        self.toast_mgr.success(f"Loaded: {path.rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}")
                        audio.play()
                elif ext in ("mid", "midi"):
                    self.toast_mgr.info("MIDI file detected — loading in MIDI Studio")
                    self._sidebar.select_page(2)
                    from core.midi_utils import load_midi as load_midi_file
                    try:
                        midi_data = load_midi_file(path)
                        self._midi_studio_view.set_midi_data(midi_data)
                    except Exception:
                        self.toast_mgr.warning("Failed to load MIDI file")
                else:
                    self.toast_mgr.warning(f"Unsupported file type: .{ext}")

    # ── Cross-Module Routing ──────────────────────────────────────────────────

    def _on_send_to_forge(self, lyrics_text: str):
        """Route lyrics to Song Forge page."""
        self._sidebar.select_page(1)  # Switch to Song Forge page
        self._song_forge_view.set_lyrics(lyrics_text)

    def _on_midi_to_forge(self, audio_path: str):
        """Route rendered MIDI audio to Song Forge as reference."""
        self._sidebar.select_page(1)
        self.toast_mgr.info("MIDI render sent to Song Forge as reference")

    def _on_send_to_vocals(self, audio_path: str):
        """Route audio from Song Forge/MIDI Studio to Vocal Suite."""
        self._sidebar.select_page(3)  # Switch to Vocals page
        self._vocal_suite_view.set_audio(audio_path)
        self.toast_mgr.info("Audio sent to Vocal Suite")

    def _on_vocal_to_forge(self, audio_path: str):
        """Route processed vocals back to Song Forge."""
        self._sidebar.select_page(1)
        self.toast_mgr.info("Vocals sent to Song Forge as reference")

    def _on_vocal_to_mixer(self, audio_path: str):
        """Route vocals to Mixer."""
        self._sidebar.select_page(5)
        self._mixer_view.add_track_from_file(audio_path)
        self.toast_mgr.info("Audio added to Mixer")

    def _on_sfx_to_mixer(self, audio_path: str):
        """Route SFX to Mixer."""
        self._sidebar.select_page(5)
        self._mixer_view.add_track_from_file(audio_path)
        self.toast_mgr.info("SFX added to Mixer")

    # ── Window Events ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """Clean up on close."""
        AudioEngine().cleanup()
        self._gpu_timer.stop()
        self._model_mgr.unload()
        from core.lyrics_db import LyricsDB
        LyricsDB().close()
        event.accept()
