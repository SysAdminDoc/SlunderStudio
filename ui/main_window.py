"""
Slunder Studio — Main Window
Studio-shell navigation, contextual workspace header, global transport,
compute status, and drag-and-drop routing.
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
from core.i18n import tr
from core.model_manager import ModelManager
from ui.theme import Palette, build_stylesheet
from ui.toast import ToastManager
from ui.accessibility import install_accessibility, set_accessible
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


PAGE_META = (
    ("CREATE", "Lyrics Engine", "Write, structure and revise lyrics with a local model."),
    ("CREATE", "Song Forge", "Turn lyrics and direction into a finished local render."),
    ("CREATE", "MIDI Studio", "Compose, arrange and humanize MIDI performances."),
    ("CREATE", "Vocal Suite", "Synthesize, convert, tune and separate vocal performances."),
    ("CREATE", "Sound Forge", "Generate production-ready sound effects from text."),
    ("FINISH", "Mix Console", "Balance tracks, shape stereo space and prepare a master."),
    ("FINISH", "AI Producer", "Orchestrate a complete local production from one brief."),
    ("LIBRARY", "Projects", "Manage sessions, assets, versions and provenance."),
    ("SYSTEM", "Model Hub", "Install and manage the local models behind each workflow."),
    ("SYSTEM", "Settings", "Control storage, compute, appearance and diagnostics."),
)


# ── Sidebar Navigation ────────────────────────────────────────────────────────

class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and label."""

    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setText(f"{icon_text}   {label}")
        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI", 10))
        set_accessible(
            self,
            tr("nav.open", label=label),
            tr("nav.switches", label=label),
        )


class Sidebar(QWidget):
    """Left sidebar with navigation buttons."""

    page_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 10)
        layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(4, 0, 4, 10)
        brand_row.setSpacing(9)
        brand_mark = QLabel("S")
        brand_mark.setObjectName("brandMark")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setFixedSize(30, 30)
        brand_row.addWidget(brand_mark)
        brand = QLabel("SLUNDER STUDIO")
        brand.setObjectName("brand")
        brand_row.addWidget(brand, 1)
        layout.addLayout(brand_row)

        # Navigation buttons
        self._buttons: list[SidebarButton] = []
        nav_items = [
            ("\u00b6", tr("nav.lyrics")),
            ("\u2726", tr("nav.song_forge")),
            ("\u266c", tr("nav.midi_studio")),
            ("\u25c9", tr("nav.vocals")),
            ("\u2248", tr("nav.sfx")),
            ("\u2301", tr("nav.mixer")),
            ("\u25c7", tr("nav.ai_producer")),
            ("\u25a3", tr("nav.projects")),
        ]

        current_section = ""
        for i, (icon, label) in enumerate(nav_items):
            section = PAGE_META[i][0]
            if section != current_section:
                section_label = QLabel(section)
                section_label.setObjectName("navSection")
                layout.addWidget(section_label)
                current_section = section
            btn = SidebarButton(icon, label)
            btn.clicked.connect(lambda checked, idx=i: self._on_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        engine_frame = QFrame()
        engine_frame.setObjectName("card")
        engine_layout = QVBoxLayout(engine_frame)
        engine_layout.setContentsMargins(10, 8, 10, 8)
        engine_layout.setSpacing(2)
        engine_status = QLabel("\u25cf  LOCAL ENGINE")
        engine_status.setObjectName("computeStatus")
        engine_layout.addWidget(engine_status)
        engine_note = QLabel("Private by default")
        engine_note.setObjectName("transportMeta")
        engine_layout.addWidget(engine_note)
        layout.addWidget(engine_frame)

        # Bottom nav
        system_label = QLabel("SYSTEM")
        system_label.setObjectName("navSection")
        layout.addWidget(system_label)
        bottom_items = [
            ("\u2b21", tr("nav.model_hub")),
            ("\u2699", tr("nav.settings")),
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
        install_accessibility(self, "Main navigation", tab_order=self._buttons)

    def _on_clicked(self, index: int):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)
        self.page_selected.emit(index)

    def select_page(self, index: int):
        """Programmatically select a page."""
        self._on_clicked(index)
        if 0 <= index < len(self._buttons):
            self._buttons[index].setFocus(Qt.FocusReason.OtherFocusReason)


# ── Transport Bar ──────────────────────────────────────────────────────────────

class TransportBar(QWidget):
    """Global audio transport bar pinned to bottom of window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("transportBar")
        self.setFixedHeight(64)
        self._audio = AudioEngine()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 16, 8)
        layout.setSpacing(10)

        track_meta = QVBoxLayout()
        track_meta.setContentsMargins(0, 0, 12, 0)
        track_meta.setSpacing(1)
        self._track_title = QLabel("GLOBAL OUTPUT")
        self._track_title.setObjectName("transportTitle")
        track_meta.addWidget(self._track_title)
        self._track_detail = QLabel("No audio loaded")
        self._track_detail.setObjectName("transportMeta")
        track_meta.addWidget(self._track_detail)
        layout.addLayout(track_meta)

        # Transport buttons
        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setObjectName("transportPrimary")
        self._play_btn.setFixedSize(40, 36)
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u25A0")
        self._stop_btn.setObjectName("transportBtn")
        self._stop_btn.setFixedSize(36, 36)
        self._stop_btn.clicked.connect(self._audio.stop)
        layout.addWidget(self._stop_btn)

        # Time display
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setObjectName("transportTime")
        self._time_label.setMinimumWidth(96)
        layout.addWidget(self._time_label)

        # Seek slider
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 1000)
        self._seek_slider.setValue(0)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        self._seek_slider.setFixedHeight(20)
        layout.addWidget(self._seek_slider, 1)

        # Loop toggle
        self._loop_btn = QPushButton("\u21bb")
        self._loop_btn.setObjectName("transportBtn")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setFixedSize(36, 36)
        self._loop_btn.toggled.connect(lambda v: self._audio.set_loop(v))
        layout.addWidget(self._loop_btn)

        # Volume
        vol_icon = QLabel("VOL")
        vol_icon.setObjectName("transportMeta")
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
        install_accessibility(
            self,
            "Global transport",
            named_controls=[
                (self._play_btn, "Play or pause audio", "Toggles playback for the loaded audio."),
                (self._stop_btn, "Stop audio", "Stops playback and returns to the start."),
                (self._seek_slider, "Audio position", "Scrubs through the loaded audio."),
                (self._loop_btn, "Loop playback", "Toggles repeat playback for the current audio."),
                (self._vol_slider, "Playback volume", "Adjusts global playback volume."),
            ],
            tab_order=[
                self._play_btn,
                self._stop_btn,
                self._seek_slider,
                self._loop_btn,
                self._vol_slider,
            ],
        )

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
        self._track_detail.setText(format_time(dur))

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

        phase_label = QLabel(tr("placeholder.coming_soon", phase=phase))
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
        self.setWindowTitle(tr("app.window_title", version=APP_VERSION))
        # Must fit a 1024x768 display, and 200% scaling of that.
        self.setMinimumSize(1024, 640)
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self._settings = Settings()
        self._model_mgr = ModelManager()

        # Toast manager
        self.toast_mgr = ToastManager(self)

        self._build_ui()
        self._start_gpu_monitor()
        self._start_autosave()

    def _build_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Content area (navigation + studio workspace)
        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        self._sidebar.page_selected.connect(self._on_page_selected)
        content.addWidget(self._sidebar)

        workspace = QFrame()
        workspace.setObjectName("studioSurface")
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)

        workspace_layout.addWidget(self._build_command_bar())
        workspace_layout.addWidget(self._build_workspace_header())

        self._pages = QStackedWidget()
        self._create_pages()
        workspace_layout.addWidget(self._pages, 1)
        content.addWidget(workspace, 1)

        main_layout.addLayout(content, 1)

        # Transport bar
        self._transport = TransportBar()
        main_layout.addWidget(self._transport)

        # Status bar
        self._status_bar = QStatusBar()
        self._status_bar.setObjectName("statusBar")
        self.setStatusBar(self._status_bar)

        self._gpu_status_label = QLabel(tr("status.gpu_detecting"))
        self._gpu_status_label.setStyleSheet(f"font-size: 11px; color: {Palette.OVERLAY0};")
        self._status_bar.addPermanentWidget(self._gpu_status_label)

        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet(f"font-size: 11px; color: {Palette.BLUE};")
        self._status_bar.addPermanentWidget(self._vram_label)
        self._status_bar.show()
        self._on_page_selected(0)
        set_accessible(self, tr("app.accessible_name"), tr("app.accessible_description"))
        set_accessible(
            self._status_bar,
            "Application status",
            "Shows GPU, VRAM, and active model status.",
        )
        set_accessible(
            self._gpu_status_label,
            tr("status.gpu_accessible_name"),
            tr("status.gpu_accessible_description"),
        )
        set_accessible(
            self._vram_label,
            tr("status.vram_accessible_name"),
            tr("status.vram_accessible_description"),
        )

    def _build_command_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("commandBar")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 18, 0)
        layout.setSpacing(12)

        project_icon = QLabel("\u25a3")
        project_icon.setObjectName("commandMeta")
        layout.addWidget(project_icon)

        project_stack = QVBoxLayout()
        project_stack.setSpacing(0)
        project_stack.setContentsMargins(0, 0, 0, 0)
        project_hint = QLabel("ACTIVE PROJECT")
        project_hint.setObjectName("commandMeta")
        project_stack.addWidget(project_hint)
        self._project_label = QLabel("No project open")
        self._project_label.setObjectName("projectName")
        project_stack.addWidget(self._project_label)
        layout.addLayout(project_stack)

        separator = QFrame()
        separator.setObjectName("commandSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedHeight(24)
        layout.addWidget(separator)

        interval = int(self._settings.get("general.auto_save_interval", 60) or 60)
        self._autosave_label = QLabel(f"Autosave interval  \u00b7  {interval}s")
        self._autosave_label.setObjectName("commandMeta")
        layout.addWidget(self._autosave_label)

        layout.addStretch()

        # Non-timed alternative to toasts (WCAG 2.2 SC 2.2.1): the last
        # notification stays readable here after the toast disappears.
        self._last_message_label = QLabel("")
        self._last_message_label.setObjectName("commandMeta")
        self._last_message_label.setMinimumWidth(0)
        layout.addWidget(self._last_message_label)
        self.toast_mgr.on_message(self._on_toast_message)

        self._compute_status_label = QLabel("Checking compute")
        self._compute_status_label.setObjectName("computeStatus")
        layout.addWidget(self._compute_status_label)

        offline = bool(self._settings.get("model_hub.offline_mode", False))
        self._local_status_label = QLabel(
            "\u25cf  Offline mode" if offline else "\u25cf  Local processing"
        )
        self._local_status_label.setObjectName("localStatus")
        layout.addWidget(self._local_status_label)

        set_accessible(
            bar,
            "Workspace command bar",
            "Shows the active project, autosave interval, compute state, and privacy mode.",
        )
        set_accessible(
            self._compute_status_label,
            "Compute status",
            "Reports whether generation is using a GPU or CPU.",
        )
        set_accessible(
            self._local_status_label,
            "Processing privacy status",
            "Reports local processing or strict offline mode.",
        )
        set_accessible(
            self._last_message_label,
            "Last notification",
            "Keeps the most recent notification readable after its toast closes.",
        )
        return bar

    def _on_toast_message(self, entry: dict):
        """Mirror a timed toast into the persistent status line."""
        text = f"{entry['type'].capitalize()}: {entry['message']}"
        self._last_message_label.setText(text)
        self._last_message_label.setToolTip(text)
        self._last_message_label.setAccessibleDescription(text)

    def _build_workspace_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("workspaceHeader")
        header.setFixedHeight(78)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(24, 8, 24, 8)
        layout.setSpacing(1)

        self._page_eyebrow = QLabel("")
        self._page_eyebrow.setObjectName("pageEyebrow")
        layout.addWidget(self._page_eyebrow)

        self._page_title = QLabel("")
        self._page_title.setObjectName("pageTitle")
        layout.addWidget(self._page_title)

        self._page_subtitle = QLabel("")
        self._page_subtitle.setObjectName("pageSubtitle")
        layout.addWidget(self._page_subtitle)
        return header

    def _create_pages(self):
        """Create all module pages (placeholders for future phases)."""
        # Page 0: Lyrics (Phase 2 — LIVE)
        self._lyrics_view = LyricsView(toast_mgr=self.toast_mgr)
        self._lyrics_view.send_to_forge.connect(self._on_send_to_forge)
        self._pages.addWidget(self._lyrics_view)

        # Page 1: Song Forge (Phase 3)
        self._song_forge_view = SongForgeView(toast_mgr=self.toast_mgr)
        self._song_forge_view.send_to_vocals.connect(self._on_song_forge_to_vocals)
        self._pages.addWidget(self._song_forge_view)

        # Page 2: MIDI Studio (Phase 4)
        self._midi_studio_view = MidiStudioView()
        self._midi_studio_view.send_to_forge.connect(self._on_midi_to_forge)
        self._midi_studio_view.send_to_vocals.connect(self._on_midi_to_vocals)
        self._pages.addWidget(self._midi_studio_view)

        # Page 3: Vocals (Phase 5)
        self._vocal_suite_view = VocalSuiteView()
        self._vocal_suite_view.send_to_forge.connect(self._on_vocal_to_forge)
        self._vocal_suite_view.send_to_mixer.connect(self._on_vocal_to_mixer)
        self._pages.addWidget(self._vocal_suite_view)

        # Page 4: SFX (Phase 6)
        self._sfx_view = SFXView(toast_mgr=self.toast_mgr)
        self._sfx_view.send_to_mixer.connect(self._on_sfx_to_mixer)
        self._pages.addWidget(self._sfx_view)

        # Page 5: Mixer (Phase 6)
        self._mixer_view = MixerView()
        self._pages.addWidget(self._mixer_view)

        # Page 6: AI Producer (Phase 7)
        self._ai_producer_view = AIProducerView()
        self._pages.addWidget(self._ai_producer_view)

        # Page 7: Projects (Phase 6)
        self._project_mgr_view = ProjectManagerView(toast_mgr=self.toast_mgr)
        self._project_mgr_view.project_opened.connect(self._on_project_opened)
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
            eyebrow, title, subtitle = PAGE_META[index]
            self._page_eyebrow.setText(eyebrow)
            self._page_title.setText(title)
            self._page_subtitle.setText(subtitle)

    def _on_project_opened(self, _project_id: str):
        from core.project import get_project_manager

        project = get_project_manager().current
        self._project_label.setText(project.name if project else "No project open")

    # ── GPU Monitoring ─────────────────────────────────────────────────────────

    def _start_gpu_monitor(self):
        """Start periodic GPU status updates."""
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_status)
        self._gpu_timer.start(2000)

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
            self._compute_status_label.setText(
                f"\u25cf  {gpu['name']}  \u00b7  {used:.1f}/{total:.1f} GB"
            )
            self._compute_status_label.setStyleSheet(f"color: {color};")

            current = gpu.get("current_model_name")
            if current:
                self._status_bar.showMessage(f"Active model: {current}", 0)
            else:
                self._status_bar.showMessage("No model loaded", 0)
        else:
            self._gpu_status_label.setText("\u26a0 No GPU")
            self._vram_label.setText("")
            self._compute_status_label.setText("\u25cf  CPU mode")
            self._compute_status_label.setStyleSheet(f"color: {Palette.TEAL};")
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

    def _build_route_artifact(self, path: str, source_module: str, **context):
        """Build a typed route payload, or report why the route cannot run."""
        from core.routing import RouteError, build_routed_artifact

        try:
            return build_routed_artifact(
                path, source_module=source_module, **context
            )
        except RouteError as exc:
            self.toast_mgr.error(f"Route cancelled: {exc}")
            return None

    def _register_routed_artifact(self, artifact, module: str):
        """Attach a routed artifact to the open project when there is one."""
        from core.routing import register_with_project

        try:
            asset_id = register_with_project(artifact, module=module)
        except Exception as exc:
            self.toast_mgr.warning(f"Could not register with project: {exc}")
            return None
        return asset_id

    def _route_to_forge_reference(self, audio_path: str, source_module: str,
                                  page: int = 1):
        artifact = self._build_route_artifact(audio_path, source_module)
        if artifact is None:
            return None
        self._sidebar.select_page(page)
        if not self._song_forge_view.receive_reference(artifact):
            self.toast_mgr.error("Song Forge could not load the routed reference.")
            return None
        asset_id = self._register_routed_artifact(artifact, "song_forge")
        suffix = " and added to the project" if asset_id else ""
        self.toast_mgr.info(
            f"Reference loaded in Song Forge: {artifact.context_summary()}{suffix}"
        )
        return artifact

    def _on_midi_to_forge(self, audio_path: str):
        """Route rendered MIDI audio to Song Forge as reference."""
        return self._route_to_forge_reference(audio_path, "midi_studio")

    def _on_song_forge_to_vocals(self, audio_path: str):
        """Route Song Forge audio to Vocal Suite."""
        return self._on_send_to_vocals(audio_path, "song_forge")

    def _on_midi_to_vocals(self, audio_path: str):
        """Route rendered MIDI audio to Vocal Suite."""
        return self._on_send_to_vocals(audio_path, "midi_studio")

    def _on_send_to_vocals(self, audio_path: str, source_module: str = "song_forge"):
        """Route audio from Song Forge or MIDI Studio to Vocal Suite."""
        artifact = self._build_route_artifact(audio_path, source_module)
        if artifact is None:
            return None
        self._sidebar.select_page(3)  # Switch to Vocals page
        self._vocal_suite_view.set_audio(artifact.path)
        asset_id = self._register_routed_artifact(artifact, "vocal_suite")
        suffix = " and added to the project" if asset_id else ""
        self.toast_mgr.info(
            f"Audio selected in Vocal Suite: {artifact.context_summary()}{suffix}"
        )
        return artifact

    def _on_vocal_to_forge(self, audio_path: str):
        """Route processed vocals back to Song Forge."""
        return self._route_to_forge_reference(audio_path, "vocal_suite")

    def _on_vocal_to_mixer(self, audio_path: str):
        """Route vocals to Mixer."""
        return self._route_to_mixer(audio_path, "vocal_suite")

    def _on_sfx_to_mixer(self, audio_path: str):
        """Route SFX to Mixer."""
        return self._route_to_mixer(audio_path, "sfx")

    def _route_to_mixer(self, audio_path: str, source_module: str):
        artifact = self._build_route_artifact(audio_path, source_module)
        if artifact is None:
            return None
        self._sidebar.select_page(5)

        def _on_import_complete(success: bool, index: int):
            if not success:
                self.toast_mgr.error(f"Mixer could not import {artifact.name}.")
                return
            self._mixer_view.select_track(index)
            asset_id = self._register_routed_artifact(artifact, "mixer")
            suffix = " and added to the project" if asset_id else ""
            self.toast_mgr.info(
                f"Track added to Mixer: {artifact.context_summary()}{suffix}"
            )

        self._mixer_view.add_track_from_file(
            artifact.path,
            on_complete=_on_import_complete,
        )
        return artifact

    # ── Window Events ──────────────────────────────────────────────────────────

    def _start_autosave(self):
        """Honour the configured autosave interval for the open project."""
        from core.autosave import AutosaveCoordinator

        self._autosave = AutosaveCoordinator(settings=self._settings, parent=self)
        self._autosave.autosaved.connect(self._on_autosaved)
        self._autosave.autosave_failed.connect(
            lambda reason: self.toast_mgr.error(f"Autosave failed: {reason}")
        )
        self._autosave.start()
        self._refresh_autosave_label()

    def _on_autosaved(self, version: int, description: str):
        self.toast_mgr.info(f"Autosaved v{version} — {description}")
        if hasattr(self, "_project_mgr_view"):
            from core.project import get_project_manager
            project = get_project_manager().current
            if project is not None:
                self._project_mgr_view.load_project(project)

    def _refresh_autosave_label(self):
        if not hasattr(self, "_autosave_label"):
            return
        if self._autosave.enabled:
            self._autosave_label.setText(
                f"Autosave interval  ·  {self._autosave.interval_seconds}s"
            )
        else:
            self._autosave_label.setText("Autosave  ·  off")

    def resizeEvent(self, event):
        """Keep transient notifications anchored to the current window bounds."""
        super().resizeEvent(event)
        if hasattr(self, "toast_mgr"):
            self.toast_mgr._reposition()

    def closeEvent(self, event):
        """Clean up on close."""
        if hasattr(self, "_autosave"):
            self._autosave.stop()
            # A dirty project must not be lost because the window closed.
            self._autosave.tick()
        AudioEngine().cleanup()
        self._gpu_timer.stop()
        self._model_mgr.unload()
        from core.lyrics_db import LyricsDB
        LyricsDB().close()
        event.accept()
