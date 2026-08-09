"""
Slunder Studio — Main Window
Studio-shell navigation, contextual workspace header, global transport,
compute status, and drag-and-drop routing.
"""
import math

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
from core.midi_controller import MidiControllerRouter, normalized_bindings
from core.midi_input import MidiInputService
from core.osc import OSCConfig, OSCMessage, OSCServer, OSC_NAMESPACE
from core.routing import is_audio_path, is_midi_path
from core.workers import shutdown_workers
from core.workers import InferenceWorker
from ui.theme import Palette, build_stylesheet
from ui.toast import ToastHistoryDialog, ToastManager
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
    ("nav.sections.create", "page.lyrics.title", "page.lyrics.subtitle"),
    ("nav.sections.create", "page.song_forge.title", "page.song_forge.subtitle"),
    ("nav.sections.create", "page.midi_studio.title", "page.midi_studio.subtitle"),
    ("nav.sections.create", "page.vocals.title", "page.vocals.subtitle"),
    ("nav.sections.create", "page.sfx.title", "page.sfx.subtitle"),
    ("nav.sections.finish", "page.mixer.title", "page.mixer.subtitle"),
    ("nav.sections.finish", "page.ai_producer.title", "page.ai_producer.subtitle"),
    ("nav.sections.library", "page.projects.title", "page.projects.subtitle"),
    ("nav.sections.system", "page.model_hub.title", "page.model_hub.subtitle"),
    ("nav.sections.system", "page.settings.title", "page.settings.subtitle"),
)


def _gpu_status_task(progress_cb=None, **_kwargs):
    """Probe the accelerator without importing torch on the GUI thread."""
    from core.model_manager import get_gpu_info

    if progress_cb:
        progress_cb(10)
    gpu = get_gpu_info()
    if progress_cb:
        progress_cb(100)
    return gpu


# ── Sidebar Navigation ────────────────────────────────────────────────────────

class SidebarButton(QPushButton):
    """Sidebar navigation button with icon and label."""

    def __init__(self, icon_text: str, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarBtn")
        self.setCheckable(True)
        self.setText(f"{icon_text}   {label}")
        self.setMinimumHeight(38)
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
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 10)
        layout.setSpacing(2)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(4, 0, 4, 10)
        brand_row.setSpacing(9)
        brand_mark = QLabel("S")
        brand_mark.setObjectName("brandMark")
        brand_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_mark.setMinimumSize(30, 30)
        brand_row.addWidget(brand_mark)
        brand = QLabel(tr("shell.sidebar.brand"))
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
                section_label = QLabel(tr(section))
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
        engine_status = QLabel(tr("shell.sidebar.local_engine"))
        engine_status.setObjectName("computeStatus")
        engine_layout.addWidget(engine_status)
        engine_note = QLabel(tr("shell.sidebar.private_by_default"))
        engine_note.setObjectName("transportMeta")
        engine_layout.addWidget(engine_note)
        layout.addWidget(engine_frame)

        # Bottom nav
        system_label = QLabel(tr("nav.sections.system"))
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
        self.setMinimumHeight(64)
        self._audio = AudioEngine()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 8, 16, 8)
        layout.setSpacing(10)

        track_meta = QVBoxLayout()
        track_meta.setContentsMargins(0, 0, 12, 0)
        track_meta.setSpacing(1)
        self._track_title = QLabel(tr("shell.transport.global_output"))
        self._track_title.setObjectName("transportTitle")
        track_meta.addWidget(self._track_title)
        self._track_detail = QLabel(tr("shell.transport.no_audio"))
        self._track_detail.setObjectName("transportMeta")
        track_meta.addWidget(self._track_detail)
        layout.addLayout(track_meta)

        # Transport buttons
        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setObjectName("transportPrimary")
        self._play_btn.setMinimumSize(40, 36)
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u25A0")
        self._stop_btn.setObjectName("transportBtn")
        self._stop_btn.setMinimumSize(36, 36)
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
        self._seek_slider.setMinimumHeight(20)
        layout.addWidget(self._seek_slider, 1)

        # Loop toggle
        self._loop_btn = QPushButton("\u21bb")
        self._loop_btn.setObjectName("transportBtn")
        self._loop_btn.setCheckable(True)
        self._loop_btn.setMinimumSize(36, 36)
        self._loop_btn.toggled.connect(lambda v: self._audio.set_loop(v))
        layout.addWidget(self._loop_btn)

        # Volume
        vol_icon = QLabel(tr("shell.transport.volume"))
        vol_icon.setObjectName("transportMeta")
        layout.addWidget(vol_icon)

        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(100)
        self._vol_slider.setMinimumWidth(100)
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
            tr("shell.transport.accessibility.name"),
            named_controls=[
                (self._play_btn, tr("shell.transport.accessibility.play_name"), tr("shell.transport.accessibility.play_description")),
                (self._stop_btn, tr("shell.transport.accessibility.stop_name"), tr("shell.transport.accessibility.stop_description")),
                (self._seek_slider, tr("shell.transport.accessibility.position_name"), tr("shell.transport.accessibility.position_description")),
                (self._loop_btn, tr("shell.transport.accessibility.loop_name"), tr("shell.transport.accessibility.loop_description")),
                (self._vol_slider, tr("shell.transport.accessibility.volume_name"), tr("shell.transport.accessibility.volume_description")),
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

    # These small public actions are also the UI-thread boundary for OSC
    # control.  They keep external control from reaching widget internals.
    def osc_play(self):
        """Start or resume playback from an external control request."""
        self._audio.play()

    def osc_pause(self):
        """Pause playback from an external control request."""
        self._audio.pause()

    def osc_stop(self):
        """Stop playback from an external control request."""
        self._audio.stop()

    def osc_toggle(self):
        """Toggle playback from an external control request."""
        self._audio.toggle_play()

    def osc_seek(self, seconds: float):
        """Seek to an absolute position from an external control request."""
        self._audio.seek(seconds)

    def osc_seek_relative(self, seconds: float):
        """Seek relative to the current position from an external request."""
        self._audio.seek_relative(seconds)

    def osc_set_loop(self, enabled: bool):
        """Set loop state from an external control request."""
        self._loop_btn.setChecked(bool(enabled))

    def osc_set_volume(self, value: float):
        """Set normalized playback volume from an external control request."""
        self._vol_slider.setValue(int(round(max(0.0, min(1.0, value)) * 100)))

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


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Slunder Studio main application window."""

    osc_message_received = Signal(object, object)
    osc_server_error = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("app.window_title", version=APP_VERSION))
        # Keep the shell inside a 1024x768 logical work area; Qt scales physical pixels per monitor.
        self.setMinimumSize(1024, 640)
        self.resize(1440, 900)
        self.setAcceptDrops(True)

        self._settings = Settings()
        self._audio = AudioEngine()
        self._model_mgr = ModelManager()
        self._gpu_worker = None
        self._gpu_workers = set()
        self._osc_server = None
        self._osc_reconfigure_pending = False
        self._midi_router = MidiControllerRouter(
            self._settings.get("midi_controller.bindings", None)
        )
        self._midi_input = MidiInputService(parent=self)
        self._midi_reconfigure_pending = False
        self._closing = False
        self._osc_settings_callback = self._on_osc_settings_changed
        self._midi_settings_callback = self._on_midi_settings_changed

        # Toast manager
        self.toast_mgr = ToastManager(self)
        self._notification_log_dialog = None

        self._build_ui()
        self.osc_message_received.connect(self._on_osc_message)
        self.osc_server_error.connect(self._on_osc_server_error)
        self._midi_input.message_received.connect(self._on_midi_message)
        self._midi_input.status_changed.connect(self._on_midi_status_changed)
        self._midi_input.error.connect(self._on_midi_input_error)
        self._settings.on_change(self._osc_settings_callback)
        self._settings.on_change(self._midi_settings_callback)
        self._configure_osc_server()
        self._configure_midi_input()
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
        self._audio.output_device_status.connect(self._on_audio_output_status)

        self._gpu_status_label = QLabel(tr("status.gpu_detecting"))
        self._gpu_status_label.setStyleSheet(f"font-size: 8.25pt; color: {Palette.OVERLAY0};")
        self._status_bar.addPermanentWidget(self._gpu_status_label)

        self._vram_label = QLabel("")
        self._vram_label.setStyleSheet(f"font-size: 8.25pt; color: {Palette.BLUE};")
        self._status_bar.addPermanentWidget(self._vram_label)
        self._status_bar.show()
        self._on_page_selected(0)
        set_accessible(self, tr("app.accessible_name"), tr("app.accessible_description"))
        set_accessible(
            self._status_bar,
            tr("shell.accessibility.status_name"),
            tr("shell.accessibility.status_description"),
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
        bar.setMinimumHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 18, 0)
        layout.setSpacing(12)

        project_icon = QLabel("\u25a3")
        project_icon.setObjectName("commandMeta")
        layout.addWidget(project_icon)

        project_stack = QVBoxLayout()
        project_stack.setSpacing(0)
        project_stack.setContentsMargins(0, 0, 0, 0)
        project_hint = QLabel(tr("shell.command.active_project"))
        project_hint.setObjectName("commandMeta")
        project_stack.addWidget(project_hint)
        self._project_label = QLabel(tr("shell.command.no_project"))
        self._project_label.setObjectName("projectName")
        project_stack.addWidget(self._project_label)
        layout.addLayout(project_stack)

        separator = QFrame()
        separator.setObjectName("commandSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setMinimumHeight(24)
        layout.addWidget(separator)

        interval = int(self._settings.get("general.auto_save_interval", 60) or 60)
        self._autosave_label = QLabel(
            tr("shell.command.autosave_interval", seconds=interval)
        )
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

        self._notification_button = QPushButton(tr("shell.command.notifications"))
        self._notification_button.setMinimumHeight(30)
        self._notification_button.clicked.connect(self._show_notification_log)
        layout.addWidget(self._notification_button)

        self._compute_status_label = QLabel(tr("shell.command.checking_compute"))
        self._compute_status_label.setObjectName("computeStatus")
        layout.addWidget(self._compute_status_label)

        offline = bool(self._settings.get("model_hub.offline_mode", False))
        self._local_status_label = QLabel(
            tr("shell.command.offline_mode")
            if offline else tr("shell.command.local_processing")
        )
        self._local_status_label.setObjectName("localStatus")
        layout.addWidget(self._local_status_label)

        set_accessible(
            bar,
            tr("shell.accessibility.command_bar_name"),
            tr("shell.accessibility.command_bar_description"),
        )
        set_accessible(
            self._compute_status_label,
            tr("shell.accessibility.compute_name"),
            tr("shell.accessibility.compute_description"),
        )
        set_accessible(
            self._local_status_label,
            tr("shell.accessibility.privacy_name"),
            tr("shell.accessibility.privacy_description"),
        )
        set_accessible(
            self._last_message_label,
            tr("shell.accessibility.last_notification_name"),
            tr("shell.accessibility.last_notification_description"),
        )
        set_accessible(
            self._notification_button,
            tr("shell.accessibility.notification_history_name"),
            tr("shell.accessibility.notification_history_description"),
        )
        return bar

    def _on_toast_message(self, entry: dict):
        """Mirror a timed toast into the persistent status line."""
        notification_type = str(entry.get("type", "info")).lower()
        type_key = {
            "info": "info",
            "warning": "warning",
            "error": "error",
        }.get(notification_type, "info")
        text = tr(
            "shell.status.notification_prefix",
            type=tr(f"shell.status.notification_types.{type_key}"),
            message=entry["message"],
        )
        self._last_message_label.setText(text)
        self._last_message_label.setToolTip(text)
        self._last_message_label.setAccessibleDescription(text)
        if hasattr(self, "_notification_button"):
            self._notification_button.setText(
                tr(
                    "shell.command.notification_count",
                    count=len(self.toast_mgr.history),
                )
            )

    def _show_notification_log(self):
        """Show retained notifications without blocking the workspace."""
        if self._notification_log_dialog is None:
            self._notification_log_dialog = ToastHistoryDialog(
                self.toast_mgr,
                self,
            )
        self._notification_log_dialog.show()
        self._notification_log_dialog.raise_()

    def _build_workspace_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("workspaceHeader")
        header.setMinimumHeight(78)
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
        self._song_forge_view.send_reference_to_midi.connect(
            self._on_reference_to_midi
        )
        self._pages.addWidget(self._song_forge_view)

        # Page 2: MIDI Studio (Phase 4)
        self._midi_studio_view = MidiStudioView(toast_mgr=self.toast_mgr)
        self._midi_studio_view.send_to_forge.connect(self._on_midi_to_forge)
        self._midi_studio_view.send_to_vocals.connect(self._on_midi_to_vocals)
        self._pages.addWidget(self._midi_studio_view)

        # Page 3: Vocals (Phase 5)
        self._vocal_suite_view = VocalSuiteView(toast_mgr=self.toast_mgr)
        self._vocal_suite_view.send_to_forge.connect(self._on_vocal_to_forge)
        self._vocal_suite_view.send_to_mixer.connect(self._on_vocal_to_mixer)
        self._pages.addWidget(self._vocal_suite_view)

        # Page 4: SFX (Phase 6)
        self._sfx_view = SFXView(toast_mgr=self.toast_mgr)
        self._sfx_view.send_to_mixer.connect(self._on_sfx_to_mixer)
        self._pages.addWidget(self._sfx_view)

        # Page 5: Mixer (Phase 6)
        self._mixer_view = MixerView(toast_mgr=self.toast_mgr)
        self._pages.addWidget(self._mixer_view)

        # Page 6: AI Producer (Phase 7)
        self._ai_producer_view = AIProducerView(toast_mgr=self.toast_mgr)
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
            self._page_eyebrow.setText(tr(eyebrow))
            self._page_title.setText(tr(title))
            self._page_subtitle.setText(tr(subtitle))

    def open_model_hub_for_onboarding(self, model_id: str, action: str = "open") -> bool:
        """Open Model Hub with the first-run model choice already selected."""
        self._sidebar.select_page(8)
        return self._model_hub.prepare_onboarding_model(model_id, action)

    def _on_project_opened(self, _project_id: str):
        from core.project import get_project_manager

        project = get_project_manager().current
        self._project_label.setText(
            project.name if project else tr("shell.command.no_project")
        )

    # ── GPU Monitoring ─────────────────────────────────────────────────────────

    def _start_gpu_monitor(self):
        """Start periodic GPU status updates."""
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._update_gpu_status)
        self._gpu_timer.start(2000)

    def _update_gpu_status(self):
        """Queue a GPU status probe; torch import and CUDA calls stay off-thread."""
        if self._gpu_worker is not None and self._gpu_worker.isRunning():
            return
        worker = InferenceWorker(_gpu_status_task)
        self._gpu_workers.add(worker)
        self._gpu_worker = worker
        worker.finished.connect(self._on_gpu_status_finished)
        worker.error.connect(self._on_gpu_status_error)
        worker.start()

    def _release_gpu_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_gpu_worker_later(worker))
            return
        self._gpu_workers.discard(worker)
        if self._gpu_worker is worker:
            self._gpu_worker = None

    def _on_gpu_status_finished(self, gpu: dict):
        worker = self._gpu_worker
        self._release_gpu_worker_later(worker)
        self._gpu_worker = None
        with self._model_mgr._state_lock:
            current_id = self._model_mgr._current_model_id
        gpu["current_model_name"] = (
            self._model_mgr._registry[current_id].name
            if current_id and current_id in self._model_mgr._registry
            else None
        )
        self._apply_gpu_status(gpu)

    def _on_gpu_status_error(self, message: str):
        worker = self._gpu_worker
        self._release_gpu_worker_later(worker)
        self._gpu_worker = None
        self._gpu_status_label.setText(tr("shell.status.gpu_unavailable"))
        self._vram_label.setText("")
        self._compute_status_label.setText(tr("shell.status.hardware_unavailable"))
        self._status_bar.showMessage(message, 5000)

    def _on_audio_output_status(self, message: str):
        """Keep device fallback visible in both the status bar and toast history."""
        self._status_bar.showMessage(message, 10000)
        if self.toast_mgr:
            self.toast_mgr.warning(message, duration_ms=10000)

    # ── OSC Control ───────────────────────────────────────────────────────────

    def _on_midi_settings_changed(self, key: str, _value, _old_value):
        """Rebind the optional MIDI input after a Settings edit."""
        if key != "*" and key != "midi_controller" and not key.startswith("midi_controller."):
            return
        if getattr(self, "_midi_reconfigure_pending", False):
            return
        self._midi_reconfigure_pending = True
        QTimer.singleShot(0, self._apply_pending_midi_settings)

    def _apply_pending_midi_settings(self):
        self._midi_reconfigure_pending = False
        self._configure_midi_input()

    def _configure_midi_input(self):
        """Apply persisted MIDI bindings and opt-in input policy."""
        self._midi_router.set_bindings(
            normalized_bindings(self._settings.get("midi_controller.bindings", None))
        )
        self._midi_input.stop()
        if not bool(self._settings.get("midi_controller.enabled", False)):
            return
        port_name = str(self._settings.get("midi_controller.port_name", "") or "")
        self._midi_input.start(port_name)

    def _on_midi_status_changed(self, message: str):
        if not getattr(self, "_closing", False):
            self._status_bar.showMessage(message, 5000)

    def _on_midi_input_error(self, message: str):
        if not getattr(self, "_closing", False):
            self._status_bar.showMessage(message, 10000)

    def _on_midi_message(self, message) -> bool:
        """Dispatch a validated MIDI message to public UI action methods."""
        if getattr(self, "_closing", False):
            return False
        events = self._midi_router.dispatch(message)
        handled = False
        for event in events:
            action = event.action
            if action == "transport.toggle":
                self._transport.osc_toggle()
            elif action == "transport.stop":
                self._transport.osc_stop()
            elif action == "mixer.volume":
                handled = self._mixer_view.set_selected_volume(event.value) or handled
                continue
            elif action == "mixer.pan":
                handled = self._mixer_view.set_selected_pan(event.value * 2.0 - 1.0) or handled
                continue
            elif action == "mixer.mute":
                handled = self._mixer_view.toggle_selected_mute() or handled
                continue
            elif action == "mixer.solo":
                handled = self._mixer_view.toggle_selected_solo() or handled
                continue
            elif action == "piano.quantize" and event.value >= 0.5:
                handled = self._midi_studio_view.controller_quantize() or handled
                continue
            elif action == "piano.swing" and event.value >= 0.5:
                handled = self._midi_studio_view.controller_swing() or handled
                continue
            elif action == "piano.humanize" and event.value >= 0.5:
                handled = self._midi_studio_view.controller_humanize() or handled
                continue
            else:
                continue
            handled = True
        return handled

    def _on_osc_settings_changed(self, key: str, _value, _old_value):
        """Rebind OSC after a relevant setting changes on the UI thread."""
        if key != "*" and key != "osc" and not key.startswith("osc."):
            return
        if getattr(self, "_osc_reconfigure_pending", False):
            return
        self._osc_reconfigure_pending = True
        QTimer.singleShot(0, self._apply_pending_osc_settings)

    def _apply_pending_osc_settings(self):
        self._osc_reconfigure_pending = False
        self._configure_osc_server()

    def _configure_osc_server(self):
        """Apply persisted OSC policy without exposing a socket by default."""
        previous = self._osc_server
        self._osc_server = None
        if previous is not None:
            previous.stop()

        config = OSCConfig.from_settings(self._settings.get_section("osc"))
        if not config.enabled:
            return

        server = OSCServer(
            config,
            self._emit_osc_message,
            error_callback=lambda message: self.osc_server_error.emit(message),
        )
        try:
            server.start()
        except OSError as exc:
            message = tr(
                "shell.osc.listen_error",
                port=config.port,
                error=exc,
            )
            self._on_osc_server_error(message)
            return
        self._osc_server = server
        bound = server.bound_address or (config.bind_host, config.port)
        self._status_bar.showMessage(
            tr("shell.osc.listening", host=bound[0], port=bound[1]),
            5000,
        )

    def _emit_osc_message(self, message: OSCMessage, source: tuple[str, int]):
        """Queue a validated worker-thread message onto the Qt UI thread."""
        if not getattr(self, "_closing", False):
            self.osc_message_received.emit(message, source)

    def _on_osc_server_error(self, message: str):
        """Surface an unexpected listener failure without touching Qt off-thread."""
        stopped = tr("shell.osc.stopped", message=message)
        self._status_bar.showMessage(stopped, 10000)
        if self.toast_mgr:
            self.toast_mgr.warning(stopped, duration_ms=10000)

    @staticmethod
    def _osc_number(arguments: tuple[object, ...]) -> float | None:
        if len(arguments) != 1 or isinstance(arguments[0], bool):
            return None
        value = arguments[0]
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def _on_osc_message(
        self,
        message: OSCMessage,
        _source: tuple[str, int],
    ) -> bool:
        """Dispatch the small, explicit transport command surface."""
        if not isinstance(message, OSCMessage):
            return False
        address = message.address
        arguments = message.arguments
        if address == f"{OSC_NAMESPACE}/ping":
            if arguments:
                return False
            self._status_bar.showMessage(tr("shell.osc.ping"), 1500)
            return True

        transport_commands = {
            f"{OSC_NAMESPACE}/transport/play": self._transport.osc_play,
            f"{OSC_NAMESPACE}/transport/pause": self._transport.osc_pause,
            f"{OSC_NAMESPACE}/transport/stop": self._transport.osc_stop,
            f"{OSC_NAMESPACE}/transport/toggle": self._transport.osc_toggle,
        }
        action = transport_commands.get(address)
        if action is not None:
            if arguments:
                return False
            action()
            return True

        if address == f"{OSC_NAMESPACE}/transport/seek":
            value = self._osc_number(arguments)
            if value is None:
                return False
            self._transport.osc_seek(value)
            return True
        if address == f"{OSC_NAMESPACE}/transport/seek_relative":
            value = self._osc_number(arguments)
            if value is None:
                return False
            self._transport.osc_seek_relative(value)
            return True
        if address == f"{OSC_NAMESPACE}/transport/loop":
            if len(arguments) != 1 or not isinstance(arguments[0], bool):
                return False
            self._transport.osc_set_loop(arguments[0])
            return True
        if address == f"{OSC_NAMESPACE}/transport/volume":
            value = self._osc_number(arguments)
            if value is None or not 0.0 <= value <= 1.0:
                return False
            self._transport.osc_set_volume(value)
            return True
        return False

    def _apply_gpu_status(self, gpu: dict):
        """Update GPU status widgets from a completed background probe."""
        if gpu.get("available"):
            self._gpu_status_label.setText(f"\U0001f4bb {gpu['name']}")
            used = gpu["used_gb"]
            total = gpu["total_gb"]
            pct = (used / total * 100) if total > 0 else 0
            color = Palette.GREEN if pct < 60 else (Palette.YELLOW if pct < 85 else Palette.RED)
            self._vram_label.setText(
                tr(
                    "shell.status.vram",
                    used=used,
                    total=total,
                    percent=pct,
                )
            )
            self._vram_label.setStyleSheet(f"font-size: 8.25pt; color: {color};")
            self._compute_status_label.setText(
                tr(
                    "shell.status.compute_gpu",
                    name=gpu["name"],
                    used=used,
                    total=total,
                )
            )
            self._compute_status_label.setStyleSheet(f"color: {color};")

            current = gpu.get("current_model_name")
            if current:
                self._status_bar.showMessage(
                    tr("shell.status.active_model", name=current), 0
                )
            else:
                self._status_bar.showMessage(tr("shell.status.no_model"), 0)
        else:
            self._gpu_status_label.setText(tr("shell.status.no_gpu"))
            self._vram_label.setText("")
            self._compute_status_label.setText(tr("shell.status.cpu_mode"))
            self._compute_status_label.setStyleSheet(f"color: {Palette.TEAL};")
            self._status_bar.showMessage(tr("shell.status.cuda_cpu"), 0)

    # ── Drag and Drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and any(
            is_audio_path(url.toLocalFile()) or is_midi_path(url.toLocalFile())
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                if is_audio_path(path):
                    self._load_dropped_audio(path)
                elif is_midi_path(path):
                    self.toast_mgr.info(tr("shell.drop.midi_detected"))
                    self._sidebar.select_page(2)
                    from core.midi_utils import load_midi as load_midi_file
                    try:
                        midi_data = load_midi_file(path)
                        self._midi_studio_view.set_midi_data(midi_data)
                    except Exception:
                        self.toast_mgr.warning(tr("shell.drop.midi_failed"))
                else:
                    suffix = path.rsplit(".", 1)[-1] if "." in path else "unknown"
                    self.toast_mgr.warning(
                        tr("shell.drop.unsupported", extension=suffix)
                    )
        event.acceptProposedAction()

    def _load_dropped_audio(self, path: str):
        """Load dropped audio into the active audio-capable workspace."""
        page = self._pages.currentIndex()
        if page == 1:
            self._route_to_forge_reference(path, "drag_drop")
            return
        if page == 3:
            self._on_send_to_vocals(path, "drag_drop")
            return
        if page == 5:
            self._route_to_mixer(path, "drag_drop")
            return

        # Pages without an audio input still get a useful, non-destructive
        # destination.  The file is loaded into Mixer rather than being
        # unexpectedly sent to the audio device.
        self.toast_mgr.info(tr("runtime.no_audio_input"))
        self._route_to_mixer(path, "drag_drop")

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
        except RouteError:
            self.toast_mgr.error(tr("runtime.load_failed"))
            return None

    def _register_routed_artifact(self, artifact, module: str):
        """Attach a routed artifact to the open project when there is one."""
        from core.routing import register_with_project

        try:
            asset_id = register_with_project(artifact, module=module)
        except Exception as exc:
            self.toast_mgr.warning(tr("runtime.register_failed", error=exc))
            return None
        return asset_id

    def _route_to_forge_reference(self, audio_path: str, source_module: str,
                                  page: int = 1):
        artifact = self._build_route_artifact(audio_path, source_module)
        if artifact is None:
            return None
        self._sidebar.select_page(page)
        if not self._song_forge_view.receive_reference(artifact):
            self.toast_mgr.error(tr("runtime.reference_failed"))
            return None
        asset_id = self._register_routed_artifact(artifact, "song_forge")
        suffix = tr("runtime.project_suffix") if asset_id else ""
        self.toast_mgr.info(
            tr(
                "runtime.reference_loaded",
                details=artifact.context_summary(),
                suffix=suffix,
            )
        )
        return artifact

    def _on_midi_to_forge(self, audio_path: str):
        """Route rendered MIDI audio to Song Forge as reference."""
        return self._route_to_forge_reference(audio_path, "midi_studio")

    def _on_reference_to_midi(self, constraints: dict):
        """Route corrected reference constraints into MIDI Studio."""
        self._sidebar.select_page(2)
        if not self._midi_studio_view.apply_reference_constraints(constraints):
            self.toast_mgr.error(tr("runtime.reference_failed"))
            return False
        effective = constraints.get("effective", constraints)
        self.toast_mgr.info(
            tr(
                "runtime.reference_constraints_routed",
                bpm=float(effective.get("bpm", 0.0) or 0.0),
                musical_key=str(effective.get("key", "") or ""),
            )
        )
        return True

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
        suffix = tr("runtime.project_suffix") if asset_id else ""
        self.toast_mgr.info(
            tr(
                "runtime.audio_selected",
                details=artifact.context_summary(),
                suffix=suffix,
            )
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
                self.toast_mgr.error(tr("runtime.mixer_failed", name=artifact.name))
                return
            self._mixer_view.select_track(index)
            asset_id = self._register_routed_artifact(artifact, "mixer")
            suffix = tr("runtime.project_suffix") if asset_id else ""
            self.toast_mgr.info(
                tr(
                    "runtime.track_added",
                    details=artifact.context_summary(),
                    suffix=suffix,
                )
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
            lambda reason: self.toast_mgr.error(
                tr("shell.autosave.failed", reason=reason)
            )
        )
        self._autosave.start()
        self._refresh_autosave_label()

    def _on_autosaved(self, version: int, description: str):
        self.toast_mgr.info(
            tr("shell.autosave.saved", version=version, description=description)
        )
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
                tr(
                    "shell.command.autosave_enabled",
                    seconds=self._autosave.interval_seconds,
                )
            )
        else:
            self._autosave_label.setText(tr("shell.command.autosave_disabled"))

    def _flush_project_before_close(self) -> bool:
        """Flush dirty editor state and keep the window open on failure."""
        project_view = getattr(self, "_project_mgr_view", None)
        sync = getattr(project_view, "sync_pending_edits", None)
        if callable(sync):
            sync()

        from core.project import get_project_manager

        projects = get_project_manager()
        if projects.current is None or not projects.is_dirty:
            return True

        version = self._autosave.flush()
        if version is not None and not projects.is_dirty:
            return True

        self.toast_mgr.error(
            tr("shell.autosave.close_failed")
        )
        if self._autosave.enabled:
            self._autosave.start()
        return False

    def resizeEvent(self, event):
        """Keep transient notifications anchored to the current window bounds."""
        super().resizeEvent(event)
        if hasattr(self, "toast_mgr"):
            self.toast_mgr._reposition()

    def closeEvent(self, event):
        """Clean up on close."""
        if hasattr(self, "_autosave"):
            self._autosave.stop()
            if not self._flush_project_before_close():
                event.ignore()
                return
        if not shutdown_workers():
            self.toast_mgr.error(
                tr("runtime.busy_unload")
            )
            event.ignore()
            return
        self._closing = True
        osc_server = getattr(self, "_osc_server", None)
        if osc_server is not None:
            osc_server.stop()
            self._osc_server = None
        midi_input = getattr(self, "_midi_input", None)
        if midi_input is not None:
            midi_input.stop()
        settings = getattr(self, "_settings", None)
        callback = getattr(self, "_osc_settings_callback", None)
        if settings is not None and callback is not None:
            settings.remove_callback(callback)
        midi_callback = getattr(self, "_midi_settings_callback", None)
        if settings is not None and midi_callback is not None:
            settings.remove_callback(midi_callback)
        AudioEngine().cleanup()
        self._gpu_timer.stop()
        self._model_mgr.unload()
        from core.lyrics_db import LyricsDB
        LyricsDB().close()
        event.accept()
