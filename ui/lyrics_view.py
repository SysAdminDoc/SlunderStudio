"""
Slunder Studio — Lyrics View
Full lyrics generation page with Quick/Guided/Pro modes, genre browser,
history panel, streaming generation, and section regeneration.
"""
import json
import time
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QSplitter, QTabWidget, QLineEdit, QTextEdit, QComboBox,
    QPushButton, QProgressBar, QScrollArea, QListWidget,
    QListWidgetItem, QGroupBox, QDoubleSpinBox, QSpinBox,
    QSlider, QPlainTextEdit, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, Slot

from ui.theme import Palette
from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget
from ui.lyrics_editor import LyricsEditor
from core.settings import Settings
from core.i18n import (
    language_code_from_label,
    language_combo_items,
    language_label,
    tr,
)
from core.workers import InferenceWorker
from core.lyrics_db import LyricsDB, LyricsEntry
from engines.lyrics_templates import (
    GENRE_TEMPLATES, MOODS, STANDARD_STRUCTURES,
    get_genre_list, get_genre_categories, get_random_theme,
)


class GenrePicker(QWidget):
    """Searchable genre picker with category tabs."""

    genre_selected = Signal(str)  # genre_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("lyrics.history.search_genres"))
        self._search.setMinimumHeight(34)
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        # Genre list
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._on_selection)
        self._list_empty = EmptyStateWidget(
            "No genres found",
            "Choose a genre to guide the next lyrics draft.",
            "Clear search",
        )
        self._list_empty.action_requested.connect(self._search.clear)
        self._list_stack = QStackedWidget()
        self._list_stack.addWidget(self._list)
        self._list_stack.addWidget(self._list_empty)
        layout.addWidget(self._list_stack, 1)

        self._populate()
        install_accessibility(
            self,
            "Genre picker",
            named_controls=[
                (self._search, "Search genres", "Filters the available genres."),
                (self._list, "Genre list", "Selects a genre for guided lyrics generation."),
            ],
        )

    def _populate(self):
        for genre in get_genre_list():
            item = QListWidgetItem(f"{genre['name']}  —  {genre['description']}")
            item.setData(Qt.ItemDataRole.UserRole, genre["id"])
            self._list.addItem(item)

    def _filter(self, text: str):
        text = text.lower()
        visible_count = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            visible = text in item.text().lower()
            item.setHidden(not visible)
            visible_count += int(visible)
        if visible_count:
            self._list_stack.setCurrentWidget(self._list)
        elif text:
            self._list_empty.set_no_matches(
                f'No genres match “{text}”. Try a broader search.',
                "Clear search",
            )
            self._list_stack.setCurrentWidget(self._list_empty)
        else:
            self._list_empty.set_state(
                "No genres available",
                "Genre guidance will appear here when the genre catalog is available.",
                "Retry search",
            )
            self._list_stack.setCurrentWidget(self._list_empty)

    def _on_selection(self, current, previous):
        if current:
            genre_id = current.data(Qt.ItemDataRole.UserRole)
            self.genre_selected.emit(genre_id)

    def set_genre(self, genre_id: str):
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == genre_id:
                self._list.setCurrentItem(item)
                break

    @property
    def current_genre(self) -> str:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else "pop"


class HistoryPanel(QWidget):
    """Sidebar panel showing lyrics generation history with search and favorites."""

    entry_selected = Signal(object)  # LyricsEntry
    create_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = LyricsDB()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel(tr("lyrics.history.title"))
        header.setStyleSheet(f"font-size: 10.5pt; font-weight: 700; color: {Palette.TEXT};")
        layout.addWidget(header)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("lyrics.history.search_placeholder"))
        self._search.setMinimumHeight(32)
        self._search.textChanged.connect(self._refresh)
        layout.addWidget(self._search)

        # Filter buttons
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        self._all_btn = QPushButton(tr("lyrics.history.all"))
        self._all_btn.setObjectName("ghostBtn")
        self._all_btn.setCheckable(True)
        self._all_btn.setChecked(True)
        self._all_btn.setMinimumHeight(26)
        self._all_btn.clicked.connect(lambda: self._set_filter("all"))
        filter_row.addWidget(self._all_btn)

        self._fav_btn = QPushButton(f"\u2605 {tr('lyrics.history.favorites')}")
        self._fav_btn.setObjectName("ghostBtn")
        self._fav_btn.setCheckable(True)
        self._fav_btn.setMinimumHeight(26)
        self._fav_btn.clicked.connect(lambda: self._set_filter("favorites"))
        filter_row.addWidget(self._fav_btn)

        filter_row.addStretch()
        layout.addLayout(filter_row)

        # List
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(self._toggle_favorite)
        self._list.setToolTip("Double-click an entry to toggle its favorite status.")
        self._empty = EmptyStateWidget(
            "No lyrics saved yet",
            "Generated drafts will stay here so you can revisit, favorite, and edit them.",
            "Generate lyrics",
        )
        self._empty.action_requested.connect(self._on_empty_action)
        self._list_stack = QStackedWidget()
        self._list_stack.addWidget(self._list)
        self._list_stack.addWidget(self._empty)
        layout.addWidget(self._list_stack, 1)

        # Count
        self._count_label = QLabel(tr("lyrics.history.entries_count", count=0))
        self._count_label.setObjectName("caption")
        layout.addWidget(self._count_label)

        self._current_filter = "all"
        self._refresh()
        install_accessibility(
            self,
            "Lyrics history",
            named_controls=[
                (self._search, "Search lyrics history", "Filters saved lyrics entries."),
                (self._all_btn, "Show all lyrics history", "Shows all saved lyrics entries."),
                (self._fav_btn, "Show favorite lyrics", "Shows only favorite lyrics entries."),
                (
                    self._list,
                    "Lyrics history list",
                    "Selects a saved lyrics entry; double-click an entry to toggle its favorite status.",
                ),
            ],
        )

    def _set_filter(self, mode: str):
        self._current_filter = mode
        self._all_btn.setChecked(mode == "all")
        self._fav_btn.setChecked(mode == "favorites")
        self._refresh()

    def _refresh(self, selected_id: Optional[int] = None):
        self._list.clear()
        query = self._search.text().strip()

        if query:
            entries = self._db.search(query)
        elif self._current_filter == "favorites":
            entries = self._db.get_favorites()
        else:
            entries = self._db.get_recent()

        for entry in entries:
            star = "\u2605 " if entry.is_favorite else ""
            item = QListWidgetItem(f"{star}{entry.genre.upper()} \u2022 {entry.timestamp_str}\n{entry.preview}")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self._list.addItem(item)

        if selected_id is not None:
            for row in range(self._list.count()):
                entry = self._list.item(row).data(Qt.ItemDataRole.UserRole)
                if entry.id == selected_id:
                    self._list.setCurrentRow(row)
                    break

        self._count_label.setText(tr("lyrics.history.entries_count", count=len(entries)))
        if entries:
            self._list_stack.setCurrentWidget(self._list)
        elif query:
            self._empty.set_no_matches(
                f'No saved lyrics match “{query}”. Clear the search to browse your history.',
                "Clear search",
            )
            self._list_stack.setCurrentWidget(self._empty)
        elif self._current_filter == "favorites":
            self._empty.set_state(
                "No favorite lyrics yet",
                "Double-click a saved draft to favorite it for quick access.",
                "Generate lyrics",
            )
            self._list_stack.setCurrentWidget(self._empty)
        else:
            self._empty.set_state(
                "No lyrics saved yet",
                "Generated drafts will stay here so you can revisit, favorite, and edit them.",
                "Generate lyrics",
            )
            self._list_stack.setCurrentWidget(self._empty)

    def _on_empty_action(self):
        if self._empty.state == "no_matches":
            self._search.clear()
            self._set_filter("all")
            return
        self.create_requested.emit()

    def _toggle_favorite(self, item: QListWidgetItem):
        """Toggle the selected history entry and persist the new state."""
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, LyricsEntry) or entry.id <= 0:
            return
        entry.is_favorite = self._db.toggle_favorite(entry.id)
        self._refresh(selected_id=entry.id)

    def _on_selection(self, current, previous):
        if current:
            entry = current.data(Qt.ItemDataRole.UserRole)
            self.entry_selected.emit(entry)

    def add_entry(self, entry: LyricsEntry):
        """Add a new entry and refresh."""
        self._db.save(entry)
        self._refresh()


# ── Main Lyrics View ───────────────────────────────────────────────────────────

class LyricsView(QWidget):
    """
    Complete lyrics generation page with:
    - Quick Mode: one-line prompt → full lyrics
    - Guided Mode: genre, mood, theme, structure pickers
    - Pro Mode: raw system prompt editor + all LLM parameters
    - Streaming token output to the editor
    - History sidebar with search and favorites
    - Section regeneration via right-click
    """

    send_to_forge = Signal(str)  # lyrics text to pass to Song Forge

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._settings = Settings()
        self._db = LyricsDB()
        self._worker: Optional[InferenceWorker] = None
        self._current_genre = "pop"
        self._build_ui()
        self._connect_signals()
        self._install_accessibility()

    def _install_accessibility(self):
        install_accessibility(
            self,
            "Lyrics workspace",
            named_controls=[
                (self._mode_tabs, "Lyrics mode tabs", "Switches between quick, guided, and pro lyrics generation."),
                (self._quick_input, "Quick lyrics prompt", "Describes the lyrics to generate in quick mode."),
                (self._quick_generate, "Generate quick lyrics", "Generates lyrics from the quick prompt."),
                (self._theme_input, "Guided lyrics theme", "Sets the theme for guided lyrics generation."),
                (self._mood_combo, "Lyrics mood", "Selects the mood for guided lyrics generation."),
                (self._structure_combo, "Lyrics structure", "Selects the structure for guided lyrics generation."),
                (self._lang_combo, "Lyrics language", "Selects the language for generated lyrics."),
                (self._guided_generate, "Generate guided lyrics", "Generates lyrics from guided settings."),
                (self._system_prompt, "Lyrics system prompt", "Sets the system instructions for pro lyrics generation."),
                (self._user_prompt, "Pro lyrics prompt", "Sets the user prompt for pro lyrics generation."),
                (self._pro_temp, "Lyrics temperature", "Adjusts pro lyrics sampling temperature."),
                (self._pro_top_p, "Lyrics top p", "Adjusts pro lyrics nucleus sampling."),
                (self._pro_top_k, "Lyrics top k", "Adjusts pro lyrics top-k sampling."),
                (self._pro_repeat, "Lyrics repeat penalty", "Adjusts repetition control for pro lyrics."),
                (self._pro_max_tokens, "Lyrics maximum tokens", "Sets the maximum pro lyrics length."),
                (self._pro_generate, "Generate pro lyrics", "Generates lyrics from the pro settings."),
                (self._cancel_btn, "Cancel lyrics generation", tr("runtime.lyrics_cancel_description")),
                (self._regen_btn, "Regenerate lyrics", "Runs the current lyrics workflow again."),
            ],
        )
        self._mode_tabs.currentChanged.connect(lambda _index: self._install_accessibility())

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main splitter: left (controls) | center (editor) | right (history)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left Panel: Input Controls ──────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        # Mode tabs
        self._mode_tabs = QTabWidget()

        # ── Quick Mode Tab ──
        quick_tab = QWidget()
        quick_layout = QVBoxLayout(quick_tab)
        quick_layout.setContentsMargins(8, 12, 8, 8)
        quick_layout.setSpacing(12)

        quick_label = QLabel(tr("lyrics.quick.label"))
        quick_label.setStyleSheet(f"font-size: 9.75pt; font-weight: 600; color: {Palette.SUBTEXT0};")
        quick_layout.addWidget(quick_label)

        self._quick_input = QTextEdit()
        self._quick_input.setPlaceholderText(tr("lyrics.quick.placeholder"))
        self._quick_input.setMaximumHeight(120)
        quick_layout.addWidget(self._quick_input)

        self._quick_generate = QPushButton(f"\U0001f3a4  {tr('lyrics.actions.generate')}")
        self._quick_generate.setMinimumHeight(40)
        self._quick_generate.clicked.connect(self._generate_quick)
        quick_layout.addWidget(self._quick_generate)

        quick_layout.addStretch()
        self._mode_tabs.addTab(quick_tab, tr("lyrics.quick.tab"))

        # ── Guided Mode Tab ──
        guided_tab = QWidget()
        guided_scroll = QScrollArea()
        guided_scroll.setWidgetResizable(True)
        guided_scroll.setFrameShape(QFrame.Shape.NoFrame)

        guided_inner = QWidget()
        guided_layout = QVBoxLayout(guided_inner)
        guided_layout.setContentsMargins(8, 12, 8, 8)
        guided_layout.setSpacing(10)

        # Theme input
        theme_label = QLabel(tr("lyrics.guided.theme"))
        theme_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(theme_label)

        self._theme_input = QLineEdit()
        self._theme_input.setPlaceholderText(tr("lyrics.guided.theme_placeholder"))
        self._theme_input.setMinimumHeight(34)
        guided_layout.addWidget(self._theme_input)

        # Genre picker
        genre_label = QLabel(tr("lyrics.guided.genre"))
        genre_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(genre_label)

        self._genre_picker = GenrePicker()
        self._genre_picker.setMaximumHeight(180)
        self._genre_picker.genre_selected.connect(self._on_genre_changed)
        self._genre_picker.set_genre("pop")
        guided_layout.addWidget(self._genre_picker)

        # Mood
        mood_label = QLabel(tr("lyrics.guided.mood"))
        mood_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(mood_label)

        self._mood_combo = QComboBox()
        self._mood_combo.addItem(tr("lyrics.guided.auto_detect"), "")
        for mood in MOODS:
            self._mood_combo.addItem(mood.capitalize(), mood)
        self._mood_combo.setMinimumHeight(34)
        guided_layout.addWidget(self._mood_combo)

        # Structure
        struct_label = QLabel(tr("lyrics.guided.structure"))
        struct_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(struct_label)

        self._structure_combo = QComboBox()
        self._structure_combo.addItem(tr("lyrics.guided.default_structure"), "")
        for key, val in STANDARD_STRUCTURES.items():
            display = key.replace("_", " ").title()
            self._structure_combo.addItem(display, val)
        self._structure_combo.setMinimumHeight(34)
        guided_layout.addWidget(self._structure_combo)

        # Language
        lang_label = QLabel(tr("lyrics.guided.language"))
        lang_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(lang_label)

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(language_combo_items())
        selected_language = language_label(self._settings.get("lyrics.default_language", "en"))
        selected_idx = self._lang_combo.findText(selected_language)
        if selected_idx >= 0:
            self._lang_combo.setCurrentIndex(selected_idx)
        self._lang_combo.setMinimumHeight(34)
        self._lang_combo.currentTextChanged.connect(self._on_language_changed)
        guided_layout.addWidget(self._lang_combo)

        # Generate button
        self._guided_generate = QPushButton(f"\U0001f3a4  {tr('lyrics.actions.generate')}")
        self._guided_generate.setMinimumHeight(40)
        self._guided_generate.clicked.connect(self._generate_guided)
        guided_layout.addWidget(self._guided_generate)

        guided_layout.addStretch()
        guided_scroll.setWidget(guided_inner)

        guided_tab_layout = QVBoxLayout(guided_tab)
        guided_tab_layout.setContentsMargins(0, 0, 0, 0)
        guided_tab_layout.addWidget(guided_scroll)
        self._mode_tabs.addTab(guided_tab, tr("lyrics.guided.tab"))

        # ── Pro Mode Tab ──
        pro_tab = QWidget()
        pro_scroll = QScrollArea()
        pro_scroll.setWidgetResizable(True)
        pro_scroll.setFrameShape(QFrame.Shape.NoFrame)

        pro_inner = QWidget()
        pro_layout = QVBoxLayout(pro_inner)
        pro_layout.setContentsMargins(8, 12, 8, 8)
        pro_layout.setSpacing(10)

        # System prompt editor
        sys_label = QLabel(tr("lyrics.pro.system_prompt"))
        sys_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        pro_layout.addWidget(sys_label)

        self._system_prompt = QPlainTextEdit()
        self._system_prompt.setPlaceholderText(tr("lyrics.pro.system_placeholder"))
        self._system_prompt.setMaximumHeight(150)
        self._system_prompt.setStyleSheet("font-family: monospace; font-size: 9pt;")
        pro_layout.addWidget(self._system_prompt)

        # User prompt
        user_label = QLabel(tr("lyrics.pro.user_prompt"))
        user_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        pro_layout.addWidget(user_label)

        self._user_prompt = QPlainTextEdit()
        self._user_prompt.setPlaceholderText(tr("lyrics.pro.user_placeholder"))
        self._user_prompt.setMaximumHeight(100)
        pro_layout.addWidget(self._user_prompt)

        # LLM Parameters
        params_group = QGroupBox(tr("lyrics.pro.parameters"))
        params_layout = QVBoxLayout(params_group)

        # Temperature
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel(tr("lyrics.pro.temperature")))
        self._pro_temp = QDoubleSpinBox()
        self._pro_temp.setRange(0.1, 2.0)
        self._pro_temp.setSingleStep(0.05)
        self._pro_temp.setValue(self._settings.get("lyrics.temperature", 0.8))
        self._pro_temp.setMinimumWidth(80)
        temp_row.addWidget(self._pro_temp)
        params_layout.addLayout(temp_row)

        # Top P
        topp_row = QHBoxLayout()
        topp_row.addWidget(QLabel(tr("lyrics.pro.top_p")))
        self._pro_top_p = QDoubleSpinBox()
        self._pro_top_p.setRange(0.1, 1.0)
        self._pro_top_p.setSingleStep(0.05)
        self._pro_top_p.setValue(self._settings.get("lyrics.top_p", 0.92))
        self._pro_top_p.setMinimumWidth(80)
        topp_row.addWidget(self._pro_top_p)
        params_layout.addLayout(topp_row)

        # Top K
        topk_row = QHBoxLayout()
        topk_row.addWidget(QLabel(tr("lyrics.pro.top_k")))
        self._pro_top_k = QSpinBox()
        self._pro_top_k.setRange(1, 200)
        self._pro_top_k.setValue(self._settings.get("lyrics.top_k", 50))
        self._pro_top_k.setMinimumWidth(80)
        topk_row.addWidget(self._pro_top_k)
        params_layout.addLayout(topk_row)

        # Repeat penalty
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel(tr("lyrics.pro.repeat_penalty")))
        self._pro_repeat = QDoubleSpinBox()
        self._pro_repeat.setRange(1.0, 2.0)
        self._pro_repeat.setSingleStep(0.05)
        self._pro_repeat.setValue(self._settings.get("lyrics.repeat_penalty", 1.1))
        self._pro_repeat.setMinimumWidth(80)
        rep_row.addWidget(self._pro_repeat)
        params_layout.addLayout(rep_row)

        # Max tokens
        tok_row = QHBoxLayout()
        tok_row.addWidget(QLabel(tr("lyrics.pro.max_tokens")))
        self._pro_max_tokens = QSpinBox()
        self._pro_max_tokens.setRange(256, 8192)
        self._pro_max_tokens.setSingleStep(256)
        self._pro_max_tokens.setValue(self._settings.get("lyrics.max_tokens", 2048))
        self._pro_max_tokens.setMinimumWidth(100)
        tok_row.addWidget(self._pro_max_tokens)
        params_layout.addLayout(tok_row)

        pro_layout.addWidget(params_group)

        # Generate button
        self._pro_generate = QPushButton(f"\U0001f3a4  {tr('lyrics.actions.generate')}")
        self._pro_generate.setMinimumHeight(40)
        self._pro_generate.clicked.connect(self._generate_pro)
        pro_layout.addWidget(self._pro_generate)

        pro_layout.addStretch()
        pro_scroll.setWidget(pro_inner)

        pro_tab_layout = QVBoxLayout(pro_tab)
        pro_tab_layout.setContentsMargins(0, 0, 0, 0)
        pro_tab_layout.addWidget(pro_scroll)
        self._mode_tabs.addTab(pro_tab, tr("lyrics.pro.tab"))

        left_layout.addWidget(self._mode_tabs, 1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setMinimumHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        # Cancel button (shown during generation)
        self._cancel_btn = QPushButton(tr("lyrics.actions.cancel"))
        self._cancel_btn.setObjectName("dangerBtn")
        self._cancel_btn.setMinimumHeight(34)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_generation)
        left_layout.addWidget(self._cancel_btn)

        # Regenerate with new seed
        regen_row = QHBoxLayout()
        self._regen_btn = QPushButton(f"\U0001f504 {tr('lyrics.actions.regenerate')}")
        self._regen_btn.setObjectName("secondaryBtn")
        self._regen_btn.setMinimumHeight(34)
        self._regen_btn.setEnabled(False)
        self._regen_btn.clicked.connect(self._regenerate)
        regen_row.addWidget(self._regen_btn)
        left_layout.addLayout(regen_row)

        splitter.addWidget(left)

        # ── Center Panel: Editor ────────────────────────────────────────────────
        self._editor = LyricsEditor()
        self._editor.send_to_song_forge.connect(self.send_to_forge.emit)
        self._editor.section_regenerate.connect(self._regenerate_section)
        splitter.addWidget(self._editor)

        # ── Right Panel: History ────────────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(280)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 16, 16, 16)

        self._history = HistoryPanel()
        self._history.entry_selected.connect(self._load_from_history)
        self._history.create_requested.connect(self._quick_generate.click)
        right_layout.addWidget(self._history)

        splitter.addWidget(right)

        # Splitter ratios
        splitter.setStretchFactor(0, 0)  # Left: fixed
        splitter.setStretchFactor(1, 1)  # Center: stretches
        splitter.setStretchFactor(2, 0)  # Right: fixed

        layout.addWidget(splitter)

    def _connect_signals(self):
        pass  # Signals connected inline in _build_ui

    def _selected_language_code(self) -> str:
        return language_code_from_label(self._lang_combo.currentText())

    def _on_language_changed(self, label: str):
        self._settings.set("lyrics.default_language", language_code_from_label(label))

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate_quick(self):
        """Generate lyrics in Quick Mode."""
        description = self._quick_input.toPlainText().strip()
        if not description:
            if self.toast_mgr:
                self.toast_mgr.warning(tr("lyrics.messages.describe_song"))
            return

        from engines.lyrics_engine import generate_lyrics_quick
        self._run_generation(
            generate_lyrics_quick,
            description,
        )

    def _generate_guided(self):
        """Generate lyrics in Guided Mode."""
        theme = self._theme_input.text().strip()
        if not theme:
            if self.toast_mgr:
                self.toast_mgr.warning(tr("lyrics.messages.enter_theme"))
            return

        genre_id = self._genre_picker.current_genre
        mood = self._mood_combo.currentData() or ""
        structure = self._structure_combo.currentData() or ""
        language = self._selected_language_code()

        from engines.lyrics_engine import generate_lyrics
        self._run_generation(
            generate_lyrics,
            theme,
            genre_id=genre_id,
            mood=mood,
            language=language,
            structure_override=structure,
        )

    def _generate_pro(self):
        """Generate lyrics in Pro Mode with custom prompts and parameters."""
        system = self._system_prompt.toPlainText().strip()
        user = self._user_prompt.toPlainText().strip()

        if not user:
            if self.toast_mgr:
                self.toast_mgr.warning(tr("lyrics.messages.enter_user_prompt"))
            return

        if not system:
            from engines.lyrics_templates import BASE_SYSTEM_PROMPT
            system = BASE_SYSTEM_PROMPT

        from engines.lyrics_engine import generate_lyrics

        self._run_generation(
            generate_lyrics,
            user,
            temperature=self._pro_temp.value(),
            top_p=self._pro_top_p.value(),
            top_k=self._pro_top_k.value(),
            repeat_penalty=self._pro_repeat.value(),
            max_tokens=self._pro_max_tokens.value(),
            system_prompt_override=system,
        )

    def _regenerate(self):
        """Re-run the last generation with a new implicit seed."""
        mode = self._mode_tabs.currentIndex()
        if mode == 0:
            self._generate_quick()
        elif mode == 1:
            self._generate_guided()
        else:
            self._generate_pro()

    def _regenerate_section(self, section_tag: str):
        """Regenerate a specific section of the current lyrics."""
        lyrics = self._editor.text
        if not lyrics.strip():
            return

        from engines.lyrics_engine import regenerate_section

        genre_id = self._genre_picker.current_genre
        mood = self._mood_combo.currentData() or ""

        def _task(
            progress_cb=None, step_cb=None, log_cb=None,
            cancel_event=None, **kw
        ):
            return regenerate_section(
                full_lyrics=lyrics,
                section_tag=section_tag.strip("[]"),
                genre_id=genre_id,
                mood=mood,
                progress_cb=progress_cb,
                step_cb=step_cb,
                log_cb=log_cb,
                cancel_event=cancel_event,
                token_cb=None,
            )

        self._set_generating(True)
        self._worker = InferenceWorker(_task)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_section_regenerated)
        self._worker.error.connect(self._on_generation_error)
        self._worker.cancelled.connect(self._on_generation_cancelled)
        self._worker.start()

    def _run_generation(self, gen_fn, prompt: str, **kwargs):
        """Run a generation function on a worker thread."""
        token_emitter = None

        def _task(
            progress_cb=None, step_cb=None, log_cb=None,
            cancel_event=None, **kw
        ):
            return gen_fn(
                prompt,
                progress_cb=progress_cb,
                step_cb=step_cb,
                log_cb=log_cb,
                cancel_event=cancel_event,
                token_cb=token_emitter,
                **kwargs,
            )

        self._set_generating(True)
        self._editor.start_streaming()

        self._worker = InferenceWorker(_task)
        self._worker.token.connect(
            self._editor.append_token,
            Qt.ConnectionType.QueuedConnection,
        )
        token_emitter = self._worker.token.emit
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_generation_complete)
        self._worker.error.connect(self._on_generation_error)
        self._worker.cancelled.connect(self._on_generation_cancelled)
        self._worker.start()

    def _cancel_generation(self):
        if self._worker:
            self._worker.cancel()
            self._set_generating(False)
            self._editor.stop_streaming()
            self._editor.set_status(tr("lyrics.messages.cancelled"), Palette.YELLOW)
            if self.toast_mgr:
                self.toast_mgr.warning(tr("lyrics.messages.cancelled"))

    def _on_generation_complete(self, result: dict):
        """Handle completed lyrics generation."""
        self._worker = None
        self._set_generating(False)
        self._editor.stop_streaming()

        if result.get("cancelled"):
            return

        lyrics = result.get("lyrics", "")
        if not lyrics:
            self._editor.set_status(tr("lyrics.messages.no_output"), Palette.RED)
            if self.toast_mgr:
                self.toast_mgr.warning(tr("lyrics.messages.empty_output"))
            return

        # Save to history
        entry = LyricsEntry(
            prompt=self._get_current_prompt(),
            genre=result.get("genre", ""),
            mood=result.get("mood", ""),
            language=result.get("language", "en"),
            model_id=result.get("model_id", ""),
            temperature=result.get("generation_params", {}).get("temperature", 0.8),
            lyrics_original=lyrics,
            generation_params=json.dumps(result.get("generation_params", {})),
        )
        self._db.save(entry)
        self._history._refresh()

        self._regen_btn.setEnabled(True)
        self._editor.set_status(tr("lyrics.messages.complete"), Palette.GREEN)
        if self.toast_mgr:
            self.toast_mgr.success(tr("lyrics.messages.generated"))

    def _on_section_regenerated(self, result: dict):
        """Handle completed section regeneration."""
        self._worker = None
        self._set_generating(False)
        tag = result.get("section_tag", "")
        new_content = result.get("new_content", "")

        if tag and new_content:
            self._editor.replace_section(f"[{tag}]", new_content)
            self._editor.set_status(f"Regenerated [{tag}]", Palette.GREEN)
            if self.toast_mgr:
                self.toast_mgr.success(tr("lyrics.messages.section_regenerated", tag=tag))

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        self._worker = None
        self._set_generating(False)
        self._editor.stop_streaming()
        self._editor.set_status(tr("lyrics.messages.error_status", error=error_msg), Palette.RED)
        if self.toast_mgr:
            self.toast_mgr.error(tr("lyrics.messages.failed", error=error_msg))

    def _on_generation_cancelled(self):
        self._worker = None
        self._editor.stop_streaming()

    def _set_generating(self, generating: bool):
        """Toggle UI state for generation in progress."""
        self._progress.setVisible(generating)
        self._cancel_btn.setVisible(generating)
        self._quick_generate.setEnabled(not generating)
        self._guided_generate.setEnabled(not generating)
        self._pro_generate.setEnabled(not generating)
        self._regen_btn.setEnabled(not generating)

        if generating:
            self._progress.setValue(0)

    def _get_current_prompt(self) -> str:
        """Get the current prompt text based on active mode."""
        mode = self._mode_tabs.currentIndex()
        if mode == 0:
            return self._quick_input.toPlainText().strip()
        elif mode == 1:
            return self._theme_input.text().strip()
        else:
            return self._user_prompt.toPlainText().strip()

    # ── History Integration ────────────────────────────────────────────────────

    def _load_from_history(self, entry: LyricsEntry):
        """Load a history entry into the editor."""
        self._editor.text = entry.lyrics
        self._editor.set_status(
            tr(
                "lyrics.messages.loaded_history",
                genre=entry.genre.upper(),
                timestamp=entry.timestamp_str,
            ),
            Palette.BLUE,
        )

    # ── Genre Change ───────────────────────────────────────────────────────────

    def _on_genre_changed(self, genre_id: str):
        self._current_genre = genre_id
