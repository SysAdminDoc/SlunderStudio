"""
Slunder Studio v0.0.2 — Lyrics View
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
    QSlider, QPlainTextEdit, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot

from ui.theme import Palette
from ui.lyrics_editor import LyricsEditor
from core.settings import Settings
from core.workers import InferenceWorker
from core.lyrics_db import LyricsDB, LyricsEntry
from engines.lyrics_templates import (
    GENRE_TEMPLATES, MOODS, STANDARD_STRUCTURES,
    get_genre_list, get_genre_categories, get_random_theme,
    build_generation_prompt,
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
        self._search.setPlaceholderText("Search genres...")
        self._search.setFixedHeight(34)
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)

        # Genre list
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._on_selection)
        layout.addWidget(self._list, 1)

        self._populate()

    def _populate(self):
        for genre in get_genre_list():
            item = QListWidgetItem(f"{genre['name']}  —  {genre['description']}")
            item.setData(Qt.ItemDataRole.UserRole, genre["id"])
            self._list.addItem(item)

    def _filter(self, text: str):
        text = text.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            visible = text in item.text().lower()
            item.setHidden(not visible)

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = LyricsDB()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        header = QLabel("History")
        header.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {Palette.TEXT};")
        layout.addWidget(header)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search history...")
        self._search.setFixedHeight(32)
        self._search.textChanged.connect(self._refresh)
        layout.addWidget(self._search)

        # Filter buttons
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)

        self._all_btn = QPushButton("All")
        self._all_btn.setObjectName("ghostBtn")
        self._all_btn.setCheckable(True)
        self._all_btn.setChecked(True)
        self._all_btn.setFixedHeight(26)
        self._all_btn.clicked.connect(lambda: self._set_filter("all"))
        filter_row.addWidget(self._all_btn)

        self._fav_btn = QPushButton("\u2605 Favorites")
        self._fav_btn.setObjectName("ghostBtn")
        self._fav_btn.setCheckable(True)
        self._fav_btn.setFixedHeight(26)
        self._fav_btn.clicked.connect(lambda: self._set_filter("favorites"))
        filter_row.addWidget(self._fav_btn)

        filter_row.addStretch()
        layout.addLayout(filter_row)

        # List
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentItemChanged.connect(self._on_selection)
        layout.addWidget(self._list, 1)

        # Count
        self._count_label = QLabel("0 entries")
        self._count_label.setObjectName("caption")
        layout.addWidget(self._count_label)

        self._current_filter = "all"
        self._refresh()

    def _set_filter(self, mode: str):
        self._current_filter = mode
        self._all_btn.setChecked(mode == "all")
        self._fav_btn.setChecked(mode == "favorites")
        self._refresh()

    def _refresh(self):
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

        self._count_label.setText(f"{len(entries)} entries")

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

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main splitter: left (controls) | center (editor) | right (history)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left Panel: Input Controls ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

        title = QLabel("Lyrics Engine")
        title.setObjectName("heading")
        left_layout.addWidget(title)

        # Mode tabs
        self._mode_tabs = QTabWidget()

        # ── Quick Mode Tab ──
        quick_tab = QWidget()
        quick_layout = QVBoxLayout(quick_tab)
        quick_layout.setContentsMargins(8, 12, 8, 8)
        quick_layout.setSpacing(12)

        quick_label = QLabel("Describe your song in a sentence")
        quick_label.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {Palette.SUBTEXT0};")
        quick_layout.addWidget(quick_label)

        self._quick_input = QTextEdit()
        self._quick_input.setPlaceholderText(
            "e.g., \"A melancholic synthwave track about driving alone at night\"\n\n"
            "or \"Upbeat summer pop song about falling in love at a beach party\""
        )
        self._quick_input.setMaximumHeight(120)
        quick_layout.addWidget(self._quick_input)

        self._quick_generate = QPushButton("\U0001f3a4  Generate Lyrics")
        self._quick_generate.setFixedHeight(40)
        self._quick_generate.clicked.connect(self._generate_quick)
        quick_layout.addWidget(self._quick_generate)

        quick_layout.addStretch()
        self._mode_tabs.addTab(quick_tab, "Quick")

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
        theme_label = QLabel("Theme / Topic")
        theme_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(theme_label)

        self._theme_input = QLineEdit()
        self._theme_input.setPlaceholderText("What is the song about?")
        self._theme_input.setFixedHeight(34)
        guided_layout.addWidget(self._theme_input)

        # Genre picker
        genre_label = QLabel("Genre")
        genre_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(genre_label)

        self._genre_picker = GenrePicker()
        self._genre_picker.setMaximumHeight(180)
        self._genre_picker.genre_selected.connect(self._on_genre_changed)
        self._genre_picker.set_genre("pop")
        guided_layout.addWidget(self._genre_picker)

        # Mood
        mood_label = QLabel("Mood")
        mood_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(mood_label)

        self._mood_combo = QComboBox()
        self._mood_combo.addItem("Auto-detect", "")
        for mood in MOODS:
            self._mood_combo.addItem(mood.capitalize(), mood)
        self._mood_combo.setFixedHeight(34)
        guided_layout.addWidget(self._mood_combo)

        # Structure
        struct_label = QLabel("Song Structure")
        struct_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(struct_label)

        self._structure_combo = QComboBox()
        self._structure_combo.addItem("Default for genre", "")
        for key, val in STANDARD_STRUCTURES.items():
            display = key.replace("_", " ").title()
            self._structure_combo.addItem(display, val)
        self._structure_combo.setFixedHeight(34)
        guided_layout.addWidget(self._structure_combo)

        # Language
        lang_label = QLabel("Language")
        lang_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        guided_layout.addWidget(lang_label)

        self._lang_combo = QComboBox()
        self._lang_combo.addItems([
            "English", "Spanish", "French", "Portuguese", "German",
            "Italian", "Japanese", "Korean", "Chinese (Mandarin)",
            "Arabic", "Hindi", "Russian", "Dutch", "Swedish",
            "Turkish", "Polish", "Thai", "Vietnamese", "Indonesian",
        ])
        self._lang_combo.setFixedHeight(34)
        guided_layout.addWidget(self._lang_combo)

        # Generate button
        self._guided_generate = QPushButton("\U0001f3a4  Generate Lyrics")
        self._guided_generate.setFixedHeight(40)
        self._guided_generate.clicked.connect(self._generate_guided)
        guided_layout.addWidget(self._guided_generate)

        guided_layout.addStretch()
        guided_scroll.setWidget(guided_inner)

        guided_tab_layout = QVBoxLayout(guided_tab)
        guided_tab_layout.setContentsMargins(0, 0, 0, 0)
        guided_tab_layout.addWidget(guided_scroll)
        self._mode_tabs.addTab(guided_tab, "Guided")

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
        sys_label = QLabel("System Prompt")
        sys_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        pro_layout.addWidget(sys_label)

        self._system_prompt = QPlainTextEdit()
        self._system_prompt.setPlaceholderText("Custom system prompt for the LLM...")
        self._system_prompt.setMaximumHeight(150)
        self._system_prompt.setStyleSheet("font-family: monospace; font-size: 12px;")
        pro_layout.addWidget(self._system_prompt)

        # User prompt
        user_label = QLabel("User Prompt")
        user_label.setStyleSheet(f"font-weight: 600; color: {Palette.SUBTEXT0};")
        pro_layout.addWidget(user_label)

        self._user_prompt = QPlainTextEdit()
        self._user_prompt.setPlaceholderText("Your creative request to the LLM...")
        self._user_prompt.setMaximumHeight(100)
        pro_layout.addWidget(self._user_prompt)

        # LLM Parameters
        params_group = QGroupBox("LLM Parameters")
        params_layout = QVBoxLayout(params_group)

        # Temperature
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temperature"))
        self._pro_temp = QDoubleSpinBox()
        self._pro_temp.setRange(0.1, 2.0)
        self._pro_temp.setSingleStep(0.05)
        self._pro_temp.setValue(self._settings.get("lyrics.temperature", 0.8))
        self._pro_temp.setFixedWidth(80)
        temp_row.addWidget(self._pro_temp)
        params_layout.addLayout(temp_row)

        # Top P
        topp_row = QHBoxLayout()
        topp_row.addWidget(QLabel("Top P"))
        self._pro_top_p = QDoubleSpinBox()
        self._pro_top_p.setRange(0.1, 1.0)
        self._pro_top_p.setSingleStep(0.05)
        self._pro_top_p.setValue(self._settings.get("lyrics.top_p", 0.92))
        self._pro_top_p.setFixedWidth(80)
        topp_row.addWidget(self._pro_top_p)
        params_layout.addLayout(topp_row)

        # Top K
        topk_row = QHBoxLayout()
        topk_row.addWidget(QLabel("Top K"))
        self._pro_top_k = QSpinBox()
        self._pro_top_k.setRange(1, 200)
        self._pro_top_k.setValue(self._settings.get("lyrics.top_k", 50))
        self._pro_top_k.setFixedWidth(80)
        topk_row.addWidget(self._pro_top_k)
        params_layout.addLayout(topk_row)

        # Repeat penalty
        rep_row = QHBoxLayout()
        rep_row.addWidget(QLabel("Repeat Penalty"))
        self._pro_repeat = QDoubleSpinBox()
        self._pro_repeat.setRange(1.0, 2.0)
        self._pro_repeat.setSingleStep(0.05)
        self._pro_repeat.setValue(self._settings.get("lyrics.repeat_penalty", 1.1))
        self._pro_repeat.setFixedWidth(80)
        rep_row.addWidget(self._pro_repeat)
        params_layout.addLayout(rep_row)

        # Max tokens
        tok_row = QHBoxLayout()
        tok_row.addWidget(QLabel("Max Tokens"))
        self._pro_max_tokens = QSpinBox()
        self._pro_max_tokens.setRange(256, 8192)
        self._pro_max_tokens.setSingleStep(256)
        self._pro_max_tokens.setValue(self._settings.get("lyrics.max_tokens", 2048))
        self._pro_max_tokens.setFixedWidth(100)
        tok_row.addWidget(self._pro_max_tokens)
        params_layout.addLayout(tok_row)

        pro_layout.addWidget(params_group)

        # Generate button
        self._pro_generate = QPushButton("\U0001f3a4  Generate Lyrics")
        self._pro_generate.setFixedHeight(40)
        self._pro_generate.clicked.connect(self._generate_pro)
        pro_layout.addWidget(self._pro_generate)

        pro_layout.addStretch()
        pro_scroll.setWidget(pro_inner)

        pro_tab_layout = QVBoxLayout(pro_tab)
        pro_tab_layout.setContentsMargins(0, 0, 0, 0)
        pro_tab_layout.addWidget(pro_scroll)
        self._mode_tabs.addTab(pro_tab, "Pro")

        left_layout.addWidget(self._mode_tabs, 1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        # Cancel button (shown during generation)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("dangerBtn")
        self._cancel_btn.setFixedHeight(34)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_generation)
        left_layout.addWidget(self._cancel_btn)

        # Regenerate with new seed
        regen_row = QHBoxLayout()
        self._regen_btn = QPushButton("\U0001f504 Regenerate")
        self._regen_btn.setObjectName("secondaryBtn")
        self._regen_btn.setFixedHeight(34)
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
        right.setFixedWidth(280)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 16, 16, 16)

        self._history = HistoryPanel()
        self._history.entry_selected.connect(self._load_from_history)
        right_layout.addWidget(self._history)

        splitter.addWidget(right)

        # Splitter ratios
        splitter.setStretchFactor(0, 0)  # Left: fixed
        splitter.setStretchFactor(1, 1)  # Center: stretches
        splitter.setStretchFactor(2, 0)  # Right: fixed

        layout.addWidget(splitter)

    def _connect_signals(self):
        pass  # Signals connected inline in _build_ui

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate_quick(self):
        """Generate lyrics in Quick Mode."""
        description = self._quick_input.toPlainText().strip()
        if not description:
            if self.toast_mgr:
                self.toast_mgr.warning("Please describe your song first.")
            return

        from engines.lyrics_engine import generate_lyrics_quick
        self._run_generation(
            generate_lyrics_quick,
            description,
            token_cb=self._editor.append_token,
        )

    def _generate_guided(self):
        """Generate lyrics in Guided Mode."""
        theme = self._theme_input.text().strip()
        if not theme:
            if self.toast_mgr:
                self.toast_mgr.warning("Please enter a theme or topic.")
            return

        genre_id = self._genre_picker.current_genre
        mood = self._mood_combo.currentData() or ""
        structure = self._structure_combo.currentData() or ""
        language = self._lang_combo.currentText().split("(")[0].strip().lower()
        if language == "english":
            language = "en"

        from engines.lyrics_engine import generate_lyrics
        self._run_generation(
            generate_lyrics,
            theme,
            genre_id=genre_id,
            mood=mood,
            language=language,
            structure_override=structure,
            token_cb=self._editor.append_token,
        )

    def _generate_pro(self):
        """Generate lyrics in Pro Mode with custom prompts and parameters."""
        system = self._system_prompt.toPlainText().strip()
        user = self._user_prompt.toPlainText().strip()

        if not user:
            if self.toast_mgr:
                self.toast_mgr.warning("Please enter a user prompt.")
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
            token_cb=self._editor.append_token,
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
        self._worker.start()

    def _run_generation(self, gen_fn, prompt: str, **kwargs):
        """Run a generation function on a worker thread."""
        token_cb = kwargs.pop("token_cb", None)

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
                token_cb=token_cb,
                **kwargs,
            )

        self._set_generating(True)
        self._editor.start_streaming()

        self._worker = InferenceWorker(_task)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_generation_complete)
        self._worker.error.connect(self._on_generation_error)
        self._worker.start()

    def _cancel_generation(self):
        if self._worker:
            self._worker.cancel()
            self._set_generating(False)
            self._editor.stop_streaming()
            self._editor.set_status("Generation cancelled", Palette.YELLOW)
            if self.toast_mgr:
                self.toast_mgr.warning("Generation cancelled")

    def _on_generation_complete(self, result: dict):
        """Handle completed lyrics generation."""
        self._set_generating(False)
        self._editor.stop_streaming()

        if result.get("cancelled"):
            return

        lyrics = result.get("lyrics", "")
        if not lyrics:
            self._editor.set_status("No output generated", Palette.RED)
            if self.toast_mgr:
                self.toast_mgr.warning("Model produced empty output. Try a different prompt.")
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
        self._editor.set_status("Generation complete \u2022 Saved to history", Palette.GREEN)
        if self.toast_mgr:
            self.toast_mgr.success("Lyrics generated!")

    def _on_section_regenerated(self, result: dict):
        """Handle completed section regeneration."""
        self._set_generating(False)
        tag = result.get("section_tag", "")
        new_content = result.get("new_content", "")

        if tag and new_content:
            self._editor.replace_section(f"[{tag}]", new_content)
            self._editor.set_status(f"Regenerated [{tag}]", Palette.GREEN)
            if self.toast_mgr:
                self.toast_mgr.success(f"[{tag}] regenerated!")

    def _on_generation_error(self, error_msg: str):
        """Handle generation error."""
        self._set_generating(False)
        self._editor.stop_streaming()
        self._editor.set_status(f"Error: {error_msg}", Palette.RED)
        if self.toast_mgr:
            self.toast_mgr.error(f"Generation failed: {error_msg}")

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
            f"Loaded from history: {entry.genre.upper()} \u2022 {entry.timestamp_str}",
            Palette.BLUE,
        )

    # ── Genre Change ───────────────────────────────────────────────────────────

    def _on_genre_changed(self, genre_id: str):
        self._current_genre = genre_id
