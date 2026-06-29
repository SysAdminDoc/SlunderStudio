"""
Slunder Studio v0.1.15 — Song Forge View
Main Song Forge page: Quick/Advanced generation modes, style tag browser,
batch generation, waveform display, seed explorer, mood curves, reference panel.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSplitter, QTabWidget, QTextEdit, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QProgressBar,
    QFileDialog, QCheckBox, QGroupBox, QGridLayout, QListWidget,
    QListWidgetItem, QPlainTextEdit, QSlider,
)
from PySide6.QtCore import Signal, Qt

from core.workers import InferenceWorker
from core.audio_engine import AudioEngine
from engines.style_tags import StyleTagDB, CATEGORIES
from ui.waveform_widget import WaveformWidget
from ui.batch_view import BatchView
from ui.seed_explorer import SeedExplorer
from ui.mood_curve_editor import MoodCurveEditor
from ui.reference_panel import ReferencePanel
from ui.accessibility import install_accessibility


class StyleTagBrowser(QWidget):
    """Searchable, categorized style tag selector with favorites."""
    tags_changed = Signal(str)  # comma-separated tag string

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = StyleTagDB()
        self._selected_tags: list[str] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search 1000+ style tags...")
        self._search.setFixedHeight(28)
        self._search.textChanged.connect(self._refresh)
        layout.addWidget(self._search)

        # Category filter
        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        self._cat_combo = QComboBox()
        self._cat_combo.addItem("All Categories")
        self._cat_combo.addItems([c.title() for c in self._db.get_categories()])
        self._cat_combo.setFixedHeight(26)
        self._cat_combo.currentIndexChanged.connect(self._refresh)
        cat_row.addWidget(self._cat_combo)

        self._fav_check = QCheckBox("Favorites")
        self._fav_check.setStyleSheet("color: #F9E2AF;")
        self._fav_check.toggled.connect(self._refresh)
        cat_row.addWidget(self._fav_check)

        layout.addLayout(cat_row)

        # Tag list
        self._tag_list = QListWidget()
        self._tag_list.setStyleSheet(
            "QListWidget { background: #181825; border: 1px solid #313244; border-radius: 4px; }"
            "QListWidget::item { padding: 3px 6px; color: #CDD6F4; }"
            "QListWidget::item:selected { background: #313244; color: #89B4FA; }"
            "QListWidget::item:hover { background: #1E1E2E; }"
        )
        self._tag_list.setSelectionMode(QListWidget.MultiSelection)
        self._tag_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tag_list, 1)

        # Selected tags display
        self._selected_label = QLabel("No tags selected")
        self._selected_label.setWordWrap(True)
        self._selected_label.setStyleSheet("color: #94E2D5; font-size: 11px; padding: 4px;")
        layout.addWidget(self._selected_label)

        self._refresh()
        install_accessibility(
            self,
            "Style tag browser",
            named_controls=[
                (self._search, "Search style tags", "Filters the style tag list."),
                (self._cat_combo, "Style tag category", "Filters tags by category."),
                (self._fav_check, "Favorite style tags only", "Shows only favorite style tags."),
                (self._tag_list, "Style tag results", "Selects one or more style tags."),
            ],
            tab_order=[
                self._search,
                self._cat_combo,
                self._fav_check,
                self._tag_list,
            ],
        )

    def _refresh(self):
        self._tag_list.clear()
        query = self._search.text()
        cat_idx = self._cat_combo.currentIndex()
        category = "" if cat_idx == 0 else self._db.get_categories()[cat_idx - 1]
        fav_only = self._fav_check.isChecked()

        results = self._db.search(query, category=category, favorites_only=fav_only)
        for item in results[:200]:  # Limit display
            tag = item["tag"]
            star = "\u2605 " if item.get("is_favorite") else ""
            li = QListWidgetItem(f"{star}{tag}")
            li.setData(Qt.UserRole, tag)
            if tag in self._selected_tags:
                li.setSelected(True)
            self._tag_list.addItem(li)

    def _on_selection_changed(self):
        self._selected_tags = []
        for item in self._tag_list.selectedItems():
            tag = item.data(Qt.UserRole)
            if tag:
                self._selected_tags.append(tag)

        if self._selected_tags:
            self._selected_label.setText(", ".join(self._selected_tags))
        else:
            self._selected_label.setText("No tags selected")

        self.tags_changed.emit(", ".join(self._selected_tags))

    def set_tags(self, tag_string: str):
        """Set selected tags from a comma-separated string."""
        self._selected_tags = [t.strip() for t in tag_string.split(",") if t.strip()]
        self._selected_label.setText(", ".join(self._selected_tags) or "No tags selected")
        self.tags_changed.emit(tag_string)
        self._refresh()

    def get_tags(self) -> str:
        return ", ".join(self._selected_tags)


class SongForgeView(QWidget):
    """
    Main Song Forge page. ACE-Step v1.5 song generation with:
    - Quick/Advanced mode tabs
    - Style tag browser
    - Waveform display
    - Batch generation panel
    - Seed explorer
    - Mood curve editor
    - Reference track panel
    """
    send_to_vocals = Signal(str)  # audio_path

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self._toast = toast_mgr
        self._current_audio_path = ""
        self._is_generating = False
        self._worker = None
        self._seed_explore_params: list[dict] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Main splitter: left controls | center output | right reference
        splitter = QSplitter(Qt.Horizontal)

        # ── Left Panel: Input Controls ────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Mode tabs
        self._mode_tabs = QTabWidget()
        self._mode_tabs.setFixedHeight(340)

        # Quick Mode
        quick_page = QWidget()
        ql = QVBoxLayout(quick_page)
        ql.setSpacing(8)

        ql.addWidget(QLabel("Lyrics"))
        self._quick_lyrics = QTextEdit()
        self._quick_lyrics.setPlaceholderText(
            "Paste lyrics from Lyrics Engine or type directly...\n"
            "Use [Verse], [Chorus], [Bridge] structure tags."
        )
        self._quick_lyrics.setStyleSheet(
            "QTextEdit { background: #181825; border: 1px solid #313244; border-radius: 6px; "
            "color: #CDD6F4; font-size: 13px; padding: 8px; }"
        )
        ql.addWidget(self._quick_lyrics)

        ql.addWidget(QLabel("Style Tags"))
        self._quick_tags = QLineEdit()
        self._quick_tags.setPlaceholderText("e.g., pop, female vocals, dreamy, 120 bpm")
        self._quick_tags.setFixedHeight(30)
        ql.addWidget(self._quick_tags)

        self._mode_tabs.addTab(quick_page, "Quick")

        # Advanced Mode
        adv_page = QWidget()
        adv_scroll = QScrollArea()
        adv_scroll.setWidgetResizable(True)
        adv_scroll.setStyleSheet("QScrollArea { border: none; }")
        adv_inner = QWidget()
        al = QVBoxLayout(adv_inner)
        al.setSpacing(6)

        al.addWidget(QLabel("Lyrics"))
        self._adv_lyrics = QTextEdit()
        self._adv_lyrics.setPlaceholderText("Lyrics with structure tags...")
        self._adv_lyrics.setMinimumHeight(120)
        self._adv_lyrics.setStyleSheet(
            "QTextEdit { background: #181825; border: 1px solid #313244; border-radius: 6px; "
            "color: #CDD6F4; font-size: 12px; padding: 6px; }"
        )
        al.addWidget(self._adv_lyrics)

        # Parameters grid
        params = QGroupBox("Generation Parameters")
        params.setStyleSheet(
            "QGroupBox { color: #A6ADC8; border: 1px solid #313244; border-radius: 6px; "
            "margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        pg = QGridLayout(params)
        pg.setSpacing(6)

        pg.addWidget(QLabel("Duration (s):"), 0, 0)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(5, 600)
        self._duration_spin.setValue(60)
        self._duration_spin.setSuffix("s")
        pg.addWidget(self._duration_spin, 0, 1)

        pg.addWidget(QLabel("CFG Scale:"), 0, 2)
        self._cfg_spin = QDoubleSpinBox()
        self._cfg_spin.setRange(1.0, 15.0)
        self._cfg_spin.setValue(5.0)
        self._cfg_spin.setSingleStep(0.5)
        pg.addWidget(self._cfg_spin, 0, 3)

        pg.addWidget(QLabel("Steps:"), 1, 0)
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(10, 100)
        self._steps_spin.setValue(60)
        pg.addWidget(self._steps_spin, 1, 1)

        pg.addWidget(QLabel("Seed:"), 1, 2)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(-1, 2**31 - 1)
        self._seed_spin.setValue(-1)
        self._seed_spin.setSpecialValueText("Random")
        pg.addWidget(self._seed_spin, 1, 3)

        pg.addWidget(QLabel("Batch:"), 2, 0)
        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 8)
        self._batch_spin.setValue(1)
        pg.addWidget(self._batch_spin, 2, 1)

        self._long_form_check = QCheckBox("Long-form stitching")
        self._long_form_check.setChecked(True)
        self._long_form_check.setToolTip(
            "For durations over 120 seconds, render sections separately and stitch them."
        )
        pg.addWidget(self._long_form_check, 2, 2, 1, 2)

        al.addWidget(params)

        # Genre fusion presets
        fusion = QGroupBox("Genre Fusion")
        fusion.setStyleSheet(
            "QGroupBox { color: #A6ADC8; border: 1px solid #313244; border-radius: 6px; "
            "margin-top: 8px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        fg = QGridLayout(fusion)
        fg.setSpacing(6)

        from engines.lyrics_templates import get_genre_list
        genres = get_genre_list()

        fg.addWidget(QLabel("Primary:"), 0, 0)
        self._fusion_primary = QComboBox()
        for genre in genres:
            self._fusion_primary.addItem(genre["name"], genre["id"])
        fg.addWidget(self._fusion_primary, 0, 1)

        fg.addWidget(QLabel("Secondary:"), 0, 2)
        self._fusion_secondary = QComboBox()
        for genre in genres:
            self._fusion_secondary.addItem(genre["name"], genre["id"])
        self._fusion_secondary.setCurrentIndex(min(2, max(0, self._fusion_secondary.count() - 1)))
        fg.addWidget(self._fusion_secondary, 0, 3)

        fg.addWidget(QLabel("Blend:"), 1, 0)
        self._fusion_slider = QSlider(Qt.Horizontal)
        self._fusion_slider.setRange(0, 100)
        self._fusion_slider.setValue(50)
        self._fusion_slider.valueChanged.connect(self._on_fusion_weight_changed)
        fg.addWidget(self._fusion_slider, 1, 1, 1, 2)

        self._fusion_weight_label = QLabel("50/50")
        self._fusion_weight_label.setStyleSheet("color: #94E2D5; font-size: 11px;")
        fg.addWidget(self._fusion_weight_label, 1, 3)

        self._fusion_apply_btn = QPushButton("Apply Fusion")
        self._fusion_apply_btn.setFixedHeight(28)
        self._fusion_apply_btn.setProperty("class", "secondary")
        self._fusion_apply_btn.clicked.connect(self._apply_genre_fusion)
        fg.addWidget(self._fusion_apply_btn, 2, 0, 1, 4)

        al.addWidget(fusion)
        adv_scroll.setWidget(adv_inner)

        adv_layout = QVBoxLayout(adv_page)
        adv_layout.setContentsMargins(0, 0, 0, 0)
        adv_layout.addWidget(adv_scroll)

        self._mode_tabs.addTab(adv_page, "Advanced")

        left_layout.addWidget(self._mode_tabs)

        # Style Tag Browser
        self._tag_browser = StyleTagBrowser()
        self._tag_browser.tags_changed.connect(self._on_tags_changed)
        left_layout.addWidget(self._tag_browser, 1)

        # Generate buttons
        gen_row = QHBoxLayout()
        gen_row.setSpacing(6)

        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setFixedHeight(36)
        self._generate_btn.clicked.connect(self._on_generate)
        gen_row.addWidget(self._generate_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setProperty("class", "danger")
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._cancel_btn.hide()
        gen_row.addWidget(self._cancel_btn)

        left_layout.addLayout(gen_row)

        # Progress
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.hide()
        left_layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("color: #6C7086; font-size: 11px;")
        left_layout.addWidget(self._status)

        splitter.addWidget(left)

        # ── Center Panel: Output & Sub-views ──────────────────────────────────
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        # Waveform
        self._waveform = WaveformWidget()
        self._waveform.setFixedHeight(150)
        center_layout.addWidget(self._waveform)

        # Output toolbar
        out_row = QHBoxLayout()
        out_row.setSpacing(6)

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedHeight(28)
        self._play_btn.setProperty("class", "secondary")
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._on_play)
        out_row.addWidget(self._play_btn)

        self._export_btn = QPushButton("Export")
        self._export_btn.setFixedHeight(28)
        self._export_btn.setProperty("class", "secondary")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        out_row.addWidget(self._export_btn)

        self._to_vocals_btn = QPushButton("Send to Vocals")
        self._to_vocals_btn.setFixedHeight(28)
        self._to_vocals_btn.setProperty("class", "secondary")
        self._to_vocals_btn.setEnabled(False)
        self._to_vocals_btn.clicked.connect(
            lambda: self.send_to_vocals.emit(self._current_audio_path)
        )
        out_row.addWidget(self._to_vocals_btn)

        out_row.addStretch()

        self._output_info = QLabel("")
        self._output_info.setStyleSheet("color: #A6ADC8; font-size: 11px;")
        out_row.addWidget(self._output_info)

        center_layout.addLayout(out_row)

        # Sub-view tabs: Batch / Seed Explorer / Mood Curve
        self._sub_tabs = QTabWidget()

        self._batch_view = BatchView()
        self._batch_view.play_requested.connect(self._play_audio)
        self._batch_view.use_result.connect(self._use_batch_result)
        self._sub_tabs.addTab(self._batch_view, "Batch Results")

        self._seed_explorer = SeedExplorer()
        self._seed_explorer.play_requested.connect(self._play_audio)
        self._seed_explorer.generate_requested.connect(self._on_seed_explore)
        self._sub_tabs.addTab(self._seed_explorer, "Seed Explorer")

        self._mood_curve = MoodCurveEditor()
        self._sub_tabs.addTab(self._mood_curve, "Mood Curve")

        center_layout.addWidget(self._sub_tabs, 1)

        splitter.addWidget(center)

        # ── Right Panel: Reference ────────────────────────────────────────────
        self._ref_panel = ReferencePanel()
        self._ref_panel.setMaximumWidth(300)
        self._ref_panel.match_requested.connect(self._on_reference_match)
        self._ref_panel.tags_extracted.connect(self._on_reference_tags)
        splitter.addWidget(self._ref_panel)

        # Splitter sizes
        splitter.setSizes([360, 500, 280])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)
        install_accessibility(
            self,
            "Song Forge",
            named_controls=[
                (self._mode_tabs, "Song Forge mode", "Switches between quick and advanced song generation controls."),
                (self._quick_lyrics, "Quick lyrics", "Lyrics or structure tags for quick song generation."),
                (self._quick_tags, "Quick style tags", "Comma-separated style prompt for quick generation."),
                (self._adv_lyrics, "Advanced lyrics", "Lyrics with structure tags for advanced generation."),
                (self._duration_spin, "Song duration", "Target generated song duration in seconds."),
                (self._cfg_spin, "CFG scale", "Controls prompt adherence for generation."),
                (self._steps_spin, "Inference steps", "Controls generation quality and speed."),
                (self._seed_spin, "Generation seed", "Use random or fixed seed for repeatable generation."),
                (self._batch_spin, "Batch count", "Number of variations to generate."),
                (self._long_form_check, "Long-form stitching", "Renders long songs by sections and stitches them."),
                (self._fusion_primary, "Primary genre", "First genre for fusion tags."),
                (self._fusion_secondary, "Secondary genre", "Second genre for fusion tags."),
                (self._fusion_slider, "Genre blend", "Balances primary and secondary genre tags."),
                (self._fusion_apply_btn, "Apply genre fusion", "Copies blended genre tags into the prompt."),
                (self._generate_btn, "Generate song", "Starts ACE-Step song generation."),
                (self._cancel_btn, "Cancel generation", "Requests cancellation for the running generation."),
                (self._play_btn, "Play generated song", "Plays the current generated output."),
                (self._export_btn, "Export generated song", "Exports the current generated output."),
                (self._to_vocals_btn, "Send generated song to vocals", "Routes generated audio to Vocal Suite."),
                (self._sub_tabs, "Song Forge result tools", "Switches between batch results, seed explorer, and mood curve."),
            ],
            tab_order=[
                self._mode_tabs,
                self._quick_lyrics,
                self._quick_tags,
                self._adv_lyrics,
                self._duration_spin,
                self._cfg_spin,
                self._steps_spin,
                self._seed_spin,
                self._batch_spin,
                self._long_form_check,
                self._fusion_primary,
                self._fusion_secondary,
                self._fusion_slider,
                self._fusion_apply_btn,
                self._tag_browser._search,
                self._tag_browser._cat_combo,
                self._tag_browser._fav_check,
                self._tag_browser._tag_list,
                self._generate_btn,
                self._cancel_btn,
                self._play_btn,
                self._export_btn,
                self._to_vocals_btn,
                self._sub_tabs,
            ],
        )

    # ── Lyrics Injection ──────────────────────────────────────────────────────

    def set_lyrics(self, lyrics_text: str):
        """Receive lyrics from Lyrics Engine via cross-module routing."""
        self._quick_lyrics.setPlainText(lyrics_text)
        self._adv_lyrics.setPlainText(lyrics_text)
        if self._toast:
            self._toast.show_toast("Lyrics loaded into Song Forge", "info")

    # ── Generation ────────────────────────────────────────────────────────────

    def _get_lyrics(self) -> str:
        if self._mode_tabs.currentIndex() == 0:
            return self._quick_lyrics.toPlainText().strip()
        return self._adv_lyrics.toPlainText().strip()

    def _get_tags(self) -> str:
        if self._mode_tabs.currentIndex() == 0:
            manual = self._quick_tags.text().strip()
            browser = self._tag_browser.get_tags()
            parts = [p for p in [manual, browser] if p]
            return ", ".join(parts)
        return self._tag_browser.get_tags()

    def _on_tags_changed(self, tags: str):
        """Sync tag browser selection to quick mode input."""
        if self._mode_tabs.currentIndex() == 0:
            existing = self._quick_tags.text().strip()
            if not existing:
                self._quick_tags.setText(tags)

    def _on_fusion_weight_changed(self, value: int):
        self._fusion_weight_label.setText(f"{100 - value}/{value}")

    def _apply_genre_fusion(self):
        from engines.lyrics_templates import blend_genre_style_tags

        primary = self._fusion_primary.currentData()
        secondary = self._fusion_secondary.currentData()
        tags = blend_genre_style_tags(primary, secondary, self._fusion_slider.value() / 100)
        tag_str = ", ".join(tags)
        self._tag_browser.set_tags(tag_str)
        self._quick_tags.setText(tag_str)
        if self._toast:
            self._toast.show_toast("Genre fusion tags applied", "success")

    def _on_generate(self):
        lyrics = self._get_lyrics()
        tags = self._get_tags()

        if not lyrics and not tags:
            if self._toast:
                self._toast.show_toast("Add lyrics or style tags first", "warning")
            return

        self._is_generating = True
        self._generate_btn.hide()
        self._cancel_btn.show()
        self._progress.setValue(0)
        self._progress.show()
        self._status.setText("Starting generation...")

        # Determine batch count
        batch_count = 1
        if self._mode_tabs.currentIndex() == 1:
            batch_count = self._batch_spin.value()

        duration = self._duration_spin.value() if self._mode_tabs.currentIndex() == 1 else 60.0
        cfg = self._cfg_spin.value() if self._mode_tabs.currentIndex() == 1 else 5.0
        steps = self._steps_spin.value() if self._mode_tabs.currentIndex() == 1 else 60
        seed = self._seed_spin.value() if self._mode_tabs.currentIndex() == 1 else -1
        long_form = (
            self._mode_tabs.currentIndex() == 1
            and duration > 120
            and self._long_form_check.isChecked()
        )
        job_inputs = {
            "duration": duration,
            "batch_count": batch_count,
            "long_form": long_form,
            "has_lyrics": bool(lyrics),
            "lyrics_chars": len(lyrics),
            "style_tags": tags[:240],
        }

        if batch_count > 1:
            from engines.ace_step_engine import generate_song_batch
            self._worker = InferenceWorker(
                generate_song_batch,
                lyrics=lyrics,
                style_tags=tags,
                count=batch_count,
                duration=duration,
                cfg_scale=cfg,
                infer_steps=steps,
                long_form=long_form,
                job_kind="song_generation",
                job_label=f"Song Forge batch ({batch_count})",
                job_inputs=job_inputs,
            )
        else:
            from engines.ace_step_engine import generate_song
            self._worker = InferenceWorker(
                generate_song,
                lyrics=lyrics,
                style_tags=tags,
                duration=duration,
                seed=seed,
                cfg_scale=cfg,
                infer_steps=steps,
                long_form=long_form,
                job_kind="song_generation",
                job_label="Song Forge generation",
                job_inputs=job_inputs,
            )

        self._worker.progress.connect(self._on_progress)
        self._worker.step_info.connect(self._on_step)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
        if self._seed_explore_params:
            for cell in self._seed_explore_params:
                self._seed_explorer.set_cell_failed(
                    int(cell.get("row", 0)),
                    int(cell.get("col", 0)),
                    "Cancelled",
                )
            self._seed_explore_params = []
        self._reset_ui(clear_worker=False)
        self._batch_view.refresh_recoverable_jobs()
        self._status.setText("Cancelled")

    def _on_progress(self, pct: int):
        self._progress.setValue(pct)

    def _on_step(self, msg: str):
        self._status.setText(msg)

    def _on_finished(self, result: dict):
        self._reset_ui()
        self._batch_view.refresh_recoverable_jobs()

        if result.get("cancelled"):
            self._status.setText("Cancelled")
            return

        # Batch result
        if "results" in result:
            self._batch_view.set_results(result["results"])
            self._sub_tabs.setCurrentWidget(self._batch_view)
            count = result.get("count", 0)
            self._status.setText(f"Generated {count} variations")
            if self._toast:
                self._toast.show_toast(f"Batch complete: {count} variations", "success")

            # Load first result into main waveform
            if result["results"]:
                first = result["results"][0]
                self._load_output(first["audio_path"], first.get("seed", 0))
        else:
            # Single result
            self._load_output(result.get("audio_path", ""), result.get("seed", 0))
            gen_time = result.get("generation_time", 0)
            if result.get("mode") == "long_form":
                sections = len(result.get("sections", []))
                self._status.setText(
                    f"Generated stitched long-form song ({sections} sections) in {gen_time:.1f}s"
                )
            else:
                self._status.setText(f"Generated in {gen_time:.1f}s (seed: {result.get('seed', '?')})")
            if self._toast:
                self._toast.show_toast(f"Song generated in {gen_time:.1f}s", "success")

    def _on_error(self, error_msg: str):
        self._reset_ui()
        self._batch_view.refresh_recoverable_jobs()
        self._status.setText(f"Error: {error_msg[:100]}")
        self._status.setStyleSheet("color: #F38BA8; font-size: 11px;")
        if self._toast:
            self._toast.show_toast(f"Generation failed: {error_msg[:80]}", "error")

    def _on_cancelled(self):
        self._worker = None
        self._batch_view.refresh_recoverable_jobs()

    def _reset_ui(self, clear_worker: bool = True):
        self._is_generating = False
        self._generate_btn.show()
        self._cancel_btn.hide()
        self._progress.hide()
        if clear_worker:
            self._worker = None
        self._status.setStyleSheet("color: #6C7086; font-size: 11px;")

    def _load_output(self, audio_path: str, seed: int = 0):
        """Load generated audio into waveform display."""
        if not audio_path:
            return
        self._current_audio_path = audio_path
        try:
            self._waveform.load_file(audio_path)
        except Exception:
            pass

        self._play_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._to_vocals_btn.setEnabled(True)

        import os
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        self._output_info.setText(f"seed: {seed} | {size_mb:.1f} MB")

    # ── Playback ──────────────────────────────────────────────────────────────

    def _on_play(self):
        if self._current_audio_path:
            self._play_audio(self._current_audio_path)

    def _play_audio(self, path: str):
        try:
            engine = AudioEngine()
            engine.load_file(path)
            engine.play()
        except Exception as e:
            if self._toast:
                self._toast.show_toast(f"Playback error: {e}", "error")

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export(self):
        if not self._current_audio_path:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audio", "",
            "WAV (*.wav);;FLAC (*.flac);;MP3 (*.mp3);;OGG (*.ogg)",
        )
        if path:
            try:
                from core.audio_export import export_audio, get_export_license_warnings, ExportSettings
                fmt = path.rsplit(".", 1)[-1].lower()
                settings = ExportSettings(format=fmt)
                license_warnings = get_export_license_warnings(self._current_audio_path)
                export_audio(self._current_audio_path, path, settings)
                if self._toast:
                    if license_warnings:
                        self._toast.show_toast(
                            f"Exported with license warning: {license_warnings[0]}",
                            "warning",
                        )
                    else:
                        self._toast.show_toast(f"Exported to {path}", "success")
            except Exception as e:
                if self._toast:
                    self._toast.show_toast(f"Export failed: {e}", "error")

    # ── Batch Result ──────────────────────────────────────────────────────────

    def _use_batch_result(self, audio_path: str):
        self._load_output(audio_path)
        if self._toast:
            self._toast.show_toast("Result loaded as primary output", "info")

    # ── Seed Explorer ─────────────────────────────────────────────────────────

    def _on_seed_explore(self, params_list: list[dict]):
        """Handle seed explorer grid generation request."""
        if self._is_generating:
            if self._toast:
                self._toast.show_toast("Wait for the current generation to finish", "warning")
            return

        lyrics = self._get_lyrics()
        tags = self._get_tags()
        if not lyrics and not tags:
            for cell in params_list:
                self._seed_explorer.set_cell_failed(
                    int(cell.get("row", 0)),
                    int(cell.get("col", 0)),
                    "Missing prompt",
                )
            if self._toast:
                self._toast.show_toast("Add lyrics or style tags before exploring seeds", "warning")
            return

        duration = self._duration_spin.value() if self._mode_tabs.currentIndex() == 1 else 60.0
        steps = self._steps_spin.value() if self._mode_tabs.currentIndex() == 1 else 60
        long_form = (
            self._mode_tabs.currentIndex() == 1
            and duration > 120
            and self._long_form_check.isChecked()
        )

        self._seed_explore_params = params_list
        self._is_generating = True
        self._generate_btn.hide()
        self._cancel_btn.show()
        self._progress.setValue(0)
        self._progress.show()
        self._sub_tabs.setCurrentWidget(self._seed_explorer)
        self._status.setText("Starting seed exploration...")

        from engines.ace_step_engine import generate_seed_grid
        self._worker = InferenceWorker(
            generate_seed_grid,
            lyrics=lyrics,
            style_tags=tags,
            params_list=params_list,
            duration=duration,
            infer_steps=steps,
            long_form=long_form,
            job_kind="song_generation",
            job_label=f"Seed Explorer grid ({len(params_list)} cells)",
            job_inputs={
                "duration": duration,
                "cell_count": len(params_list),
                "long_form": long_form,
                "has_lyrics": bool(lyrics),
                "lyrics_chars": len(lyrics),
                "style_tags": tags[:240],
            },
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.step_info.connect(self._on_step)
        self._worker.finished.connect(self._on_seed_finished)
        self._worker.error.connect(self._on_seed_error)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

        if self._toast:
            count = len(params_list)
            self._toast.show_toast(
                f"Seed exploration started: {count} cells", "info"
            )

    def _on_seed_finished(self, result: dict):
        """Handle completed seed explorer generation."""
        self._reset_ui()
        self._batch_view.refresh_recoverable_jobs()
        self._seed_explore_params = []

        if result.get("cancelled"):
            self._status.setText("Seed exploration cancelled")
            return

        results = result.get("results", [])
        successes = []
        failures = 0
        for item in results:
            row = int(item.get("row", 0))
            col = int(item.get("col", 0))
            if item.get("error"):
                failures += 1
                self._seed_explorer.set_cell_failed(row, col, item.get("error", "Failed"))
                continue

            audio_path = item.get("audio_path", "")
            seed = int(item.get("seed", 0))
            self._seed_explorer.set_cell_result(row, col, audio_path, seed)
            successes.append(item)

        if successes:
            first = successes[0]
            self._load_output(first.get("audio_path", ""), int(first.get("seed", 0)))

        self._status.setText(
            f"Seed exploration complete: {len(successes)} generated, {failures} failed"
        )
        if self._toast:
            self._toast.show_toast(
                f"Seed exploration complete: {len(successes)} generated",
                "success" if successes else "warning",
            )

    def _on_seed_error(self, error_msg: str):
        """Handle fatal seed explorer worker errors."""
        self._reset_ui()
        self._batch_view.refresh_recoverable_jobs()
        for cell in self._seed_explore_params:
            self._seed_explorer.set_cell_failed(
                int(cell.get("row", 0)),
                int(cell.get("col", 0)),
                error_msg,
            )
        self._seed_explore_params = []
        self._status.setText(f"Seed exploration error: {error_msg[:100]}")
        self._status.setStyleSheet("color: #F38BA8; font-size: 11px;")
        if self._toast:
            self._toast.show_toast(f"Seed exploration failed: {error_msg[:80]}", "error")

    # ── Reference Panel ───────────────────────────────────────────────────────

    def _on_reference_match(self, analysis: dict):
        """Auto-populate parameters from reference track analysis."""
        tags = analysis.get("suggested_tags", [])
        tempo_tag = analysis.get("suggested_tempo_tag", "")
        if tempo_tag:
            tags.append(tempo_tag)

        tag_str = ", ".join(tags)
        self._quick_tags.setText(tag_str)
        self._tag_browser.set_tags(tag_str)

        # Set duration to match reference
        duration = analysis.get("duration", 60.0)
        self._duration_spin.setValue(min(300, duration))

        # Overlay energy curve on mood editor
        energy = analysis.get("energy_curve", [])
        if energy:
            self._mood_curve.set_reference_curve(energy)

        if self._toast:
            self._toast.show_toast("Reference analysis applied to parameters", "success")

    def _on_reference_tags(self, tag_str: str):
        """Just use the extracted tags without full match."""
        self._quick_tags.setText(tag_str)
        self._tag_browser.set_tags(tag_str)
        if self._toast:
            self._toast.show_toast("Reference tags applied", "info")
