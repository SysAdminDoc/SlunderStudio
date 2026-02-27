"""
Slunder Studio v0.0.2 — SFX Generator View
Text-to-SFX generation with preset categories, batch generation,
waveform preview, and drag-to-mixer support.
"""
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QScrollArea,
    QGridLayout, QSlider, QLineEdit, QFileDialog,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine
from ui.waveform_widget import WaveformWidget, MiniWaveform
from engines.sfx_engine import SFXParams, SFXResult, SFX_CATEGORIES


# ── SFX Card ───────────────────────────────────────────────────────────────────

class SFXCard(QFrame):
    """Card for a generated SFX result."""

    play_requested = Signal(object)    # SFXResult
    use_requested = Signal(object)     # SFXResult
    delete_requested = Signal(object)  # self

    def __init__(self, result: SFXResult, parent=None):
        super().__init__(parent)
        self.result = result

        t = ThemeEngine.get_colors()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            SFXCard {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
            SFXCard:hover {{
                border-color: {t['accent']};
            }}
        """)
        self.setFixedHeight(80)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Mini waveform
        self._waveform = MiniWaveform()
        if result.audio is not None:
            import numpy as np
            mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
            self._waveform.set_audio(mono, result.sample_rate)
        self._waveform.setFixedWidth(120)
        layout.addWidget(self._waveform)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)
        seed_label = QLabel(f"Seed: {result.seed}")
        seed_label.setStyleSheet(f"color: {t['text']}; font-size: 10px; font-weight: bold;")
        dur_label = QLabel(f"{result.duration:.1f}s | {result.generation_time:.1f}s gen")
        dur_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
        info.addWidget(seed_label)
        info.addWidget(dur_label)
        info.addStretch()
        layout.addLayout(info, 1)

        # Buttons
        btn_col = QVBoxLayout()
        btn_col.setSpacing(3)

        btn_style = f"""
            QPushButton {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 9px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        play_btn = QPushButton("Play")
        play_btn.setStyleSheet(btn_style)
        play_btn.clicked.connect(lambda: self.play_requested.emit(self.result))

        use_btn = QPushButton("Use")
        use_btn.setStyleSheet(btn_style.replace(t['background'], '#238636').replace(t['text'], 'white'))
        use_btn.clicked.connect(lambda: self.use_requested.emit(self.result))

        del_btn = QPushButton("X")
        del_btn.setFixedSize(20, 20)
        del_btn.setStyleSheet(btn_style)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self))

        btn_col.addWidget(play_btn)
        btn_col.addWidget(use_btn)
        btn_col.addWidget(del_btn)
        layout.addLayout(btn_col)


# ── SFX View ───────────────────────────────────────────────────────────────────

class SFXView(QWidget):
    """SFX Generator page."""

    send_to_mixer = Signal(str)  # audio file path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[SFXResult] = []
        self._cards: list[SFXCard] = []

        t = ThemeEngine.get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Left: Controls ─────────────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ctrl_layout = QVBoxLayout(ctrl_frame)
        ctrl_layout.setContentsMargins(12, 10, 12, 10)
        ctrl_layout.setSpacing(6)

        title = QLabel("Text-to-SFX")
        title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 13px; border: none;")
        ctrl_layout.addWidget(title)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText("Describe the sound effect...\ne.g. 'rain falling on a tin roof'")
        self._prompt.setMaximumHeight(60)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 12px;
            }}
        """)
        ctrl_layout.addWidget(self._prompt)

        # Negative prompt
        self._neg_prompt = QLineEdit()
        self._neg_prompt.setPlaceholderText("Negative prompt (optional)")
        self._neg_prompt.setStyleSheet(f"""
            QLineEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 8px; font-size: 11px;
            }}
        """)
        ctrl_layout.addWidget(self._neg_prompt)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 11px; border: none; }}
        """

        # Category presets
        cat_row = QHBoxLayout()
        cl = QLabel("Category:")
        cl.setFixedWidth(60)
        cl.setStyleSheet(param_style)
        self._category = QComboBox()
        self._category.addItem("Custom")
        self._category.addItems(SFX_CATEGORIES.keys())
        self._category.currentTextChanged.connect(self._on_category_changed)
        self._category.setStyleSheet(param_style)
        cat_row.addWidget(cl)
        cat_row.addWidget(self._category)
        ctrl_layout.addLayout(cat_row)

        # Preset prompts
        self._preset_combo = QComboBox()
        self._preset_combo.setStyleSheet(param_style)
        self._preset_combo.currentTextChanged.connect(self._on_preset_selected)
        self._preset_combo.setVisible(False)
        ctrl_layout.addWidget(self._preset_combo)

        # Duration + Steps
        row1 = QHBoxLayout()
        dl = QLabel("Duration:")
        dl.setFixedWidth(60)
        dl.setStyleSheet(param_style)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(0.5, 47.0)
        self._duration.setValue(5.0)
        self._duration.setSuffix("s")
        self._duration.setStyleSheet(param_style)

        sl = QLabel("Steps:")
        sl.setFixedWidth(38)
        sl.setStyleSheet(param_style)
        self._steps = QSpinBox()
        self._steps.setRange(20, 200)
        self._steps.setValue(100)
        self._steps.setStyleSheet(param_style)

        row1.addWidget(dl)
        row1.addWidget(self._duration)
        row1.addWidget(sl)
        row1.addWidget(self._steps)
        ctrl_layout.addLayout(row1)

        # CFG + Batch
        row2 = QHBoxLayout()
        cfl = QLabel("CFG:")
        cfl.setFixedWidth(60)
        cfl.setStyleSheet(param_style)
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 20.0)
        self._cfg.setValue(7.0)
        self._cfg.setSingleStep(0.5)
        self._cfg.setStyleSheet(param_style)

        bl = QLabel("Batch:")
        bl.setFixedWidth(38)
        bl.setStyleSheet(param_style)
        self._batch = QSpinBox()
        self._batch.setRange(1, 8)
        self._batch.setValue(1)
        self._batch.setStyleSheet(param_style)

        row2.addWidget(cfl)
        row2.addWidget(self._cfg)
        row2.addWidget(bl)
        row2.addWidget(self._batch)
        ctrl_layout.addLayout(row2)

        # Generate button
        self._gen_btn = QPushButton("Generate SFX")
        self._gen_btn.setFixedHeight(36)
        self._gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #fab387, stop:1 #f9e2af);
                color: #1e1e2e; border: none; border-radius: 6px;
                font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._gen_btn.clicked.connect(self._on_generate)
        ctrl_layout.addWidget(self._gen_btn)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px; border: none;")
        ctrl_layout.addWidget(self._status)

        left.addWidget(ctrl_frame)

        # Main output waveform
        self._main_waveform = WaveformWidget()
        left.addWidget(self._main_waveform, 1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(340)
        layout.addWidget(left_w)

        # ── Right: Results Grid ────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        results_header = QHBoxLayout()
        rl = QLabel("Generated SFX")
        rl.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px;")

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface']}; color: {t['text_secondary']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 10px; font-size: 10px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """)
        self._clear_btn.clicked.connect(self._clear_results)

        results_header.addWidget(rl)
        results_header.addStretch()
        results_header.addWidget(self._clear_btn)
        right.addLayout(results_header)

        # Scrollable results
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._results_container = QWidget()
        self._results_layout = QVBoxLayout(self._results_container)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(4)
        self._results_layout.addStretch()

        self._scroll.setWidget(self._results_container)
        right.addWidget(self._scroll, 1)

        right_w = QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, 1)

    # ── Events ─────────────────────────────────────────────────────────────────

    def _on_category_changed(self, category: str):
        self._preset_combo.clear()
        if category in SFX_CATEGORIES:
            self._preset_combo.addItems(SFX_CATEGORIES[category])
            self._preset_combo.setVisible(True)
        else:
            self._preset_combo.setVisible(False)

    def _on_preset_selected(self, text: str):
        if text:
            self._prompt.setPlainText(text)

    def _on_generate(self):
        from engines.sfx_engine import generate_sfx

        params = SFXParams(
            prompt=self._prompt.toPlainText().strip(),
            negative_prompt=self._neg_prompt.text().strip(),
            duration=self._duration.value(),
            cfg_scale=self._cfg.value(),
            steps=self._steps.value(),
            batch_size=self._batch.value(),
        )

        if not params.prompt:
            self._status.setText("Enter a prompt first")
            return

        self._gen_btn.setEnabled(False)
        self._status.setText("Generating...")

        try:
            batch_count = params.batch_size
            for i in range(batch_count):
                p = SFXParams(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    duration=params.duration,
                    cfg_scale=params.cfg_scale,
                    steps=params.steps,
                    seed=None,
                )
                result = generate_sfx(p)

                if result.error:
                    self._status.setText(f"Error: {result.error}")
                    continue

                self._results.append(result)
                self._add_result_card(result)

                # Show first result in main waveform
                if i == 0 and result.audio is not None:
                    import numpy as np
                    mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
                    self._main_waveform.load_audio(mono, result.sample_rate)

            self._status.setText(f"Generated {batch_count} SFX ({params.duration:.1f}s each)")

        except Exception as e:
            self._status.setText(f"Error: {e}")
        finally:
            self._gen_btn.setEnabled(True)

    def _add_result_card(self, result: SFXResult):
        card = SFXCard(result)
        card.use_requested.connect(self._on_use_sfx)
        card.delete_requested.connect(self._on_delete_card)
        self._cards.append(card)
        self._results_layout.insertWidget(self._results_layout.count() - 1, card)

    def _on_use_sfx(self, result: SFXResult):
        if result.file_path:
            self.send_to_mixer.emit(result.file_path)

    def _on_delete_card(self, card: SFXCard):
        if card in self._cards:
            self._cards.remove(card)
        if card.result in self._results:
            self._results.remove(card.result)
        self._results_layout.removeWidget(card)
        card.deleteLater()

    def _clear_results(self):
        for card in self._cards:
            self._results_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._results.clear()
