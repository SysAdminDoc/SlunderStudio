"""
Slunder Studio v0.0.2 — AI Producer View
One-prompt-to-full-song interface with creative brief input,
live pipeline stage visualization, and final output preview.
"""
import os
import time
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QScrollArea,
    QLineEdit, QCheckBox, QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QTimer

from ui.theme import ThemeEngine
from ui.waveform_widget import WaveformWidget
from engines.ai_producer import (
    ProducerBrief, ProducerResult, PipelineStage, PIPELINE_ORDER,
    GENRE_DEFAULTS, MOOD_TAGS, produce_song,
)
from core.mastering import PRESETS


# ── Stage Card ─────────────────────────────────────────────────────────────────

STAGE_ICONS = {
    PipelineStage.PLANNING: "01",
    PipelineStage.LYRICS: "02",
    PipelineStage.STYLE: "03",
    PipelineStage.SONG_GEN: "04",
    PipelineStage.VOCALS: "05",
    PipelineStage.SFX: "06",
    PipelineStage.MIXING: "07",
    PipelineStage.MASTERING: "08",
}

STAGE_LABELS = {
    PipelineStage.PLANNING: "Planning",
    PipelineStage.LYRICS: "Lyrics",
    PipelineStage.STYLE: "Style Tags",
    PipelineStage.SONG_GEN: "Song Generation",
    PipelineStage.VOCALS: "Vocals",
    PipelineStage.SFX: "SFX Layer",
    PipelineStage.MIXING: "Mixing",
    PipelineStage.MASTERING: "Mastering",
}


class StageIndicator(QFrame):
    """Visual indicator for a pipeline stage."""

    def __init__(self, stage: PipelineStage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self._status = "pending"

        t = ThemeEngine.get_colors()
        self.setFixedHeight(40)
        self._base_style = f"""
            StageIndicator {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(self._base_style)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Step number
        num = STAGE_ICONS.get(stage, "??")
        self._num_label = QLabel(num)
        self._num_label.setFixedSize(24, 24)
        self._num_label.setAlignment(Qt.AlignCenter)
        self._num_label.setStyleSheet(f"""
            background: {t['border']};
            color: {t['text_secondary']};
            border-radius: 12px;
            font-size: 10px; font-weight: bold;
        """)
        layout.addWidget(self._num_label)

        # Stage name
        self._name_label = QLabel(STAGE_LABELS.get(stage, stage.value))
        self._name_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        layout.addWidget(self._name_label, 1)

        # Status indicator
        self._status_label = QLabel("")
        self._status_label.setFixedWidth(60)
        self._status_label.setAlignment(Qt.AlignRight)
        self._status_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px;")
        layout.addWidget(self._status_label)

    def set_status(self, status: str, duration: float = 0.0):
        """Update the stage status display."""
        self._status = status
        t = ThemeEngine.get_colors()

        if status == "running":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['accent']}15;
                    border: 1px solid {t['accent']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['accent']};
                color: white; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 11px; font-weight: bold;")
            self._status_label.setText("Running...")
            self._status_label.setStyleSheet(f"color: {t['accent']}; font-size: 10px;")

        elif status == "complete":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['surface']};
                    border: 1px solid #238636;
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: #238636;
                color: white; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 11px;")
            dur_str = f"{duration:.1f}s" if duration > 0 else ""
            self._status_label.setText(dur_str)
            self._status_label.setStyleSheet(f"color: #238636; font-size: 10px;")

        elif status == "skipped":
            self._name_label.setStyleSheet(f"color: {t['muted']}; font-size: 11px;")
            self._status_label.setText("Skipped")
            self._status_label.setStyleSheet(f"color: {t['muted']}; font-size: 10px;")

        elif status == "failed":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: #f38ba815;
                    border: 1px solid #f38ba8;
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: #f38ba8;
                color: white; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._status_label.setText("Failed")
            self._status_label.setStyleSheet("color: #f38ba8; font-size: 10px;")


# ── AI Producer View ───────────────────────────────────────────────────────────

class AIProducerView(QWidget):
    """AI Producer page — one prompt to full song."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[ProducerResult] = None
        self._stage_indicators: dict[PipelineStage, StageIndicator] = {}

        t = ThemeEngine.get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Left: Creative Brief ───────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        # Title
        title_frame = QFrame()
        title_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t['accent']}22, stop:1 #a371f722);
                border: 1px solid {t['accent']}44;
                border-radius: 10px;
            }}
        """)
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(16, 12, 16, 12)
        title_label = QLabel("AI Producer")
        title_label.setStyleSheet(f"color: {t['text']}; font-size: 18px; font-weight: bold; border: none;")
        subtitle = QLabel("One prompt to full song. Describe your vision.")
        subtitle.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px; border: none;")
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle)
        left.addWidget(title_frame)

        # Main prompt
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        ctrl = QVBoxLayout(ctrl_frame)
        ctrl.setContentsMargins(12, 10, 12, 10)
        ctrl.setSpacing(6)

        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText(
            "Describe your song...\n"
            "e.g. 'A dreamy lo-fi hip-hop track about rainy nights in Tokyo, "
            "with mellow piano chords and vinyl crackle'"
        )
        self._prompt.setMaximumHeight(80)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px; font-size: 13px;
            }}
        """)
        ctrl.addWidget(self._prompt)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 11px;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 11px; border: none; }}
        """

        # Genre + Mood
        row1 = QHBoxLayout()
        gl = QLabel("Genre:")
        gl.setFixedWidth(42)
        gl.setStyleSheet(param_style)
        self._genre = QComboBox()
        self._genre.addItem("Auto-detect")
        self._genre.addItems(sorted(GENRE_DEFAULTS.keys()))
        self._genre.setStyleSheet(param_style)

        ml = QLabel("Mood:")
        ml.setFixedWidth(36)
        ml.setStyleSheet(param_style)
        self._mood = QComboBox()
        self._mood.addItem("Auto-detect")
        self._mood.addItems(sorted(MOOD_TAGS.keys()))
        self._mood.setStyleSheet(param_style)

        row1.addWidget(gl)
        row1.addWidget(self._genre)
        row1.addWidget(ml)
        row1.addWidget(self._mood)
        ctrl.addLayout(row1)

        # Duration + Vocals
        row2 = QHBoxLayout()
        dl = QLabel("Length:")
        dl.setFixedWidth(42)
        dl.setStyleSheet(param_style)
        self._duration = QSpinBox()
        self._duration.setRange(30, 600)
        self._duration.setValue(180)
        self._duration.setSuffix("s")
        self._duration.setStyleSheet(param_style)

        vl = QLabel("Vocals:")
        vl.setFixedWidth(46)
        vl.setStyleSheet(param_style)
        self._vocals = QComboBox()
        self._vocals.addItems(["None", "Male", "Female"])
        self._vocals.setStyleSheet(param_style)

        row2.addWidget(dl)
        row2.addWidget(self._duration)
        row2.addWidget(vl)
        row2.addWidget(self._vocals)
        ctrl.addLayout(row2)

        # Mastering preset + SFX toggle
        row3 = QHBoxLayout()
        mpl = QLabel("Master:")
        mpl.setFixedWidth(42)
        mpl.setStyleSheet(param_style)
        self._master_preset = QComboBox()
        self._master_preset.addItems(PRESETS.keys())
        self._master_preset.setCurrentText("Balanced")
        self._master_preset.setStyleSheet(param_style)

        self._sfx_check = QCheckBox("Add SFX")
        self._sfx_check.setChecked(True)
        self._sfx_check.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")

        row3.addWidget(mpl)
        row3.addWidget(self._master_preset)
        row3.addWidget(self._sfx_check)
        ctrl.addLayout(row3)

        # PRODUCE button
        self._produce_btn = QPushButton("PRODUCE")
        self._produce_btn.setFixedHeight(44)
        self._produce_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t['accent']}, stop:0.5 #a371f7, stop:1 #f38ba8);
                color: white; border: none; border-radius: 8px;
                font-weight: bold; font-size: 16px;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{ opacity: 0.9; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._produce_btn.clicked.connect(self._on_produce)
        ctrl.addWidget(self._produce_btn)

        left.addWidget(ctrl_frame)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background: {t['border']};
                border: none; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {t['accent']}, stop:1 #a371f7);
                border-radius: 3px;
            }}
        """)
        left.addWidget(self._progress)

        # Pipeline stages
        stages_label = QLabel("Pipeline")
        stages_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px;")
        left.addWidget(stages_label)

        for stage in PIPELINE_ORDER:
            indicator = StageIndicator(stage)
            self._stage_indicators[stage] = indicator
            left.addWidget(indicator)

        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(380)
        layout.addWidget(left_w)

        # ── Right: Output ──────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        # Output info
        self._output_frame = QFrame()
        self._output_frame.setStyleSheet(f"""
            QFrame {{ background: {t['surface']}; border: 1px solid {t['border']};
                border-radius: 8px; }}
        """)
        out_layout = QVBoxLayout(self._output_frame)
        out_layout.setContentsMargins(12, 10, 12, 10)
        out_layout.setSpacing(4)

        self._output_title = QLabel("Output")
        self._output_title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 14px; border: none;")
        out_layout.addWidget(self._output_title)

        self._output_info = QLabel("Run the producer to generate a song")
        self._output_info.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px; border: none;")
        self._output_info.setWordWrap(True)
        out_layout.addWidget(self._output_info)

        right.addWidget(self._output_frame)

        # Waveform
        self._waveform = WaveformWidget()
        right.addWidget(self._waveform, 1)

        # Lyrics preview
        lyrics_label = QLabel("Generated Lyrics")
        lyrics_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px;")
        right.addWidget(lyrics_label)

        self._lyrics_preview = QTextEdit()
        self._lyrics_preview.setReadOnly(True)
        self._lyrics_preview.setMaximumHeight(120)
        self._lyrics_preview.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px; font-size: 11px;
            }}
        """)
        right.addWidget(self._lyrics_preview)

        # Export
        self._export_btn = QPushButton("Export Final Song")
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(f"""
            QPushButton {{
                background: #238636; color: white; border: none;
                border-radius: 6px; padding: 8px 16px;
                font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ background: #2ea043; }}
            QPushButton:disabled {{ background: {t['border']}; color: #555; }}
        """)
        self._export_btn.clicked.connect(self._on_export)
        right.addWidget(self._export_btn)

        right_w = QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, 1)

    # ── Production ─────────────────────────────────────────────────────────────

    def _on_produce(self):
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._output_info.setText("Enter a prompt to begin")
            return

        genre = self._genre.currentText()
        mood = self._mood.currentText()
        vocal_map = {"None": "none", "Male": "male", "Female": "female"}

        brief = ProducerBrief(
            prompt=prompt,
            genre="" if genre == "Auto-detect" else genre,
            mood="" if mood == "Auto-detect" else mood,
            duration_seconds=self._duration.value(),
            vocal_style=vocal_map.get(self._vocals.currentText(), "none"),
            include_sfx=self._sfx_check.isChecked(),
            mastering_preset=self._master_preset.currentText(),
        )

        # Reset indicators
        for indicator in self._stage_indicators.values():
            indicator.set_status("pending")
        self._progress.setValue(0)
        self._lyrics_preview.clear()

        self._produce_btn.setEnabled(False)
        self._output_info.setText("Producing...")

        try:
            result = produce_song(brief, self._on_progress)
            self._result = result
            self._display_result(result)
        except Exception as e:
            self._output_info.setText(f"Error: {e}")
        finally:
            self._produce_btn.setEnabled(True)

    def _on_progress(self, progress: float, message: str):
        """Update progress from pipeline."""
        self._progress.setValue(int(progress * 100))
        self._output_info.setText(message)

        # Update stage indicators
        if self._result:
            for step in self._result.steps:
                if step.stage in self._stage_indicators:
                    self._stage_indicators[step.stage].set_status(
                        step.status, step.duration
                    )

    def _display_result(self, result: ProducerResult):
        """Display pipeline results."""
        # Update all stage indicators
        for step in result.steps:
            if step.stage in self._stage_indicators:
                self._stage_indicators[step.stage].set_status(
                    step.status, step.duration
                )

        self._progress.setValue(100 if result.stage == PipelineStage.COMPLETE else
                                int(result.progress * 100))

        # Lyrics
        if result.lyrics_text:
            self._lyrics_preview.setPlainText(result.lyrics_text)

        # Output info
        if result.stage == PipelineStage.COMPLETE:
            self._output_title.setText("Production Complete")
            info_parts = [f"Total time: {result.total_time:.1f}s"]
            info_parts.append(f"Stages: {len(result.completed_stages)}/{len(PIPELINE_ORDER)}")
            if result.style_tags:
                info_parts.append(f"Style: {', '.join(result.style_tags[:6])}")

            master_step = result.get_step(PipelineStage.MASTERING)
            if master_step and master_step.output_data:
                lufs = master_step.output_data.get("output_lufs", 0)
                info_parts.append(f"Loudness: {lufs:.1f} LUFS")

            self._output_info.setText(" | ".join(info_parts))
        elif result.error:
            self._output_title.setText("Production Failed")
            self._output_info.setText(result.error)

        # Load waveform
        if result.final_audio_path and os.path.isfile(result.final_audio_path):
            try:
                import wave
                import numpy as np
                with wave.open(result.final_audio_path, "r") as wf:
                    frames = wf.readframes(wf.getnframes())
                    sr = wf.getframerate()
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if wf.getnchannels() == 2:
                        audio = audio.reshape(-1, 2)[:, 0]
                    self._waveform.load_audio(audio, sr)
            except Exception:
                pass

            self._export_btn.setEnabled(True)

    def _on_export(self):
        if not self._result or not self._result.final_audio_path:
            return

        from PySide6.QtWidgets import QFileDialog
        import shutil

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Final Song", "song.wav", "WAV (*.wav)"
        )
        if path:
            shutil.copy2(self._result.final_audio_path, path)
            self._output_info.setText(f"Exported: {path}")
