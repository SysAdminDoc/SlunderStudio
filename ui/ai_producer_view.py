"""
Slunder Studio — AI Producer View
One-prompt-to-full-song interface with creative brief input,
live pipeline stage visualization, and final output preview.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QFrame, QCheckBox, QProgressBar,
)
from PySide6.QtCore import Qt

from ui.theme import Palette, ThemeEngine
from ui.accessibility import install_accessibility
from ui.waveform_widget import WaveformWidget
from engines.ai_producer import (
    ProducerBrief, ProducerResult, PipelineStage, PIPELINE_ORDER,
    GENRE_DEFAULTS, MOOD_TAGS, produce_song,
)
from core.mastering import PRESETS
from core.workers import InferenceWorker
from core.engine_contract import (
    ArtifactKind,
    CAP_PRODUCER_RUN,
    EngineArtifact,
    EngineRunResult,
    adapt_engine_result,
)
from core.model_manager import ModelManager


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
        self.setFixedHeight(34)
        self._base_style = f"""
            StageIndicator {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
        """
        self.setStyleSheet(self._base_style)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 3, 7, 3)
        layout.setSpacing(7)

        # Step number
        num = STAGE_ICONS.get(stage, "??")
        self._num_label = QLabel(num)
        self._num_label.setFixedSize(22, 22)
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

        if status == "pending":
            self.setStyleSheet(self._base_style)
            self._num_label.setStyleSheet(f"""
                background: {t['border']};
                color: {t['text_secondary']};
                border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._name_label.setStyleSheet(
                f"color: {t['text_secondary']}; font-size: 11px;"
            )
            self._status_label.clear()
            self._status_label.setStyleSheet(
                f"color: {t['text_secondary']}; font-size: 10px;"
            )
        elif status == "running":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['accent']}15;
                    border: 1px solid {t['accent']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['accent']};
                color: {t['background']}; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 11px; font-weight: bold;")
            self._status_label.setText("Running...")
            self._status_label.setStyleSheet(f"color: {t['accent']}; font-size: 10px;")

        elif status == "complete":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['surface']};
                    border: 1px solid {t['success']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['success']};
                color: {t['background']}; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 11px;")
            dur_str = f"{duration:.1f}s" if duration > 0 else ""
            self._status_label.setText(dur_str)
            self._status_label.setStyleSheet(
                f"color: {t['success']}; font-size: 10px;"
            )

        elif status == "skipped":
            self._name_label.setStyleSheet(f"color: {t['muted']}; font-size: 11px;")
            self._status_label.setText("Skipped")
            self._status_label.setStyleSheet(f"color: {t['muted']}; font-size: 10px;")

        elif status == "failed":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['surface']};
                    border: 1px solid {t['error']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['error']};
                color: {t['background']}; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._status_label.setText("Failed")
            self._status_label.setStyleSheet(f"color: {t['error']}; font-size: 10px;")
        elif status == "cancelled":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {t['surface']};
                    border: 1px solid {t['warning']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['warning']};
                color: {t['background']}; border-radius: 12px;
                font-size: 10px; font-weight: bold;
            """)
            self._status_label.setText("Cancelled")
            self._status_label.setStyleSheet(
                f"color: {t['warning']}; font-size: 10px;"
            )


# ── AI Producer View ───────────────────────────────────────────────────────────

class AIProducerView(QWidget):
    """AI Producer page — one prompt to full song."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[ProducerResult] = None
        self._contract_result: Optional[EngineRunResult] = None
        self._worker: Optional[InferenceWorker] = None
        self._last_job_id = ""
        self._stage_indicators: dict[PipelineStage, StageIndicator] = {}
        self._model_mgr = ModelManager()

        t = ThemeEngine.get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Left: Creative Brief ───────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        # Creative brief header
        title_frame = QFrame()
        title_frame.setStyleSheet(f"""
            QFrame {{
                background: {t['surface']};
                border: none;
                border-bottom: 1px solid {t['border']};
                border-radius: 0;
            }}
        """)
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(16, 12, 16, 12)
        title_label = QLabel("Production brief")
        title_label.setStyleSheet(f"color: {t['text']}; font-size: 16px; font-weight: bold; border: none;")
        subtitle = QLabel("Describe the song, then review each local pipeline stage.")
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
        self._prompt.setMinimumHeight(56)
        self._prompt.setMaximumHeight(80)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px; font-size: 13px;
            }}
        """)
        self._prompt.textChanged.connect(self._refresh_capability_state)
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

        row4 = QHBoxLayout()
        self._demo_fallback_check = QCheckBox("Demo Fallback")
        self._demo_fallback_check.setChecked(False)
        self._demo_fallback_check.setToolTip(
            "When enabled, a silent placeholder is used if song generation fails. "
            "When disabled, the pipeline stops on failure."
        )
        self._demo_fallback_check.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        self._demo_fallback_check.toggled.connect(self._refresh_capability_state)
        row4.addWidget(self._demo_fallback_check)
        row4.addStretch()
        ctrl.addLayout(row4)

        # PRODUCE button
        self._produce_btn = QPushButton("PRODUCE")
        self._produce_btn.setObjectName("primaryAction")
        self._produce_btn.setFixedHeight(44)
        self._produce_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']};
                color: {t['background']}; border: none; border-radius: 5px;
                font-weight: bold; font-size: 14px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._produce_btn.clicked.connect(self._on_produce)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(44)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface_hover']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 5px;
                font-weight: bold; padding: 0 12px;
            }}
            QPushButton:hover {{ border-color: {t['warning']}; }}
            QPushButton:disabled {{ color: {t['muted']}; }}
        """)
        self._cancel_btn.clicked.connect(self._on_cancel)

        self._retry_btn = QPushButton("Retry")
        self._retry_btn.setFixedHeight(44)
        self._retry_btn.setEnabled(False)
        self._retry_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface_hover']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 5px;
                font-weight: bold; padding: 0 12px;
            }}
            QPushButton:hover {{ border-color: {t['accent']}; }}
            QPushButton:disabled {{ color: {t['muted']}; }}
        """)
        self._retry_btn.clicked.connect(self._on_produce)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.addWidget(self._produce_btn, 1)
        action_row.addWidget(self._cancel_btn)
        action_row.addWidget(self._retry_btn)
        ctrl.addLayout(action_row)

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
                background: {t['accent']};
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
                background: {t['success']}; color: {t['background']}; border: none;
                border-radius: 6px; padding: 8px 16px;
                font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ border: 1px solid {t['text']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._export_btn.clicked.connect(self._on_export)
        right.addWidget(self._export_btn)

        right_w = QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, 1)
        self._model_mgr.status_changed.connect(self._on_model_status_changed)
        self._refresh_capability_state()
        install_accessibility(
            self,
            "AI Producer",
            named_controls=[
                (self._prompt, "Producer brief", "Describes the song for the AI producer pipeline."),
                (self._genre, "Producer genre", "Selects the song genre."),
                (self._mood, "Producer mood", "Selects the song mood."),
                (self._duration, "Producer duration", "Sets the target song duration in seconds."),
                (self._vocals, "Producer vocals", "Selects the vocal arrangement."),
                (self._master_preset, "Producer mastering preset", "Selects the mastering preset for the final song."),
                (self._sfx_check, "Add producer sound effects", "Includes sound effects in the generated production."),
                (self._demo_fallback_check, "Enable producer demo fallback", "Allows a declared demo fallback when production cannot complete."),
                (self._produce_btn, "Produce song", "Runs the complete AI producer pipeline."),
                (self._cancel_btn, "Cancel production", "Cancels the running producer pipeline."),
                (self._retry_btn, "Retry production", "Retries the producer pipeline."),
                (self._lyrics_preview, "Generated lyrics preview", "Shows the lyrics produced by the pipeline."),
                (self._export_btn, "Export produced song", "Exports the verified producer result."),
            ],
        )

    # ── Production ─────────────────────────────────────────────────────────────

    def _on_produce(self):
        if self._worker and self._worker.isRunning():
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._output_info.setText("Enter a prompt to begin")
            return
        readiness = self._model_mgr.get_capability_readiness(
            CAP_PRODUCER_RUN,
            allow_demo=self._demo_fallback_check.isChecked(),
        )
        if not readiness.can_run:
            self._output_info.setText(readiness.remedy)
            self._refresh_capability_state()
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
            demo_fallback=self._demo_fallback_check.isChecked(),
        )

        self._reset_for_run()
        self._produce_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._retry_btn.setEnabled(False)
        self._output_title.setText("Production in progress")
        self._output_info.setText("Planning...")

        prior_job_id = self._last_job_id
        self._worker = InferenceWorker(
            produce_song,
            brief,
            job_kind="ai_producer",
            job_label="AI Producer pipeline",
            job_inputs={
                "duration_seconds": brief.duration_seconds,
                "genre": brief.genre or "auto",
                "mood": brief.mood or "auto",
                "vocal_style": brief.vocal_style,
                "include_sfx": brief.include_sfx,
                "demo_fallback": brief.demo_fallback,
                "prompt_chars": len(brief.prompt),
            },
            job_metadata={
                "module": "ai_producer",
                **({"retry_of": prior_job_id} if prior_job_id else {}),
            },
        )
        self._last_job_id = self._worker.job_id
        self._worker.progress.connect(self._on_progress)
        self._worker.step_info.connect(self._on_step)
        self._worker.finished.connect(self._on_produce_finished)
        self._worker.error.connect(self._on_produce_error)
        self._worker.cancelled.connect(self._on_produce_cancelled)
        self._worker.start()

    def _reset_for_run(self):
        """Clear every routable artifact before a new run can start."""
        self._result = None
        self._contract_result = None
        self._export_btn.setEnabled(False)
        self._export_btn.setText("Export Final Song")
        self._waveform.clear()
        for indicator in self._stage_indicators.values():
            indicator.set_status("pending")
        self._progress.setValue(0)
        self._lyrics_preview.clear()

    def _on_produce_finished(self, result):
        self._result = result
        artifacts = []
        if result.can_export:
            artifacts.append(EngineArtifact(
                kind=ArtifactKind.AUDIO,
                path=result.final_audio_path,
                routable=True,
            ))
        self._contract_result = adapt_engine_result(
            CAP_PRODUCER_RUN,
            result,
            artifacts,
            model_id="ace-step-v1.5",
        )
        self._display_result(result)
        self._finish_worker_ui()

    def _on_produce_error(self, error_msg):
        self._result = None
        self._contract_result = EngineRunResult.failure(
            CAP_PRODUCER_RUN,
            error_msg,
            model_id="ace-step-v1.5",
        )
        self._output_title.setText("Production failed")
        self._output_info.setText(f"Error: {error_msg}")
        self._export_btn.setEnabled(False)
        self._retry_btn.setEnabled(True)
        self._finish_worker_ui(keep_retry=True)

    def _on_produce_cancelled(self):
        self._result = None
        self._contract_result = EngineRunResult.cancelled(
            CAP_PRODUCER_RUN,
            "Cancellation completed; partial artifacts were removed.",
            model_id="ace-step-v1.5",
        )
        for indicator in self._stage_indicators.values():
            if indicator._status == "running":
                indicator.set_status("cancelled")
        self._output_title.setText("Production cancelled")
        self._output_info.setText(
            "Cancellation completed. Partial artifacts were removed; Retry starts a new job."
        )
        self._export_btn.setEnabled(False)
        self._retry_btn.setEnabled(True)
        self._finish_worker_ui(keep_retry=True)

    def _on_cancel(self):
        if not self._worker or not self._worker.isRunning():
            return
        self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._output_info.setText("Cancellation requested; finishing the active stage safely...")

    def _on_progress(self, progress: int):
        """Update persisted overall progress from the worker."""
        self._progress.setValue(max(0, min(100, int(progress))))

    def _on_step(self, message: str):
        """Keep the current detailed stage message and indicator visible."""
        self._output_info.setText(message)
        normalized = message.casefold()
        for stage, indicator in self._stage_indicators.items():
            label = stage.value.replace("_", " ").casefold()
            if not normalized.startswith(label):
                continue
            if ": failed" in normalized:
                indicator.set_status("failed")
            elif ": cancelled" in normalized:
                indicator.set_status("cancelled")
            elif ": skipped" in normalized:
                indicator.set_status("skipped")
            elif ": complete" in normalized:
                indicator.set_status("complete")
            else:
                indicator.set_status("running")
            break

    def _finish_worker_ui(self, *, keep_retry: bool = False):
        self._cancel_btn.setEnabled(False)
        if not keep_retry and self._result:
            self._retry_btn.setEnabled(
                not self._result.is_success
                or self._result.is_demo
                or self._result.is_degraded
            )
        self._worker = None
        self._refresh_capability_state()

    def _on_model_status_changed(self, model_id: str, _status: str):
        if model_id == "ace-step-v1.5":
            self._refresh_capability_state()

    def _refresh_capability_state(self):
        if not hasattr(self, "_produce_btn"):
            return
        readiness = self._model_mgr.get_capability_readiness(
            CAP_PRODUCER_RUN,
            allow_demo=self._demo_fallback_check.isChecked(),
        )
        idle = not (self._worker and self._worker.isRunning())
        has_prompt = bool(self._prompt.toPlainText().strip())
        self._produce_btn.setEnabled(idle and readiness.can_run and has_prompt)
        self._produce_btn.setToolTip(
            (
                f"Produces declared outputs: {readiness.output_summary}."
                if readiness.can_run and has_prompt
                else "Enter a production brief."
                if not has_prompt
                else readiness.remedy
            )
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
        if result.is_success:
            if result.is_demo:
                self._output_title.setText("Demo production complete")
            elif result.is_degraded:
                self._output_title.setText("Production complete with limitations")
            else:
                self._output_title.setText("Production complete")
            info_parts = [f"Total time: {result.total_time:.1f}s"]
            info_parts.append(f"Stages: {len(result.completed_stages)}/{len(PIPELINE_ORDER)}")
            info_parts.append(f"Output: {result.output_kind}")
            if result.style_tags:
                info_parts.append(f"Style: {', '.join(result.style_tags[:6])}")

            master_step = result.get_step(PipelineStage.MASTERING)
            if master_step and master_step.output_data:
                lufs = master_step.output_data.get("output_lufs", 0)
                info_parts.append(f"Loudness: {lufs:.1f} LUFS")
            if result.degraded_reasons:
                info_parts.append(
                    "Limitations: " + "; ".join(result.degraded_reasons[:3])
                )

            self._output_info.setText(" | ".join(info_parts))
        elif result.cancelled or result.stage == PipelineStage.CANCELLED:
            self._output_title.setText("Production cancelled")
            self._output_info.setText("Cancellation completed; no result is exportable.")
        else:
            self._output_title.setText("Production failed")
            stage_errors = [
                f"{STAGE_LABELS.get(step.stage, step.stage.value)}: {step.error}"
                for step in result.steps if step.error
            ]
            self._output_info.setText(
                " | ".join([result.error or "The pipeline did not complete", *stage_errors])
            )

        # Load waveform
        if result.can_export:
            self._waveform.load_audio(result.final_audio_path)
            self._export_btn.setEnabled(True)
            self._export_btn.setText(
                "Export Demo" if result.is_demo else "Export Final Song"
            )
        else:
            self._export_btn.setEnabled(False)

    def _on_export(self):
        if not self._result or not self._result.can_export:
            self._export_btn.setEnabled(False)
            self._output_info.setText(
                "The verified result is no longer available. Retry production before exporting."
            )
            return

        from PySide6.QtWidgets import QFileDialog
        import shutil

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Final Song", "song.wav", "WAV (*.wav)"
        )
        if path:
            shutil.copy2(self._result.final_audio_path, path)
            self._output_info.setText(f"Exported: {path}")
