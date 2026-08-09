"""
Slunder Studio — AI Producer View
One-prompt-to-full-song interface with creative brief input,
live pipeline stage visualization, and final output preview.
"""
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QFrame, QCheckBox, QProgressBar, QFileDialog,
)
from PySide6.QtCore import Qt, QTimer

from ui.theme import Palette, ThemeEngine, rgba
from ui.accessibility import install_accessibility
from ui.widgets import ElidedLabel
from ui.waveform_widget import WaveformWidget
from engines.ai_producer import (
    ProducerBrief, ProducerResult, PipelineStage, PIPELINE_ORDER,
    GENRE_DEFAULTS, MOOD_TAGS, produce_song,
)
from core.mastering import PRESETS
from core.workers import InferenceWorker
from core.i18n import tr, user_facing_readiness
from ui.file_dialogs import ensure_extension, save_audio_file
from core.engine_contract import (
    ArtifactKind,
    CAP_PRODUCER_RUN,
    EngineArtifact,
    EngineRunResult,
    adapt_engine_result,
)
from core.model_manager import ModelManager
from core.song_generator_registry import active_song_generator_model_ids


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

STAGE_LABEL_KEYS = {
    PipelineStage.PLANNING: "producer.stages.planning",
    PipelineStage.LYRICS: "producer.stages.lyrics",
    PipelineStage.STYLE: "producer.stages.style",
    PipelineStage.SONG_GEN: "producer.stages.song_generation",
    PipelineStage.VOCALS: "producer.stages.vocals",
    PipelineStage.SFX: "producer.stages.sfx",
    PipelineStage.MIXING: "producer.stages.mixing",
    PipelineStage.MASTERING: "producer.stages.mastering",
}


def _stage_label(stage: PipelineStage) -> str:
    key = STAGE_LABEL_KEYS.get(stage)
    return tr(key) if key else stage.value


def _producer_export_task(
    source_path: str,
    output_path: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
) -> str:
    """Export a verified producer artifact without blocking the GUI."""
    from core.audio_export import ExportSettings, export_audio
    from core.workers import CancelledJobError

    fmt = os.path.splitext(output_path)[1].lower().lstrip(".") or "wav"
    try:
        return export_audio(
            source_path,
            output_path,
            ExportSettings(format=fmt),
            module="ai_producer",
            operation="producer_export",
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )
    except CancelledJobError:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


class StageIndicator(QFrame):
    """Visual indicator for a pipeline stage."""

    def __init__(self, stage: PipelineStage, parent=None):
        super().__init__(parent)
        self.stage = stage
        self._status = "pending"

        t = ThemeEngine.get_colors()
        self.setMinimumHeight(34)
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
        self._num_label.setMinimumSize(22, 22)
        self._num_label.setAlignment(Qt.AlignCenter)
        self._num_label.setStyleSheet(f"""
            background: {t['border']};
            color: {t['text_secondary']};
            border-radius: 12px;
            font-size: 7.5pt; font-weight: bold;
        """)
        layout.addWidget(self._num_label)

        # Stage name
        self._name_label = QLabel(_stage_label(stage))
        self._name_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(self._name_label, 1)

        # Status indicator
        self._status_label = ElidedLabel("", minimum_width=60)
        self._status_label.setAlignment(Qt.AlignRight)
        self._status_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt;")
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
                font-size: 7.5pt; font-weight: bold;
            """)
            self._name_label.setStyleSheet(
                f"color: {t['text_secondary']}; font-size: 8.25pt;"
            )
            self._status_label.clear()
            self._status_label.setStyleSheet(
                f"color: {t['text_secondary']}; font-size: 7.5pt;"
            )
        elif status == "running":
            self.setStyleSheet(f"""
                StageIndicator {{
                    background: {rgba(t['accent'], 21)};
                    border: 1px solid {t['accent']};
                    border-radius: 6px;
                }}
            """)
            self._num_label.setStyleSheet(f"""
                background: {t['accent']};
                color: {t['background']}; border-radius: 12px;
                font-size: 7.5pt; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt; font-weight: bold;")
            self._status_label.setText(tr("producer.stage_status.running"))
            self._status_label.setStyleSheet(f"color: {t['accent']}; font-size: 7.5pt;")

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
                font-size: 7.5pt; font-weight: bold;
            """)
            self._name_label.setStyleSheet(f"color: {t['text']}; font-size: 8.25pt;")
            dur_str = tr("producer.duration", seconds=duration) if duration > 0 else ""
            self._status_label.setText(dur_str)
            self._status_label.setStyleSheet(
                f"color: {t['success']}; font-size: 7.5pt;"
            )

        elif status == "skipped":
            self._name_label.setStyleSheet(f"color: {t['muted']}; font-size: 8.25pt;")
            self._status_label.setText(tr("producer.stage_status.skipped"))
            self._status_label.setStyleSheet(f"color: {t['muted']}; font-size: 7.5pt;")

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
                font-size: 7.5pt; font-weight: bold;
            """)
            self._status_label.setText(tr("producer.stage_status.failed"))
            self._status_label.setStyleSheet(f"color: {t['error']}; font-size: 7.5pt;")
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
                font-size: 7.5pt; font-weight: bold;
            """)
            self._status_label.setText(tr("producer.stage_status.cancelled"))
            self._status_label.setStyleSheet(
                f"color: {t['warning']}; font-size: 7.5pt;"
            )


# ── AI Producer View ───────────────────────────────────────────────────────────

class AIProducerView(QWidget):
    """AI Producer page — one prompt to full song."""

    def __init__(self, parent=None, toast_mgr=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._result: Optional[ProducerResult] = None
        self._contract_result: Optional[EngineRunResult] = None
        self._worker: Optional[InferenceWorker] = None
        self._export_worker: Optional[InferenceWorker] = None
        self._export_workers = set()
        self._last_job_id = ""
        active_model_ids = active_song_generator_model_ids()
        self._generator_model_id = active_model_ids[0] if active_model_ids else ""
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
        title_label = QLabel(tr("producer.title"))
        title_label.setStyleSheet(f"color: {t['text']}; font-size: 12pt; font-weight: bold; border: none;")
        subtitle = QLabel(tr("runtime.production_subtitle"))
        subtitle.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
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
            tr("producer.prompt_placeholder")
        )
        self._prompt.setMinimumHeight(56)
        self._prompt.setMaximumHeight(80)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px; font-size: 9.75pt;
            }}
        """)
        self._prompt.textChanged.connect(self._refresh_capability_state)
        ctrl.addWidget(self._prompt)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 8.25pt;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 8.25pt; border: none; }}
        """

        # Genre + Mood
        row1 = QHBoxLayout()
        gl = QLabel(tr("producer.genre_label"))
        gl.setMinimumWidth(42)
        gl.setStyleSheet(param_style)
        self._genre = QComboBox()
        self._genre.addItem(tr("producer.auto_detect"), "")
        for genre in sorted(GENRE_DEFAULTS.keys()):
            # Genre identifiers are engine taxonomy data; keep the raw value in UserRole.
            self._genre.addItem(genre, genre)
        self._genre.setStyleSheet(param_style)

        ml = QLabel(tr("producer.mood_label"))
        ml.setMinimumWidth(36)
        ml.setStyleSheet(param_style)
        self._mood = QComboBox()
        self._mood.addItem(tr("producer.auto_detect"), "")
        for mood in sorted(MOOD_TAGS.keys()):
            # Mood identifiers are engine taxonomy data; keep the raw value in UserRole.
            self._mood.addItem(mood, mood)
        self._mood.setStyleSheet(param_style)

        row1.addWidget(gl)
        row1.addWidget(self._genre)
        row1.addWidget(ml)
        row1.addWidget(self._mood)
        ctrl.addLayout(row1)

        # Duration + Vocals
        row2 = QHBoxLayout()
        dl = QLabel(tr("producer.length_label"))
        dl.setMinimumWidth(42)
        dl.setStyleSheet(param_style)
        self._duration = QSpinBox()
        self._duration.setRange(30, 600)
        self._duration.setValue(180)
        self._duration.setSuffix("s")
        self._duration.setStyleSheet(param_style)

        vl = QLabel(tr("producer.vocals_label"))
        vl.setMinimumWidth(46)
        vl.setStyleSheet(param_style)
        self._vocals = QComboBox()
        for value, key in (
            ("none", "producer.vocals_none"),
            ("male", "producer.vocals_male"),
            ("female", "producer.vocals_female"),
        ):
            self._vocals.addItem(tr(key), value)
        self._vocals.setStyleSheet(param_style)

        row2.addWidget(dl)
        row2.addWidget(self._duration)
        row2.addWidget(vl)
        row2.addWidget(self._vocals)
        ctrl.addLayout(row2)

        # Mastering preset + SFX toggle
        row3 = QHBoxLayout()
        mpl = QLabel(tr("producer.master_label"))
        mpl.setMinimumWidth(42)
        mpl.setStyleSheet(param_style)
        self._master_preset = QComboBox()
        self._preset_display_keys = {
            "Balanced": "producer.presets.balanced",
            "Loud / Radio": "producer.presets.loud_radio",
            "Warm / Analog": "producer.presets.warm_analog",
            "Bright / Crisp": "producer.presets.bright_crisp",
            "Hip-Hop / Trap": "producer.presets.hip_hop_trap",
            "Cinematic": "producer.presets.cinematic",
            "Lo-Fi": "producer.presets.lo_fi",
            "Streaming (Spotify)": "producer.presets.streaming_spotify",
        }
        for preset in PRESETS:
            self._master_preset.addItem(
                tr(self._preset_display_keys.get(preset, "producer.preset_unknown")),
                preset,
            )
        self._master_preset.setCurrentIndex(
            max(0, self._master_preset.findData("Balanced"))
        )
        self._master_preset.setStyleSheet(param_style)

        self._sfx_check = QCheckBox(tr("producer.add_sfx"))
        self._sfx_check.setChecked(True)
        self._sfx_check.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")

        row3.addWidget(mpl)
        row3.addWidget(self._master_preset)
        row3.addWidget(self._sfx_check)
        ctrl.addLayout(row3)

        row4 = QHBoxLayout()
        self._demo_fallback_check = QCheckBox(tr("producer.demo_fallback"))
        self._demo_fallback_check.setChecked(False)
        self._demo_fallback_check.setToolTip(
            tr("runtime.production_demo_fallback")
        )
        self._demo_fallback_check.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        self._demo_fallback_check.toggled.connect(self._refresh_capability_state)
        row4.addWidget(self._demo_fallback_check)
        row4.addStretch()
        ctrl.addLayout(row4)

        # PRODUCE button
        self._produce_btn = QPushButton(tr("producer.produce"))
        self._produce_btn.setObjectName("primaryAction")
        self._produce_btn.setMinimumHeight(44)
        self._produce_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']};
                color: {t['background']}; border: none; border-radius: 5px;
                font-weight: bold; font-size: 10.5pt;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._produce_btn.clicked.connect(self._on_produce)

        self._cancel_btn = QPushButton(tr("producer.cancel"))
        self._cancel_btn.setMinimumHeight(44)
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

        self._retry_btn = QPushButton(tr("producer.retry"))
        self._retry_btn.setMinimumHeight(44)
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
        self._progress.setMinimumHeight(6)
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

        # Production steps
        stages_label = QLabel(tr("runtime.production_steps"))
        stages_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        left.addWidget(stages_label)

        for stage in PIPELINE_ORDER:
            indicator = StageIndicator(stage)
            self._stage_indicators[stage] = indicator
            left.addWidget(indicator)

        left.addStretch()

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(380)
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

        self._output_title = QLabel(tr("producer.output_title"))
        self._output_title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 10.5pt; border: none;")
        out_layout.addWidget(self._output_title)

        self._output_info = QLabel(tr("producer.output_empty"))
        self._output_info.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt; border: none;")
        self._output_info.setWordWrap(True)
        out_layout.addWidget(self._output_info)

        right.addWidget(self._output_frame)

        # Waveform
        self._waveform = WaveformWidget()
        right.addWidget(self._waveform, 1)

        # Lyrics preview
        lyrics_label = QLabel(tr("producer.lyrics_title"))
        lyrics_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        right.addWidget(lyrics_label)

        self._lyrics_preview = QTextEdit()
        self._lyrics_preview.setReadOnly(True)
        self._lyrics_preview.setMaximumHeight(120)
        self._lyrics_preview.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px; font-size: 8.25pt;
            }}
        """)
        right.addWidget(self._lyrics_preview)

        # Export
        self._export_btn = QPushButton(tr("producer.export_final"))
        self._export_btn.setEnabled(False)
        self._export_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['success']}; color: {t['background']}; border: none;
                border-radius: 6px; padding: 8px 16px;
                font-weight: bold; font-size: 9pt;
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
            tr("producer.accessibility.name"),
            named_controls=[
                (self._prompt, tr("producer.accessibility.brief_name"), tr("runtime.production_brief_description")),
                (self._genre, tr("producer.accessibility.genre_name"), tr("producer.accessibility.genre_description")),
                (self._mood, tr("producer.accessibility.mood_name"), tr("producer.accessibility.mood_description")),
                (self._duration, tr("producer.accessibility.duration_name"), tr("producer.accessibility.duration_description")),
                (self._vocals, tr("producer.accessibility.vocals_name"), tr("producer.accessibility.vocals_description")),
                (self._master_preset, tr("producer.accessibility.master_name"), tr("producer.accessibility.master_description")),
                (self._sfx_check, tr("producer.accessibility.sfx_name"), tr("producer.accessibility.sfx_description")),
                (self._demo_fallback_check, tr("producer.accessibility.demo_name"), tr("runtime.production_demo_description")),
                (self._produce_btn, tr("producer.accessibility.produce_name"), tr("runtime.production_run_description")),
                (self._cancel_btn, tr("producer.accessibility.cancel_name"), tr("runtime.production_cancel_description")),
                (self._retry_btn, tr("producer.accessibility.retry_name"), tr("runtime.production_retry_description")),
                (self._lyrics_preview, tr("producer.accessibility.lyrics_name"), tr("runtime.production_lyrics_description")),
                (self._export_btn, tr("producer.accessibility.export_name"), tr("producer.accessibility.export_description")),
            ],
        )

    # ── Production ─────────────────────────────────────────────────────────────

    def _on_produce(self):
        if self._worker and self._worker.isRunning():
            return
        prompt = self._prompt.toPlainText().strip()
        if not prompt:
            self._output_info.setText(tr("runtime.enter_producer_brief"))
            return
        readiness = self._model_mgr.get_capability_readiness(
            CAP_PRODUCER_RUN,
            allow_demo=self._demo_fallback_check.isChecked(),
        )
        if not readiness.can_run:
            self._output_info.setText(self._readiness_message(readiness))
            self._refresh_capability_state()
            return
        self._generator_model_id = readiness.model_id or self._generator_model_id

        genre = str(self._genre.currentData() or "")
        mood = str(self._mood.currentData() or "")
        vocal_style = str(self._vocals.currentData() or "none")

        brief = ProducerBrief(
            prompt=prompt,
            genre=genre,
            mood=mood,
            duration_seconds=self._duration.value(),
            vocal_style=vocal_style,
            include_sfx=self._sfx_check.isChecked(),
            mastering_preset=str(self._master_preset.currentData() or "Balanced"),
            demo_fallback=self._demo_fallback_check.isChecked(),
            song_generator_model_id=self._generator_model_id,
        )

        self._reset_for_run()
        self._produce_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._retry_btn.setEnabled(False)
        self._output_title.setText(tr("producer.status.in_progress"))
        self._output_info.setText(tr("producer.status.planning"))

        prior_job_id = self._last_job_id
        self._worker = InferenceWorker(
            produce_song,
            brief,
            job_kind="ai_producer",
            job_label=tr("producer.jobs.production"),
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
        self._export_btn.setText(tr("producer.export_final"))
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
            model_id=result.song_model_id or self._generator_model_id,
        )
        self._display_result(result)
        self._finish_worker_ui()

    def _on_produce_error(self, error_msg):
        self._result = None
        self._contract_result = EngineRunResult.failure(
            CAP_PRODUCER_RUN,
            error_msg,
            model_id=self._generator_model_id,
        )
        message = tr("producer.status.failed", error=error_msg)
        self._output_title.setText(tr("producer.status.failed_title"))
        self._output_info.setText(message)
        if self.toast_mgr is not None:
            self.toast_mgr.error(message)
        self._export_btn.setEnabled(False)
        self._retry_btn.setEnabled(True)
        self._finish_worker_ui(keep_retry=True)

    def _on_produce_cancelled(self):
        self._result = None
        self._contract_result = EngineRunResult.cancelled(
            CAP_PRODUCER_RUN,
            "Production cancelled before a complete result was available.",
            model_id=self._generator_model_id,
        )
        for indicator in self._stage_indicators.values():
            if indicator._status == "running":
                indicator.set_status("cancelled")
        self._output_title.setText(tr("producer.status.cancelled_title"))
        self._output_info.setText(tr("runtime.partial_files_removed"))
        self._export_btn.setEnabled(False)
        self._retry_btn.setEnabled(True)
        self._finish_worker_ui(keep_retry=True)

    def _on_cancel(self):
        if not self._worker or not self._worker.isRunning():
            return
        self._worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._output_info.setText(tr("runtime.cancel_production"))

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
        if model_id in active_song_generator_model_ids():
            self._refresh_capability_state()

    def _readiness_message(self, readiness) -> str:
        info = self._model_mgr.get_model_info(readiness.model_id)
        return user_facing_readiness(
            readiness,
            model_name=info.name if info is not None else "",
        )

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
                tr("runtime.ready")
                if readiness.can_run and has_prompt
                else tr("runtime.enter_producer_brief")
                if not has_prompt
                else self._readiness_message(readiness)
            )
        )

    def _display_result(self, result: ProducerResult):
        """Display production results."""
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
                self._output_title.setText(tr("producer.status.demo_complete"))
            elif result.is_degraded:
                self._output_title.setText(tr("producer.status.complete_limited"))
            else:
                self._output_title.setText(tr("producer.status.complete"))
            info_parts = [tr("producer.info.total_time", seconds=result.total_time)]
            info_parts.append(
                tr(
                    "producer.info.steps",
                    completed=len(result.completed_stages),
                    total=len(PIPELINE_ORDER),
                )
            )
            info_parts.append(tr("producer.info.output", kind=result.output_kind))
            if result.style_tags:
                info_parts.append(
                    tr("producer.info.style", tags=", ".join(result.style_tags[:6]))
                )

            master_step = result.get_step(PipelineStage.MASTERING)
            if master_step and master_step.output_data:
                lufs = master_step.output_data.get("output_lufs", 0)
                info_parts.append(tr("producer.info.loudness", lufs=lufs))
            if result.degraded_reasons:
                info_parts.append(
                    tr(
                        "producer.info.limitations",
                        reasons="; ".join(result.degraded_reasons[:3]),
                    )
                )

            self._output_info.setText(tr("producer.info.separator").join(info_parts))
        elif result.cancelled or result.stage == PipelineStage.CANCELLED:
            self._output_title.setText(tr("producer.status.cancelled_title"))
            self._output_info.setText(tr("producer.status.cancelled_no_export"))
        else:
            self._output_title.setText(tr("producer.status.failed_title"))
            stage_errors = [
                tr("producer.info.stage_error", stage=_stage_label(step.stage), error=step.error)
                for step in result.steps if step.error
            ]
            self._output_info.setText(
                tr("producer.info.separator").join([
                    result.error or tr("producer.status.did_not_complete"),
                    *stage_errors,
                ])
            )

        # Load waveform
        if result.can_export:
            self._waveform.load_audio(result.final_audio_path)
            self._export_btn.setEnabled(True)
            self._export_btn.setText(
                tr("producer.export_demo" if result.is_demo else "producer.export_final")
            )
        else:
            self._export_btn.setEnabled(False)

    def _on_export(self):
        if self._export_worker is not None and self._export_worker.isRunning():
            self._export_worker.cancel()
            self._output_info.setText(tr("producer.status.cancelling_export"))
            return
        if not self._result or not self._result.can_export:
            self._export_btn.setEnabled(False)
            self._output_info.setText(
                tr("producer.status.result_unavailable")
            )
            return

        path, selected_filter = save_audio_file(
            self,
            tr("producer.export_final"),
            "song.wav",
            operation_kind="ai_producer_export",
            dialog=QFileDialog,
        )
        if path:
            path = ensure_extension(path, selected_filter)
            worker = InferenceWorker(
                _producer_export_task,
                self._result.final_audio_path,
                path,
            )
            worker.progress.connect(
                lambda pct: self._output_info.setText(
                    tr("producer.status.exporting", percent=pct)
                )
            )
            worker.finished.connect(self._on_export_finished)
            worker.error.connect(self._on_export_error)
            worker.cancelled.connect(self._on_export_cancelled)
            self._export_workers.add(worker)
            self._export_worker = worker
            self._export_btn.setEnabled(False)
            self._export_btn.setText(tr("producer.cancel_export"))
            self._output_info.setText(tr("producer.status.exporting", percent=0))
            worker.start()

    def _release_export_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_export_worker_later(worker))
            return
        self._export_workers.discard(worker)
        if self._export_worker is worker:
            self._export_worker = None

    def _restore_export_button(self):
        if self._result and self._result.can_export:
            self._export_btn.setEnabled(True)
            self._export_btn.setText(
                tr("producer.export_demo" if self._result.is_demo else "producer.export_final")
            )

    def _on_export_finished(self, path: str):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        self._output_info.setText(tr("producer.status.exported", path=path))
        self._restore_export_button()

    def _on_export_error(self, message: str):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        self._output_info.setText(tr("producer.status.export_failed", error=message))
        self._restore_export_button()

    def _on_export_cancelled(self):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        self._output_info.setText(tr("producer.status.export_cancelled"))
        self._restore_export_button()
