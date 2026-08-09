"""
Slunder Studio — SFX Generator View
Text-to-SFX generation with preset categories, batch generation,
waveform preview, and drag-to-mixer support.
"""
import os
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QFrame, QScrollArea,
    QGridLayout, QSlider, QLineEdit, QFileDialog, QCheckBox,
)
from PySide6.QtCore import Qt, Signal, QTimer

from ui.theme import Palette, ThemeEngine
from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget, OperationProgressWidget
from core.i18n import tr, user_facing_readiness
from ui.waveform_widget import WaveformWidget, MiniWaveform
from core.engine_contract import (
    ArtifactKind,
    CAP_SFX_GENERATE,
    EngineArtifact,
    EngineBatchResult,
    EngineRunResult,
    RunMode,
    adapt_engine_result,
)
from core.model_manager import ModelManager
from core.audio_engine import AudioEngine, decode_playback_file
from core.workers import CancelledJobError, InferenceWorker
from engines.sfx_engine import SFXParams, SFXResult, SFX_CATEGORIES


_SFX_CATEGORY_LABEL_KEYS = {
    "Nature": "sfx.categories.nature",
    "Urban": "sfx.categories.urban",
    "Sci-Fi": "sfx.categories.sci_fi",
    "Musical": "sfx.categories.musical",
    "UI / Game": "sfx.categories.ui_game",
    "Foley": "sfx.categories.foley",
}


def _sfx_playback_task(path: str, **kwargs):
    return decode_playback_file(path, **kwargs)


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
        self.setMinimumHeight(92)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Mini waveform
        self._waveform = MiniWaveform()
        if result.audio is not None:
            import numpy as np
            mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
            self._waveform.load_audio(mono, result.sample_rate)
        self._waveform.setMinimumWidth(120)
        layout.addWidget(self._waveform)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)
        prefix = tr("sfx.card.demo_prefix") if result.is_demo else ""
        seed_label = QLabel(
            tr("sfx.card.seed", prefix=prefix, seed=result.seed)
        )
        seed_label.setStyleSheet(f"color: {t['text']}; font-size: 7.5pt; font-weight: bold;")
        dur_label = QLabel(
            tr(
                "sfx.card.duration",
                duration=result.duration,
                generation_time=result.generation_time,
            )
        )
        dur_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 6.75pt;")
        info.addWidget(seed_label)
        info.addWidget(dur_label)
        if result.is_demo:
            demo_label = QLabel(tr("sfx.card.demo_synthesis"))
            demo_label.setStyleSheet(f"color: {Palette.YELLOW}; font-size: 6.75pt;")
            info.addWidget(demo_label)
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
                font-size: 6.75pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        self._play_btn = QPushButton(tr("sfx.actions.play"))
        self._play_btn.setStyleSheet(btn_style)
        self._play_btn.clicked.connect(lambda: self.play_requested.emit(self.result))

        self._use_btn = QPushButton(
            tr("sfx.actions.use_demo") if result.is_demo else tr("sfx.actions.use")
        )
        self._use_btn.setProperty("class", "success")
        self._use_btn.setEnabled(result.can_route)
        self._use_btn.clicked.connect(lambda: self.use_requested.emit(self.result))

        self._delete_btn = QPushButton(tr("sfx.actions.remove_short"))
        self._delete_btn.setMinimumSize(20, 20)
        self._delete_btn.setStyleSheet(btn_style)
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self))

        btn_col.addWidget(self._play_btn)
        btn_col.addWidget(self._use_btn)
        btn_col.addWidget(self._delete_btn)
        layout.addLayout(btn_col)

        install_accessibility(
            self,
            tr("sfx.accessibility.card_name"),
            named_controls=[
                (
                    self._play_btn,
                    tr("sfx.accessibility.play_name"),
                    tr("sfx.accessibility.play_description"),
                ),
                (
                    self._use_btn,
                    tr("sfx.accessibility.use_name"),
                    tr("sfx.accessibility.use_description"),
                ),
                (
                    self._delete_btn,
                    tr("sfx.accessibility.delete_name"),
                    tr("sfx.accessibility.delete_description"),
                ),
            ],
        )


# ── SFX View ───────────────────────────────────────────────────────────────────

class SFXView(QWidget):
    """SFX Generator page."""

    send_to_mixer = Signal(str)  # audio file path

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._results: list[SFXResult] = []
        self._cards: list[SFXCard] = []
        self._model_mgr = ModelManager()
        self._generation_worker: Optional[InferenceWorker] = None
        self._playback_worker = None
        self._playback_workers = set()
        self._contract_batch: Optional[EngineBatchResult] = None

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

        title = QLabel(tr("sfx.title"))
        title.setStyleSheet(f"color: {t['accent']}; font-weight: bold; font-size: 9.75pt; border: none;")
        ctrl_layout.addWidget(title)

        # Prompt
        self._prompt = QTextEdit()
        self._prompt.setPlaceholderText(tr("sfx.prompt_placeholder"))
        self._prompt.setMaximumHeight(60)
        self._prompt.setStyleSheet(f"""
            QTextEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 9pt;
            }}
        """)
        self._prompt.textChanged.connect(self._refresh_capability_state)
        ctrl_layout.addWidget(self._prompt)

        # Negative prompt
        self._neg_prompt = QLineEdit()
        self._neg_prompt.setPlaceholderText(tr("sfx.negative_prompt_placeholder"))
        self._neg_prompt.setStyleSheet(f"""
            QLineEdit {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 8px; font-size: 8.25pt;
            }}
        """)
        ctrl_layout.addWidget(self._neg_prompt)

        param_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {t['background']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 3px;
                padding: 3px 6px; font-size: 8.25pt;
            }}
            QLabel {{ color: {t['text_secondary']}; font-size: 8.25pt; border: none; }}
        """

        # Category presets
        cat_row = QHBoxLayout()
        cl = QLabel(tr("sfx.category_label"))
        cl.setMinimumWidth(60)
        cl.setStyleSheet(param_style)
        self._category = QComboBox()
        self._category.addItem(tr("sfx.categories.custom"), "")
        for category in SFX_CATEGORIES:
            self._category.addItem(
                tr(_SFX_CATEGORY_LABEL_KEYS.get(category, "sfx.categories.custom")),
                category,
            )
        self._category.currentIndexChanged.connect(self._on_category_changed)
        self._category.setStyleSheet(param_style)
        cat_row.addWidget(cl)
        cat_row.addWidget(self._category)
        ctrl_layout.addLayout(cat_row)

        # Preset prompts
        self._preset_combo = QComboBox()
        self._preset_combo.setStyleSheet(param_style)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self._preset_combo.setVisible(False)
        ctrl_layout.addWidget(self._preset_combo)

        # Duration + Steps
        row1 = QHBoxLayout()
        dl = QLabel(tr("sfx.duration_label"))
        dl.setMinimumWidth(60)
        dl.setStyleSheet(param_style)
        self._duration = QDoubleSpinBox()
        self._duration.setRange(0.5, 47.0)
        self._duration.setValue(5.0)
        self._duration.setSuffix(tr("sfx.seconds_suffix"))
        self._duration.setStyleSheet(param_style)

        sl = QLabel(tr("sfx.steps_label"))
        sl.setMinimumWidth(38)
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
        cfl = QLabel(tr("sfx.cfg_label"))
        cfl.setMinimumWidth(60)
        cfl.setStyleSheet(param_style)
        self._cfg = QDoubleSpinBox()
        self._cfg.setRange(1.0, 20.0)
        self._cfg.setValue(7.0)
        self._cfg.setSingleStep(0.5)
        self._cfg.setStyleSheet(param_style)

        bl = QLabel(tr("sfx.batch_label"))
        bl.setMinimumWidth(38)
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
        self._gen_btn = QPushButton(tr("sfx.actions.generate"))
        self._gen_btn.setMinimumHeight(36)
        self._gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']};
                color: {t['background']}; border: none; border-radius: 5px;
                font-weight: bold; font-size: 9.75pt;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._gen_btn.clicked.connect(self._on_generate)
        ctrl_layout.addWidget(self._gen_btn)

        self._demo_checkbox = QCheckBox(tr("sfx.demo_checkbox"))
        self._demo_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {t['text_secondary']}; border: none; font-size: 7.5pt;
            }}
            QCheckBox::indicator {{
                width: 13px; height: 13px;
            }}
        """)
        self._demo_checkbox.toggled.connect(self._refresh_capability_state)
        ctrl_layout.addWidget(self._demo_checkbox)

        # Status
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(28)
        self._status.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt; border: none;")
        ctrl_layout.addWidget(self._status)

        self._operation_progress = OperationProgressWidget()
        self._operation_progress.cancel_requested.connect(
            self._cancel_active_operation
        )
        ctrl_layout.addWidget(self._operation_progress)

        left.addWidget(ctrl_frame)

        # Main output waveform
        self._main_waveform = WaveformWidget()
        left.addWidget(self._main_waveform, 1)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(340)
        layout.addWidget(left_w)

        # ── Right: Results Grid ────────────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        results_header = QHBoxLayout()
        rl = QLabel(tr("sfx.results_title"))
        rl.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")

        self._clear_btn = QPushButton(tr("sfx.actions.clear_all"))
        self._clear_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface']}; color: {t['text_secondary']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 4px 10px; font-size: 7.5pt;
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
        self._results_empty = EmptyStateWidget(
            tr("sfx.empty.title"),
            tr("sfx.empty.description"),
            tr("sfx.actions.generate_sound_effect"),
        )
        self._results_empty.action_requested.connect(self._gen_btn.click)
        self._results_layout.addWidget(self._results_empty)
        self._results_layout.addStretch()

        self._scroll.setWidget(self._results_container)
        right.addWidget(self._scroll, 1)

        right_w = QWidget()
        right_w.setLayout(right)
        layout.addWidget(right_w, 1)
        self._model_mgr.status_changed.connect(self._on_model_status_changed)
        self._refresh_capability_state()
        self._update_results_empty_state()
        install_accessibility(
            self,
            tr("sfx.accessibility.view_name"),
            named_controls=[
                (self._prompt, tr("sfx.accessibility.prompt_name"), tr("sfx.accessibility.prompt_description")),
                (self._neg_prompt, tr("sfx.accessibility.negative_name"), tr("sfx.accessibility.negative_description")),
                (self._category, tr("sfx.accessibility.category_name"), tr("sfx.accessibility.category_description")),
                (self._preset_combo, tr("sfx.accessibility.preset_name"), tr("sfx.accessibility.preset_description")),
                (self._duration, tr("sfx.accessibility.duration_name"), tr("sfx.accessibility.duration_description")),
                (self._steps, tr("sfx.accessibility.steps_name"), tr("sfx.accessibility.steps_description")),
                (self._cfg, tr("sfx.accessibility.cfg_name"), tr("sfx.accessibility.cfg_description")),
                (self._batch, tr("sfx.accessibility.batch_name"), tr("sfx.accessibility.batch_description")),
                (self._gen_btn, tr("sfx.accessibility.generate_name"), tr("sfx.accessibility.generate_description")),
                (self._operation_progress.cancel_button, tr("sfx.accessibility.cancel_name"), tr("sfx.accessibility.cancel_description")),
                (self._demo_checkbox, tr("sfx.accessibility.demo_name"), tr("sfx.accessibility.demo_description")),
                (self._clear_btn, tr("sfx.accessibility.clear_name"), tr("sfx.accessibility.clear_description")),
            ],
        )

    # ── Events ─────────────────────────────────────────────────────────────────

    def _cancel_active_operation(self):
        """Request cancellation for generation or an asynchronous preview load."""
        worker = self._generation_worker or self._playback_worker
        if worker is None:
            self._operation_progress.finish()
            return
        self._operation_progress.mark_cancelling()
        worker.cancel()
        if worker is self._generation_worker:
            self._gen_btn.setEnabled(False)
            self._status.setText(tr("sfx.status.cancelling_generation"))
        else:
            self._status.setText(tr("sfx.status.cancelling_preview"))

    def _on_category_changed(self, category: str):
        category = self._category.currentData() or category
        self._preset_combo.clear()
        if category in SFX_CATEGORIES:
            for prompt in SFX_CATEGORIES[category]:
                self._preset_combo.addItem(prompt, prompt)
            self._preset_combo.setVisible(True)
        else:
            self._preset_combo.setVisible(False)
        install_accessibility(
            self,
            tr("sfx.accessibility.view_name"),
            named_controls=[
                (
                    self._preset_combo,
                    tr("sfx.accessibility.preset_name"),
                    tr("sfx.accessibility.preset_description"),
                ),
            ],
        )

    def _on_preset_selected(self, _index: int):
        prompt = self._preset_combo.currentData()
        if prompt:
            self._prompt.setPlainText(str(prompt))

    def _on_generate(self):
        if self._generation_worker is not None:
            self._cancel_active_operation()
            return

        readiness = self._model_mgr.get_capability_readiness(
            CAP_SFX_GENERATE,
            allow_demo=self._demo_checkbox.isChecked(),
        )
        if not readiness.can_run:
            self._status.setText(self._readiness_message(readiness))
            self._refresh_capability_state()
            return

        params = SFXParams(
            prompt=self._prompt.toPlainText().strip(),
            negative_prompt=self._neg_prompt.text().strip(),
            duration=self._duration.value(),
            cfg_scale=self._cfg.value(),
            steps=self._steps.value(),
            batch_size=self._batch.value(),
            allow_demo_output=self._demo_checkbox.isChecked(),
        )

        if not params.prompt:
            self._status.setText(tr("sfx.status.enter_prompt"))
            return

        self._status.setText(
            tr("sfx.status.generating_demo")
            if readiness.mode == RunMode.DEMO
            else tr("sfx.status.generating_model")
        )
        self._generation_worker = InferenceWorker(
            self._run_generation_batch,
            params,
            readiness.model_id,
            job_kind="sfx_generation",
            job_label=tr("sfx.jobs.generation_batch"),
            job_inputs={
                "prompt_chars": len(params.prompt),
                "duration": params.duration,
                "batch_size": params.batch_size,
                "demo": params.allow_demo_output,
            },
            job_metadata={
                "module": "sfx",
                "capability_id": CAP_SFX_GENERATE,
            },
        )
        self._generation_worker.progress.connect(
            self._on_generation_progress
        )
        self._generation_worker.step_info.connect(self._on_generation_step)
        self._generation_worker.finished.connect(self._on_generation_finished)
        self._generation_worker.error.connect(self._on_generation_error)
        self._generation_worker.cancelled.connect(self._on_generation_cancelled)
        self._operation_progress.start(tr("sfx.progress.generation"), determinate=True)
        self._generation_worker.start()
        self._refresh_capability_state()

    def _on_generation_progress(self, percent: int):
        self._operation_progress.set_progress(percent, tr("sfx.progress.generation"))
        self._status.setText(tr("sfx.status.generation_progress", percent=percent))

    def _on_generation_step(self, message: str):
        self._operation_progress.set_step(message)
        self._status.setText(message)

    def _run_generation_batch(
        self,
        params: SFXParams,
        model_id: str,
        progress_cb=None,
        step_cb=None,
        log_cb=None,
        cancel_event=None,
    ) -> EngineBatchResult:
        from engines.sfx_engine import generate_sfx

        runs: list[EngineRunResult] = []

        def _verified_paths() -> list[str]:
            """Completed variations that are actually readable on disk."""
            paths = []
            for run in runs:
                if not run.is_success:
                    continue
                for path in run.output_paths:
                    if path and os.path.isfile(path):
                        paths.append(path)
            return paths

        for index in range(params.batch_size):
            if cancel_event and cancel_event.is_set():
                batch = EngineBatchResult(
                    CAP_SFX_GENERATE, runs,
                    error=f"Cancelled after {len(runs)} of {params.batch_size}",
                )
                verified = _verified_paths()
                raise CancelledJobError(
                    f"SFX generation cancelled after {len(runs)} of "
                    f"{params.batch_size} variation(s); {len(verified)} kept",
                    outputs=batch.output_paths,
                    preserved=verified,
                    result=batch,
                )
            item = SFXParams(
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                duration=params.duration,
                cfg_scale=params.cfg_scale,
                steps=params.steps,
                seed=None,
                allow_demo_output=params.allow_demo_output,
            )

            def progress(value: float, message: str = ""):
                if progress_cb:
                    overall = (index + max(0.0, min(1.0, value))) / params.batch_size
                    progress_cb(int(round(overall * 100)))
                if step_cb and message:
                    step_cb(f"{index + 1}/{params.batch_size}: {message}")

            result = generate_sfx(item, progress_callback=progress)
            artifacts = []
            if result.audio is not None or result.file_path:
                artifacts.append(EngineArtifact(
                    kind=ArtifactKind.AUDIO,
                    path=result.file_path or "",
                    payload=result.audio,
                    provenance_path=result.provenance_path,
                    routable=result.can_route,
                ))
            runs.append(adapt_engine_result(
                CAP_SFX_GENERATE,
                result,
                artifacts,
                model_id=model_id,
            ))

        errors = [run.error for run in runs if not run.is_success and run.error]
        return EngineBatchResult(
            capability_id=CAP_SFX_GENERATE,
            runs=runs,
            error="; ".join(errors),
        )

    def _on_generation_finished(self, batch: EngineBatchResult):
        self._generation_worker = None
        self._operation_progress.finish()
        self._contract_batch = batch
        successful = batch.successful_runs
        if not successful:
            self._status.setText(
                tr(
                    "sfx.status.generation_failed",
                    error=batch.error or tr("sfx.status.no_output"),
                )
            )
            self._refresh_capability_state()
            return

        for run in successful:
            result = run.source_result
            self._results.append(result)
            self._add_result_card(result)
        first = successful[0].source_result
        if first.audio is not None:
            self._main_waveform.load_audio(first.audio, first.sample_rate)
        demo_count = sum(run.is_demo for run in successful)
        suffix = tr("sfx.status.demo_count", count=demo_count) if demo_count else ""
        self._status.setText(
            tr(
                "sfx.status.generated",
                count=len(successful),
                total=len(batch.runs),
                suffix=suffix,
                duration=first.duration,
            )
        )
        self._refresh_capability_state()

    def _on_generation_error(self, error: str):
        self._generation_worker = None
        self._operation_progress.finish()
        self._contract_batch = EngineBatchResult(
            capability_id=CAP_SFX_GENERATE,
            error=error,
        )
        self._status.setText(tr("sfx.status.generation_failed", error=error))
        self._refresh_capability_state()

    def _on_generation_cancelled(self):
        """Keep the variations that finished; only the in-flight one is dropped."""
        worker = self._generation_worker
        self._generation_worker = None
        self._operation_progress.finish()
        partial = getattr(worker, "result", None) if worker is not None else None
        kept = 0
        if isinstance(partial, EngineBatchResult):
            self._contract_batch = partial
            for run in partial.successful_runs:
                result = run.source_result
                if result is None:
                    continue
                paths = [p for p in run.output_paths if p and os.path.isfile(p)]
                if run.output_paths and not paths:
                    continue  # its file did not survive; do not advertise it
                self._results.append(result)
                self._add_result_card(result)
                kept += 1
            if kept:
                first = partial.successful_runs[0].source_result
                if first is not None and first.audio is not None:
                    self._main_waveform.load_audio(first.audio, first.sample_rate)
        self._status.setText(
            tr("sfx.status.cancelled_kept", count=kept)
            if kept else tr("sfx.status.cancelled")
        )
        self._refresh_capability_state()

    def _on_model_status_changed(self, model_id: str, _status: str):
        if model_id == "stable-audio-open":
            self._refresh_capability_state()

    def _refresh_capability_state(self):
        if not hasattr(self, "_gen_btn"):
            return
        readiness = self._model_mgr.get_capability_readiness(
            CAP_SFX_GENERATE,
            allow_demo=self._demo_checkbox.isChecked(),
        )
        if self._generation_worker is not None:
            self._gen_btn.setText(tr("sfx.actions.cancel_generation"))
            self._gen_btn.setEnabled(True)
            self._gen_btn.setToolTip(tr("runtime.cancel_generation"))
            return
        has_prompt = bool(self._prompt.toPlainText().strip())
        self._gen_btn.setText(tr("sfx.actions.generate"))
        self._gen_btn.setEnabled(readiness.can_run and has_prompt)
        self._gen_btn.setToolTip(
            (
                tr("runtime.ready")
                if readiness.can_run and has_prompt
                else tr("runtime.enter_sfx")
                if not has_prompt
                else self._readiness_message(readiness)
            )
        )
        if not readiness.can_run and not self._status.text():
            self._status.setText(self._readiness_message(readiness))

    def _readiness_message(self, readiness) -> str:
        info = self._model_mgr.get_model_info(readiness.model_id)
        return user_facing_readiness(
            readiness,
            model_name=info.name if info is not None else "",
        )

    def _add_result_card(self, result: SFXResult, index: Optional[int] = None):
        card = SFXCard(result)
        card.play_requested.connect(self._on_play_sfx)
        card.use_requested.connect(self._on_use_sfx)
        card.delete_requested.connect(self._on_delete_card)
        if index is None:
            index = len(self._cards)
        index = max(0, min(index, len(self._cards)))
        self._cards.insert(index, card)
        self._results_layout.insertWidget(index, card)
        self._update_results_empty_state()

    def _update_results_empty_state(self):
        if self._cards:
            self._results_empty.hide()
            return
        self._results_empty.set_state(
            tr("sfx.empty.title"),
            tr("sfx.empty.description"),
            tr("sfx.actions.generate_sound_effect"),
        )
        self._results_empty.show()

    @staticmethod
    def _remove_identity(items, target):
        for index, item in enumerate(items):
            if item is target:
                del items[index]
                return

    def _snapshot_sfx_cards(self, cards: list[SFXCard]) -> list[dict]:
        """Capture result order before removing cards from the visible list."""
        snapshots = []
        for card in cards:
            if card not in self._cards:
                continue
            result = card.result
            snapshots.append({
                "index": self._cards.index(card),
                "result": result,
                "original_path": result.file_path or "",
                "entry": None,
            })
        return snapshots

    def _trash_sfx_snapshots(self, snapshots: list[dict]):
        from core.trash import TrashManager

        requests = []
        request_snapshots = []
        for snapshot in snapshots:
            path = snapshot["original_path"]
            if not path or not os.path.exists(path):
                continue
            result = snapshot["result"]
            requests.append({
                "path": path,
                "category": "generated_asset",
                "label": os.path.basename(path),
                "metadata": {
                    "module": "sfx",
                    "seed": result.seed,
                    "duration": result.duration,
                    "sample_rate": result.sample_rate,
                    "is_demo": result.is_demo,
                },
            })
            request_snapshots.append(snapshot)

        entries = TrashManager().trash_paths(requests)
        for snapshot, entry in zip(request_snapshots, entries):
            snapshot["entry"] = entry
        return entries

    def _remove_sfx_snapshots(self, snapshots: list[dict]):
        for snapshot in sorted(snapshots, key=lambda item: item["index"], reverse=True):
            result = snapshot["result"]
            card = next(
                (candidate for candidate in self._cards if candidate.result is result),
                None,
            )
            if card is None:
                continue
            self._remove_identity(self._cards, card)
            self._remove_identity(self._results, result)
            self._results_layout.removeWidget(card)
            card.deleteLater()
        self._update_results_empty_state()

    def _restore_sfx_snapshots(self, snapshots: list[dict]):
        from core.trash import TrashManager

        trash = TrashManager()
        errors = []
        for snapshot in sorted(snapshots, key=lambda item: item["index"]):
            entry = snapshot.get("entry")
            if entry is None:
                continue
            try:
                if trash.get_entry(entry.id) is not None:
                    restored = trash.restore(entry.id)
                    snapshot["result"].file_path = restored.original_path
                elif not os.path.exists(snapshot["original_path"]):
                    raise RuntimeError("trash entry is no longer available")
            except Exception as exc:
                errors.append(str(exc))

        for snapshot in sorted(snapshots, key=lambda item: item["index"]):
            result = snapshot["result"]
            if any(existing is result for existing in self._results):
                continue
            index = min(snapshot["index"], len(self._results))
            self._results.insert(index, result)
            self._add_result_card(result, index=index)

        if errors:
            self._status.setText(tr("sfx.status.restore_failed", error=errors[0]))
            if self.toast_mgr:
                self.toast_mgr.error(tr("sfx.messages.restore_some_failed"))
        elif self.toast_mgr:
            self.toast_mgr.success(tr("sfx.messages.restored"))

    def _on_play_sfx(self, result: SFXResult):
        if result.audio is not None:
            self._play_decoded_sfx(result.audio, result.sample_rate)
            return
        if not result.file_path:
            self._status.setText(tr("sfx.status.playback_load_failed"))
            return
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._cancel_active_operation()
            return
        worker = InferenceWorker(_sfx_playback_task, result.file_path)
        self._playback_workers.add(worker)
        self._playback_worker = worker
        worker.progress.connect(
            self._on_sfx_playback_progress
        )
        worker.step_info.connect(self._on_sfx_playback_step)
        worker.finished.connect(self._on_sfx_playback_ready)
        worker.error.connect(self._on_sfx_playback_error)
        worker.cancelled.connect(self._on_sfx_playback_cancelled)
        self._operation_progress.start(tr("sfx.progress.loading_preview"), determinate=True)
        worker.start()

    def _on_sfx_playback_progress(self, percent: int):
        self._operation_progress.set_progress(percent, tr("sfx.progress.loading_preview"))
        self._status.setText(tr("sfx.status.playback_progress", percent=percent))

    def _on_sfx_playback_step(self, message: str):
        self._operation_progress.set_step(message)
        self._status.setText(message)

    def _play_decoded_sfx(self, audio, sample_rate):
        try:
            engine = AudioEngine()
            if not engine.load_array(audio, sample_rate):
                raise RuntimeError("Audio playback rejected the decoded buffer")
            engine.play()
            self._status.setText(tr("sfx.status.playing_preview"))
        except Exception as exc:
            self._status.setText(tr("sfx.status.playback_error", error=exc))

    def _release_playback_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_playback_worker_later(worker))
            return
        self._playback_workers.discard(worker)
        if self._playback_worker is worker:
            self._playback_worker = None

    def _on_sfx_playback_ready(self, payload):
        worker = self._playback_worker
        self._operation_progress.finish()
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._play_decoded_sfx(*payload)

    def _on_sfx_playback_error(self, message: str):
        worker = self._playback_worker
        self._operation_progress.finish()
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._status.setText(tr("sfx.status.playback_error", error=message))

    def _on_sfx_playback_cancelled(self):
        worker = self._playback_worker
        self._operation_progress.finish()
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._status.setText(tr("sfx.status.playback_cancelled"))

    def _on_use_sfx(self, result: SFXResult):
        if result.file_path and result.can_route:
            self.send_to_mixer.emit(result.file_path)
        else:
            self._status.setText(tr("runtime.send_to_mixer_unavailable"))

    def _on_delete_card(self, card: SFXCard):
        snapshots = self._snapshot_sfx_cards([card])
        if not snapshots:
            return
        try:
            self._trash_sfx_snapshots(snapshots)
        except Exception as exc:
            self._status.setText(tr("sfx.status.delete_failed", error=exc))
            if self.toast_mgr:
                self.toast_mgr.error(tr("sfx.messages.file_delete_failed"))
            return

        self._remove_sfx_snapshots(snapshots)
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("sfx.messages.moved_to_trash"),
                duration_ms=8000,
                action_label=tr("sfx.actions.undo"),
                action_callback=lambda items=snapshots: self._restore_sfx_snapshots(items),
            )

    def _restore_sfx_card(self, trash_entry_id: str, result: SFXResult):
        snapshot = {
            "index": len(self._results),
            "result": result,
            "original_path": result.file_path or "",
            "entry": type("Entry", (), {"id": trash_entry_id})(),
        }
        self._restore_sfx_snapshots([snapshot])

    def _clear_results(self):
        snapshots = self._snapshot_sfx_cards(list(self._cards))
        if not snapshots:
            return
        try:
            self._trash_sfx_snapshots(snapshots)
        except Exception as exc:
            self._status.setText(tr("sfx.status.clear_failed", error=exc))
            if self.toast_mgr:
                self.toast_mgr.error(tr("sfx.messages.clear_delete_failed"))
            return

        self._remove_sfx_snapshots(snapshots)
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("sfx.messages.results_moved_to_trash"),
                duration_ms=8000,
                action_label=tr("sfx.actions.undo"),
                action_callback=lambda items=snapshots: self._restore_sfx_snapshots(items),
            )
