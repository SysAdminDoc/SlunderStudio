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
        self.setFixedHeight(92)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        # Mini waveform
        self._waveform = MiniWaveform()
        if result.audio is not None:
            import numpy as np
            mono = result.audio[:, 0] if result.audio.ndim == 2 else result.audio
            self._waveform.load_audio(mono, result.sample_rate)
        self._waveform.setFixedWidth(120)
        layout.addWidget(self._waveform)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)
        prefix = "DEMO " if result.is_demo else ""
        seed_label = QLabel(f"{prefix}Seed: {result.seed}")
        seed_label.setStyleSheet(f"color: {t['text']}; font-size: 10px; font-weight: bold;")
        dur_label = QLabel(f"{result.duration:.1f}s | {result.generation_time:.1f}s gen")
        dur_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 9px;")
        info.addWidget(seed_label)
        info.addWidget(dur_label)
        if result.is_demo:
            demo_label = QLabel("Demo synthesis")
            demo_label.setStyleSheet(f"color: {Palette.YELLOW}; font-size: 9px;")
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
                font-size: 9px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        self._play_btn = QPushButton("Play")
        self._play_btn.setStyleSheet(btn_style)
        self._play_btn.clicked.connect(lambda: self.play_requested.emit(self.result))

        self._use_btn = QPushButton("Use Demo" if result.is_demo else "Use")
        self._use_btn.setProperty("class", "success")
        self._use_btn.setEnabled(result.can_route)
        self._use_btn.clicked.connect(lambda: self.use_requested.emit(self.result))

        self._delete_btn = QPushButton("X")
        self._delete_btn.setFixedSize(20, 20)
        self._delete_btn.setStyleSheet(btn_style)
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self))

        btn_col.addWidget(self._play_btn)
        btn_col.addWidget(self._use_btn)
        btn_col.addWidget(self._delete_btn)
        layout.addLayout(btn_col)

        install_accessibility(
            self,
            "SFX variation",
            named_controls=[
                (self._play_btn, "Play sound effect", "Plays this generated sound effect."),
                (self._use_btn, "Use sound effect", "Routes this sound effect to the mixer."),
                (self._delete_btn, "Delete sound effect", "Removes this generated sound effect."),
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
        self._prompt.textChanged.connect(self._refresh_capability_state)
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
                background: {t['accent']};
                color: {t['background']}; border: none; border-radius: 5px;
                font-weight: bold; font-size: 13px;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
            QPushButton:disabled {{ background: {t['border']}; color: {t['muted']}; }}
        """)
        self._gen_btn.clicked.connect(self._on_generate)
        ctrl_layout.addWidget(self._gen_btn)

        self._demo_checkbox = QCheckBox("Enable demo synthesis (no AI model)")
        self._demo_checkbox.setStyleSheet(f"""
            QCheckBox {{
                color: {t['text_secondary']}; border: none; font-size: 10px;
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
        self._model_mgr.status_changed.connect(self._on_model_status_changed)
        self._refresh_capability_state()
        install_accessibility(
            self,
            "SFX Generator",
            named_controls=[
                (self._prompt, "SFX prompt", "Describes the sound effect to generate."),
                (self._neg_prompt, "SFX negative prompt", "Describes sounds to avoid."),
                (self._category, "SFX category", "Selects a sound effect category."),
                (self._preset_combo, "SFX preset", "Selects a category prompt preset."),
                (self._duration, "SFX duration", "Sets the generated sound duration in seconds."),
                (self._steps, "SFX diffusion steps", "Sets the number of synthesis steps."),
                (self._cfg, "SFX guidance scale", "Sets synthesis guidance strength."),
                (self._batch, "SFX batch size", "Sets how many variations to generate."),
                (self._gen_btn, "Generate sound effects", "Generates the requested sound effect variations."),
                (self._demo_checkbox, "Enable SFX demo synthesis", "Allows a local demo fallback without the AI model."),
                (self._clear_btn, "Clear generated sound effects", "Removes all generated sound effect cards."),
            ],
        )

    # ── Events ─────────────────────────────────────────────────────────────────

    def _on_category_changed(self, category: str):
        self._preset_combo.clear()
        if category in SFX_CATEGORIES:
            self._preset_combo.addItems(SFX_CATEGORIES[category])
            self._preset_combo.setVisible(True)
        else:
            self._preset_combo.setVisible(False)
        install_accessibility(
            self,
            "SFX Generator",
            named_controls=[
                (self._preset_combo, "SFX preset", "Selects a category prompt preset."),
            ],
        )

    def _on_preset_selected(self, text: str):
        if text:
            self._prompt.setPlainText(text)

    def _on_generate(self):
        if self._generation_worker is not None:
            self._generation_worker.cancel()
            self._gen_btn.setEnabled(False)
            self._status.setText("Cancelling SFX generation...")
            return

        readiness = self._model_mgr.get_capability_readiness(
            CAP_SFX_GENERATE,
            allow_demo=self._demo_checkbox.isChecked(),
        )
        if not readiness.can_run:
            self._status.setText(readiness.remedy)
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
            self._status.setText("Enter a prompt first")
            return

        self._status.setText(
            "Generating explicit demo SFX..."
            if readiness.mode == RunMode.DEMO
            else "Generating with Stable Audio Open..."
        )
        self._generation_worker = InferenceWorker(
            self._run_generation_batch,
            params,
            readiness.model_id,
            job_kind="sfx_generation",
            job_label="SFX generation batch",
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
            lambda pct: self._status.setText(f"SFX generation... {pct}%")
        )
        self._generation_worker.step_info.connect(self._status.setText)
        self._generation_worker.finished.connect(self._on_generation_finished)
        self._generation_worker.error.connect(self._on_generation_error)
        self._generation_worker.cancelled.connect(self._on_generation_cancelled)
        self._generation_worker.start()
        self._refresh_capability_state()

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
        self._contract_batch = batch
        successful = batch.successful_runs
        if not successful:
            self._status.setText(
                f"SFX generation failed: {batch.error or 'No output was produced'}"
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
        suffix = f" ({demo_count} demo)" if demo_count else ""
        self._status.setText(
            f"Generated {len(successful)}/{len(batch.runs)} SFX{suffix} "
            f"({first.duration:.1f}s each)"
        )
        self._refresh_capability_state()

    def _on_generation_error(self, error: str):
        self._generation_worker = None
        self._contract_batch = EngineBatchResult(
            capability_id=CAP_SFX_GENERATE,
            error=error,
        )
        self._status.setText(f"SFX generation failed: {error}")
        self._refresh_capability_state()

    def _on_generation_cancelled(self):
        """Keep the variations that finished; only the in-flight one is dropped."""
        worker = self._generation_worker
        self._generation_worker = None
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
            f"SFX generation cancelled - kept {kept} completed variation(s). "
            "Generate again to retry the rest."
            if kept else "SFX generation cancelled"
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
            self._gen_btn.setText("Cancel Generation")
            self._gen_btn.setEnabled(True)
            self._gen_btn.setToolTip("Cancel the running SFX generation job.")
            return
        has_prompt = bool(self._prompt.toPlainText().strip())
        self._gen_btn.setText("Generate SFX")
        self._gen_btn.setEnabled(readiness.can_run and has_prompt)
        self._gen_btn.setToolTip(
            (
                f"Produces declared outputs: {readiness.output_summary}."
                if readiness.can_run and has_prompt
                else "Enter an SFX prompt."
                if not has_prompt
                else readiness.remedy
            )
        )
        if not readiness.can_run and not self._status.text():
            self._status.setText(readiness.remedy)

    def _add_result_card(self, result: SFXResult):
        card = SFXCard(result)
        card.play_requested.connect(self._on_play_sfx)
        card.use_requested.connect(self._on_use_sfx)
        card.delete_requested.connect(self._on_delete_card)
        self._cards.append(card)
        self._results_layout.insertWidget(self._results_layout.count() - 1, card)

    def _on_play_sfx(self, result: SFXResult):
        if result.audio is not None:
            self._play_decoded_sfx(result.audio, result.sample_rate)
            return
        if not result.file_path:
            self._status.setText("Could not load SFX audio for playback")
            return
        if self._playback_worker is not None and self._playback_worker.isRunning():
            self._playback_worker.cancel()
        worker = InferenceWorker(_sfx_playback_task, result.file_path)
        self._playback_workers.add(worker)
        self._playback_worker = worker
        worker.progress.connect(
            lambda pct: self._status.setText(f"Loading SFX preview... {pct}%")
        )
        worker.finished.connect(self._on_sfx_playback_ready)
        worker.error.connect(self._on_sfx_playback_error)
        worker.cancelled.connect(self._on_sfx_playback_cancelled)
        self._status.setText("Loading SFX preview... 0%")
        worker.start()

    def _play_decoded_sfx(self, audio, sample_rate):
        try:
            engine = AudioEngine()
            if not engine.load_array(audio, sample_rate):
                raise RuntimeError("Audio playback rejected the decoded buffer")
            engine.play()
            self._status.setText("Playing SFX preview")
        except Exception as exc:
            self._status.setText(f"SFX playback error: {exc}")

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
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._play_decoded_sfx(*payload)

    def _on_sfx_playback_error(self, message: str):
        worker = self._playback_worker
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._status.setText(f"SFX playback error: {message}")

    def _on_sfx_playback_cancelled(self):
        worker = self._playback_worker
        self._release_playback_worker_later(worker)
        self._playback_worker = None
        self._status.setText("SFX preview loading cancelled")

    def _on_use_sfx(self, result: SFXResult):
        if result.file_path and result.can_route:
            self.send_to_mixer.emit(result.file_path)
        else:
            self._status.setText("SFX cannot be routed to the mixer")

    def _on_delete_card(self, card: SFXCard):
        result = card.result
        entry = None
        if result.file_path and os.path.exists(result.file_path):
            try:
                from core.trash import TrashManager
                entry = TrashManager().trash_path(
                    result.file_path,
                    category="generated_asset",
                    label=os.path.basename(result.file_path),
                    metadata={
                        "module": "sfx",
                        "seed": result.seed,
                        "duration": result.duration,
                        "sample_rate": result.sample_rate,
                        "is_demo": result.is_demo,
                    },
                )
            except Exception as e:
                self._status.setText(f"Delete failed: {e}")
                if self.toast_mgr:
                    self.toast_mgr.error("SFX file could not be moved to trash.")
                return

        if card in self._cards:
            self._cards.remove(card)
        if result in self._results:
            self._results.remove(result)
        self._results_layout.removeWidget(card)
        card.deleteLater()
        if entry and self.toast_mgr:
            self.toast_mgr.info(
                "SFX moved to trash.",
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda entry_id=entry.id, res=result: self._restore_sfx_card(entry_id, res),
            )

    def _restore_sfx_card(self, trash_entry_id: str, result: SFXResult):
        try:
            from core.trash import TrashManager
            entry = TrashManager().restore(trash_entry_id)
            result.file_path = entry.original_path
        except Exception as e:
            self._status.setText(f"Restore failed: {e}")
            if self.toast_mgr:
                self.toast_mgr.error("SFX restore failed.")
            return

        if result not in self._results:
            self._results.append(result)
            self._add_result_card(result)
        if self.toast_mgr:
            self.toast_mgr.success("SFX restored.")

    def _clear_results(self):
        for card in self._cards:
            self._results_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._results.clear()
