"""
Slunder Studio — Batch View
Grid display for batch-generated song variations.
Mini waveform cards with one-click playback, star/rank, delete, and "Best of" refinement.
"""
import json
import os
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QScrollArea, QSpinBox,
)
from PySide6.QtCore import Signal, Qt, QTimer

from ui.accessibility import install_accessibility
from ui.theme import Palette
from ui.waveform_widget import MiniWaveform
from core.job_state import JobStatus, JobStore
from core.i18n import tr
from core.settings import get_config_dir
from core.audio_buffers import decode_audio_file
from core.workers import CancelledJobError, InferenceWorker


def _prepare_batch_card_task(
    audio_path: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Decode once, then score the exact buffer shown by a batch card."""
    from engines.audio_analyzer import score_audio_buffer

    def _decode_progress(value: int):
        if progress_cb:
            progress_cb(int(value * 0.7))

    audio, sample_rate = decode_audio_file(
        audio_path,
        target_channels=2,
        progress_cb=_decode_progress,
        cancel_event=cancel_event,
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("Batch preview cancelled")

    def _score_progress(value: int):
        if progress_cb:
            progress_cb(70 + int(value * 0.3))

    quality = score_audio_buffer(
        audio,
        sample_rate,
        progress_cb=_score_progress,
        cancel_event=cancel_event,
    )
    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "quality": quality,
    }


class BatchCard(QFrame):
    """Card for a single batch result."""
    play_requested = Signal(str)
    star_toggled = Signal(int, bool)
    delete_requested = Signal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self._index = index
        self._audio_path = ""
        self._seed = 0
        self._gen_time = 0.0
        self._quality_score = 0.0
        self._quality_token = 0
        self._quality_worker = None
        self._quality_workers = set()
        self._closed = False
        self._is_starred = False
        self._is_playing = False

        self.setMinimumSize(200, 140)
        self.setMaximumWidth(350)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        header.setSpacing(4)

        self._title = QLabel(tr("batch.variation", index=self._index + 1))
        self._title.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 9pt;")
        header.addWidget(self._title)

        header.addStretch()

        self._star_btn = QPushButton("\u2606")
        self._star_btn.setMinimumSize(24, 24)
        self._star_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {Palette.OVERLAY0}; font-size: 12pt; }}"
            f"QPushButton:hover {{ color: {Palette.YELLOW}; }}"
        )
        self._star_btn.clicked.connect(self._toggle_star)
        header.addWidget(self._star_btn)

        self._delete_btn = QPushButton("\u2715")
        self._delete_btn.setMinimumSize(24, 24)
        self._delete_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {Palette.OVERLAY0}; font-size: 10.5pt; }}"
            f"QPushButton:hover {{ color: {Palette.RED}; }}"
        )
        self._delete_btn.clicked.connect(lambda: self.delete_requested.emit(self._index))
        header.addWidget(self._delete_btn)

        layout.addLayout(header)

        # Waveform
        self._waveform = MiniWaveform()
        self._waveform.clicked.connect(self._on_play)
        layout.addWidget(self._waveform)

        # Info row
        info = QHBoxLayout()
        self._seed_label = QLabel("")
        self._seed_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 7.5pt;")
        info.addWidget(self._seed_label)

        info.addStretch()

        self._score_label = QLabel("")
        self._score_label.setStyleSheet(f"color: {Palette.GREEN}; font-size: 7.5pt; font-weight: bold;")
        info.addWidget(self._score_label)

        self._time_label = QLabel("")
        self._time_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 7.5pt;")
        info.addWidget(self._time_label)

        layout.addLayout(info)

        install_accessibility(
            self,
            tr("batch.accessibility.card_name", index=self._index + 1),
            named_controls=[
                (self._star_btn, tr("batch.accessibility.star_name", state=tr("batch.star"), index=self._index + 1), tr("batch.accessibility.star_description")),
                (self._delete_btn, tr("batch.accessibility.delete_name", index=self._index + 1), tr("batch.accessibility.delete_description")),
            ],
            tab_order=[self._star_btn, self._delete_btn],
        )

    def _update_style(self):
        if self._is_playing:
            border = f"2px solid {Palette.GREEN}"
        elif self._is_starred:
            border = f"2px solid {Palette.YELLOW}"
        else:
            border = f"1px solid {Palette.SURFACE0}"

        self.setStyleSheet(
            f"QFrame {{ background: {Palette.BASE}; border: {border}; border-radius: 8px; }}"
        )

    def set_result(self, audio_path: str, seed: int, gen_time: float = 0.0):
        self._audio_path = audio_path
        self._seed = seed
        self._gen_time = gen_time
        self._seed_label.setText(tr("batch.seed", seed=seed))
        if gen_time > 0:
            self._time_label.setText(tr("batch.duration", seconds=gen_time))

        self._start_quality_job()

    def _start_quality_job(self):
        path = self._audio_path
        if not path or not os.path.isfile(path):
            self._score_label.setText(tr("batch.quality_unavailable"))
            return

        self._quality_token += 1
        token = self._quality_token
        worker = InferenceWorker(_prepare_batch_card_task, path)
        self._quality_workers.add(worker)
        self._quality_worker = worker
        self._score_label.setText(tr("batch.quality_progress", percent=0))
        worker.progress.connect(
            lambda percent, t=token: self._on_quality_progress(t, percent)
        )
        worker.finished.connect(
            lambda payload, t=token, w=worker: self._on_quality_finished(t, w, payload)
        )
        worker.error.connect(
            lambda message, t=token, w=worker: self._on_quality_error(t, w, message)
        )
        worker.cancelled.connect(
            lambda t=token, w=worker: self._on_quality_cancelled(t, w)
        )
        try:
            worker.start()
        except Exception as exc:
            self._on_quality_error(token, worker, f"{type(exc).__name__}: {exc}")

    def _forget_quality_worker_later(self, worker):
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._forget_quality_worker_later(worker))
            return
        self._quality_workers.discard(worker)

    def _on_quality_progress(self, token: int, percent: int):
        if token == self._quality_token and not self._closed:
            self._score_label.setText(tr("batch.quality_progress", percent=percent))

    def _on_quality_finished(self, token: int, worker, payload: dict):
        self._forget_quality_worker_later(worker)
        if token != self._quality_token or self._closed:
            return
        self._quality_worker = None
        try:
            self._waveform.set_audio(payload["audio"], payload["sample_rate"])
            self.set_quality_score(payload["quality"].total)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            self._on_quality_error(token, worker, str(exc))

    def _on_quality_error(self, token: int, worker, message: str):
        self._forget_quality_worker_later(worker)
        if token != self._quality_token or self._closed:
            return
        self._quality_worker = None
        self._score_label.setText(tr("batch.quality_unavailable"))

    def _on_quality_cancelled(self, token: int, worker):
        self._forget_quality_worker_later(worker)
        if token != self._quality_token or self._closed:
            return
        self._quality_worker = None
        self._score_label.setText(tr("batch.quality_cancelled"))

    def cancel_quality_score(self):
        """Request cancellation without waiting on the GUI thread."""
        worker = self._quality_worker
        if worker is None or not worker.isRunning():
            return
        self._quality_token += 1
        worker.cancel()
        self._score_label.setText(tr("batch.quality_cancelling"))

    def closeEvent(self, event):
        self._closed = True
        for worker in tuple(self._quality_workers):
            try:
                worker.cancel()
                worker.progress.disconnect()
                worker.finished.disconnect()
                worker.error.disconnect()
                worker.cancelled.disconnect()
            except (RuntimeError, TypeError):
                pass
        self._quality_worker = None
        super().closeEvent(event)

    def _on_play(self):
        if self._audio_path:
            self.play_requested.emit(self._audio_path)

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self._update_style()

    def _toggle_star(self):
        self.set_starred(not self._is_starred)
        self.star_toggled.emit(self._index, self._is_starred)

    def set_starred(self, starred: bool):
        """Set star state without emitting the persistence signal."""
        self._is_starred = bool(starred)
        self._star_btn.setText("\u2605" if self._is_starred else "\u2606")
        self._star_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {Palette.YELLOW if self._is_starred else Palette.OVERLAY0}; font-size: 12pt; }}"
            f" QPushButton:hover {{ color: {Palette.YELLOW}; }}"
        )
        state = tr("batch.unstar") if self._is_starred else tr("batch.star")
        self._star_btn.setAccessibleName(
            tr("batch.accessibility.star_name", state=state, index=self._index + 1)
        )
        self._update_style()

    @property
    def audio_path(self) -> str:
        return self._audio_path

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_starred(self) -> bool:
        return self._is_starred

    @property
    def favorite_key(self) -> str:
        if not self._audio_path:
            return ""
        normalized_path = os.path.normcase(os.path.abspath(self._audio_path))
        return f"{normalized_path}|{self._seed}"

    @property
    def quality_score(self) -> float:
        return self._quality_score

    def set_quality_score(self, score: float):
        self._quality_score = score
        self._score_label.setText(tr("batch.quality_value", score=score))
        if score >= 70:
            self._score_label.setStyleSheet(f"color: {Palette.GREEN}; font-size: 7.5pt; font-weight: bold;")
        elif score >= 40:
            self._score_label.setStyleSheet(f"color: {Palette.YELLOW}; font-size: 7.5pt; font-weight: bold;")
        else:
            self._score_label.setStyleSheet(f"color: {Palette.RED}; font-size: 7.5pt; font-weight: bold;")

    @property
    def index(self) -> int:
        return self._index


class BatchView(QWidget):
    """
    Grid view of batch-generated song variations.
    Shows mini waveform cards with playback, star, and delete.
    """
    play_requested = Signal(str)
    use_result = Signal(str)  # audio_path of selected result

    def __init__(self, parent=None, toast_mgr=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._cards: list[BatchCard] = []
        self._playing_index = -1
        self._job_store = JobStore()
        self._starred_keys = self._load_starred_keys()
        self._setup_ui()
        # Startup recovery is intentionally separate from later banner refreshes.
        self._job_store.recover_stale_jobs()
        self.refresh_recoverable_jobs()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel(tr("batch.title"))
        title.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 9.75pt;")
        header.addWidget(title)

        self._count_label = QLabel(tr("batch.count", count=0))
        self._count_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 8.25pt;")
        header.addWidget(self._count_label)

        header.addStretch()

        self._use_best_btn = QPushButton(tr("batch.use_best"))
        self._use_best_btn.setMinimumHeight(28)
        self._use_best_btn.setEnabled(False)
        self._use_best_btn.clicked.connect(self._use_best)
        header.addWidget(self._use_best_btn)

        self._clear_btn = QPushButton(tr("batch.clear_all"))
        self._clear_btn.setMinimumHeight(28)
        self._clear_btn.setProperty("class", "secondary")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)

        layout.addLayout(header)

        self._recovery_label = QLabel("")
        self._recovery_label.setWordWrap(True)
        self._recovery_label.setVisible(False)
        self._recovery_label.setStyleSheet(
            f"background: rgba(249, 226, 175, 28); color: {Palette.YELLOW}; "
            "border: 1px solid rgba(249, 226, 175, 70); border-radius: 6px; "
            "padding: 7px 9px; font-size: 8.25pt;"
        )
        layout.addWidget(self._recovery_label)

        # Scrollable grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(8)

        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll, 1)

        # Empty state
        self._empty_label = QLabel(tr("batch.empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 9pt; padding: 40px;")
        self._grid_layout.addWidget(self._empty_label, 0, 0, 1, 2)

        install_accessibility(
            self,
            tr("batch.accessibility.name"),
            named_controls=[
                (self._use_best_btn, tr("batch.accessibility.use_best_name"), tr("batch.accessibility.use_best_description")),
                (self._clear_btn, tr("batch.accessibility.clear_name"), tr("batch.accessibility.clear_description")),
            ],
            tab_order=[self._use_best_btn, self._clear_btn],
        )

    def _reflow_cards(self):
        for index, card in enumerate(self._cards):
            card._index = index
            card._title.setText(tr("batch.variation", index=index + 1))
            self._grid_layout.addWidget(card, index // 2, index % 2)
        self._count_label.setText(tr("batch.count", count=len(self._cards)))
        self._use_best_btn.setEnabled(bool(self._cards))
        self._empty_label.setVisible(not self._cards)

    def add_result(
        self,
        audio_path: str,
        seed: int,
        gen_time: float = 0.0,
        *,
        index: Optional[int] = None,
        starred: Optional[bool] = None,
    ):
        """Add a new batch result card."""
        if self._empty_label.isVisible():
            self._empty_label.hide()

        idx = len(self._cards) if index is None else max(0, min(index, len(self._cards)))
        card = BatchCard(idx)
        card.set_result(audio_path, seed, gen_time)
        card.set_starred(
            card.favorite_key in self._starred_keys
            if starred is None else starred
        )
        card.play_requested.connect(self._on_play)
        card.star_toggled.connect(self._on_star_toggled)
        card.delete_requested.connect(self._on_delete)

        self._cards.insert(idx, card)
        self._reflow_cards()

    def set_results(self, results: list[dict]):
        """Set all results at once from batch generation."""
        self.clear()
        for r in results:
            self.add_result(
                r.get("audio_path", ""),
                r.get("seed", 0),
                r.get("generation_time", 0.0),
            )

    def _on_play(self, audio_path: str):
        # Reset all cards
        for i, card in enumerate(self._cards):
            card.set_playing(card.audio_path == audio_path)
            if card.audio_path == audio_path:
                self._playing_index = i

        self.play_requested.emit(audio_path)

    def _on_star_toggled(self, index: int, starred: bool):
        if not 0 <= index < len(self._cards):
            return
        key = self._cards[index].favorite_key
        if not key:
            return
        if starred:
            self._starred_keys.add(key)
        else:
            self._starred_keys.discard(key)
        self._save_starred_keys()

    @staticmethod
    def _starred_path() -> Path:
        return get_config_dir() / "batch_favorites.json"

    def _load_starred_keys(self) -> set[str]:
        try:
            data = json.loads(self._starred_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return set()
        if not isinstance(data, list):
            return set()
        return {item for item in data if isinstance(item, str) and item}

    def _save_starred_keys(self):
        path = self._starred_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_name(path.name + ".tmp")
            temp_path.write_text(
                json.dumps(sorted(self._starred_keys), indent=2),
                encoding="utf-8",
            )
            os.replace(temp_path, path)
        except OSError:
            # Starring should remain usable even when the config directory is read-only.
            pass

    def _snapshot_cards(self, cards: list[BatchCard]) -> list[dict]:
        snapshots = []
        for card in cards:
            if card not in self._cards:
                continue
            snapshots.append({
                "index": self._cards.index(card),
                "card": card,
                "audio_path": card.audio_path,
                "seed": card.seed,
                "generation_time": card._gen_time,
                "starred": card.is_starred,
                "entry": None,
            })
        return snapshots

    def _trash_snapshots(self, snapshots: list[dict]):
        from core.trash import TrashManager

        for snapshot in snapshots:
            snapshot["card"].cancel_quality_score()
        requests = []
        request_snapshots = []
        for snapshot in snapshots:
            path = snapshot["audio_path"]
            if not path or not os.path.exists(path):
                continue
            requests.append({
                "path": path,
                "category": "generated_asset",
                "label": os.path.basename(path),
                "metadata": {
                    "module": "song_forge_batch",
                    "seed": snapshot["seed"],
                    "generation_time": snapshot["generation_time"],
                    "starred": snapshot["starred"],
                },
            })
            request_snapshots.append(snapshot)

        entries = TrashManager().trash_paths(requests)
        for snapshot, entry in zip(request_snapshots, entries):
            snapshot["entry"] = entry
        return entries

    def _remove_snapshots(self, snapshots: list[dict]):
        for snapshot in sorted(snapshots, key=lambda item: item["index"], reverse=True):
            card = snapshot["card"]
            if card not in self._cards:
                continue
            card.cancel_quality_score()
            self._grid_layout.removeWidget(card)
            self._cards.remove(card)
            card.deleteLater()
        self._reflow_cards()

    def _restore_snapshots(self, snapshots: list[dict]):
        from core.trash import TrashManager

        trash = TrashManager()
        errors = []
        for snapshot in sorted(snapshots, key=lambda item: item["index"]):
            entry = snapshot.get("entry")
            if entry is None:
                continue
            try:
                if trash.get_entry(entry.id) is not None:
                    trash.restore(entry.id)
                elif not os.path.exists(snapshot["audio_path"]):
                    raise RuntimeError("trash entry is no longer available")
            except Exception as exc:
                errors.append(str(exc))

        for snapshot in sorted(snapshots, key=lambda item: item["index"]):
            if any(
                card.audio_path == snapshot["audio_path"]
                and card.seed == snapshot["seed"]
                for card in self._cards
            ):
                continue
            self.add_result(
                snapshot["audio_path"],
                snapshot["seed"],
                snapshot["generation_time"],
                index=snapshot["index"],
                starred=snapshot["starred"],
            )

        if errors:
            if self.toast_mgr:
                self.toast_mgr.error(tr("batch.restore_failed"))
        elif self.toast_mgr:
            self.toast_mgr.success(tr("batch.restored"))

    def _remove_with_undo(self, snapshots: list[dict], message: str):
        if not snapshots:
            return False
        try:
            self._trash_snapshots(snapshots)
        except Exception as exc:
            if self.toast_mgr:
                self.toast_mgr.error(tr("batch.delete_failed", error=exc))
            return False

        self._remove_snapshots(snapshots)
        if self.toast_mgr:
            self.toast_mgr.info(
                message,
                duration_ms=8000,
                action_label=tr("batch.undo"),
                action_callback=lambda items=snapshots: self._restore_snapshots(items),
            )
        return True

    def _on_delete(self, index: int):
        if 0 <= index < len(self._cards):
            self._remove_with_undo(
                self._snapshot_cards([self._cards[index]]),
                tr("batch.moved_to_trash"),
            )

    def _use_best(self):
        """Use the first starred result, or the highest quality-scored result."""
        for card in self._cards:
            if card.is_starred and card.audio_path:
                self.use_result.emit(card.audio_path)
                return
        scored = [c for c in self._cards if c.audio_path]
        if scored:
            best = max(scored, key=lambda c: c.quality_score)
            self.use_result.emit(best.audio_path)

    def get_starred(self) -> list[dict]:
        return [
            {"audio_path": c.audio_path, "seed": c.seed, "index": c.index}
            for c in self._cards if c.is_starred
        ]

    def clear(self):
        self._remove_with_undo(
            self._snapshot_cards(list(self._cards)),
            tr("batch.moved_to_trash_plural"),
        )
        self._playing_index = -1

    def refresh_recoverable_jobs(self):
        records = self._job_store.list_records(
            status=JobStatus.RECOVERABLE,
            kind="song_generation",
        )
        if not records:
            self._recovery_label.setVisible(False)
            return

        labels = [record.label for record in records[:3]]
        suffix = (
            tr("batch.recoverable_more", count=len(records) - 3)
            if len(records) > 3 else ""
        )
        self._recovery_label.setText(
            tr(
                "batch.recoverable",
                tasks=tr("runtime.recoverable_tasks"),
                labels=", ".join(labels),
                suffix=suffix,
            )
        )
        self._recovery_label.setVisible(True)

    @property
    def count(self) -> int:
        return len(self._cards)
