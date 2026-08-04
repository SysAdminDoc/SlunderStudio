"""
Slunder Studio — Seed Interpolation Explorer
2D grid where each cell represents a generation with varying parameters.
Progressive generation, click to play, star favorites, zoom into regions.
"""
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QComboBox,
    QSlider, QFileDialog, QStackedWidget,
    QLineEdit,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QKeyEvent

from ui.theme import Palette
from ui.accessibility import FOCUS_RING_COLOR, install_accessibility, set_accessible
from ui.widgets import EmptyStateWidget
from ui.waveform_widget import MiniWaveform
from core.provenance import sidecar_path_for
from core.workers import CancelledJobError, InferenceWorker


def _export_starred_task(
    starred: list[dict],
    destination: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Copy starred audio and sidecars away from the GUI thread."""
    destination_path = Path(destination)
    copied = sidecars = skipped = 0
    total = max(1, len(starred))
    for position, item in enumerate(starred, 1):
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledJobError("Starred export cancelled")
        source = Path(item["audio_path"])
        if not source.is_file():
            skipped += 1
            continue
        target = destination_path / (
            f"seed_{item['row']}_{item['col']}_{item['seed']}{source.suffix}"
        )
        try:
            if target.resolve() == source.resolve():
                target = target.with_name(f"{target.stem}_export{target.suffix}")
            shutil.copy2(source, target)
            copied += 1
            source_sidecar = sidecar_path_for(source)
            if source_sidecar.is_file():
                shutil.copy2(source_sidecar, sidecar_path_for(target))
                sidecars += 1
        except OSError:
            skipped += 1
        if progress_cb:
            progress_cb(int(position * 100 / total))
    return {"copied": copied, "sidecars": sidecars, "skipped": skipped}


class SeedCell(QFrame):
    """A single cell in the seed grid."""
    clicked = Signal(int, int)  # row, col
    play_requested = Signal(str)  # audio_path
    star_toggled = Signal(int, int, bool)

    def __init__(self, row: int, col: int, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self._audio_path = ""
        self._seed = 0
        self._is_starred = False
        self._is_generating = False
        self._is_generated = False
        self._is_playing = False

        self.setFixedSize(140, 110)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_style("idle")
        self._setup_ui()
        self._update_accessibility()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._waveform = MiniWaveform()
        self._waveform.setFixedHeight(50)
        self._waveform.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._waveform.clicked.connect(self._on_click)
        layout.addWidget(self._waveform)

        info = QHBoxLayout()
        info.setSpacing(2)
        self._seed_label = QLabel("")
        self._seed_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 9px;")
        info.addWidget(self._seed_label)

        info.addStretch()

        self._star_btn = QPushButton("")
        self._star_btn.setFixedSize(20, 20)
        self._star_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; color: {Palette.OVERLAY0}; font-size: 14px; }}"
            f"QPushButton:hover {{ color: {Palette.YELLOW}; }}"
        )
        self._star_btn.setText("\u2606")  # empty star
        self._star_btn.clicked.connect(self._toggle_star)
        self._star_btn.hide()
        info.addWidget(self._star_btn)

        layout.addLayout(info)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 10px;")
        layout.addWidget(self._status_label)
        self._set_star_accessibility()

    def _set_star_accessibility(self):
        action = "Unstar" if self._is_starred else "Star"
        set_accessible(
            self._star_btn,
            f"{action} seed variation",
            f"{action}s this variation in the starred export set.",
        )
        self._star_btn.setToolTip(f"{action} this generated seed variation")
        style = self._star_btn.styleSheet() or ""
        if ":focus" not in style:
            self._star_btn.setStyleSheet(
                f"{style}\nQPushButton:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}"
            )

    def _update_accessibility(self):
        status = ""
        if self._is_playing:
            status = " Playing."
        elif self._is_generating:
            status = " Generating."
        elif self._status_label.text() == "Failed":
            status = " Failed."
        seed = f" Seed {self._seed}." if self._is_generated else ""
        set_accessible(
            self,
            f"Seed variation row {self.row + 1}, column {self.col + 1}",
            "Press Enter or Space to play this variation. Press S to toggle its "
            f"favorite state.{seed}{status}",
        )

    def _update_style(self, state: str):
        focus_style = (
            f" QFrame:focus {{ border: 2px solid {FOCUS_RING_COLOR}; }}"
        )
        styles = {
            "idle": f"QFrame {{ background: {Palette.BASE}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; }}{focus_style}",
            "generating": f"QFrame {{ background: {Palette.BASE}; border: 2px solid {Palette.BLUE}; border-radius: 6px; }}{focus_style}",
            "done": f"QFrame {{ background: {Palette.BASE}; border: 1px solid {Palette.SURFACE1}; border-radius: 6px; }}{focus_style}",
            "starred": f"QFrame {{ background: {Palette.BASE}; border: 2px solid {Palette.YELLOW}; border-radius: 6px; }}{focus_style}",
            "playing": f"QFrame {{ background: {Palette.BASE}; border: 2px solid {Palette.GREEN}; border-radius: 6px; }}{focus_style}",
        }
        self.setStyleSheet(styles.get(state, styles["idle"]))

    def set_generating(self):
        self._is_playing = False
        self._is_generating = True
        self._status_label.setText("Generating...")
        self._update_style("generating")
        self._update_accessibility()

    def set_result(self, audio_path: str, seed: int):
        self._audio_path = audio_path
        self._seed = seed
        self._is_generating = False
        self._is_generated = True
        self._is_playing = False
        self._seed_label.setText(f"seed: {seed}")
        self._status_label.setText("")
        self._status_label.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 10px;")
        self._star_btn.show()
        self._update_style("done")
        self._set_star_accessibility()
        self._update_accessibility()

        # Load waveform
        try:
            self._waveform.load_audio(audio_path)
        except Exception:
            pass

    def set_failed(self, error: str = ""):
        self._is_playing = False
        self._is_generating = False
        self._status_label.setText("Failed")
        self._status_label.setStyleSheet(f"color: {Palette.RED}; font-size: 10px;")
        self._update_style("idle")
        self._update_accessibility()

    def _on_click(self):
        if self._audio_path:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
            self.play_requested.emit(self._audio_path)
            self._is_playing = True
            self._status_label.setText("▶ Playing")
            self._update_style("playing")
            self._update_accessibility()
        self.clicked.emit(self.row, self.col)

    def keyPressEvent(self, event: QKeyEvent):
        """Play or favorite a generated variation without a pointer."""
        key = event.key()
        modifiers = event.modifiers()
        if key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._on_click()
            event.accept()
            return
        if key == Qt.Key_S and not modifiers & (
            Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier
        ):
            if self._is_generated:
                self._toggle_star()
            event.accept()
            return
        super().keyPressEvent(event)

    def _toggle_star(self):
        self._is_starred = not self._is_starred
        self._star_btn.setText("\u2605" if self._is_starred else "\u2606")
        self._star_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; "
            f"color: {Palette.YELLOW if self._is_starred else Palette.OVERLAY0}; font-size: 14px; }}"
            f" QPushButton:hover {{ color: {Palette.YELLOW}; }}"
        )
        self._update_style("starred" if self._is_starred else "done")
        self._set_star_accessibility()
        self._update_accessibility()
        self.star_toggled.emit(self.row, self.col, self._is_starred)

    def reset_playing(self):
        if self._is_generated:
            self._is_playing = False
            self._status_label.setText("")
            self._update_style("starred" if self._is_starred else "done")
            self._update_accessibility()

    @property
    def audio_path(self) -> str:
        return self._audio_path

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_starred(self) -> bool:
        return self._is_starred

    def snapshot_state(self) -> dict:
        """Capture the visible cell state for an in-memory undo."""
        return {
            "audio_path": self._audio_path,
            "seed": self._seed,
            "is_starred": self._is_starred,
            "is_generating": self._is_generating,
            "is_generated": self._is_generated,
            "status": self._status_label.text(),
            "status_style": self._status_label.styleSheet(),
        }

    def restore_state(self, state: dict):
        """Restore a previously captured state without starting I/O."""
        self._audio_path = str(state.get("audio_path", ""))
        self._seed = int(state.get("seed", 0))
        self._is_starred = bool(state.get("is_starred", False))
        self._is_generating = bool(state.get("is_generating", False))
        self._is_generated = bool(state.get("is_generated", False))
        self._is_playing = False
        self._seed_label.setText(
            f"seed: {self._seed}" if self._is_generated else ""
        )
        self._status_label.setText(str(state.get("status", "")))
        self._status_label.setStyleSheet(
            state.get("status_style")
            or f"color: {Palette.OVERLAY0}; font-size: 10px;"
        )
        if self._is_generated:
            self._star_btn.show()
            if self._audio_path:
                try:
                    self._waveform.load_audio(self._audio_path)
                except Exception:
                    pass
            self._update_style("starred" if self._is_starred else "done")
        else:
            self._star_btn.hide()
            self._update_style("generating" if self._is_generating else "idle")
        self._star_btn.setText("\u2605" if self._is_starred else "\u2606")
        self._set_star_accessibility()
        self._update_accessibility()


class SeedExplorer(QWidget):
    """
    2D grid seed interpolation explorer.
    X-axis: seed range, Y-axis: ACE-Step timestep shift.
    Each cell generates with those parameters and shows a mini waveform.
    """
    generate_requested = Signal(list)  # list of param dicts for batch generation
    play_requested = Signal(str)  # audio path
    zoom_requested = Signal(int, int)  # row, col to zoom into

    def __init__(self, parent=None, toast_mgr=None):
        super().__init__(parent)
        self.toast_mgr = toast_mgr
        self._grid_size = 3  # 3x3 default
        self._cells: list[list[SeedCell]] = []
        self._export_worker = None
        self._export_workers = set()
        self._center_seed = 42
        self._seed_range = 100
        self._shift_min = 1.0
        self._shift_max = 3.0
        self._ignore_active_generation_results = False
        self._last_replaced_grid_snapshot = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        lbl = QLabel("Seed Explorer")
        lbl.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 13px;")
        ctrl.addWidget(lbl)

        ctrl.addWidget(QLabel("Grid:"))
        self._grid_combo = QComboBox()
        self._grid_combo.addItems(["2x2", "3x3", "4x4"])
        self._grid_combo.setCurrentIndex(1)
        self._grid_combo.setFixedWidth(70)
        self._grid_combo.currentIndexChanged.connect(self._rebuild_grid)
        ctrl.addWidget(self._grid_combo)

        ctrl.addWidget(QLabel("Center seed:"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2**31 - 1)
        self._seed_spin.setValue(42)
        self._seed_spin.setFixedWidth(100)
        ctrl.addWidget(self._seed_spin)

        ctrl.addWidget(QLabel("Seed range:"))
        self._range_spin = QSpinBox()
        self._range_spin.setRange(1, 10000)
        self._range_spin.setValue(100)
        self._range_spin.setFixedWidth(80)
        self._range_spin.valueChanged.connect(self._on_range_spin_changed)
        ctrl.addWidget(self._range_spin)

        ctrl.addWidget(QLabel("Distance:"))
        self._distance_slider = QSlider(Qt.Horizontal)
        self._distance_slider.setRange(1, 10000)
        self._distance_slider.setValue(100)
        self._distance_slider.setFixedWidth(110)
        self._distance_slider.setToolTip("How far generated variants can drift from the center seed")
        self._distance_slider.valueChanged.connect(self._on_distance_changed)
        ctrl.addWidget(self._distance_slider)

        ctrl.addWidget(QLabel("Shift:"))
        self._shift_min_spin = QDoubleSpinBox()
        self._shift_min_spin.setRange(1.0, 3.0)
        self._shift_min_spin.setValue(1.0)
        self._shift_min_spin.setSingleStep(1.0)
        self._shift_min_spin.setDecimals(1)
        self._shift_min_spin.setFixedWidth(65)
        ctrl.addWidget(self._shift_min_spin)

        ctrl.addWidget(QLabel("-"))
        self._shift_max_spin = QDoubleSpinBox()
        self._shift_max_spin.setRange(1.0, 3.0)
        self._shift_max_spin.setValue(3.0)
        self._shift_max_spin.setSingleStep(1.0)
        self._shift_max_spin.setDecimals(1)
        self._shift_max_spin.setFixedWidth(65)
        ctrl.addWidget(self._shift_max_spin)

        ctrl.addStretch()

        self._explore_btn = QPushButton("Explore")
        self._explore_btn.setFixedHeight(30)
        self._explore_btn.clicked.connect(self._start_exploration)
        ctrl.addWidget(self._explore_btn)

        self._export_btn = QPushButton("Export Starred")
        self._export_btn.setFixedHeight(30)
        self._export_btn.setProperty("class", "secondary")
        self._export_btn.clicked.connect(self._export_starred)
        ctrl.addWidget(self._export_btn)

        layout.addLayout(ctrl)

        # Axis labels
        axis_layout = QHBoxLayout()
        axis_layout.addSpacing(30)
        self._x_label = QLabel("Seed -->")
        self._x_label.setStyleSheet(f"color: {Palette.BLUE}; font-size: 10px;")
        axis_layout.addWidget(self._x_label)
        axis_layout.addStretch()
        layout.addLayout(axis_layout)

        # Grid area
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(6)

        # Y-axis label
        y_label = QLabel("SHIFT\n||\nV")
        y_label.setStyleSheet(f"color: {Palette.YELLOW}; font-size: 10px;")
        y_label.setAlignment(Qt.AlignCenter)
        self._grid_layout.addWidget(y_label, 0, 0, self._grid_size, 1)

        self._grid_empty = EmptyStateWidget(
            "No seed variations yet",
            "Adjust the seed range and timestep shift, then explore to compare generated variations.",
            "Explore seeds",
        )
        self._grid_empty.action_requested.connect(self._explore_btn.click)
        self._grid_stack = QStackedWidget()
        self._grid_stack.addWidget(self._grid_widget)
        self._grid_stack.addWidget(self._grid_empty)
        self._grid_stack.setCurrentWidget(self._grid_empty)
        self._scroll.setWidget(self._grid_stack)
        layout.addWidget(self._scroll, 1)

        # Info bar
        self._info = QLabel("Configure grid parameters and click Explore to generate variations")
        self._info.setStyleSheet(f"color: {Palette.OVERLAY0}; font-size: 11px;")
        layout.addWidget(self._info)

        self._rebuild_grid(1)  # Start with 3x3

        install_accessibility(
            self,
            "Seed Explorer",
            named_controls=[
                (self._grid_combo, "Grid size", "Selects the number of seed variations in each dimension."),
                (self._seed_spin, "Center seed", "Sets the seed at the center of the exploration grid."),
                (self._seed_spin.findChild(QLineEdit), "Center seed value", "Edits the center seed value."),
                (self._range_spin, "Seed range", "Sets the seed distance across the exploration grid."),
                (self._range_spin.findChild(QLineEdit), "Seed range value", "Edits the seed range value."),
                (self._distance_slider, "Seed distance", "Adjusts how far variations can drift from the center seed."),
                (self._shift_min_spin, "Minimum timestep shift", "Sets the lowest timestep shift for generated variations."),
                (self._shift_min_spin.findChild(QLineEdit), "Minimum shift value", "Edits the minimum timestep shift."),
                (self._shift_max_spin, "Maximum timestep shift", "Sets the highest timestep shift for generated variations."),
                (self._shift_max_spin.findChild(QLineEdit), "Maximum shift value", "Edits the maximum timestep shift."),
                (self._explore_btn, "Explore seeds", "Generates the configured seed variation grid."),
                (self._export_btn, "Export starred seeds", "Copies starred audio variations and provenance to a selected folder."),
                (self._info, "Seed generation status", "Reports generation, playback, and export progress."),
                (self._grid_empty.action_button, "Explore seeds from empty state", "Starts the configured seed variation grid."),
            ],
            tab_order=[],
            include_descendants=False,
        )
        self._set_cell_tab_order()

    def _set_cell_tab_order(self):
        """Keep every generated cell in the keyboard traversal order."""
        controls = [
            self._grid_combo,
            self._seed_spin,
            self._range_spin,
            self._distance_slider,
            self._shift_min_spin,
            self._shift_max_spin,
            self._explore_btn,
            self._export_btn,
        ]
        cells = [cell for row in self._cells for cell in row]
        for first, second in zip(controls + cells, controls[1:] + cells):
            QWidget.setTabOrder(first, second)

    def _rebuild_grid(self, index: int = None, *, _show_undo: bool = True):
        """Rebuild the grid with new size."""
        previous = self.snapshot_grid() if self._cells else None
        sizes = [2, 3, 4]
        if index is not None:
            self._grid_size = sizes[min(index, 2)]

        # Clear existing cells
        for row in self._cells:
            for cell in row:
                self._grid_layout.removeWidget(cell)
                cell.deleteLater()
        self._cells.clear()

        # Build new grid
        for r in range(self._grid_size):
            row = []
            for c in range(self._grid_size):
                cell = SeedCell(r, c)
                cell.play_requested.connect(self.play_requested.emit)
                cell.clicked.connect(self._on_cell_clicked)
                self._grid_layout.addWidget(cell, r, c + 1)  # +1 for Y-axis label
                cell._set_star_accessibility()
                row.append(cell)
            self._cells.append(row)
        if hasattr(self, "_explore_btn"):
            self._set_cell_tab_order()
            if not previous or not previous.get("has_content"):
                self._grid_stack.setCurrentWidget(self._grid_empty)
        if (
            _show_undo
            and previous
            and previous["has_content"]
            and self.toast_mgr
        ):
            self._last_replaced_grid_snapshot = previous
            self.toast_mgr.info(
                "Seed grid replaced.",
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda item=previous: self.restore_grid(item),
            )

    def snapshot_grid(self) -> dict:
        """Capture the current grid, including generated paths and stars."""
        cells = {
            (row, col): self._cells[row][col].snapshot_state()
            for row in range(len(self._cells))
            for col in range(len(self._cells[row]))
        }
        return {
            "grid_size": self._grid_size,
            "cells": cells,
            "info": self._info.text() if hasattr(self, "_info") else "",
            "has_content": any(
                state["is_generated"] or state["audio_path"] or state["is_starred"]
                for state in cells.values()
            ),
        }

    def restore_grid(self, snapshot: dict):
        """Restore a grid snapshot and ignore stale results from its replacement."""
        if not isinstance(snapshot, dict):
            return
        self._ignore_active_generation_results = True
        target_size = int(snapshot.get("grid_size", self._grid_size))
        if target_size != self._grid_size:
            index = {2: 0, 3: 1, 4: 2}.get(target_size, 1)
            self._grid_combo.blockSignals(True)
            self._grid_combo.setCurrentIndex(index)
            self._grid_combo.blockSignals(False)
            self._rebuild_grid(index, _show_undo=False)
        for (row, col), state in snapshot.get("cells", {}).items():
            if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
                self._cells[row][col].restore_state(state)
        self._last_replaced_grid_snapshot = snapshot
        self._set_info(snapshot.get("info", ""))

    def _set_info(self, text: str):
        """Update the persistent status line and announce its new value."""
        self._info.setText(text)
        self._info.setAccessibleDescription(text)
        try:
            from PySide6.QtGui import QAccessible, QAccessibleValueChangeEvent

            if QAccessible.isActive():
                QAccessible.updateAccessibility(
                    QAccessibleValueChangeEvent(self._info, text)
                )
        except Exception:
            # Accessibility announcements are best-effort and must not break UI updates.
            pass

    def _start_exploration(self):
        """Generate parameters for each grid cell and emit generation request."""
        previous = self.snapshot_grid()
        self._ignore_active_generation_results = False
        self._grid_stack.setCurrentWidget(self._grid_widget)
        center_seed = self._seed_spin.value()
        seed_range = self._range_spin.value()
        shift_min = self._shift_min_spin.value()
        shift_max = self._shift_max_spin.value()

        n = self._grid_size
        params_list = []

        for r in range(n):
            shift = shift_min + (shift_max - shift_min) * r / max(1, n - 1)
            for c in range(n):
                seed_offset = -seed_range // 2 + int(seed_range * c / max(1, n - 1))
                seed = center_seed + seed_offset

                self._cells[r][c].set_generating()
                params_list.append({
                    "row": r, "col": c,
                    "seed": seed,
                    "shift": round(shift, 2),
                })

        self._set_info(f"Generating {len(params_list)} variations...")
        self.generate_requested.emit(params_list)
        if previous["has_content"] and self.toast_mgr:
            self._last_replaced_grid_snapshot = previous
            self.toast_mgr.info(
                "Previous seed grid replaced.",
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda item=previous: self.restore_grid(item),
            )

    def _on_distance_changed(self, value: int):
        if self._range_spin.value() != value:
            self._range_spin.setValue(value)

    def _on_range_spin_changed(self, value: int):
        if self._distance_slider.value() != value:
            self._distance_slider.setValue(value)

    def set_cell_result(self, row: int, col: int, audio_path: str, seed: int):
        """Set result for a specific grid cell."""
        if self._ignore_active_generation_results:
            return
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            self._cells[row][col].set_result(audio_path, seed)
            self._grid_stack.setCurrentWidget(self._grid_widget)
            # Count completed
            done = sum(1 for r in self._cells for c in r if c.audio_path)
            total = self._grid_size ** 2
            self._set_info(f"Generated {done}/{total} variations")

    def set_cell_failed(self, row: int, col: int, error: str = ""):
        if self._ignore_active_generation_results:
            return
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            self._cells[row][col].set_failed(error)

    def _on_cell_clicked(self, row: int, col: int):
        """Reset all cells playing state, highlight clicked."""
        for r in self._cells:
            for c in r:
                if (c.row, c.col) != (row, col):
                    c.reset_playing()

    def _export_starred(self):
        """Copy starred audio results and any adjacent provenance sidecars."""
        starred = []
        for r in self._cells:
            for c in r:
                if c.is_starred and c.audio_path:
                    starred.append({
                        "audio_path": c.audio_path,
                        "seed": c.seed,
                        "row": c.row,
                        "col": c.col,
                    })
        if starred:
            destination = QFileDialog.getExistingDirectory(
                self, "Export Starred Variations"
            )
            if not destination:
                return

            if self._export_worker is not None and self._export_worker.isRunning():
                return
            worker = InferenceWorker(_export_starred_task, starred, destination)
            worker.progress.connect(
                lambda pct: self._set_info(f"Exporting starred variations... {pct}%")
            )
            worker.finished.connect(
                lambda result, d=destination: self._on_starred_export_finished(result, d)
            )
            worker.error.connect(self._on_starred_export_error)
            worker.cancelled.connect(self._on_starred_export_cancelled)
            self._export_workers.add(worker)
            self._export_worker = worker
            self._export_btn.setEnabled(False)
            self._set_info("Exporting starred variations... 0%")
            worker.start()
        else:
            self._set_info("No starred cells to export")

    def _release_export_worker_later(self, worker):
        if worker is None:
            return
        if worker.isRunning():
            QTimer.singleShot(10, lambda: self._release_export_worker_later(worker))
            return
        self._export_workers.discard(worker)
        if self._export_worker is worker:
            self._export_worker = None
        self._export_btn.setEnabled(True)

    def _on_starred_export_finished(self, result: dict, destination: str):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        copied = result["copied"]
        if copied:
            detail = f"; skipped {result['skipped']}" if result["skipped"] else ""
            self._set_info(
                f"Exported {copied} starred variation(s) to {destination}"
                f" ({result['sidecars']} provenance sidecar(s){detail})"
            )
        else:
            self._set_info("No starred audio files were available to export")

    def _on_starred_export_error(self, message: str):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        self._set_info(f"Starred export failed: {message}")

    def _on_starred_export_cancelled(self):
        worker = self._export_worker
        self._release_export_worker_later(worker)
        self._export_worker = None
        self._set_info("Starred export cancelled")

    def zoom_into(self, row: int, col: int):
        """Zoom into a cell - re-center seed and narrow ranges."""
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            cell = self._cells[row][col]
            if cell.seed:
                self._seed_spin.setValue(cell.seed)
                self._range_spin.setValue(max(5, self._range_spin.value() // 3))
                shift = self._shift_min_spin.value() + (
                    (self._shift_max_spin.value() - self._shift_min_spin.value())
                    * row
                    / max(1, self._grid_size - 1)
                )
                spread = 1.0
                self._shift_min_spin.setValue(max(1.0, shift - spread))
                self._shift_max_spin.setValue(min(3.0, shift + spread))
