"""
Slunder Studio v0.0.2 — Seed Interpolation Explorer
2D grid where each cell represents a generation with varying parameters.
Progressive generation, click to play, star favorites, zoom into regions.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QFrame, QScrollArea, QComboBox,
)
from PySide6.QtCore import Signal, Qt

from ui.waveform_widget import MiniWaveform


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

        self.setFixedSize(140, 110)
        self.setCursor(Qt.PointingHandCursor)
        self._update_style("idle")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._waveform = MiniWaveform()
        self._waveform.setFixedHeight(50)
        self._waveform.clicked.connect(self._on_click)
        layout.addWidget(self._waveform)

        info = QHBoxLayout()
        info.setSpacing(2)
        self._seed_label = QLabel("")
        self._seed_label.setStyleSheet("color: #6C7086; font-size: 9px;")
        info.addWidget(self._seed_label)

        info.addStretch()

        self._star_btn = QPushButton("")
        self._star_btn.setFixedSize(20, 20)
        self._star_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #6C7086; font-size: 14px; }"
            "QPushButton:hover { color: #F9E2AF; }"
        )
        self._star_btn.setText("\u2606")  # empty star
        self._star_btn.clicked.connect(self._toggle_star)
        self._star_btn.hide()
        info.addWidget(self._star_btn)

        layout.addLayout(info)

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("color: #6C7086; font-size: 10px;")
        layout.addWidget(self._status_label)

    def _update_style(self, state: str):
        styles = {
            "idle": "QFrame { background: #1E1E2E; border: 1px solid #313244; border-radius: 6px; }",
            "generating": "QFrame { background: #1E1E2E; border: 2px solid #89B4FA; border-radius: 6px; }",
            "done": "QFrame { background: #1E1E2E; border: 1px solid #45475A; border-radius: 6px; }",
            "starred": "QFrame { background: #1E1E2E; border: 2px solid #F9E2AF; border-radius: 6px; }",
            "playing": "QFrame { background: #1E1E2E; border: 2px solid #A6E3A1; border-radius: 6px; }",
        }
        self.setStyleSheet(styles.get(state, styles["idle"]))

    def set_generating(self):
        self._is_generating = True
        self._status_label.setText("Generating...")
        self._update_style("generating")

    def set_result(self, audio_path: str, seed: int):
        self._audio_path = audio_path
        self._seed = seed
        self._is_generating = False
        self._is_generated = True
        self._seed_label.setText(f"seed: {seed}")
        self._status_label.setText("")
        self._star_btn.show()
        self._update_style("done")

        # Load waveform
        try:
            self._waveform.load_file(audio_path)
        except Exception:
            pass

    def set_failed(self, error: str = ""):
        self._is_generating = False
        self._status_label.setText("Failed")
        self._status_label.setStyleSheet("color: #F38BA8; font-size: 10px;")
        self._update_style("idle")

    def _on_click(self):
        if self._audio_path:
            self.play_requested.emit(self._audio_path)
            self._update_style("playing")
        self.clicked.emit(self.row, self.col)

    def _toggle_star(self):
        self._is_starred = not self._is_starred
        self._star_btn.setText("\u2605" if self._is_starred else "\u2606")
        self._star_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {'#F9E2AF' if self._is_starred else '#6C7086'}; font-size: 14px; }}"
            "QPushButton:hover { color: #F9E2AF; }"
        )
        self._update_style("starred" if self._is_starred else "done")
        self.star_toggled.emit(self.row, self.col, self._is_starred)

    def reset_playing(self):
        if self._is_generated:
            self._update_style("starred" if self._is_starred else "done")

    @property
    def audio_path(self) -> str:
        return self._audio_path

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def is_starred(self) -> bool:
        return self._is_starred


class SeedExplorer(QWidget):
    """
    2D grid seed interpolation explorer.
    X-axis: seed range, Y-axis: CFG scale (or custom parameter).
    Each cell generates with those parameters and shows a mini waveform.
    """
    generate_requested = Signal(list)  # list of param dicts for batch generation
    play_requested = Signal(str)  # audio path
    zoom_requested = Signal(int, int)  # row, col to zoom into

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid_size = 3  # 3x3 default
        self._cells: list[list[SeedCell]] = []
        self._center_seed = 42
        self._seed_range = 100
        self._cfg_min = 3.0
        self._cfg_max = 8.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        lbl = QLabel("Seed Explorer")
        lbl.setStyleSheet("color: #CDD6F4; font-weight: bold; font-size: 13px;")
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
        ctrl.addWidget(self._range_spin)

        ctrl.addWidget(QLabel("CFG:"))
        self._cfg_min_spin = QDoubleSpinBox()
        self._cfg_min_spin.setRange(1.0, 15.0)
        self._cfg_min_spin.setValue(3.0)
        self._cfg_min_spin.setSingleStep(0.5)
        self._cfg_min_spin.setFixedWidth(65)
        ctrl.addWidget(self._cfg_min_spin)

        ctrl.addWidget(QLabel("-"))
        self._cfg_max_spin = QDoubleSpinBox()
        self._cfg_max_spin.setRange(1.0, 15.0)
        self._cfg_max_spin.setValue(8.0)
        self._cfg_max_spin.setSingleStep(0.5)
        self._cfg_max_spin.setFixedWidth(65)
        ctrl.addWidget(self._cfg_max_spin)

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
        self._x_label.setStyleSheet("color: #89B4FA; font-size: 10px;")
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
        y_label = QLabel("CFG\n||\nV")
        y_label.setStyleSheet("color: #F9E2AF; font-size: 10px;")
        y_label.setAlignment(Qt.AlignCenter)
        self._grid_layout.addWidget(y_label, 0, 0, self._grid_size, 1)

        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll, 1)

        # Info bar
        self._info = QLabel("Configure grid parameters and click Explore to generate variations")
        self._info.setStyleSheet("color: #6C7086; font-size: 11px;")
        layout.addWidget(self._info)

        self._rebuild_grid(1)  # Start with 3x3

    def _rebuild_grid(self, index: int = None):
        """Rebuild the grid with new size."""
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
                row.append(cell)
            self._cells.append(row)

    def _start_exploration(self):
        """Generate parameters for each grid cell and emit generation request."""
        center_seed = self._seed_spin.value()
        seed_range = self._range_spin.value()
        cfg_min = self._cfg_min_spin.value()
        cfg_max = self._cfg_max_spin.value()

        n = self._grid_size
        params_list = []

        for r in range(n):
            cfg = cfg_min + (cfg_max - cfg_min) * r / max(1, n - 1)
            for c in range(n):
                seed_offset = -seed_range // 2 + int(seed_range * c / max(1, n - 1))
                seed = center_seed + seed_offset

                self._cells[r][c].set_generating()
                params_list.append({
                    "row": r, "col": c,
                    "seed": seed,
                    "cfg_scale": round(cfg, 2),
                })

        self._info.setText(f"Generating {len(params_list)} variations...")
        self.generate_requested.emit(params_list)

    def set_cell_result(self, row: int, col: int, audio_path: str, seed: int):
        """Set result for a specific grid cell."""
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            self._cells[row][col].set_result(audio_path, seed)
            # Count completed
            done = sum(1 for r in self._cells for c in r if c.audio_path)
            total = self._grid_size ** 2
            self._info.setText(f"Generated {done}/{total} variations")

    def set_cell_failed(self, row: int, col: int, error: str = ""):
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            self._cells[row][col].set_failed(error)

    def _on_cell_clicked(self, row: int, col: int):
        """Reset all cells playing state, highlight clicked."""
        for r in self._cells:
            for c in r:
                c.reset_playing()

    def _export_starred(self):
        """Get all starred results."""
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
            self._info.setText(f"Exporting {len(starred)} starred variations...")
        else:
            self._info.setText("No starred cells to export")

    def zoom_into(self, row: int, col: int):
        """Zoom into a cell - re-center seed and narrow ranges."""
        if 0 <= row < len(self._cells) and 0 <= col < len(self._cells[row]):
            cell = self._cells[row][col]
            if cell.seed:
                self._seed_spin.setValue(cell.seed)
                self._range_spin.setValue(max(5, self._range_spin.value() // 3))
                cfg = self._cfg_min_spin.value() + (
                    (self._cfg_max_spin.value() - self._cfg_min_spin.value()) * row / max(1, self._grid_size - 1)
                )
                spread = 1.0
                self._cfg_min_spin.setValue(max(1.0, cfg - spread))
                self._cfg_max_spin.setValue(min(15.0, cfg + spread))
