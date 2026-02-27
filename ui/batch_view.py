"""
Slunder Studio v0.0.2 — Batch View
Grid display for batch-generated song variations.
Mini waveform cards with one-click playback, star/rank, delete, and "Best of" refinement.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
    QFrame, QScrollArea, QSpinBox,
)
from PySide6.QtCore import Signal, Qt

from ui.waveform_widget import MiniWaveform


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

        self._title = QLabel(f"Variation {self._index + 1}")
        self._title.setStyleSheet("color: #CDD6F4; font-weight: bold; font-size: 12px;")
        header.addWidget(self._title)

        header.addStretch()

        self._star_btn = QPushButton("\u2606")
        self._star_btn.setFixedSize(24, 24)
        self._star_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #6C7086; font-size: 16px; }"
            "QPushButton:hover { color: #F9E2AF; }"
        )
        self._star_btn.clicked.connect(self._toggle_star)
        header.addWidget(self._star_btn)

        self._delete_btn = QPushButton("\u2715")
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; color: #6C7086; font-size: 14px; }"
            "QPushButton:hover { color: #F38BA8; }"
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
        self._seed_label.setStyleSheet("color: #6C7086; font-size: 10px;")
        info.addWidget(self._seed_label)

        info.addStretch()

        self._time_label = QLabel("")
        self._time_label.setStyleSheet("color: #6C7086; font-size: 10px;")
        info.addWidget(self._time_label)

        layout.addLayout(info)

    def _update_style(self):
        if self._is_playing:
            border = "2px solid #A6E3A1"
        elif self._is_starred:
            border = "2px solid #F9E2AF"
        else:
            border = "1px solid #313244"

        self.setStyleSheet(
            f"QFrame {{ background: #1E1E2E; border: {border}; border-radius: 8px; }}"
        )

    def set_result(self, audio_path: str, seed: int, gen_time: float = 0.0):
        self._audio_path = audio_path
        self._seed = seed
        self._gen_time = gen_time
        self._seed_label.setText(f"seed: {seed}")
        if gen_time > 0:
            self._time_label.setText(f"{gen_time:.1f}s")

        try:
            self._waveform.load_file(audio_path)
        except Exception:
            pass

    def _on_play(self):
        if self._audio_path:
            self.play_requested.emit(self._audio_path)

    def set_playing(self, playing: bool):
        self._is_playing = playing
        self._update_style()

    def _toggle_star(self):
        self._is_starred = not self._is_starred
        self._star_btn.setText("\u2605" if self._is_starred else "\u2606")
        self._star_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; "
            f"color: {'#F9E2AF' if self._is_starred else '#6C7086'}; font-size: 16px; }}"
            "QPushButton:hover { color: #F9E2AF; }"
        )
        self._update_style()
        self.star_toggled.emit(self._index, self._is_starred)

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
    def index(self) -> int:
        return self._index


class BatchView(QWidget):
    """
    Grid view of batch-generated song variations.
    Shows mini waveform cards with playback, star, and delete.
    """
    play_requested = Signal(str)
    regenerate_similar = Signal(int)  # seed to regenerate around
    use_result = Signal(str)  # audio_path of selected result

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[BatchCard] = []
        self._playing_index = -1
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Batch Results")
        title.setStyleSheet("color: #CDD6F4; font-weight: bold; font-size: 13px;")
        header.addWidget(title)

        self._count_label = QLabel("0 variations")
        self._count_label.setStyleSheet("color: #6C7086; font-size: 11px;")
        header.addWidget(self._count_label)

        header.addStretch()

        self._use_best_btn = QPushButton("Use Best")
        self._use_best_btn.setFixedHeight(28)
        self._use_best_btn.setEnabled(False)
        self._use_best_btn.clicked.connect(self._use_best)
        header.addWidget(self._use_best_btn)

        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setFixedHeight(28)
        self._clear_btn.setProperty("class", "secondary")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)

        layout.addLayout(header)

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
        self._empty_label = QLabel("Generate batch variations to see results here")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #6C7086; font-size: 12px; padding: 40px;")
        self._grid_layout.addWidget(self._empty_label, 0, 0, 1, 2)

    def add_result(self, audio_path: str, seed: int, gen_time: float = 0.0):
        """Add a new batch result card."""
        if self._empty_label.isVisible():
            self._empty_label.hide()

        idx = len(self._cards)
        card = BatchCard(idx)
        card.set_result(audio_path, seed, gen_time)
        card.play_requested.connect(self._on_play)
        card.star_toggled.connect(self._on_star_toggled)
        card.delete_requested.connect(self._on_delete)

        row = idx // 2
        col = idx % 2
        self._grid_layout.addWidget(card, row, col)
        self._cards.append(card)

        self._count_label.setText(f"{len(self._cards)} variations")
        self._use_best_btn.setEnabled(True)

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
        pass

    def _on_delete(self, index: int):
        if 0 <= index < len(self._cards):
            card = self._cards[index]
            self._grid_layout.removeWidget(card)
            card.deleteLater()
            self._cards.pop(index)

            # Re-index remaining cards
            for i, c in enumerate(self._cards):
                c._index = i
                c._title.setText(f"Variation {i + 1}")

            # Re-layout
            for i, c in enumerate(self._cards):
                self._grid_layout.addWidget(c, i // 2, i % 2)

            self._count_label.setText(f"{len(self._cards)} variations")
            if not self._cards:
                self._empty_label.show()
                self._use_best_btn.setEnabled(False)

    def _use_best(self):
        """Use the first starred result, or the first result."""
        for card in self._cards:
            if card.is_starred and card.audio_path:
                self.use_result.emit(card.audio_path)
                return
        if self._cards and self._cards[0].audio_path:
            self.use_result.emit(self._cards[0].audio_path)

    def get_starred(self) -> list[dict]:
        return [
            {"audio_path": c.audio_path, "seed": c.seed, "index": c.index}
            for c in self._cards if c.is_starred
        ]

    def clear(self):
        for card in self._cards:
            self._grid_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()
        self._playing_index = -1
        self._count_label.setText("0 variations")
        self._use_best_btn.setEnabled(False)
        self._empty_label.show()

    @property
    def count(self) -> int:
        return len(self._cards)
