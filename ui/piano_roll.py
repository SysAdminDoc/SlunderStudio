"""
Slunder Studio v0.1.26 — Piano Roll Widget
QGraphicsView-based MIDI piano roll editor with note creation, editing,
selection, quantization, and snap-to-grid.
"""
from typing import Optional
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsLineItem, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QSpinBox, QDoubleSpinBox, QLabel, QPushButton, QGraphicsItem,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QColor, QPen, QBrush, QPainter, QWheelEvent, QMouseEvent, QKeyEvent,
)

from core.midi_utils import CCEvent, NoteData, TrackData, get_pitch_range
from ui.theme import ThemeEngine


# ── Constants ──────────────────────────────────────────────────────────────────

NOTE_HEIGHT = 14
PIXELS_PER_BEAT = 80
KEY_WIDTH = 48
MIN_PITCH = 21   # A0
MAX_PITCH = 108  # C8
TOTAL_KEYS = MAX_PITCH - MIN_PITCH + 1

SNAP_VALUES = {
    "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5,
    "1/16": 0.25, "1/32": 0.125, "Off": 0.0,
}

CC_LANES = {
    "Mod Wheel (CC1)": 1,
    "Volume (CC7)": 7,
    "Pan (CC10)": 10,
    "Expression (CC11)": 11,
    "Sustain (CC64)": 64,
}

# Piano key colors
BLACK_KEYS = {1, 3, 6, 8, 10}  # C#, D#, F#, G#, A#
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def is_black_key(pitch: int) -> bool:
    return (pitch % 12) in BLACK_KEYS


# ── Note Item ──────────────────────────────────────────────────────────────────

class NoteItem(QGraphicsRectItem):
    """Editable MIDI note rectangle on the piano roll."""

    def __init__(self, note_data: NoteData, tempo: float, parent_roll: "PianoRollScene"):
        self.note_data = note_data
        self._tempo = tempo
        self._roll = parent_roll
        self._dragging = False
        self._resizing = False
        self._drag_offset = QPointF()

        beat_dur = 60.0 / tempo
        x = note_data.start / beat_dur * PIXELS_PER_BEAT
        y = (MAX_PITCH - note_data.pitch) * NOTE_HEIGHT
        w = note_data.duration / beat_dur * PIXELS_PER_BEAT
        h = NOTE_HEIGHT - 1

        super().__init__(x, y, max(w, 4), h)

        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

        self._update_color()

    def _update_color(self):
        vel_factor = self.note_data.velocity / 127.0
        r = int(60 + 140 * vel_factor)
        g = int(180 - 60 * vel_factor)
        b = int(220 - 80 * vel_factor)
        base = QColor(r, g, b)
        if self.isSelected():
            base = QColor(255, 180, 60)
        self.setBrush(QBrush(base))
        self.setPen(QPen(base.darker(130), 1))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemSelectedChange:
            # Defer color update
            from PySide6.QtCore import QTimer
            QTimer.singleShot(10, self._update_color)
        return super().itemChange(change, value)

    def hoverMoveEvent(self, event):
        # Show resize cursor near right edge
        if event.pos().x() > self.rect().width() - 6:
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if event.pos().x() > self.rect().width() - 6:
                self._resizing = True
            else:
                self._dragging = True
                self._drag_offset = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            new_w = max(4, event.pos().x())
            rect = self.rect()
            rect.setWidth(new_w)
            self.setRect(rect)
            self._sync_to_data()
        elif self._dragging:
            delta = event.scenePos() - event.lastScenePos()
            new_x = self.pos().x() + self.rect().x() + delta.x()
            new_y = self.pos().y() + self.rect().y() + delta.y()

            # Snap
            snap = self._roll.snap_value
            if snap > 0:
                beat_dur = 60.0 / self._tempo
                grid_px = snap * PIXELS_PER_BEAT
                new_x = round(new_x / grid_px) * grid_px

            # Snap pitch
            new_y = round(new_y / NOTE_HEIGHT) * NOTE_HEIGHT

            # Clamp
            new_x = max(0, new_x)
            new_y = max(0, min((TOTAL_KEYS - 1) * NOTE_HEIGHT, new_y))

            rect = self.rect()
            rect.moveTopLeft(QPointF(new_x, new_y))
            self.setRect(rect)
            self._sync_to_data()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging = False
        self._resizing = False
        self._sync_to_data()
        self._roll.notes_changed.emit()
        super().mouseReleaseEvent(event)

    def _sync_to_data(self):
        """Update NoteData from visual position."""
        beat_dur = 60.0 / self._tempo
        rect = self.rect()
        self.note_data.start = max(0, rect.x() / PIXELS_PER_BEAT * beat_dur)
        self.note_data.pitch = max(MIN_PITCH, min(MAX_PITCH,
            MAX_PITCH - int(rect.y() / NOTE_HEIGHT)))
        duration = rect.width() / PIXELS_PER_BEAT * beat_dur
        self.note_data.end = self.note_data.start + max(0.01, duration)


# ── Piano Roll Scene ───────────────────────────────────────────────────────────

class PianoRollScene(QGraphicsScene):
    """Scene containing the piano roll grid and notes."""

    notes_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.track: Optional[TrackData] = None
        self.tempo: float = 120.0
        self.bars: int = 16
        self.snap_value: float = 0.25  # 1/16 note in beats
        self.default_velocity: int = 100
        self._note_items: list[NoteItem] = []
        self._drawing = False

        self._draw_grid()

    def _draw_grid(self):
        """Draw piano roll background grid."""
        t = ThemeEngine.get_colors()
        bg = QColor(t.get("background", "#0d1117"))
        grid_color = QColor(t.get("border", "#1e2733"))
        bar_color = QColor(t.get("accent", "#58a6ff")).darker(200)

        total_height = TOTAL_KEYS * NOTE_HEIGHT
        beat_dur = 60.0 / self.tempo
        bar_beats = 4  # assuming 4/4
        total_beats = self.bars * bar_beats
        total_width = total_beats * PIXELS_PER_BEAT

        self.setSceneRect(0, 0, total_width + KEY_WIDTH, total_height)

        # Piano keys background
        for i in range(TOTAL_KEYS):
            pitch = MAX_PITCH - i
            y = i * NOTE_HEIGHT
            if is_black_key(pitch):
                self.addRect(0, y, total_width, NOTE_HEIGHT,
                             QPen(Qt.NoPen), QBrush(bg.lighter(115)))
            else:
                self.addRect(0, y, total_width, NOTE_HEIGHT,
                             QPen(Qt.NoPen), QBrush(bg.lighter(105)))

            # Horizontal lines
            line = self.addLine(0, y, total_width, y, QPen(grid_color, 0.5))
            line.setZValue(1)

            # C note labels
            if pitch % 12 == 0:
                label = self.addText(f"C{pitch // 12 - 1}")
                label.setDefaultTextColor(QColor("#8b949e"))
                label.setPos(-42, y - 2)
                label.setZValue(20)
                # Brighter horizontal line at C
                bright = self.addLine(0, y, total_width, y, QPen(grid_color.lighter(150), 1))
                bright.setZValue(2)

        # Vertical beat/bar lines
        for beat in range(total_beats + 1):
            x = beat * PIXELS_PER_BEAT
            is_bar = beat % bar_beats == 0
            pen = QPen(bar_color if is_bar else grid_color, 1 if is_bar else 0.5)
            line = self.addLine(x, 0, x, total_height, pen)
            line.setZValue(2)

            # Bar numbers
            if is_bar:
                bar_num = beat // bar_beats + 1
                label = self.addText(str(bar_num))
                label.setDefaultTextColor(QColor("#58a6ff"))
                label.setPos(x + 3, -18)
                label.setZValue(20)

        # Sub-beat grid (16th notes)
        sub_pen = QPen(grid_color.darker(130), 0.3)
        for beat in range(total_beats):
            for sub in range(1, 4):
                x = (beat + sub * 0.25) * PIXELS_PER_BEAT
                line = self.addLine(x, 0, x, total_height, sub_pen)
                line.setZValue(1)

    def load_track(self, track: TrackData, tempo: float = 120.0, bars: int = 16):
        """Load a track's notes into the scene."""
        self.track = track
        self.tempo = tempo
        self.bars = bars

        # Remove old notes
        for item in self._note_items:
            self.removeItem(item)
        self._note_items.clear()

        # Recalculate grid
        self.clear()
        self._draw_grid()

        # Add notes
        for note in track.notes:
            item = NoteItem(note, tempo, self)
            self.addItem(item)
            self._note_items.append(item)

    def add_note(self, pitch: int, start: float, duration: float = 0.25,
                 velocity: int = 100) -> NoteItem:
        """Add a new note to the scene and track."""
        channel = self.track.channel if self.track else 0
        note = NoteData(
            pitch=pitch,
            start=start,
            end=start + duration,
            velocity=velocity,
            channel=channel,
        )
        if self.track:
            self.track.notes.append(note)
        item = NoteItem(note, self.tempo, self)
        self.addItem(item)
        self._note_items.append(item)
        self.notes_changed.emit()
        return item

    def delete_selected(self):
        """Remove selected notes."""
        to_remove = [item for item in self._note_items if item.isSelected()]
        for item in to_remove:
            if self.track and item.note_data in self.track.notes:
                self.track.notes.remove(item.note_data)
            self.removeItem(item)
            self._note_items.remove(item)
        if to_remove:
            self.notes_changed.emit()

    def select_all(self):
        for item in self._note_items:
            item.setSelected(True)

    def get_notes(self) -> list[NoteData]:
        """Get all notes from the scene."""
        return [item.note_data for item in self._note_items]

    def mousePressEvent(self, event):
        # If clicking on empty space with no modifiers, add a note
        item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else __import__("PySide6.QtGui", fromlist=["QTransform"]).QTransform())
        if item is None and event.button() == Qt.LeftButton:
            pos = event.scenePos()
            beat_dur = 60.0 / self.tempo
            pitch = MAX_PITCH - int(pos.y() / NOTE_HEIGHT)
            pitch = max(MIN_PITCH, min(MAX_PITCH, pitch))

            start_time = pos.x() / PIXELS_PER_BEAT * beat_dur
            if self.snap_value > 0:
                grid = self.snap_value * beat_dur
                start_time = round(start_time / grid) * grid

            self.add_note(
                pitch,
                max(0, start_time),
                self.snap_value * beat_dur if self.snap_value > 0 else 0.25 * beat_dur,
                self.default_velocity,
            )
            event.accept()
        else:
            super().mousePressEvent(event)


# ── Piano Roll View ────────────────────────────────────────────────────────────

class PianoRollView(QGraphicsView):
    """Scrollable, zoomable piano roll view."""

    def __init__(self, scene: PianoRollScene, parent=None):
        super().__init__(scene, parent)
        self._scene = scene
        self._zoom_x = 1.0
        self._zoom_y = 1.0

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        t = ThemeEngine.get_colors()
        bg = t.get("background", "#0d1117")
        self.setStyleSheet(f"""
            QGraphicsView {{
                background: {bg};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 6px;
            }}
        """)

        # Center on middle C
        middle_c_y = (MAX_PITCH - 60) * NOTE_HEIGHT
        self.centerOn(0, middle_c_y)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with Ctrl+scroll, scroll otherwise."""
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            if event.modifiers() & Qt.ShiftModifier:
                # Vertical zoom
                self._zoom_y *= factor
                self._zoom_y = max(0.3, min(3.0, self._zoom_y))
            else:
                # Horizontal zoom
                self._zoom_x *= factor
                self._zoom_x = max(0.2, min(5.0, self._zoom_x))
            self.resetTransform()
            self.scale(self._zoom_x, self._zoom_y)
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self._scene.delete_selected()
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self._scene.select_all()
        else:
            super().keyPressEvent(event)


# ── Piano Roll Widget (with toolbar) ──────────────────────────────────────────

class CCAutomationLane(QWidget):
    """Compact control-change lane editor for the loaded track."""

    cc_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._track: Optional[TrackData] = None
        self._tempo: float = 120.0

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        label = QLabel("CC Lane:")
        label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._controller_combo = QComboBox()
        for name, controller in CC_LANES.items():
            self._controller_combo.addItem(name, controller)
        self._controller_combo.currentIndexChanged.connect(self._refresh)

        beat_label = QLabel("Beat:")
        beat_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._beat_spin = QDoubleSpinBox()
        self._beat_spin.setRange(0.0, 512.0)
        self._beat_spin.setDecimals(2)
        self._beat_spin.setSingleStep(0.25)
        self._beat_spin.setValue(0.0)
        self._beat_spin.setFixedWidth(72)

        value_label = QLabel("Value:")
        value_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._value_spin = QSpinBox()
        self._value_spin.setRange(0, 127)
        self._value_spin.setValue(64)
        self._value_spin.setFixedWidth(58)

        btn_style = f"""
            QPushButton {{
                background: {t.get('surface', '#161b22')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: {t.get('surface_hover', '#1c2333')}; }}
        """
        self._add_cc_btn = QPushButton("Add CC")
        self._add_cc_btn.setStyleSheet(btn_style)
        self._add_cc_btn.clicked.connect(self._on_add_event)
        self._clear_lane_btn = QPushButton("Clear Lane")
        self._clear_lane_btn.setStyleSheet(btn_style)
        self._clear_lane_btn.clicked.connect(self._on_clear_lane)

        controls.addWidget(label)
        controls.addWidget(self._controller_combo)
        controls.addWidget(beat_label)
        controls.addWidget(self._beat_spin)
        controls.addWidget(value_label)
        controls.addWidget(self._value_spin)
        controls.addWidget(self._add_cc_btn)
        controls.addWidget(self._clear_lane_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Beat", "CC", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(94)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {t.get('background', '#0d1117')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 4px;
                gridline-color: {t.get('border', '#1e2733')};
                font-size: 11px;
            }}
            QHeaderView::section {{
                background: {t.get('surface', '#161b22')};
                color: {t.get('text_secondary', '#8b949e')};
                border: none;
                padding: 3px 6px;
            }}
        """)
        layout.addWidget(self._table)

    def load_track(self, track: TrackData, tempo: float):
        self._track = track
        self._tempo = tempo
        self._refresh()

    def _on_add_event(self):
        if self._track is None:
            return
        beat_dur = 60.0 / self._tempo
        event = CCEvent(
            controller=int(self._controller_combo.currentData()),
            value=self._value_spin.value(),
            time=round(self._beat_spin.value() * beat_dur, 6),
            channel=self._track.channel,
        )
        self._track.cc_events.append(event)
        self._track.cc_events.sort(key=lambda cc: (cc.time, cc.controller, cc.value))
        self._refresh()
        self.cc_changed.emit()

    def _on_clear_lane(self):
        if self._track is None:
            return
        controller = int(self._controller_combo.currentData())
        before = len(self._track.cc_events)
        self._track.cc_events = [
            event for event in self._track.cc_events
            if event.controller != controller
        ]
        if len(self._track.cc_events) != before:
            self._refresh()
            self.cc_changed.emit()

    def _refresh(self):
        if self._track is None:
            self._table.setRowCount(0)
            return
        controller = int(self._controller_combo.currentData())
        events = [event for event in self._track.cc_events if event.controller == controller]
        self._table.setRowCount(len(events))
        beat_dur = 60.0 / self._tempo
        for row, event in enumerate(events):
            beat = event.time / beat_dur if beat_dur > 0 else 0.0
            self._table.setItem(row, 0, QTableWidgetItem(f"{beat:.2f}"))
            self._table.setItem(row, 1, QTableWidgetItem(f"CC{event.controller}"))
            self._table.setItem(row, 2, QTableWidgetItem(str(event.value)))


class PianoRollWidget(QWidget):
    """Complete piano roll widget with toolbar controls."""

    notes_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = PianoRollScene()
        self._view = PianoRollView(self._scene)
        self._scene.notes_changed.connect(self.notes_changed.emit)

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Snap selector
        snap_label = QLabel("Snap:")
        snap_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._snap_combo = QComboBox()
        self._snap_combo.addItems(SNAP_VALUES.keys())
        self._snap_combo.setCurrentText("1/16")
        self._snap_combo.currentTextChanged.connect(self._on_snap_changed)
        self._snap_combo.setFixedWidth(70)

        # Velocity
        vel_label = QLabel("Velocity:")
        vel_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._velocity_spin = QSpinBox()
        self._velocity_spin.setRange(1, 127)
        self._velocity_spin.setValue(100)
        self._velocity_spin.setFixedWidth(60)
        self._velocity_spin.valueChanged.connect(self._on_velocity_changed)

        swing_label = QLabel("Swing:")
        swing_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._swing_spin = QSpinBox()
        self._swing_spin.setRange(0, 75)
        self._swing_spin.setValue(33)
        self._swing_spin.setSuffix("%")
        self._swing_spin.setFixedWidth(64)

        human_label = QLabel("Human:")
        human_label.setStyleSheet(f"color: {t.get('text_secondary', '#8b949e')};")
        self._humanize_spin = QSpinBox()
        self._humanize_spin.setRange(0, 32)
        self._humanize_spin.setValue(8)
        self._humanize_spin.setFixedWidth(54)

        # Quantize button
        btn_style = f"""
            QPushButton {{
                background: {t.get('surface', '#161b22')};
                color: {t.get('text', '#e6edf3')};
                border: 1px solid {t.get('border', '#1e2733')};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: {t.get('surface_hover', '#1c2333')}; }}
        """
        self._quantize_btn = QPushButton("Quantize")
        self._quantize_btn.setStyleSheet(btn_style)
        self._quantize_btn.clicked.connect(self._on_quantize)

        self._swing_btn = QPushButton("Swing")
        self._swing_btn.setStyleSheet(btn_style)
        self._swing_btn.clicked.connect(self._on_apply_swing)

        self._humanize_btn = QPushButton("Humanize")
        self._humanize_btn.setStyleSheet(btn_style)
        self._humanize_btn.clicked.connect(self._on_humanize_velocity)

        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setStyleSheet(btn_style)
        self._select_all_btn.clicked.connect(self._scene.select_all)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.setStyleSheet(btn_style)
        self._delete_btn.clicked.connect(self._scene.delete_selected)

        toolbar.addWidget(snap_label)
        toolbar.addWidget(self._snap_combo)
        toolbar.addWidget(vel_label)
        toolbar.addWidget(self._velocity_spin)
        toolbar.addWidget(swing_label)
        toolbar.addWidget(self._swing_spin)
        toolbar.addWidget(human_label)
        toolbar.addWidget(self._humanize_spin)
        toolbar.addStretch()
        toolbar.addWidget(self._quantize_btn)
        toolbar.addWidget(self._swing_btn)
        toolbar.addWidget(self._humanize_btn)
        toolbar.addWidget(self._select_all_btn)
        toolbar.addWidget(self._delete_btn)

        layout.addLayout(toolbar)
        layout.addWidget(self._view, 1)
        self._automation_lane = CCAutomationLane()
        self._automation_lane.cc_changed.connect(self.notes_changed.emit)
        layout.addWidget(self._automation_lane)

    def load_track(self, track: TrackData, tempo: float = 120.0, bars: int = 16):
        self._scene.load_track(track, tempo, bars)
        self._automation_lane.load_track(track, tempo)

    def get_notes(self) -> list[NoteData]:
        return self._scene.get_notes()

    def _on_snap_changed(self, text: str):
        self._scene.snap_value = SNAP_VALUES.get(text, 0.25)

    def _on_velocity_changed(self, value: int):
        self._scene.default_velocity = value

    def _on_quantize(self):
        """Quantize all notes to current snap grid."""
        from core.midi_utils import quantize_notes
        if self._scene.track is None:
            return

        snap = self._scene.snap_value
        if snap <= 0:
            return

        self._replace_track_notes(quantize_notes(self._scene.track.notes, snap, self._scene.tempo))

    def _on_apply_swing(self):
        from core.midi_utils import apply_swing_to_notes
        if self._scene.track is None:
            return
        snap = self._scene.snap_value
        if snap <= 0:
            return
        amount = self._swing_spin.value() / 100.0
        self._replace_track_notes(
            apply_swing_to_notes(self._scene.track.notes, snap, self._scene.tempo, amount)
        )

    def _on_humanize_velocity(self):
        from core.midi_utils import humanize_note_velocities
        if self._scene.track is None:
            return
        self._replace_track_notes(
            humanize_note_velocities(self._scene.track.notes, self._humanize_spin.value())
        )

    def _replace_track_notes(self, notes: list[NoteData]):
        if self._scene.track is None:
            return
        self._scene.track.notes = notes
        self._scene.load_track(self._scene.track, self._scene.tempo, self._scene.bars)
        self._automation_lane.load_track(self._scene.track, self._scene.tempo)
        self.notes_changed.emit()
