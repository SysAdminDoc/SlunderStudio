"""
Slunder Studio — Piano Roll Widget
QGraphicsView-based MIDI piano roll editor with note creation, editing,
selection, quantization, and snap-to-grid.
"""
from collections import deque
from copy import deepcopy
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
from core.i18n import tr
from ui.accessibility import install_accessibility
from ui.theme import Palette, ThemeEngine


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
            base = QColor(Palette.PEACH)
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
    edit_started = Signal()

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
        bg = QColor(t["background"])
        grid_color = QColor(t["border"])
        bar_color = QColor(t["accent"]).darker(200)

        total_height = TOTAL_KEYS * NOTE_HEIGHT
        beat_dur = 60.0 / self.tempo
        bar_beats = 4  # assuming 4/4
        total_beats = self.bars * bar_beats
        total_width = total_beats * PIXELS_PER_BEAT

        self.setSceneRect(-KEY_WIDTH, -20, total_width + KEY_WIDTH, total_height + 20)

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
                label.setDefaultTextColor(QColor(Palette.SUBTEXT0))
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
                label.setDefaultTextColor(QColor(Palette.BLUE))
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
        if to_remove:
            self.edit_started.emit()
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

    undo_requested = Signal()

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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setProperty("accessibility_canvas", True)

        t = ThemeEngine.get_colors()
        bg = t["background"]
        self.setStyleSheet(f"""
            QGraphicsView {{
                background: {bg};
                border: 1px solid {t['border']};
                border-radius: 6px;
            }}
        """)

        # Center on middle C
        middle_c_y = (MAX_PITCH - 60) * NOTE_HEIGHT
        self.centerOn(0, middle_c_y)
        install_accessibility(
            self,
            tr("midi.piano_roll.accessibility.canvas_name"),
            named_controls=[
                (self, tr("midi.piano_roll.accessibility.canvas_name"), tr("midi.piano_roll.accessibility.canvas_description")),
            ],
        )

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
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_requested.emit()
            event.accept()
        elif event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self._scene.delete_selected()
            event.accept()
        elif event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self._scene.select_all()
            event.accept()
        else:
            super().keyPressEvent(event)


# ── Piano Roll Widget (with toolbar) ──────────────────────────────────────────

class CCAutomationLane(QWidget):
    """Compact control-change lane editor for the loaded track."""

    cc_changed = Signal()
    edit_started = Signal()

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

        label = QLabel(tr("midi.piano_roll.cc_lane_label"))
        label.setStyleSheet(f"color: {t['text_secondary']};")
        self._controller_combo = QComboBox()
        for _name, controller in CC_LANES.items():
            self._controller_combo.addItem(
                tr(f"midi.piano_roll.cc_{controller}"), controller
            )
        self._controller_combo.currentIndexChanged.connect(self._refresh)

        beat_label = QLabel(tr("midi.piano_roll.beat_label"))
        beat_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._beat_spin = QDoubleSpinBox()
        self._beat_spin.setRange(0.0, 512.0)
        self._beat_spin.setDecimals(2)
        self._beat_spin.setSingleStep(0.25)
        self._beat_spin.setValue(0.0)
        self._beat_spin.setMinimumWidth(72)

        value_label = QLabel(tr("midi.piano_roll.value_label"))
        value_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._value_spin = QSpinBox()
        self._value_spin.setRange(0, 127)
        self._value_spin.setValue(64)
        self._value_spin.setMinimumWidth(58)

        btn_style = f"""
            QPushButton {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 8.25pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """
        self._add_cc_btn = QPushButton(tr("midi.piano_roll.add_cc"))
        self._add_cc_btn.setStyleSheet(btn_style)
        self._add_cc_btn.clicked.connect(self._on_add_event)
        self._clear_lane_btn = QPushButton(tr("midi.piano_roll.clear_lane"))
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
        self._table.setHorizontalHeaderLabels([
            tr("midi.piano_roll.beat_header"),
            tr("midi.piano_roll.cc_header"),
            tr("midi.piano_roll.value_header"),
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setMaximumHeight(94)
        self._table.setStyleSheet(f"""
            QTableWidget {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                gridline-color: {t['border']};
                font-size: 8.25pt;
            }}
            QHeaderView::section {{
                background: {t['surface']};
                color: {t['text_secondary']};
                border: none;
                padding: 3px 6px;
            }}
        """)
        layout.addWidget(self._table)

        install_accessibility(
            self,
            tr("midi.piano_roll.cc_automation_title"),
            named_controls=[
                (self._controller_combo, tr("midi.piano_roll.accessibility.controller_name"), tr("midi.piano_roll.accessibility.controller_description")),
                (self._beat_spin, tr("midi.piano_roll.accessibility.beat_name"), tr("midi.piano_roll.accessibility.beat_description")),
                (self._value_spin, tr("midi.piano_roll.accessibility.value_name"), tr("midi.piano_roll.accessibility.value_description")),
                (self._add_cc_btn, tr("midi.piano_roll.accessibility.add_name"), tr("midi.piano_roll.accessibility.add_description")),
                (self._clear_lane_btn, tr("midi.piano_roll.accessibility.clear_name"), tr("midi.piano_roll.accessibility.clear_description")),
                (self._table, tr("midi.piano_roll.accessibility.table_name"), tr("midi.piano_roll.accessibility.table_description")),
            ],
            tab_order=[
                self._controller_combo, self._beat_spin, self._value_spin,
                self._add_cc_btn, self._clear_lane_btn, self._table,
            ],
        )

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
        filtered_events = [
            event for event in self._track.cc_events
            if event.controller != controller
        ]
        if len(filtered_events) != before:
            self.edit_started.emit()
            self._track.cc_events = filtered_events
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

    UNDO_LIMIT = 32
    notes_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = PianoRollScene()
        self._view = PianoRollView(self._scene)
        self._undo_stack: deque[tuple[list[NoteData], list[CCEvent]]] = deque(
            maxlen=self.UNDO_LIMIT
        )
        self._scene.notes_changed.connect(self.notes_changed.emit)
        self._scene.edit_started.connect(self._record_undo)
        self._view.undo_requested.connect(self.undo)

        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        # Snap selector
        snap_label = QLabel(tr("midi.piano_roll.snap_label"))
        snap_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._snap_combo = QComboBox()
        for snap in SNAP_VALUES:
            snap_key = "off" if snap == "Off" else snap.replace("/", "_")
            self._snap_combo.addItem(tr(f"midi.piano_roll.snap_{snap_key}"), snap)
        self._snap_combo.setCurrentIndex(self._snap_combo.findData("1/16"))
        self._snap_combo.currentTextChanged.connect(self._on_snap_changed)
        self._snap_combo.setMinimumWidth(70)

        # Velocity
        vel_label = QLabel(tr("midi.piano_roll.velocity_label"))
        vel_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._velocity_spin = QSpinBox()
        self._velocity_spin.setRange(1, 127)
        self._velocity_spin.setValue(100)
        self._velocity_spin.setMinimumWidth(60)
        self._velocity_spin.valueChanged.connect(self._on_velocity_changed)

        swing_label = QLabel(tr("midi.piano_roll.swing_label"))
        swing_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._swing_spin = QSpinBox()
        self._swing_spin.setRange(0, 75)
        self._swing_spin.setValue(33)
        self._swing_spin.setSuffix("%")
        self._swing_spin.setMinimumWidth(64)

        human_label = QLabel(tr("midi.piano_roll.humanize_label"))
        human_label.setStyleSheet(f"color: {t['text_secondary']};")
        self._humanize_spin = QSpinBox()
        self._humanize_spin.setRange(0, 32)
        self._humanize_spin.setValue(8)
        self._humanize_spin.setMinimumWidth(54)

        # Quantize button
        btn_style = f"""
            QPushButton {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 8.25pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """
        self._quantize_btn = QPushButton(tr("midi.piano_roll.quantize"))
        self._quantize_btn.setStyleSheet(btn_style)
        self._quantize_btn.clicked.connect(self._on_quantize)

        self._swing_btn = QPushButton(tr("midi.piano_roll.swing"))
        self._swing_btn.setStyleSheet(btn_style)
        self._swing_btn.clicked.connect(self._on_apply_swing)

        self._humanize_btn = QPushButton(tr("midi.piano_roll.humanize"))
        self._humanize_btn.setStyleSheet(btn_style)
        self._humanize_btn.clicked.connect(self._on_humanize_velocity)

        self._select_all_btn = QPushButton(tr("midi.piano_roll.select_all"))
        self._select_all_btn.setStyleSheet(btn_style)
        self._select_all_btn.clicked.connect(self._scene.select_all)

        self._delete_btn = QPushButton(tr("midi.piano_roll.delete"))
        self._delete_btn.setStyleSheet(btn_style)
        self._delete_btn.clicked.connect(self._scene.delete_selected)

        self._undo_btn = QPushButton(tr("midi.piano_roll.undo"))
        self._undo_btn.setStyleSheet(btn_style)
        self._undo_btn.setToolTip(tr("midi.piano_roll.undo_tooltip"))
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo)

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
        toolbar.addWidget(self._undo_btn)

        layout.addLayout(toolbar)
        layout.addWidget(self._view, 1)
        self._automation_lane = CCAutomationLane()
        self._automation_lane.edit_started.connect(self._record_undo)
        self._automation_lane.cc_changed.connect(self.notes_changed.emit)
        layout.addWidget(self._automation_lane)

        install_accessibility(
            self,
            tr("midi.piano_roll.title"),
            named_controls=[
                (self._snap_combo, tr("midi.piano_roll.accessibility.snap_name"), tr("midi.piano_roll.accessibility.snap_description")),
                (self._velocity_spin, tr("midi.piano_roll.accessibility.velocity_name"), tr("midi.piano_roll.accessibility.velocity_description")),
                (self._swing_spin, tr("midi.piano_roll.accessibility.swing_name"), tr("midi.piano_roll.accessibility.swing_description")),
                (self._humanize_spin, tr("midi.piano_roll.accessibility.humanize_range_name"), tr("midi.piano_roll.accessibility.humanize_range_description")),
                (self._quantize_btn, tr("midi.piano_roll.accessibility.quantize_name"), tr("midi.piano_roll.accessibility.quantize_description")),
                (self._swing_btn, tr("midi.piano_roll.accessibility.apply_swing_name"), tr("midi.piano_roll.accessibility.apply_swing_description")),
                (self._humanize_btn, tr("midi.piano_roll.accessibility.humanize_name"), tr("midi.piano_roll.accessibility.humanize_description")),
                (self._select_all_btn, tr("midi.piano_roll.accessibility.select_all_name"), tr("midi.piano_roll.accessibility.select_all_description")),
                (self._delete_btn, tr("midi.piano_roll.accessibility.delete_name"), tr("midi.piano_roll.accessibility.delete_description")),
                (self._undo_btn, tr("midi.piano_roll.accessibility.undo_name"), tr("midi.piano_roll.accessibility.undo_description")),
            ],
            tab_order=[
                self._snap_combo, self._velocity_spin, self._swing_spin, self._humanize_spin,
                self._quantize_btn, self._swing_btn, self._humanize_btn,
                self._select_all_btn, self._delete_btn, self._undo_btn, self._view,
            ],
        )

    def load_track(self, track: TrackData, tempo: float = 120.0, bars: int = 16):
        self._undo_stack.clear()
        self._undo_btn.setEnabled(False)
        self._scene.load_track(track, tempo, bars)
        self._automation_lane.load_track(track, tempo)

    def get_notes(self) -> list[NoteData]:
        return self._scene.get_notes()

    def _on_snap_changed(self, text: str):
        self._scene.snap_value = SNAP_VALUES.get(
            self._snap_combo.currentData(), SNAP_VALUES.get(text, 0.25)
        )

    def _on_velocity_changed(self, value: int):
        self._scene.default_velocity = value

    def controller_quantize(self) -> bool:
        """Apply quantize through the public controller-action boundary."""
        return self._on_quantize()

    def controller_swing(self) -> bool:
        """Apply swing through the public controller-action boundary."""
        return self._on_apply_swing()

    def controller_humanize(self) -> bool:
        """Apply velocity humanization through the controller boundary."""
        return self._on_humanize_velocity()

    def _on_quantize(self) -> bool:
        """Quantize all notes to current snap grid."""
        from core.midi_utils import quantize_notes
        if self._scene.track is None:
            return False

        snap = self._scene.snap_value
        if snap <= 0:
            return False

        return self._replace_track_notes(
            quantize_notes(self._scene.track.notes, snap, self._scene.tempo)
        )

    def _on_apply_swing(self) -> bool:
        from core.midi_utils import apply_swing_to_notes
        if self._scene.track is None:
            return False
        snap = self._scene.snap_value
        if snap <= 0:
            return False
        amount = self._swing_spin.value() / 100.0
        return self._replace_track_notes(
            apply_swing_to_notes(self._scene.track.notes, snap, self._scene.tempo, amount)
        )

    def _on_humanize_velocity(self) -> bool:
        from core.midi_utils import humanize_note_velocities
        if self._scene.track is None:
            return False
        return self._replace_track_notes(
            humanize_note_velocities(self._scene.track.notes, self._humanize_spin.value())
        )

    def _replace_track_notes(self, notes: list[NoteData]) -> bool:
        if self._scene.track is None:
            return False
        self._record_undo()
        self._scene.track.notes = notes
        self._scene.load_track(self._scene.track, self._scene.tempo, self._scene.bars)
        self._automation_lane.load_track(self._scene.track, self._scene.tempo)
        self.notes_changed.emit()
        return True

    def _record_undo(self):
        """Save the current note and CC state before a destructive edit."""
        if self._scene.track is None:
            return
        self._undo_stack.append((
            deepcopy(self._scene.track.notes),
            deepcopy(self._scene.track.cc_events),
        ))
        self._undo_btn.setEnabled(True)

    def undo(self) -> bool:
        """Restore the most recent piano-roll note and CC snapshot."""
        if self._scene.track is None or not self._undo_stack:
            return False

        notes, cc_events = self._undo_stack.pop()
        self._scene.track.notes = deepcopy(notes)
        self._scene.track.cc_events = deepcopy(cc_events)
        self._scene.load_track(self._scene.track, self._scene.tempo, self._scene.bars)
        self._automation_lane.load_track(self._scene.track, self._scene.tempo)
        self._undo_btn.setEnabled(bool(self._undo_stack))
        self.notes_changed.emit()
        return True
