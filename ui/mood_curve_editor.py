"""
Slunder Studio — Mood/Energy Curve Editor
Visual curve editor for drawing song energy arcs.
Draggable control points with Bezier interpolation.
Preset curves and reference overlay.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPathItem,
)
from PySide6.QtCore import Signal, Qt, QPointF, QRectF
from PySide6.QtGui import QKeyEvent, QPainterPath, QPen, QBrush, QColor, QLinearGradient

import numpy as np

from ui.theme import Palette
from ui.accessibility import install_accessibility
from core.i18n import tr

# ── Preset Curves ──────────────────────────────────────────────────────────────

PRESETS = {
    "Classic Pop Build": [0.2, 0.3, 0.5, 0.7, 0.9, 0.6, 0.85, 0.95, 0.4],
    "Slow Burn": [0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0],
    "Intro Drop": [0.8, 0.9, 1.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.6],
    "Epic Crescendo": [0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95, 1.0],
    "Flat Energy": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    "Roller Coaster": [0.3, 0.8, 0.4, 0.9, 0.3, 0.85, 0.5, 0.95, 0.6],
    "Ballad Arc": [0.2, 0.3, 0.45, 0.7, 0.5, 0.6, 0.85, 0.7, 0.3],
    "EDM Build-Drop": [0.2, 0.4, 0.6, 0.8, 1.0, 0.2, 0.6, 0.9, 0.5],
}


class DraggablePoint(QGraphicsEllipseItem):
    """A draggable control point on the curve."""

    def __init__(self, x, y, index, callback, parent=None):
        size = 10
        super().__init__(-size/2, -size/2, size, size, parent)
        self.setPos(x, y)
        self.setFlag(QGraphicsEllipseItem.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsEllipseItem.ItemSendsGeometryChanges, True)
        self.setBrush(QBrush(QColor(Palette.BLUE)))
        self.setPen(QPen(QColor(Palette.TEXT), 1.5))
        self.setZValue(10)
        self.setCursor(Qt.SizeAllCursor)
        self.setToolTip(
            tr("batch.mood_curve.point_tooltip", index=index + 1)
        )
        self._index = index
        self._callback = callback
        self._scene_width = 1.0
        self._scene_height = 1.0

    def set_bounds(self, width, height):
        self._scene_width = width
        self._scene_height = height

    def itemChange(self, change, value):
        if change == QGraphicsEllipseItem.ItemPositionChange:
            # Constrain to scene bounds
            new_pos = value
            x = max(0, min(self._scene_width, new_pos.x()))
            y = max(0, min(self._scene_height, new_pos.y()))
            new_pos = QPointF(x, y)
            if self._callback:
                self._callback(self._index, new_pos)
            return new_pos
        return super().itemChange(change, value)


class MoodCurveView(QGraphicsView):
    """Graphics canvas with a keyboard route for selecting and nudging points."""

    keyboard_navigation = Signal(int, bool)

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key in (
            Qt.Key_Left,
            Qt.Key_Right,
            Qt.Key_Up,
            Qt.Key_Down,
            Qt.Key_Tab,
            Qt.Key_Home,
            Qt.Key_End,
        ):
            self.keyboard_navigation.emit(
                key,
                bool(event.modifiers() & Qt.ShiftModifier),
            )
            event.accept()
            return
        super().keyPressEvent(event)


class MoodCurveEditor(QWidget):
    """
    Visual energy/mood curve editor for song generation.
    X-axis: song timeline (0 to duration)
    Y-axis: energy level (0 = minimal, 1 = max intensity)
    """
    curve_changed = Signal(list)  # emits list of (normalized_x, energy_y) tuples

    SCENE_W = 600
    SCENE_H = 200

    def __init__(self, parent=None, n_points: int = 9):
        super().__init__(parent)
        self._n_points = n_points
        self._points: list[DraggablePoint] = []
        self._curve_path: QGraphicsPathItem = None
        self._reference_path: QGraphicsPathItem = None
        self._fill_item: QGraphicsPathItem = None
        self._duration = 60.0
        self._selected_point_index = 0

        self._setup_ui()
        self._set_preset("Classic Pop Build")
        install_accessibility(
            self,
            tr("batch.mood_curve.accessibility.name"),
            named_controls=[
                (self._preset_combo, tr("batch.mood_curve.accessibility.preset_name"), tr("batch.mood_curve.accessibility.preset_description")),
                (self._reset_btn, tr("batch.mood_curve.accessibility.reset_name"), tr("batch.mood_curve.accessibility.reset_description")),
                (self._view, tr("batch.mood_curve.accessibility.canvas_name"), tr("batch.mood_curve.accessibility.canvas_description")),
            ],
        )

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        lbl = QLabel(tr("batch.mood_curve.title"))
        lbl.setStyleSheet(f"color: {Palette.TEXT}; font-weight: bold; font-size: 9pt;")
        ctrl.addWidget(lbl)

        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(160)
        for name in PRESETS:
            preset_key = name.lower().replace(" ", "_").replace("-", "_")
            self._preset_combo.addItem(
                tr(f"batch.mood_curve.presets.{preset_key}"), name
            )
        self._preset_combo.currentIndexChanged.connect(
            lambda index: self._set_preset(
                self._preset_combo.itemData(index)
                or self._preset_combo.itemText(index)
            )
        )
        ctrl.addWidget(self._preset_combo)

        self._reset_btn = QPushButton(tr("batch.mood_curve.reset"))
        self._reset_btn.setMinimumWidth(60)
        self._reset_btn.setMinimumHeight(26)
        self._reset_btn.setProperty("class", "secondary")
        self._reset_btn.clicked.connect(lambda: self._set_preset("Flat Energy"))
        ctrl.addWidget(self._reset_btn)

        ctrl.addStretch()

        self._energy_label = QLabel("")
        self._energy_label.setStyleSheet(f"color: {Palette.SUBTEXT0}; font-size: 8.25pt;")
        ctrl.addWidget(self._energy_label)

        layout.addLayout(ctrl)

        # Graphics view
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, self.SCENE_W, self.SCENE_H)

        self._view = MoodCurveView(self._scene)
        self._view.keyboard_navigation.connect(self._on_keyboard_navigation)
        self._view.setRenderHint(self._view.renderHints())
        self._view.setStyleSheet(
            f"QGraphicsView {{ background: {Palette.MANTLE}; border: 1px solid {Palette.SURFACE0}; border-radius: 6px; }}"
        )
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setMinimumHeight(self.SCENE_H + 20)
        self._view.setProperty("accessibility_canvas", True)

        layout.addWidget(self._view)

        # Draw grid
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._draw_grid()

    def _draw_grid(self):
        """Draw background grid lines."""
        pen = QPen(QColor(Palette.SURFACE0), 1, Qt.DotLine)
        # Horizontal lines (energy levels)
        for i in range(1, 4):
            y = i * self.SCENE_H / 4
            self._scene.addLine(0, y, self.SCENE_W, y, pen)
        # Vertical lines (time sections)
        for i in range(1, self._n_points):
            x = i * self.SCENE_W / self._n_points
            self._scene.addLine(x, 0, x, self.SCENE_H, pen)

    def _set_preset(self, name: str):
        """Load a preset curve."""
        values = PRESETS.get(name, PRESETS["Flat Energy"])
        # Clear existing points
        for p in self._points:
            self._scene.removeItem(p)
        self._points.clear()

        # Create new points
        for i, val in enumerate(values[:self._n_points]):
            x = i * self.SCENE_W / (self._n_points - 1)
            y = (1.0 - val) * self.SCENE_H  # Invert: top = high energy
            point = DraggablePoint(x, y, i, self._on_point_moved)
            point.set_bounds(self.SCENE_W, self.SCENE_H)
            self._scene.addItem(point)
            self._points.append(point)

        self._selected_point_index = 0
        self._select_point(self._selected_point_index)
        self._update_curve()

    def _select_point(self, index: int):
        """Select one control point and expose its current value to assistive tech."""
        if not self._points:
            return
        self._selected_point_index = index % len(self._points)
        self._scene.clearSelection()
        self._points[self._selected_point_index].setSelected(True)
        self._announce_selected_point()

    def _on_selection_changed(self):
        """Keep keyboard selection aligned when a point is selected with the mouse."""
        for item in self._scene.selectedItems():
            if isinstance(item, DraggablePoint):
                self._selected_point_index = item._index
                self._announce_selected_point()
                return

    def _announce_selected_point(self):
        if not self._points or not hasattr(self, "_view"):
            return
        point = self._points[self._selected_point_index]
        time = point.pos().x() / self.SCENE_W
        energy = 1.0 - point.pos().y() / self.SCENE_H
        self._view.setAccessibleDescription(
            f"Keyboard-focusable canvas. Curve point {self._selected_point_index + 1} "
            f"of {len(self._points)} selected at {time:.0%} of the song and "
            f"{energy:.0%} energy. Tab selects another point; arrow keys nudge it."
        )

    def _on_keyboard_navigation(self, key: int, large_step: bool):
        """Select points with Tab and move the selected point with the arrow keys."""
        if not self._points:
            return

        if key == Qt.Key_Tab:
            direction = -1 if large_step else 1
            self._select_point(self._selected_point_index + direction)
            return
        if key == Qt.Key_Home:
            self._select_point(0)
            return
        if key == Qt.Key_End:
            self._select_point(len(self._points) - 1)
            return

        point = self._points[self._selected_point_index]
        position = point.pos()
        x_step = self.SCENE_W / max(1, len(self._points) - 1) / (
            1 if large_step else 4
        )
        y_step = self.SCENE_H / (5 if large_step else 20)
        x = position.x()
        y = position.y()
        if key == Qt.Key_Left:
            x -= x_step
        elif key == Qt.Key_Right:
            x += x_step
        elif key == Qt.Key_Up:
            y -= y_step
        elif key == Qt.Key_Down:
            y += y_step
        else:
            return

        point.setPos(QPointF(x, y))
        point.setSelected(True)
        self._announce_selected_point()

    def _on_point_moved(self, index: int, pos: QPointF):
        """Called when a control point is dragged."""
        self._selected_point_index = index
        self._update_curve()
        self.curve_changed.emit(self.get_values())
        self._announce_selected_point()

    def _update_curve(self):
        """Redraw the smooth curve through control points."""
        if len(self._points) < 2:
            return

        # Remove old curve
        if self._curve_path:
            self._scene.removeItem(self._curve_path)
        if self._fill_item:
            self._scene.removeItem(self._fill_item)

        # Get sorted point positions
        positions = [(p.pos().x(), p.pos().y()) for p in self._points]
        positions.sort(key=lambda p: p[0])

        # Build smooth path using cubic bezier
        path = QPainterPath()
        path.moveTo(positions[0][0], positions[0][1])

        for i in range(1, len(positions)):
            x0, y0 = positions[i-1]
            x1, y1 = positions[i]
            cx = (x0 + x1) / 2
            path.cubicTo(cx, y0, cx, y1, x1, y1)

        # Draw filled area under curve
        fill_path = QPainterPath(path)
        fill_path.lineTo(positions[-1][0], self.SCENE_H)
        fill_path.lineTo(positions[0][0], self.SCENE_H)
        fill_path.closeSubpath()

        grad = QLinearGradient(0, 0, 0, self.SCENE_H)
        gradient_start = QColor(Palette.BLUE)
        gradient_start.setAlpha(60)
        gradient_end = QColor(Palette.BLUE)
        gradient_end.setAlpha(10)
        grad.setColorAt(0, gradient_start)
        grad.setColorAt(1, gradient_end)

        self._fill_item = self._scene.addPath(
            fill_path, QPen(Qt.NoPen), QBrush(grad)
        )
        self._fill_item.setZValue(1)

        # Draw curve line
        self._curve_path = self._scene.addPath(
            path, QPen(QColor(Palette.BLUE), 2.5, Qt.SolidLine)
        )
        self._curve_path.setZValue(5)

        # Update label
        vals = self.get_values()
        avg = np.mean([v[1] for v in vals])
        self._energy_label.setText(tr("batch.mood_curve.average", average=avg))

    def set_reference_curve(self, energy_values: list[float]):
        """Overlay a reference track's energy curve (semi-transparent)."""
        if self._reference_path:
            self._scene.removeItem(self._reference_path)
            self._reference_path = None

        if not energy_values:
            return

        path = QPainterPath()
        n = len(energy_values)
        for i, val in enumerate(energy_values):
            x = i * self.SCENE_W / max(1, n - 1)
            y = (1.0 - val) * self.SCENE_H
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        self._reference_path = self._scene.addPath(
            path, QPen(QColor(Palette.YELLOW), 1.5, Qt.DashLine)
        )
        self._reference_path.setZValue(3)

    def clear_reference(self):
        if self._reference_path:
            self._scene.removeItem(self._reference_path)
            self._reference_path = None

    def get_values(self) -> list[tuple[float, float]]:
        """Get curve as list of (normalized_time, energy) tuples."""
        if not self._points:
            return [(0.0, 0.5), (1.0, 0.5)]

        positions = [(p.pos().x(), p.pos().y()) for p in self._points]
        positions.sort(key=lambda p: p[0])

        values = []
        for x, y in positions:
            norm_x = x / self.SCENE_W
            energy = 1.0 - (y / self.SCENE_H)
            values.append((round(norm_x, 3), round(max(0, min(1, energy)), 3)))

        return values

    def get_energy_at(self, normalized_time: float) -> float:
        """Interpolate energy value at a given normalized time."""
        values = self.get_values()
        if not values:
            return 0.5

        for i in range(len(values) - 1):
            t0, e0 = values[i]
            t1, e1 = values[i + 1]
            if t0 <= normalized_time <= t1:
                frac = (normalized_time - t0) / max(0.001, t1 - t0)
                return e0 + frac * (e1 - e0)

        return values[-1][1]

    def energy_to_tags(self) -> dict[str, list[str]]:
        """
        Map energy curve to per-section ACE-Step dynamics tags.
        Returns dict with section labels and suggested intensity tags.
        """
        values = self.get_values()
        sections = {}
        section_labels = ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Chorus 2", "Bridge", "Final Chorus", "Outro"]

        n_sections = min(len(section_labels), len(values))
        for i in range(n_sections):
            energy = values[i][1] if i < len(values) else 0.5
            label = section_labels[i]

            if energy < 0.25:
                tags = ["soft", "gentle", "quiet", "minimal"]
            elif energy < 0.5:
                tags = ["moderate", "building"]
            elif energy < 0.75:
                tags = ["energetic", "driving", "powerful"]
            else:
                tags = ["intense", "epic", "maximum energy", "climactic"]

            sections[label] = tags

        return sections

    def set_duration(self, seconds: float):
        self._duration = seconds
