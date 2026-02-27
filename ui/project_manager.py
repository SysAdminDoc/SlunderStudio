"""
Slunder Studio v0.0.2 — Project Manager View
Project browser with create, open, delete, asset management,
version history, and auto-save controls.
"""
import os
import time
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QLineEdit, QTextEdit, QFileDialog,
    QListWidget, QListWidgetItem, QStackedWidget, QInputDialog,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine
from core.project import ProjectManager, Project, ProjectAsset, get_project_manager


# ── Project Card ───────────────────────────────────────────────────────────────

class ProjectCard(QFrame):
    """Clickable project card in the browser."""

    open_requested = Signal(str)    # project_id
    delete_requested = Signal(str)  # project_id

    def __init__(self, project_info: dict, parent=None):
        super().__init__(parent)
        self._project_id = project_info["id"]

        t = ThemeEngine.get_colors()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            ProjectCard {{
                background: {t['surface']};
                border: 1px solid {t['border']};
                border-radius: 8px;
            }}
            ProjectCard:hover {{
                border-color: {t['accent']};
                background: {t['surface_hover']};
            }}
        """)
        self.setFixedHeight(72)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)

        name = QLabel(project_info.get("name", "Untitled"))
        name.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 13px;")
        info.addWidget(name)

        updated = project_info.get("updated_at", 0)
        if updated:
            time_str = time.strftime("%b %d, %Y %I:%M %p", time.localtime(updated))
        else:
            time_str = "Unknown"
        date_label = QLabel(f"Last modified: {time_str}")
        date_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px;")
        info.addWidget(date_label)

        layout.addLayout(info, 1)

        # Buttons
        btn_style = f"""
            QPushButton {{
                background: {t['background']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 10px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        open_btn = QPushButton("Open")
        open_btn.setStyleSheet(btn_style.replace(t['background'], t['accent']).replace(t['text'] + ';', 'white;'))
        open_btn.clicked.connect(lambda: self.open_requested.emit(self._project_id))

        del_btn = QPushButton("Delete")
        del_btn.setStyleSheet(btn_style)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self._project_id))

        layout.addWidget(open_btn)
        layout.addWidget(del_btn)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_requested.emit(self._project_id)
        super().mousePressEvent(event)


# ── Project Detail Panel ───────────────────────────────────────────────────────

class ProjectDetailPanel(QWidget):
    """Shows details of the currently open project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Project info header
        self._name_label = QLabel("No Project Open")
        self._name_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 16px;")
        layout.addWidget(self._name_label)

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 11px;")
        layout.addWidget(self._meta_label)

        # Notes
        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Project notes...")
        self._notes.setMaximumHeight(80)
        self._notes.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 11px;
            }}
        """)
        layout.addWidget(self._notes)

        # Assets list
        assets_label = QLabel("Assets")
        assets_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px;")
        layout.addWidget(assets_label)

        self._asset_list = QListWidget()
        self._asset_list.setStyleSheet(f"""
            QListWidget {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {t['border']};
            }}
            QListWidget::item:selected {{
                background: {t['accent']}33;
            }}
        """)
        layout.addWidget(self._asset_list, 1)

        # Version history
        ver_label = QLabel("Version History")
        ver_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12px;")
        layout.addWidget(ver_label)

        self._version_list = QListWidget()
        self._version_list.setMaximumHeight(120)
        self._version_list.setStyleSheet(self._asset_list.styleSheet())
        layout.addWidget(self._version_list)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_style = f"""
            QPushButton {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 11px;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        self._save_btn = QPushButton("Save")
        self._save_btn.setStyleSheet(btn_style.replace(t['surface'] + ';', '#238636;').replace(t['text'] + ';', 'white;'))
        self._save_btn.clicked.connect(self._on_save)

        self._snapshot_btn = QPushButton("Save Version")
        self._snapshot_btn.setStyleSheet(btn_style)
        self._snapshot_btn.clicked.connect(self._on_snapshot)

        self._import_btn = QPushButton("Import Asset")
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.clicked.connect(self._on_import_asset)

        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._snapshot_btn)
        btn_row.addWidget(self._import_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def load_project(self, project: Project):
        """Display project details."""
        self._name_label.setText(project.name)

        created = time.strftime("%b %d, %Y", time.localtime(project.created_at))
        self._meta_label.setText(
            f"Created: {created} | "
            f"Tempo: {project.tempo} BPM | "
            f"Key: {project.key} | "
            f"Assets: {project.asset_count} | "
            f"Versions: {project.version_count}"
        )

        self._notes.setPlainText(project.notes)

        # Assets
        self._asset_list.clear()
        for asset in project.assets:
            item = QListWidgetItem(
                f"[{asset.asset_type}] {asset.name} ({asset.module})"
            )
            self._asset_list.addItem(item)

        # Versions
        self._version_list.clear()
        for ver in reversed(project.versions):
            ts = time.strftime("%b %d %I:%M %p", time.localtime(ver.timestamp))
            auto = " (auto)" if ver.auto_save else ""
            item = QListWidgetItem(f"v{ver.version} - {ts}{auto}: {ver.description}")
            self._version_list.addItem(item)

    def clear(self):
        self._name_label.setText("No Project Open")
        self._meta_label.setText("")
        self._notes.clear()
        self._asset_list.clear()
        self._version_list.clear()

    def _on_save(self):
        mgr = get_project_manager()
        if mgr.current:
            mgr.current.notes = self._notes.toPlainText()
            mgr.save()

    def _on_snapshot(self):
        mgr = get_project_manager()
        if mgr.current:
            desc, ok = QInputDialog.getText(self, "Version Description",
                                            "Description for this version:")
            if ok:
                mgr.current.notes = self._notes.toPlainText()
                mgr.save()
                mgr.create_version(desc or "Manual save")
                self.load_project(mgr.current)

    def _on_import_asset(self):
        mgr = get_project_manager()
        if not mgr.current:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Asset", "",
            "Audio (*.wav *.flac *.mp3);;MIDI (*.mid *.midi);;All (*.*)"
        )
        if path:
            ext = path.lower().rsplit(".", 1)[-1]
            atype = "midi" if ext in ("mid", "midi") else "audio"
            mgr.import_asset(path, atype, "project_manager")
            self.load_project(mgr.current)


# ── Project Manager View ───────────────────────────────────────────────────────

class ProjectManagerView(QWidget):
    """Main project management page."""

    project_opened = Signal(str)  # project_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cards: list[ProjectCard] = []

        t = ThemeEngine.get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ── Left: Project Browser ──────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Projects")
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 16px;")

        self._new_btn = QPushButton("+ New Project")
        self._new_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: white; border: none;
                border-radius: 5px; padding: 6px 14px;
                font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._new_btn.clicked.connect(self._on_new_project)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._new_btn)
        left.addLayout(header)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search projects...")
        self._search.setStyleSheet(f"""
            QLineEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px 12px; font-size: 12px;
            }}
        """)
        self._search.textChanged.connect(self._on_search)
        left.addWidget(self._search)

        # Project list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._list_layout.addStretch()

        self._scroll.setWidget(self._list_container)
        left.addWidget(self._scroll, 1)

        # Count
        self._count_label = QLabel("")
        self._count_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 10px;")
        left.addWidget(self._count_label)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setFixedWidth(400)
        layout.addWidget(left_w)

        # ── Right: Project Detail ──────────────────────────────────────────
        self._detail = ProjectDetailPanel()
        layout.addWidget(self._detail, 1)

        # Load initial project list
        self._refresh_list()

    def _refresh_list(self):
        """Reload project list from ProjectManager."""
        # Clear existing cards
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        mgr = get_project_manager()
        projects = mgr.list_projects()

        for info in projects:
            card = ProjectCard(info)
            card.open_requested.connect(self._on_open_project)
            card.delete_requested.connect(self._on_delete_project)
            self._cards.append(card)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        self._count_label.setText(f"{len(projects)} project{'s' if len(projects) != 1 else ''}")

    def _on_new_project(self):
        name, ok = QInputDialog.getText(self, "New Project", "Project name:")
        if ok and name.strip():
            mgr = get_project_manager()
            project = mgr.create(name.strip())
            self._refresh_list()
            self._detail.load_project(project)
            self.project_opened.emit(project.id)

    def _on_open_project(self, project_id: str):
        mgr = get_project_manager()
        project = mgr.open(project_id)
        if project:
            self._detail.load_project(project)
            self.project_opened.emit(project_id)

    def _on_delete_project(self, project_id: str):
        mgr = get_project_manager()
        mgr.delete(project_id)
        self._detail.clear()
        self._refresh_list()

    def _on_search(self, text: str):
        query = text.lower()
        for card in self._cards:
            visible = not query or query in card._project_id.lower()
            # Check card's name label
            for child in card.findChildren(QLabel):
                if child.styleSheet() and "bold" in child.styleSheet():
                    if query in child.text().lower():
                        visible = True
                    break
            card.setVisible(visible)
