"""
Slunder Studio — Project Manager View
Project browser with create, open, delete, asset management,
version history, and auto-save controls.
"""
import os
import json
import time
from typing import Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QLineEdit, QTextEdit, QFileDialog,
    QListWidget, QListWidgetItem, QStackedWidget, QInputDialog,
    QDialog, QPlainTextEdit, QMessageBox, QComboBox,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine, rgba
from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget
from core.project import ProjectManager, Project, ProjectAsset, get_project_manager
from core.dawproject import (
    DAWProjectSpec,
    export_dawproject,
    spec_from_project,
    validate_dawproject,
)
from core.i18n import tr
from core.disclosure import (
    format_human_contributions,
    parse_human_contributions,
    write_disclosure_report,
)
from core.provenance import (
    check_provenance_compatibility,
    provenance_replayability,
    read_provenance_sidecar,
    rerender_from_provenance,
)
from core.workers import CancelledJobError, InferenceWorker
from core.routing import is_midi_path
from ui.file_dialogs import choose_directory, ensure_extension, open_project_files, save_file


def _rerender_provenance_task(
    artifact_path: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Run a provenance render away from the Project Manager GUI thread."""
    return rerender_from_provenance(
        artifact_path,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
    )


def _export_dawproject_task(
    spec: DAWProjectSpec,
    output_path: str,
    progress_cb=None,
    cancel_event=None,
    **_kwargs,
):
    """Export and validate a DAWproject archive away from the GUI thread."""
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("DAWproject export cancelled", outputs=[output_path])
    if progress_cb:
        progress_cb(10)
    written = export_dawproject(spec, output_path)
    if progress_cb:
        progress_cb(75)
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledJobError("DAWproject export cancelled", outputs=[written])
    validation = validate_dawproject(written)
    if not validation.valid:
        try:
            os.remove(written)
        except OSError:
            pass
        raise ValueError(
            "DAWproject validation failed: " + "; ".join(validation.errors)
        )
    if progress_cb:
        progress_cb(100)
    return {
        "path": written,
        "track_count": len(spec.tracks),
        "entries": len(validation.entries),
    }


# ── Project Card ───────────────────────────────────────────────────────────────

class ProjectCard(QFrame):
    """Clickable project card in the browser."""

    open_requested = Signal(str)    # project_id
    delete_requested = Signal(str)  # project_id

    def __init__(self, project_info: dict, parent=None):
        super().__init__(parent)
        self._project_id = project_info["id"]
        self._project_name = str(
            project_info.get("name") or tr("project_manager.data.untitled")
        )
        self._project_notes = str(project_info.get("notes") or "")

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
        self.setMinimumHeight(72)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)

        name = QLabel(self._project_name)
        name.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9.75pt;")
        info.addWidget(name)

        updated = project_info.get("updated_at", 0)
        if updated:
            time_str = time.strftime("%b %d, %Y %I:%M %p", time.localtime(updated))
        else:
            time_str = tr("project_manager.data.unknown")
        date_label = QLabel(
            tr("project_manager.card.last_modified", time=time_str)
        )
        date_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt;")
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
                font-size: 7.5pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        open_btn = QPushButton(tr("project_manager.actions.open"))
        open_btn.setStyleSheet(btn_style.replace(t['background'], t['accent']).replace(t['text'] + ';', 'white;'))
        open_btn.clicked.connect(lambda: self.open_requested.emit(self._project_id))

        del_btn = QPushButton(tr("project_manager.actions.delete"))
        del_btn.setStyleSheet(btn_style)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self._project_id))

        layout.addWidget(open_btn)
        layout.addWidget(del_btn)

        self._open_btn = open_btn
        self._delete_btn = del_btn
        install_accessibility(
            self,
            tr("project_manager.accessibility.card_name", project=self._project_name),
            named_controls=[
                (
                    self._open_btn,
                    tr("project_manager.accessibility.open_name"),
                    tr("project_manager.accessibility.open_description"),
                ),
                (
                    self._delete_btn,
                    tr("project_manager.accessibility.delete_name"),
                    tr("project_manager.accessibility.delete_description"),
                ),
            ],
        )

    def matches_query(self, query: str) -> bool:
        """Match only user-facing project metadata, never the internal ID."""
        normalized = query.casefold().strip()
        if not normalized:
            return True
        searchable = "\n".join((self._project_name, self._project_notes)).casefold()
        return normalized in searchable

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.open_requested.emit(self._project_id)
        super().mousePressEvent(event)


# ── Project Detail Panel ───────────────────────────────────────────────────────

class ProjectDetailPanel(QWidget):
    """Shows details of the currently open project."""

    create_requested = Signal()

    def __init__(self, parent=None, toast_mgr=None):
        super().__init__(parent)
        t = ThemeEngine.get_colors()
        self.toast_mgr = toast_mgr
        self._asset_by_id: dict[str, ProjectAsset] = {}
        self._rerender_worker = None
        self._rerender_workers = set()
        self._dawproject_worker = None
        self._dawproject_workers = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Project info header
        self._name_label = QLabel(tr("project_manager.detail.no_project"))
        self._name_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12pt;")
        layout.addWidget(self._name_label)

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(self._meta_label)

        self._rerender_capability_label = QLabel("")
        self._rerender_capability_label.setWordWrap(True)
        self._rerender_capability_label.setStyleSheet(
            f"color: {t['text_secondary']}; font-size: 8pt; padding: 1px 0;"
        )
        layout.addWidget(self._rerender_capability_label)

        # Notes
        self._notes = QTextEdit()
        self._notes.setPlaceholderText(tr("project_manager.detail.notes_placeholder"))
        self._notes.setMaximumHeight(80)
        self._notes.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 8.25pt;
            }}
        """)
        layout.addWidget(self._notes)

        contributions_label = QLabel(
            tr("project_manager.detail.contributions_label")
        )
        contributions_label.setStyleSheet(
            f"color: {t['text']}; font-weight: bold; font-size: 9pt;"
        )
        layout.addWidget(contributions_label)

        self._contributions = QTextEdit()
        self._contributions.setPlaceholderText(
            tr("project_manager.detail.contributions_placeholder")
        )
        self._contributions.setMaximumHeight(92)
        self._contributions.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 8.25pt;
            }}
        """)
        layout.addWidget(self._contributions)

        # Assets list
        assets_label = QLabel(tr("project_manager.detail.assets"))
        assets_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(assets_label)

        self._asset_list = QListWidget()
        self._asset_list.setStyleSheet(f"""
            QListWidget {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                font-size: 8.25pt;
            }}
            QListWidget::item {{
                padding: 4px 8px;
                border-bottom: 1px solid {t['border']};
            }}
            QListWidget::item:selected {{
                background: {rgba(t['accent'], 51)};
            }}
        """)
        self._asset_list.currentItemChanged.connect(self._on_asset_selected)
        self._asset_empty = EmptyStateWidget(
            tr("project_manager.empty.assets_title"),
            tr("project_manager.empty.assets_description"),
            tr("project_manager.actions.import_asset"),
        )
        self._asset_empty.action_requested.connect(self._on_asset_empty_action)
        self._asset_stack = QStackedWidget()
        self._asset_stack.addWidget(self._asset_list)
        self._asset_stack.addWidget(self._asset_empty)
        layout.addWidget(self._asset_stack, 1)

        # Version history
        ver_label = QLabel(tr("project_manager.detail.version_history"))
        ver_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(ver_label)

        self._version_list = QListWidget()
        self._version_list.setMaximumHeight(120)
        self._version_list.setStyleSheet(self._asset_list.styleSheet())
        self._version_list.currentItemChanged.connect(self._on_version_selected)
        self._version_empty = EmptyStateWidget(
            tr("project_manager.empty.versions_title"),
            tr("project_manager.empty.versions_description"),
            tr("project_manager.actions.save_version"),
        )
        self._version_empty.action_requested.connect(self._on_version_empty_action)
        self._version_stack = QStackedWidget()
        self._version_stack.addWidget(self._version_list)
        self._version_stack.addWidget(self._version_empty)
        layout.addWidget(self._version_stack)

        self._version_preview = QLabel(tr("project_manager.version.select_preview"))
        self._version_preview.setWordWrap(True)
        self._version_preview.setStyleSheet(
            f"color: {t['text_secondary']}; font-size: 8.25pt; padding: 2px 0;"
        )
        layout.addWidget(self._version_preview)

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
                font-size: 8.25pt;
            }}
            QPushButton:hover {{ background: {t['surface_hover']}; }}
        """

        self._save_btn = QPushButton(tr("project_manager.actions.save"))
        self._save_btn.setProperty("class", "success")
        self._save_btn.clicked.connect(self._on_save)

        self._snapshot_btn = QPushButton(tr("project_manager.actions.save_version"))
        self._snapshot_btn.setStyleSheet(btn_style)
        self._snapshot_btn.clicked.connect(self._on_snapshot)

        self._import_btn = QPushButton(tr("project_manager.actions.import_asset"))
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.clicked.connect(self._on_import_asset)

        self._delete_asset_btn = QPushButton(tr("project_manager.actions.delete_asset"))
        self._delete_asset_btn.setProperty("class", "dangerBtn")
        self._delete_asset_btn.setEnabled(False)
        self._delete_asset_btn.clicked.connect(self._on_delete_asset)

        self._provenance_btn = QPushButton(tr("project_manager.actions.open_provenance"))
        self._provenance_btn.setStyleSheet(btn_style)
        self._provenance_btn.setEnabled(False)
        self._provenance_btn.clicked.connect(self._on_open_provenance)

        self._rerender_btn = QPushButton(tr("project_manager.actions.rerender"))
        self._rerender_btn.setStyleSheet(btn_style)
        self._rerender_btn.setEnabled(False)
        self._rerender_btn.clicked.connect(self._on_rerender_from_provenance)

        self._disclosure_btn = QPushButton(tr("project_manager.actions.export_disclosure"))
        self._disclosure_btn.setStyleSheet(btn_style)
        self._disclosure_btn.setEnabled(False)
        self._disclosure_btn.clicked.connect(self._on_export_disclosure)

        self._dawproject_btn = QPushButton(tr("project_manager.actions.export_dawproject"))
        self._dawproject_btn.setStyleSheet(btn_style)
        self._dawproject_btn.setEnabled(False)
        self._dawproject_btn.clicked.connect(self._on_export_dawproject)

        self._restore_btn = QPushButton(tr("project_manager.actions.restore_version"))
        self._restore_btn.setStyleSheet(btn_style)
        self._restore_btn.setEnabled(False)
        self._restore_btn.clicked.connect(self._on_restore_version)

        btn_row.addWidget(self._save_btn)
        btn_row.addWidget(self._snapshot_btn)
        btn_row.addWidget(self._restore_btn)
        btn_row.addWidget(self._import_btn)
        btn_row.addWidget(self._delete_asset_btn)
        btn_row.addWidget(self._provenance_btn)
        btn_row.addWidget(self._rerender_btn)
        btn_row.addWidget(self._disclosure_btn)
        btn_row.addWidget(self._dawproject_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._update_detail_empty_states(False, False, project_open=False)

        install_accessibility(
            self,
            tr("project_manager.accessibility.detail_name"),
            named_controls=[
                (
                    self._notes,
                    tr("project_manager.accessibility.notes_name"),
                    tr("project_manager.accessibility.notes_description"),
                ),
                (
                    self._contributions,
                    tr("project_manager.accessibility.contributions_name"),
                    tr("project_manager.accessibility.contributions_description"),
                ),
                (
                    self._asset_list,
                    tr("project_manager.accessibility.assets_name"),
                    tr("project_manager.accessibility.assets_description"),
                ),
                (
                    self._version_list,
                    tr("project_manager.accessibility.versions_name"),
                    tr("project_manager.accessibility.versions_description"),
                ),
                (
                    self._save_btn,
                    tr("project_manager.accessibility.save_name"),
                    tr("project_manager.accessibility.save_description"),
                ),
                (
                    self._snapshot_btn,
                    tr("project_manager.accessibility.snapshot_name"),
                    tr("project_manager.accessibility.snapshot_description"),
                ),
                (
                    self._restore_btn,
                    tr("project_manager.accessibility.restore_name"),
                    tr("project_manager.accessibility.restore_description"),
                ),
                (
                    self._import_btn,
                    tr("project_manager.accessibility.import_name"),
                    tr("project_manager.accessibility.import_description"),
                ),
                (
                    self._delete_asset_btn,
                    tr("project_manager.accessibility.delete_asset_name"),
                    tr("project_manager.accessibility.delete_asset_description"),
                ),
                (
                    self._provenance_btn,
                    tr("project_manager.accessibility.provenance_name"),
                    tr("project_manager.accessibility.provenance_description"),
                ),
                (
                    self._rerender_btn,
                    tr("project_manager.accessibility.rerender_name"),
                    tr("runtime.rerender_selected_file"),
                ),
                (
                    self._disclosure_btn,
                    tr("project_manager.accessibility.disclosure_name"),
                    tr("project_manager.accessibility.disclosure_description"),
                ),
                (
                    self._dawproject_btn,
                    tr("project_manager.accessibility.dawproject_name"),
                    tr("project_manager.accessibility.dawproject_description"),
                ),
            ],
        )

    def load_project(self, project: Project):
        """Display project details."""
        self._name_label.setText(project.name)

        created = time.strftime("%b %d, %Y", time.localtime(project.created_at))
        self._meta_label.setText(
            tr(
                "project_manager.detail.metadata",
                created=created,
                tempo=project.tempo,
                musical_key=project.key,
                assets=project.asset_count,
                versions=project.version_count,
            )
        )

        self._notes.setPlainText(project.notes)
        self._contributions.setPlainText(
            format_human_contributions(project.human_contributions)
        )
        self._disclosure_btn.setEnabled(True)
        self._dawproject_btn.setEnabled(True)

        # Assets
        self._asset_list.clear()
        self._asset_by_id = {}
        for asset in project.assets:
            self._asset_by_id[asset.id] = asset
            item = QListWidgetItem(
                tr(
                    "project_manager.detail.asset_item",
                    asset_type=asset.asset_type,
                    name=asset.name,
                    module=asset.module,
                )
            )
            item.setData(Qt.UserRole, asset.id)
            if asset.provenance_path:
                item.setToolTip(asset.provenance_path)
            self._asset_list.addItem(item)
        self._provenance_btn.setEnabled(False)
        self._delete_asset_btn.setEnabled(False)
        self._rerender_btn.setEnabled(False)
        self._rerender_capability_label.setText("")

        # Versions
        self._version_list.clear()
        for ver in sorted(project.versions, key=lambda v: v.version, reverse=True):
            ts = time.strftime("%b %d %I:%M %p", time.localtime(ver.timestamp))
            item = QListWidgetItem(
                tr(
                    "project_manager.version.item",
                    version=ver.version,
                    timestamp=ts,
                    label=ver.label,
                    description=ver.description,
                )
            )
            item.setData(Qt.UserRole, ver.version)
            self._version_list.addItem(item)
        self._restore_btn.setEnabled(False)
        self._version_preview.setText(
            tr("project_manager.version.select_preview") if project.versions
            else tr("project_manager.empty.versions_title")
        )
        self._update_detail_empty_states(bool(project.assets), bool(project.versions))

    def clear(self):
        self._name_label.setText(tr("project_manager.detail.no_project"))
        self._meta_label.setText("")
        self._rerender_capability_label.setText("")
        self._notes.clear()
        self._contributions.clear()
        self._asset_list.clear()
        self._asset_by_id = {}
        self._provenance_btn.setEnabled(False)
        self._delete_asset_btn.setEnabled(False)
        self._rerender_btn.setEnabled(False)
        self._disclosure_btn.setEnabled(False)
        self._dawproject_btn.setEnabled(False)
        self._version_list.clear()
        self._update_detail_empty_states(False, False, project_open=False)

    def _update_detail_empty_states(
        self,
        has_assets: bool,
        has_versions: bool,
        *,
        project_open: bool = True,
    ) -> None:
        if has_assets:
            self._asset_stack.setCurrentWidget(self._asset_list)
        elif project_open:
            self._asset_empty.set_state(
                tr("project_manager.empty.assets_title"),
                tr("project_manager.empty.assets_description"),
                tr("project_manager.actions.import_asset"),
            )
            self._asset_stack.setCurrentWidget(self._asset_empty)
        else:
            self._asset_empty.set_state(
                tr("project_manager.empty.open_assets_title"),
                tr("project_manager.empty.open_assets_description"),
                tr("project_manager.actions.create_project"),
            )
            self._asset_stack.setCurrentWidget(self._asset_empty)

        if has_versions:
            self._version_stack.setCurrentWidget(self._version_list)
        elif project_open:
            self._version_empty.set_state(
                tr("project_manager.empty.versions_title"),
                tr("project_manager.empty.versions_description"),
                tr("project_manager.actions.save_version"),
            )
            self._version_stack.setCurrentWidget(self._version_empty)
        else:
            self._version_empty.set_state(
                tr("project_manager.empty.open_versions_title"),
                tr("project_manager.empty.open_versions_description"),
                tr("project_manager.actions.create_project"),
            )
            self._version_stack.setCurrentWidget(self._version_empty)

    def _on_asset_empty_action(self):
        if get_project_manager().current is None:
            self.create_requested.emit()
        else:
            self._on_import_asset()

    def _on_version_empty_action(self):
        if get_project_manager().current is None:
            self.create_requested.emit()
        else:
            self._on_snapshot()

    def sync_pending_edits(self):
        """Copy editor contents into the open project before a shell flush."""
        project = get_project_manager().current
        if project is not None:
            project.notes = self._notes.toPlainText()
            contributions = getattr(self, "_contributions", None)
            if contributions is not None:
                project.human_contributions = parse_human_contributions(
                    contributions.toPlainText()
                )

    def _on_save(self):
        mgr = get_project_manager()
        if mgr.current:
            self.sync_pending_edits()
            saved = mgr.save()
            if self.toast_mgr:
                if saved:
                    self.toast_mgr.success(tr("project_manager.messages.saved"))
                else:
                    self.toast_mgr.error(tr("project_manager.messages.save_failed"))
            if saved:
                self.load_project(mgr.current)
            return saved
        if self.toast_mgr:
            self.toast_mgr.error(tr("project_manager.messages.no_project_save"))
        return False

    def _on_snapshot(self):
        mgr = get_project_manager()
        if mgr.current:
            desc, ok = QInputDialog.getText(
                self,
                tr("project_manager.version.dialog_title"),
                tr("project_manager.version.dialog_prompt"),
            )
            if ok:
                mgr.current.notes = self._notes.toPlainText()
                version = mgr.create_version(
                    desc or tr("project_manager.version.manual_save")
                )
                if version is None and self.toast_mgr:
                    self.toast_mgr.error(tr("project_manager.messages.snapshot_failed"))
                elif version is not None and self.toast_mgr:
                    self.toast_mgr.success(
                        tr("project_manager.messages.version_saved", version=version.version)
                    )
                self.load_project(mgr.current)

    def _selected_version(self) -> Optional[int]:
        item = self._version_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.UserRole)
        return int(value) if value is not None else None

    def _on_version_selected(self, current, previous):
        version = self._selected_version()
        if version is None:
            self._restore_btn.setEnabled(False)
            self._version_preview.setText(tr("project_manager.version.select_preview"))
            return
        preview = get_project_manager().version_preview(version)
        if preview is None:
            self._restore_btn.setEnabled(False)
            self._version_preview.setText(
                tr("project_manager.version.missing", version=version)
            )
            return
        assets = preview["asset_names"][:4]
        asset_summary = ", ".join(a for a in assets if a) or tr("project_manager.data.none")
        if preview["asset_count"] > len(assets):
            asset_summary += tr(
                "project_manager.version.more_assets",
                count=preview["asset_count"] - len(assets),
            )
        self._version_preview.setText(
            tr(
                "project_manager.version.preview",
                version=version,
                kind=preview["kind"],
                name=preview["name"],
                tempo=preview["tempo"],
                musical_key=preview["key"],
                asset_count=preview["asset_count"],
                asset_summary=asset_summary,
                mixer_track_count=preview["mixer_track_count"],
            )
        )
        self._restore_btn.setEnabled(True)

    def _on_restore_version(self):
        mgr = get_project_manager()
        version = self._selected_version()
        if mgr.current is None or version is None:
            return
        confirm = QMessageBox.question(
            self,
            tr("project_manager.version.restore_title"),
            tr("project_manager.version.restore_prompt", version=version),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        restored = mgr.restore_version(version)
        if restored is None:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.restore_failed", version=version)
                )
            return
        self.load_project(restored)
        if self.toast_mgr:
            self.toast_mgr.success(
                tr("project_manager.messages.restored", version=version)
            )

    def _on_import_asset(self):
        mgr = get_project_manager()
        if not mgr.current:
            return

        fallback_dir = (
            os.path.dirname(mgr.current.assets[0].file_path)
            if mgr.current.assets else None
        )
        paths, _ = open_project_files(
            self,
            tr("project_manager.dialogs.import_assets_title"),
            operation_kind="project_asset_import",
            dialog=QFileDialog,
            fallback_dir=fallback_dir,
        )
        if paths:
            imported = 0
            for path in paths:
                atype = "midi" if is_midi_path(path) else "audio"
                try:
                    asset_id = mgr.import_asset(path, atype, "project_manager")
                except Exception as exc:
                    if self.toast_mgr:
                        self.toast_mgr.error(
                            tr("project_manager.messages.asset_import_failed", error=exc)
                        )
                    continue
                if asset_id:
                    imported += 1
                    continue
                if self.toast_mgr:
                    self.toast_mgr.error(
                        tr("project_manager.messages.asset_import_no_project")
                    )
            if imported:
                self.load_project(mgr.current)
                if self.toast_mgr:
                    self.toast_mgr.success(
                        tr("project_manager.messages.assets_imported", count=imported)
                    )

    def _on_delete_asset(self):
        mgr = get_project_manager()
        asset = self._selected_asset()
        if mgr.current is None or asset is None:
            return

        entry = mgr.delete_asset(asset.id)
        if entry is None:
            if self.toast_mgr:
                self.toast_mgr.error(tr("project_manager.messages.asset_delete_failed"))
            return

        self.load_project(mgr.current)
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("project_manager.messages.asset_moved_to_trash"),
                duration_ms=8000,
                action_label=tr("project_manager.actions.undo"),
                action_callback=lambda entry_id=entry.id: self._restore_asset(entry_id),
            )

    def _restore_asset(self, trash_entry_id: str):
        mgr = get_project_manager()
        if not mgr.restore_deleted_asset(trash_entry_id):
            if self.toast_mgr:
                self.toast_mgr.error(tr("project_manager.messages.asset_restore_failed"))
            return
        if mgr.current is not None:
            self.load_project(mgr.current)
        if self.toast_mgr:
            self.toast_mgr.success(tr("project_manager.messages.asset_restored"))

    def _selected_asset(self) -> Optional[ProjectAsset]:
        item = self._asset_list.currentItem()
        if item is None:
            return None
        return self._asset_by_id.get(item.data(Qt.UserRole))

    def _on_asset_selected(self, current, previous):
        asset = self._selected_asset()
        has_provenance = bool(asset and asset.provenance_path and os.path.isfile(asset.provenance_path))
        self._provenance_btn.setEnabled(has_provenance)
        self._delete_asset_btn.setEnabled(asset is not None)
        self._set_rerender_capability(asset if has_provenance else None)

    def _set_rerender_capability(self, asset: Optional[ProjectAsset]) -> None:
        """Show the recorded operation's replay contract before a long run."""
        if asset is None or not asset.provenance_path:
            self._rerender_capability_label.setText("")
            self._rerender_btn.setToolTip("")
            self._rerender_btn.setEnabled(False)
            return
        record = read_provenance_sidecar(asset.provenance_path)
        capability = provenance_replayability(record)
        label = capability.state.replace("_", " ").capitalize()
        self._rerender_btn.setToolTip(capability.reason)
        if capability.state == "not_replayable":
            self._rerender_capability_label.setText(
                tr("runtime.rerender_capability_unavailable", reason=capability.reason)
            )
            self._rerender_btn.setEnabled(False)
        else:
            self._rerender_capability_label.setText(
                tr(
                    "runtime.rerender_capability",
                    state=label,
                    reason=capability.reason,
                )
            )
            self._rerender_btn.setEnabled(self._rerender_worker is None)

    def _on_open_provenance(self):
        asset = self._selected_asset()
        if not asset or not asset.provenance_path:
            return

        record = read_provenance_sidecar(asset.provenance_path)
        if not record:
            record = asset.metadata.get("provenance", {})

        dialog = QDialog(self)
        dialog.setWindowTitle(
            tr("project_manager.dialogs.provenance_title", asset=asset.name)
        )
        dialog.resize(720, 520)
        t = ThemeEngine.get_colors()
        layout = QVBoxLayout(dialog)

        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(json.dumps(record, indent=2, ensure_ascii=False))
        editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background: {t['surface']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 8.25pt;
            }}
        """)
        layout.addWidget(editor, 1)

        close_btn = QPushButton(tr("project_manager.actions.close"))
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        dialog.exec()

    def _on_rerender_from_provenance(self):
        asset = self._selected_asset()
        if asset is None or not asset.provenance_path:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.select_provenance_asset")
                )
            return
        if self._rerender_worker is not None:
            if self.toast_mgr:
                self.toast_mgr.warning(
                    tr("project_manager.messages.rerender_already_running")
                )
            return

        provenance = read_provenance_sidecar(asset.provenance_path)
        capability = provenance_replayability(provenance)
        if capability.state == "not_replayable":
            self._set_rerender_capability(asset)
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr(
                        "runtime.rerender_capability_unavailable",
                        reason=capability.reason,
                    )
                )
            return
        compatibility = check_provenance_compatibility(provenance)
        if not compatibility.compatible:
            detail = "\n".join(diff.format() for diff in compatibility.diffs[:5])
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.rerender_incompatible") + "\n" + detail
                )
            return

        worker = InferenceWorker(_rerender_provenance_task, asset.file_path)
        self._rerender_worker = worker
        self._rerender_workers.add(worker)
        worker.progress.connect(
            lambda percent: self._meta_label.setText(
                tr(
                    "project_manager.messages.rerender_progress",
                    asset=asset.name,
                    percent=percent,
                )
            )
        )
        worker.finished.connect(
            lambda result, item=asset, current_worker=worker:
            self._on_rerender_finished(item, current_worker, result)
        )
        worker.error.connect(
            lambda message, current_worker=worker:
            self._on_rerender_error(current_worker, message)
        )
        worker.cancelled.connect(
            lambda current_worker=worker: self._on_rerender_cancelled(current_worker)
        )
        worker.thread_stopped.connect(
            lambda current_worker=worker: self._rerender_workers.discard(current_worker)
        )
        self._rerender_btn.setEnabled(False)
        worker.start()
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("project_manager.messages.rerender_started", asset=asset.name)
            )

    def _on_rerender_finished(self, source_asset: ProjectAsset, worker, result):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._set_rerender_capability(source_asset)
        if not result.identical:
            detail = "\n".join(diff.format() for diff in result.differences)
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("runtime.rerender_bytes_differ") + "\n" + detail
                )
            return

        manager = get_project_manager()
        if manager.current is None:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.rerender_no_project")
                )
            return
        try:
            asset_id = manager.import_asset(
                result.rerendered_path,
                source_asset.asset_type or "audio",
                "provenance_rerender",
                name=f"{source_asset.name} (re-rendered)",
            )
        except Exception as exc:
            if self.toast_mgr:
                self.toast_mgr.error(tr("runtime.rerender_import_failed", error=exc))
            return
        if not asset_id:
            if self.toast_mgr:
                self.toast_mgr.error(tr("runtime.rerender_project_failed"))
            return
        self.load_project(manager.current)
        if self.toast_mgr:
            self.toast_mgr.success(
                tr(
                    "project_manager.messages.rerender_succeeded",
                    asset=source_asset.name,
                )
            )

    def _on_rerender_error(self, worker, message: str):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._set_rerender_capability(self._selected_asset())
        if self.toast_mgr:
            self.toast_mgr.error(
                tr("project_manager.messages.rerender_failed", error=message)
            )

    def _on_rerender_cancelled(self, worker):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._set_rerender_capability(self._selected_asset())
        if self.toast_mgr:
            self.toast_mgr.warning(tr("project_manager.messages.rerender_cancelled"))

    def _on_export_disclosure(self):
        """Write the project disclosure record as JSON and TSV."""
        manager = get_project_manager()
        project = manager.current
        if project is None:
            if self.toast_mgr:
                self.toast_mgr.error(tr("project_manager.messages.no_project_report"))
            return

        self.sync_pending_edits()
        output_dir = choose_directory(
            self,
            tr("project_manager.dialogs.export_disclosure_title"),
            operation_kind="project_disclosure_export",
            fallback_dir=(
                os.path.dirname(project.assets[0].file_path)
                if project.assets else None
            ),
            dialog=QFileDialog,
        )
        if not output_dir:
            return
        if not manager.save(project):
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.disclosure_save_failed")
                )
            return
        try:
            json_path, tsv_path = write_disclosure_report(project, output_dir)
        except (OSError, TypeError, ValueError) as exc:
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.disclosure_failed", error=exc)
                )
            return
        if self.toast_mgr:
            self.toast_mgr.success(
                tr(
                    "project_manager.messages.disclosure_exported",
                    json=json_path.name,
                    tsv=tsv_path.name,
                )
            )

    def _on_export_dawproject(self):
        """Export the open project's existing audio assets for a DAW."""
        manager = get_project_manager()
        project = manager.current
        if project is None:
            if self.toast_mgr:
                self.toast_mgr.error(tr("project_manager.messages.dawproject_no_project"))
            return
        if self._dawproject_worker is not None and self._dawproject_worker.isRunning():
            if self.toast_mgr:
                self.toast_mgr.warning(
                    tr("project_manager.messages.dawproject_already_running")
                )
            return

        self.sync_pending_edits()
        spec = spec_from_project(project)
        if not spec.tracks:
            if self.toast_mgr:
                self.toast_mgr.warning(
                    tr("project_manager.messages.dawproject_no_audio")
                )
            return
        if not manager.save(project):
            if self.toast_mgr:
                self.toast_mgr.error(
                    tr("project_manager.messages.dawproject_save_failed")
                )
            return

        fallback_dir = os.path.dirname(spec.tracks[0].media_file)
        path, selected_filter = save_file(
            self,
            tr("project_manager.dialogs.export_dawproject_title"),
            f"{project.name or tr('project_manager.data.slunder_project')}.dawproject",
            "DAWproject (*.dawproject);;All Files (*)",
            "project_dawproject_export",
            dialog=QFileDialog,
            fallback_dir=fallback_dir,
        )
        if not path:
            return
        path = ensure_extension(path, selected_filter, default="dawproject")
        worker = InferenceWorker(
            _export_dawproject_task,
            spec,
            path,
            job_kind="project_dawproject_export",
            job_label=tr(
                "project_manager.jobs.dawproject_export",
                project=project.name or tr("project_manager.data.untitled"),
            ),
            job_inputs={"track_count": len(spec.tracks), "output_path": path},
        )
        self._dawproject_worker = worker
        self._dawproject_workers.add(worker)
        worker.progress.connect(
            lambda percent: self._meta_label.setText(
                tr(
                    "project_manager.messages.dawproject_progress",
                    percent=percent,
                )
            )
        )
        worker.finished.connect(self._on_export_dawproject_finished)
        worker.error.connect(self._on_export_dawproject_error)
        worker.cancelled.connect(self._on_export_dawproject_cancelled)
        worker.thread_stopped.connect(
            lambda current_worker=worker: self._dawproject_workers.discard(current_worker)
        )
        self._dawproject_btn.setEnabled(False)
        self._meta_label.setText(tr("project_manager.messages.dawproject_exporting"))
        worker.start()

    def _on_export_dawproject_finished(self, result: dict):
        worker = self._dawproject_worker
        self._dawproject_worker = None
        self._dawproject_btn.setEnabled(True)
        path = str(result.get("path", ""))
        self._meta_label.setText(
            tr(
                "project_manager.messages.dawproject_validated",
                count=result.get("track_count", 0),
            )
        )
        if self.toast_mgr:
            self.toast_mgr.success(
                tr("project_manager.messages.dawproject_exported", path=path)
            )
        if worker is not None and worker.isRunning():
            self._dawproject_workers.add(worker)

    def _on_export_dawproject_error(self, message: str):
        self._dawproject_worker = None
        self._dawproject_btn.setEnabled(True)
        self._meta_label.setText(tr("project_manager.messages.dawproject_export_failed"))
        if self.toast_mgr:
            self.toast_mgr.error(
                tr("project_manager.messages.dawproject_failed", error=message)
            )

    def _on_export_dawproject_cancelled(self):
        self._dawproject_worker = None
        self._dawproject_btn.setEnabled(True)
        self._meta_label.setText(tr("project_manager.messages.dawproject_cancelled"))
        if self.toast_mgr:
            self.toast_mgr.warning(tr("project_manager.messages.dawproject_cancelled"))


# ── Project Manager View ───────────────────────────────────────────────────────

class ProjectManagerView(QWidget):
    """Main project management page."""

    project_opened = Signal(str)  # project_id

    def __init__(self, toast_mgr=None, parent=None):
        super().__init__(parent)
        self._cards: list[ProjectCard] = []
        self.toast_mgr = toast_mgr
        self._last_repair_notice = ""

        t = ThemeEngine.get_colors()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # ── Left: Project Browser ──────────────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(tr("project_manager.library.title"))
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12pt;")

        self._new_btn = QPushButton(tr("project_manager.actions.new_project"))
        self._new_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: {t['background']}; border: none;
                border-radius: 5px; padding: 6px 14px;
                font-weight: bold; font-size: 9pt;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._new_btn.clicked.connect(self._on_new_project)

        self._rescan_btn = QPushButton(tr("project_manager.actions.rescan"))
        self._rescan_btn.setToolTip(
            tr("project_manager.library.rescan_tooltip")
        )
        self._rescan_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 5px;
                padding: 6px 12px; font-weight: bold; font-size: 8.25pt;
            }}
            QPushButton:hover {{ border-color: {t['accent']}; }}
        """)
        self._rescan_btn.clicked.connect(self._on_rescan_projects)

        header.addWidget(title)
        header.addStretch()
        header.addWidget(self._rescan_btn)
        header.addWidget(self._new_btn)
        left.addLayout(header)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText(tr("project_manager.library.search_placeholder"))
        self._search.setStyleSheet(f"""
            QLineEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px 12px; font-size: 9pt;
            }}
        """)
        self._search.textChanged.connect(self._on_search)
        left.addWidget(self._search)

        sort_row = QHBoxLayout()
        sort_label = QLabel(tr("project_manager.library.sort_by"))
        sort_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        sort_row.addWidget(sort_label)
        self._sort_combo = QComboBox()
        self._sort_combo.addItem(tr("project_manager.library.sort_newest"), "updated_desc")
        self._sort_combo.addItem(tr("project_manager.library.sort_oldest"), "updated_asc")
        self._sort_combo.addItem(tr("project_manager.library.sort_name_asc"), "name_asc")
        self._sort_combo.addItem(tr("project_manager.library.sort_name_desc"), "name_desc")
        self._sort_combo.setMinimumHeight(30)
        self._sort_combo.currentIndexChanged.connect(self._refresh_list)
        sort_row.addWidget(self._sort_combo, 1)
        left.addLayout(sort_row)

        # Project list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(4)
        self._library_empty = EmptyStateWidget(
            tr("project_manager.empty.library_title"),
            tr("project_manager.empty.library_description"),
            tr("project_manager.actions.new_project"),
        )
        self._library_empty.action_requested.connect(self._on_library_empty_action)
        self._list_layout.addWidget(self._library_empty)
        self._list_layout.addStretch()

        self._scroll.setWidget(self._list_container)
        left.addWidget(self._scroll, 1)

        # Count
        self._count_label = QLabel("")
        self._count_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 7.5pt;")
        left.addWidget(self._count_label)

        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(400)
        layout.addWidget(left_w)

        # ── Right: Project Detail ──────────────────────────────────────────
        self._detail = ProjectDetailPanel(toast_mgr=toast_mgr)
        self._detail.create_requested.connect(self._on_new_project)
        layout.addWidget(self._detail, 1)

        # Load initial project list
        self._refresh_list()
        install_accessibility(
            self,
            tr("project_manager.accessibility.library_name"),
            named_controls=[
                (
                    self._new_btn,
                    tr("project_manager.accessibility.new_name"),
                    tr("project_manager.accessibility.new_description"),
                ),
                (
                    self._rescan_btn,
                    tr("project_manager.accessibility.rescan_name"),
                    tr("project_manager.accessibility.rescan_description"),
                ),
                (
                    self._search,
                    tr("project_manager.accessibility.search_name"),
                    tr("project_manager.accessibility.search_description"),
                ),
                (
                    self._sort_combo,
                    tr("project_manager.accessibility.sort_name"),
                    tr("project_manager.accessibility.sort_description"),
                ),
            ],
        )

    @staticmethod
    def _sort_projects(projects: list[dict], sort_key: str | None) -> list[dict]:
        """Return a deterministic project order without changing the stored index."""
        sort_key = sort_key or "updated_desc"
        if sort_key in {"name_asc", "name_desc"}:
            return sorted(
                projects,
                key=lambda item: (
                    str(item.get("name") or "Untitled").casefold(),
                    str(item.get("id") or "").casefold(),
                ),
                reverse=sort_key == "name_desc",
            )
        return sorted(
            projects,
            key=lambda item: (
                float(item.get("updated_at") or 0),
                str(item.get("id") or "").casefold(),
            ),
            reverse=sort_key != "updated_asc",
        )

    def _refresh_list(self):
        """Reload project list from ProjectManager."""
        # Clear existing cards
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        mgr = get_project_manager()
        projects = self._sort_projects(
            mgr.list_projects(), self._sort_combo.currentData()
        )

        for info in projects:
            card = ProjectCard(info)
            card.open_requested.connect(self._on_open_project)
            card.delete_requested.connect(self._on_delete_project)
            self._cards.append(card)
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        count_text = tr("project_manager.library.count", count=len(projects))
        repair_text = self._repair_status_text(mgr.last_repair_status)
        self._count_label.setText(
            f"{count_text} | {repair_text}" if repair_text else count_text
        )
        self._count_label.setToolTip(
            self._repair_status_details(mgr.last_repair_status)
        )
        if (
            repair_text
            and repair_text != self._last_repair_notice
            and self.toast_mgr
        ):
            self.toast_mgr.warning(repair_text)
        if repair_text:
            self._last_repair_notice = repair_text
        self._on_search(self._search.text())

    def sync_pending_edits(self):
        """Flush the detail editor into ProjectManager without writing yet."""
        self._detail.sync_pending_edits()

    def _on_rescan_projects(self):
        mgr = get_project_manager()
        count = mgr.rebuild_index()
        self._refresh_list()
        repair_text = self._repair_status_text(mgr.last_repair_status)
        if self.toast_mgr and not repair_text:
            self.toast_mgr.success(
                tr("project_manager.messages.index_verified", count=count)
            )

    def _on_new_project(self):
        name, ok = QInputDialog.getText(
            self,
            tr("project_manager.dialogs.new_title"),
            tr("project_manager.dialogs.project_name"),
        )
        if ok and name.strip():
            mgr = get_project_manager()
            project = mgr.create(name.strip())
            self._refresh_list()
            self._detail.load_project(project)
            self.project_opened.emit(project.id)

    def _on_open_project(self, project_id: str):
        mgr = get_project_manager()
        project = mgr.open(project_id)
        repair_text = self._repair_status_text(mgr.last_repair_status)
        if project:
            self._detail.load_project(project)
            self.project_opened.emit(project_id)
            if repair_text:
                self._count_label.setText(repair_text)
                if self.toast_mgr:
                    self.toast_mgr.warning(repair_text)
        else:
            if repair_text:
                self._count_label.setText(repair_text)
            if self.toast_mgr:
                self.toast_mgr.error(
                    repair_text or tr("project_manager.messages.open_failed")
                )

    def _on_delete_project(self, project_id: str):
        mgr = get_project_manager()
        entry = mgr.delete(project_id)
        if not entry:
            if self.toast_mgr:
                self.toast_mgr.error(tr("project_manager.messages.delete_failed"))
            return

        self._detail.clear()
        self._refresh_list()
        if self.toast_mgr:
            self.toast_mgr.info(
                tr("project_manager.messages.moved_to_trash"),
                duration_ms=8000,
                action_label=tr("project_manager.actions.undo"),
                action_callback=lambda entry_id=entry.id: self._restore_project(entry_id),
            )

    def _restore_project(self, trash_entry_id: str):
        mgr = get_project_manager()
        if mgr.restore_deleted_project(trash_entry_id):
            self._refresh_list()
            if self.toast_mgr:
                self.toast_mgr.success(tr("project_manager.messages.restored_project"))
        elif self.toast_mgr:
            self.toast_mgr.error(tr("project_manager.messages.restore_project_failed"))

    def _repair_status_text(self, status: dict) -> str:
        state = status.get("status", "ok")
        if state == "ok":
            return ""
        label = {
            "migrated": tr("project_manager.recovery.migrated"),
            "repaired": tr("project_manager.recovery.repaired"),
            "error": tr("project_manager.recovery.error"),
        }.get(state, tr("project_manager.recovery.error"))
        if status.get("backup_paths"):
            label += tr("project_manager.recovery.backup_saved")
        return label

    @staticmethod
    def _repair_status_details(status: dict) -> str:
        messages = status.get("messages") or []
        backups = status.get("backup_paths") or []
        details = "\n".join(str(message) for message in messages)
        if backups:
            details += ("\n" if details else "") + f"Latest backup: {backups[-1]}"
        return details

    def _on_search(self, text: str):
        query = text.casefold().strip()
        visible_count = 0
        for card in self._cards:
            visible = card.matches_query(query)
            card.setVisible(visible)
            visible_count += int(visible)
        if visible_count:
            self._library_empty.hide()
        elif query:
            self._library_empty.set_no_matches(
                tr("project_manager.empty.no_matches", query=text.strip()),
                tr("project_manager.actions.clear_search"),
            )
            self._library_empty.show()
        else:
            self._library_empty.set_state(
                tr("project_manager.empty.library_title"),
                tr("project_manager.empty.library_description"),
                tr("project_manager.actions.new_project"),
            )
            self._library_empty.show()

    def _on_library_empty_action(self):
        if self._library_empty.state == "no_matches":
            self._search.clear()
        else:
            self._on_new_project()
