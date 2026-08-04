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
    QDialog, QPlainTextEdit, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from ui.theme import ThemeEngine, rgba
from ui.accessibility import install_accessibility
from ui.widgets import EmptyStateWidget
from core.project import ProjectManager, Project, ProjectAsset, get_project_manager
from core.i18n import tr
from core.disclosure import (
    format_human_contributions,
    parse_human_contributions,
    write_disclosure_report,
)
from core.provenance import (
    check_provenance_compatibility,
    read_provenance_sidecar,
    rerender_from_provenance,
)
from core.workers import InferenceWorker
from core.routing import is_midi_path
from ui.file_dialogs import choose_directory, open_project_files


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
        self.setMinimumHeight(72)
        self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        # Info
        info = QVBoxLayout()
        info.setSpacing(2)

        name = QLabel(project_info.get("name", "Untitled"))
        name.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9.75pt;")
        info.addWidget(name)

        updated = project_info.get("updated_at", 0)
        if updated:
            time_str = time.strftime("%b %d, %Y %I:%M %p", time.localtime(updated))
        else:
            time_str = "Unknown"
        date_label = QLabel(f"Last modified: {time_str}")
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

        open_btn = QPushButton("Open")
        open_btn.setStyleSheet(btn_style.replace(t['background'], t['accent']).replace(t['text'] + ';', 'white;'))
        open_btn.clicked.connect(lambda: self.open_requested.emit(self._project_id))

        del_btn = QPushButton("Delete")
        del_btn.setStyleSheet(btn_style)
        del_btn.clicked.connect(lambda: self.delete_requested.emit(self._project_id))

        layout.addWidget(open_btn)
        layout.addWidget(del_btn)

        self._open_btn = open_btn
        self._delete_btn = del_btn
        install_accessibility(
            self,
            f"Project {project_info.get('name', 'Untitled')}",
            named_controls=[
                (self._open_btn, "Open project", "Opens this project in the studio."),
                (self._delete_btn, "Delete project", "Moves this project to recoverable trash."),
            ],
        )

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Project info header
        self._name_label = QLabel("No Project Open")
        self._name_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12pt;")
        layout.addWidget(self._name_label)

        self._meta_label = QLabel("")
        self._meta_label.setStyleSheet(f"color: {t['text_secondary']}; font-size: 8.25pt;")
        layout.addWidget(self._meta_label)

        # Notes
        self._notes = QTextEdit()
        self._notes.setPlaceholderText("Project notes...")
        self._notes.setMaximumHeight(80)
        self._notes.setStyleSheet(f"""
            QTextEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 4px;
                padding: 6px; font-size: 8.25pt;
            }}
        """)
        layout.addWidget(self._notes)

        contributions_label = QLabel("Human contributions (registration evidence)")
        contributions_label.setStyleSheet(
            f"color: {t['text']}; font-weight: bold; font-size: 9pt;"
        )
        layout.addWidget(contributions_label)

        self._contributions = QTextEdit()
        self._contributions.setPlaceholderText(
            "One declaration per line, for example:\n"
            "lyrics: wrote the chorus\n"
            "midi: drew the bass notes\n"
            "edits: chose the final vocal take"
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
        assets_label = QLabel("Assets")
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
            "No project assets yet",
            "Import audio or MIDI into the open project to see it here.",
            "Import asset",
        )
        self._asset_empty.action_requested.connect(self._on_asset_empty_action)
        self._asset_stack = QStackedWidget()
        self._asset_stack.addWidget(self._asset_list)
        self._asset_stack.addWidget(self._asset_empty)
        layout.addWidget(self._asset_stack, 1)

        # Version history
        ver_label = QLabel("Version History")
        ver_label.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 9pt;")
        layout.addWidget(ver_label)

        self._version_list = QListWidget()
        self._version_list.setMaximumHeight(120)
        self._version_list.setStyleSheet(self._asset_list.styleSheet())
        self._version_list.currentItemChanged.connect(self._on_version_selected)
        self._version_empty = EmptyStateWidget(
            "No saved versions yet",
            "Save a version when you want a recoverable project checkpoint.",
            "Save version",
        )
        self._version_empty.action_requested.connect(self._on_version_empty_action)
        self._version_stack = QStackedWidget()
        self._version_stack.addWidget(self._version_list)
        self._version_stack.addWidget(self._version_empty)
        layout.addWidget(self._version_stack)

        self._version_preview = QLabel("Select a version to preview it.")
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

        self._save_btn = QPushButton("Save")
        self._save_btn.setProperty("class", "success")
        self._save_btn.clicked.connect(self._on_save)

        self._snapshot_btn = QPushButton("Save Version")
        self._snapshot_btn.setStyleSheet(btn_style)
        self._snapshot_btn.clicked.connect(self._on_snapshot)

        self._import_btn = QPushButton("Import Asset")
        self._import_btn.setStyleSheet(btn_style)
        self._import_btn.clicked.connect(self._on_import_asset)

        self._delete_asset_btn = QPushButton("Delete Asset")
        self._delete_asset_btn.setProperty("class", "dangerBtn")
        self._delete_asset_btn.setEnabled(False)
        self._delete_asset_btn.clicked.connect(self._on_delete_asset)

        self._provenance_btn = QPushButton("Open Provenance")
        self._provenance_btn.setStyleSheet(btn_style)
        self._provenance_btn.setEnabled(False)
        self._provenance_btn.clicked.connect(self._on_open_provenance)

        self._rerender_btn = QPushButton("Re-render from Provenance")
        self._rerender_btn.setStyleSheet(btn_style)
        self._rerender_btn.setEnabled(False)
        self._rerender_btn.clicked.connect(self._on_rerender_from_provenance)

        self._disclosure_btn = QPushButton("Export AI Disclosure")
        self._disclosure_btn.setStyleSheet(btn_style)
        self._disclosure_btn.setEnabled(False)
        self._disclosure_btn.clicked.connect(self._on_export_disclosure)

        self._restore_btn = QPushButton("Restore Version")
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
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._update_detail_empty_states(False, False, project_open=False)

        install_accessibility(
            self,
            "Project details",
            named_controls=[
                (self._notes, "Project notes", "Edits notes saved with the current project."),
                (
                    self._contributions,
                    "Human contributions",
                    "Records user-declared authorship evidence for the disclosure report.",
                ),
                (self._asset_list, "Project assets", "Lists assets in the current project."),
                (self._version_list, "Project versions", "Lists saved project versions."),
                (self._save_btn, "Save project", "Saves the current project data."),
                (self._snapshot_btn, "Save project version", "Creates a version snapshot."),
                (self._restore_btn, "Restore project version", "Restores the selected version."),
                (self._import_btn, "Import project asset", "Imports an asset into the project."),
                (self._delete_asset_btn, "Delete project asset", "Moves the selected asset to recoverable trash."),
                (self._provenance_btn, "Open asset provenance", "Opens provenance for the selected asset."),
                (self._rerender_btn, "Re-render from provenance", tr("runtime.rerender_selected_file")),
                (
                    self._disclosure_btn,
                    "Export AI disclosure",
                    "Exports a JSON and copy-pasteable TSV AI disclosure and human-authorship record.",
                ),
            ],
        )

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
        self._contributions.setPlainText(
            format_human_contributions(project.human_contributions)
        )
        self._disclosure_btn.setEnabled(True)

        # Assets
        self._asset_list.clear()
        self._asset_by_id = {}
        for asset in project.assets:
            self._asset_by_id[asset.id] = asset
            item = QListWidgetItem(
                f"[{asset.asset_type}] {asset.name} ({asset.module})"
            )
            item.setData(Qt.UserRole, asset.id)
            if asset.provenance_path:
                item.setToolTip(asset.provenance_path)
            self._asset_list.addItem(item)
        self._provenance_btn.setEnabled(False)
        self._delete_asset_btn.setEnabled(False)
        self._rerender_btn.setEnabled(False)

        # Versions
        self._version_list.clear()
        for ver in sorted(project.versions, key=lambda v: v.version, reverse=True):
            ts = time.strftime("%b %d %I:%M %p", time.localtime(ver.timestamp))
            item = QListWidgetItem(
                f"v{ver.version} - {ts} ({ver.label}): {ver.description}"
            )
            item.setData(Qt.UserRole, ver.version)
            self._version_list.addItem(item)
        self._restore_btn.setEnabled(False)
        self._version_preview.setText(
            "Select a version to preview it." if project.versions
            else "No saved versions yet."
        )
        self._update_detail_empty_states(bool(project.assets), bool(project.versions))

    def clear(self):
        self._name_label.setText("No Project Open")
        self._meta_label.setText("")
        self._notes.clear()
        self._contributions.clear()
        self._asset_list.clear()
        self._asset_by_id = {}
        self._provenance_btn.setEnabled(False)
        self._delete_asset_btn.setEnabled(False)
        self._rerender_btn.setEnabled(False)
        self._disclosure_btn.setEnabled(False)
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
                "No project assets yet",
                "Import audio or MIDI into the open project to see it here.",
                "Import asset",
            )
            self._asset_stack.setCurrentWidget(self._asset_empty)
        else:
            self._asset_empty.set_state(
                "Open a project to manage assets",
                "Create or open a project, then import audio or MIDI here.",
                "Create project",
            )
            self._asset_stack.setCurrentWidget(self._asset_empty)

        if has_versions:
            self._version_stack.setCurrentWidget(self._version_list)
        elif project_open:
            self._version_empty.set_state(
                "No saved versions yet",
                "Save a version when you want a recoverable project checkpoint.",
                "Save version",
            )
            self._version_stack.setCurrentWidget(self._version_empty)
        else:
            self._version_empty.set_state(
                "Open a project to save versions",
                "Create or open a project before saving a checkpoint.",
                "Create project",
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
                    self.toast_mgr.success("Project saved.")
                else:
                    self.toast_mgr.error("Could not save the project; your changes remain dirty.")
            if saved:
                self.load_project(mgr.current)
            return saved
        if self.toast_mgr:
            self.toast_mgr.error("No project is open to save.")
        return False

    def _on_snapshot(self):
        mgr = get_project_manager()
        if mgr.current:
            desc, ok = QInputDialog.getText(self, "Version Description",
                                            "Description for this version:")
            if ok:
                mgr.current.notes = self._notes.toPlainText()
                version = mgr.create_version(desc or "Manual save")
                if version is None and self.toast_mgr:
                    self.toast_mgr.error("Could not write the version snapshot.")
                elif version is not None and self.toast_mgr:
                    self.toast_mgr.success(f"Saved version v{version.version}.")
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
            self._version_preview.setText("Select a version to preview it.")
            return
        preview = get_project_manager().version_preview(version)
        if preview is None:
            self._restore_btn.setEnabled(False)
            self._version_preview.setText(
                f"v{version} snapshot file is missing; it cannot be previewed or restored."
            )
            return
        assets = preview["asset_names"][:4]
        asset_summary = ", ".join(a for a in assets if a) or "none"
        if preview["asset_count"] > len(assets):
            asset_summary += f", +{preview['asset_count'] - len(assets)} more"
        self._version_preview.setText(
            f"v{version} ({preview['kind']}) - {preview['name']} - "
            f"{preview['tempo']:.0f} BPM, {preview['key']} - "
            f"{preview['asset_count']} asset(s): {asset_summary} - "
            f"{preview['mixer_track_count']} mixer track(s)"
        )
        self._restore_btn.setEnabled(True)

    def _on_restore_version(self):
        mgr = get_project_manager()
        version = self._selected_version()
        if mgr.current is None or version is None:
            return
        confirm = QMessageBox.question(
            self,
            "Restore Version",
            f"Restore v{version}? The current state is snapshotted first so this "
            "can be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        restored = mgr.restore_version(version)
        if restored is None:
            if self.toast_mgr:
                self.toast_mgr.error(f"Could not restore v{version}.")
            return
        self.load_project(restored)
        if self.toast_mgr:
            self.toast_mgr.success(
                f"Restored v{version}. The previous state was saved as a "
                "pre-restore version."
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
            "Import Project Assets",
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
                        self.toast_mgr.error(f"Asset import failed: {exc}")
                    continue
                if asset_id:
                    imported += 1
                    continue
                if self.toast_mgr:
                    self.toast_mgr.error("Asset import failed: no project is open.")
            if imported:
                self.load_project(mgr.current)
                if self.toast_mgr:
                    suffix = "s" if imported != 1 else ""
                    self.toast_mgr.success(
                        f"Imported {imported} project asset{suffix}."
                    )

    def _on_delete_asset(self):
        mgr = get_project_manager()
        asset = self._selected_asset()
        if mgr.current is None or asset is None:
            return

        entry = mgr.delete_asset(asset.id)
        if entry is None:
            if self.toast_mgr:
                self.toast_mgr.error("Asset could not be moved to trash.")
            return

        self.load_project(mgr.current)
        if self.toast_mgr:
            self.toast_mgr.info(
                "Asset moved to trash.",
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda entry_id=entry.id: self._restore_asset(entry_id),
            )

    def _restore_asset(self, trash_entry_id: str):
        mgr = get_project_manager()
        if not mgr.restore_deleted_asset(trash_entry_id):
            if self.toast_mgr:
                self.toast_mgr.error("Asset restore failed.")
            return
        if mgr.current is not None:
            self.load_project(mgr.current)
        if self.toast_mgr:
            self.toast_mgr.success("Asset restored.")

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
        self._rerender_btn.setEnabled(has_provenance and self._rerender_worker is None)

    def _on_open_provenance(self):
        asset = self._selected_asset()
        if not asset or not asset.provenance_path:
            return

        record = read_provenance_sidecar(asset.provenance_path)
        if not record:
            record = asset.metadata.get("provenance", {})

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Provenance - {asset.name}")
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

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
        dialog.exec()

    def _on_rerender_from_provenance(self):
        asset = self._selected_asset()
        if asset is None or not asset.provenance_path:
            if self.toast_mgr:
                self.toast_mgr.error("Select an asset with a provenance sidecar first.")
            return
        if self._rerender_worker is not None:
            if self.toast_mgr:
                self.toast_mgr.warning("A provenance re-render is already running.")
            return

        provenance = read_provenance_sidecar(asset.provenance_path)
        compatibility = check_provenance_compatibility(provenance)
        if not compatibility.compatible:
            detail = "\n".join(diff.format() for diff in compatibility.diffs[:5])
            if self.toast_mgr:
                self.toast_mgr.error(
                    "Re-render refused; recorded inputs do not match this installation.\n"
                    + detail
                )
            return

        worker = InferenceWorker(_rerender_provenance_task, asset.file_path)
        self._rerender_worker = worker
        self._rerender_workers.add(worker)
        worker.progress.connect(
            lambda percent: self._meta_label.setText(
                f"Re-rendering {asset.name}: {percent}%"
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
            self.toast_mgr.info(f"Re-rendering {asset.name} from provenance...")

    def _on_rerender_finished(self, source_asset: ProjectAsset, worker, result):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._rerender_btn.setEnabled(bool(source_asset.provenance_path))
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
                self.toast_mgr.error("Re-render completed, but no project is open to import it.")
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
                f"Re-rendered {source_asset.name} bit-identically and added it to the project."
            )

    def _on_rerender_error(self, worker, message: str):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._rerender_btn.setEnabled(True)
        if self.toast_mgr:
            self.toast_mgr.error(f"Provenance re-render failed: {message}")

    def _on_rerender_cancelled(self, worker):
        if self._rerender_worker is worker:
            self._rerender_worker = None
        self._rerender_btn.setEnabled(True)
        if self.toast_mgr:
            self.toast_mgr.warning("Provenance re-render cancelled.")

    def _on_export_disclosure(self):
        """Write the project disclosure record as JSON and TSV."""
        manager = get_project_manager()
        project = manager.current
        if project is None:
            if self.toast_mgr:
                self.toast_mgr.error("No project is open to report.")
            return

        self.sync_pending_edits()
        output_dir = choose_directory(
            self,
            "Export AI Disclosure",
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
                    "Could not save the project before exporting its disclosure record."
                )
            return
        try:
            json_path, tsv_path = write_disclosure_report(project, output_dir)
        except (OSError, TypeError, ValueError) as exc:
            if self.toast_mgr:
                self.toast_mgr.error(f"AI disclosure export failed: {exc}")
            return
        if self.toast_mgr:
            self.toast_mgr.success(
                f"AI disclosure exported: {json_path.name} and {tsv_path.name}"
            )


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
        title = QLabel("Project library")
        title.setStyleSheet(f"color: {t['text']}; font-weight: bold; font-size: 12pt;")

        self._new_btn = QPushButton("+ New Project")
        self._new_btn.setStyleSheet(f"""
            QPushButton {{
                background: {t['accent']}; color: {t['background']}; border: none;
                border-radius: 5px; padding: 6px 14px;
                font-weight: bold; font-size: 9pt;
            }}
            QPushButton:hover {{ background: {t['accent_hover']}; }}
        """)
        self._new_btn.clicked.connect(self._on_new_project)

        self._rescan_btn = QPushButton("Rescan")
        self._rescan_btn.setToolTip(
            "Rebuild the project index from valid project folders and backups."
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
        self._search.setPlaceholderText("Search projects...")
        self._search.setStyleSheet(f"""
            QLineEdit {{
                background: {t['surface']}; color: {t['text']};
                border: 1px solid {t['border']}; border-radius: 6px;
                padding: 8px 12px; font-size: 9pt;
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
        self._library_empty = EmptyStateWidget(
            "No projects yet",
            "Create a project to keep lyrics, MIDI, audio, versions, and provenance together.",
            "New project",
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
            "Project library",
            named_controls=[
                (self._new_btn, "New project", "Creates a new project."),
                (self._rescan_btn, "Rescan projects", "Rebuilds the project index from project folders."),
                (self._search, "Search projects", "Filters the project library by name."),
            ],
        )

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

        count_text = f"{len(projects)} project{'s' if len(projects) != 1 else ''}"
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
                f"Project index verified: {count} project"
                f"{'s' if count != 1 else ''}."
            )

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
                self.toast_mgr.error(repair_text or "Project could not be opened.")

    def _on_delete_project(self, project_id: str):
        mgr = get_project_manager()
        entry = mgr.delete(project_id)
        if not entry:
            if self.toast_mgr:
                self.toast_mgr.error("Project could not be moved to trash.")
            return

        self._detail.clear()
        self._refresh_list()
        if self.toast_mgr:
            self.toast_mgr.info(
                "Project moved to trash.",
                duration_ms=8000,
                action_label="Undo",
                action_callback=lambda entry_id=entry.id: self._restore_project(entry_id),
            )

    def _restore_project(self, trash_entry_id: str):
        mgr = get_project_manager()
        if mgr.restore_deleted_project(trash_entry_id):
            self._refresh_list()
            if self.toast_mgr:
                self.toast_mgr.success("Project restored.")
        elif self.toast_mgr:
            self.toast_mgr.error("Project restore failed.")

    def _repair_status_text(self, status: dict) -> str:
        state = status.get("status", "ok")
        if state == "ok":
            return ""
        label = {
            "migrated": "Project data upgraded",
            "repaired": "Project library repaired",
            "error": "Project recovery needs attention",
        }.get(state, "Project recovery needs attention")
        if status.get("backup_paths"):
            label += " — backup saved"
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
        query = text.lower()
        visible_count = 0
        for card in self._cards:
            visible = not query or query in card._project_id.lower()
            # Check card's name label
            for child in card.findChildren(QLabel):
                if child.styleSheet() and "bold" in child.styleSheet():
                    if query in child.text().lower():
                        visible = True
                    break
            card.setVisible(visible)
            visible_count += int(visible)
        if visible_count:
            self._library_empty.hide()
        elif query:
            self._library_empty.set_no_matches(
                f'No projects match “{text.strip()}”. Clear the search to browse the library.',
                "Clear search",
            )
            self._library_empty.show()
        else:
            self._library_empty.set_state(
                "No projects yet",
                "Create a project to keep lyrics, MIDI, audio, versions, and provenance together.",
                "New project",
            )
            self._library_empty.show()

    def _on_library_empty_action(self):
        if self._library_empty.state == "no_matches":
            self._search.clear()
        else:
            self._on_new_project()
