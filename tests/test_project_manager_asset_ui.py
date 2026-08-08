import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.project import Project, ProjectAsset
from core.provenance import write_provenance_sidecar
from ui.project_manager import ProjectDetailPanel


class _ToastStub:
    def __init__(self):
        self.actions = []
        self.messages = []

    def info(self, message, **kwargs):
        self.messages.append(message)
        if kwargs.get("action_callback"):
            self.actions.append(kwargs["action_callback"])

    def success(self, message, **_kwargs):
        self.messages.append(message)

    def error(self, message, **_kwargs):
        self.messages.append(message)


class ProjectManagerAssetUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_selected_asset_delete_uses_undo_and_restore(self):
        toast = _ToastStub()
        project = Project(
            id="project-ui",
            name="UI project",
            assets=[
                ProjectAsset(
                    id="asset-ui",
                    name="take.wav",
                    asset_type="audio",
                    file_path="C:/project/take.wav",
                    module="mixer",
                )
            ],
        )
        manager = mock.Mock()
        manager.current = project
        manager.delete_asset.return_value = SimpleNamespace(id="trash-entry")
        manager.restore_deleted_asset.return_value = True

        panel = ProjectDetailPanel(toast_mgr=toast)
        try:
            panel.load_project(project)
            panel._asset_list.setCurrentRow(0)
            self.assertTrue(panel._delete_asset_btn.isEnabled())

            with mock.patch("ui.project_manager.get_project_manager", return_value=manager):
                panel._on_delete_asset()
                manager.delete_asset.assert_called_once_with("asset-ui")
                self.assertTrue(toast.actions)
                toast.actions[-1]()
                manager.restore_deleted_asset.assert_called_once_with("trash-entry")
                self.assertIn("Asset restored.", toast.messages)
        finally:
            panel.deleteLater()

    def test_import_failure_is_reported_without_propagating(self):
        toast = _ToastStub()
        project = Project(id="project-import", name="Import project")
        manager = mock.Mock()
        manager.current = project
        manager.import_asset.side_effect = OSError("read-only project")
        panel = ProjectDetailPanel(toast_mgr=toast)
        try:
            with (
                mock.patch("ui.project_manager.get_project_manager", return_value=manager),
                mock.patch(
                    "ui.project_manager.QFileDialog.getOpenFileNames",
                    return_value=(["C:/source.wav"], "Audio Files (*.wav)"),
                ),
            ):
                panel._on_import_asset()

            manager.import_asset.assert_called_once()
            self.assertIn("Asset import failed: read-only project", toast.messages)
        finally:
            panel.deleteLater()

    def test_unsupported_provenance_disables_misleading_rerender_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "master.wav"
            artifact.write_bytes(b"audio")
            sidecar = write_provenance_sidecar(
                artifact,
                module="mixer",
                operation="export_master",
                export_format="wav",
            )
            project = Project(
                id="project-replayability",
                name="Replayability project",
                assets=[
                    ProjectAsset(
                        id="asset-replayability",
                        name=artifact.name,
                        asset_type="audio",
                        file_path=str(artifact),
                        module="mixer",
                        provenance_path=str(sidecar),
                    )
                ],
            )
            panel = ProjectDetailPanel(toast_mgr=_ToastStub())
            try:
                panel.load_project(project)
                panel._asset_list.setCurrentRow(0)
                self.assertTrue(panel._provenance_btn.isEnabled())
                self.assertFalse(panel._rerender_btn.isEnabled())
                self.assertIn("Replay unavailable", panel._rerender_capability_label.text())
            finally:
                panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
