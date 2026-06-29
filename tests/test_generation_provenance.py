import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.project import ProjectManager
from core.provenance import (
    project_metadata_from_provenance,
    read_provenance_sidecar,
    write_provenance_sidecar,
)
from core.trash import TrashManager
from engines.sfx_engine import SFXEngine, SFXParams


class GenerationProvenanceTests(unittest.TestCase):
    def test_sidecar_contains_reproducibility_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "render.wav"
            artifact.write_bytes(b"rendered audio")

            sidecar = write_provenance_sidecar(
                artifact,
                module="song_forge",
                operation="generate",
                model_id="ace-step-v1.5",
                seed=1234,
                prompt="dark trap metal",
                lyrics="[Chorus]\nTest",
                parameters={"cfg_scale": 7.5},
                source_asset_ids=["asset_source"],
                export_format="wav",
            )

            data = read_provenance_sidecar(artifact)

            self.assertEqual(sidecar, Path(str(artifact) + ".provenance.json"))
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["module"], "song_forge")
            self.assertEqual(data["operation"], "generate")
            self.assertEqual(data["model"]["id"], "ace-step-v1.5")
            self.assertEqual(data["seed"], 1234)
            self.assertEqual(data["prompt"], "dark trap metal")
            self.assertEqual(data["lyrics"], "[Chorus]\nTest")
            self.assertEqual(data["parameters"]["cfg_scale"], 7.5)
            self.assertEqual(data["source_asset_ids"], ["asset_source"])
            self.assertEqual(data["artifact"]["sha256"], "56724ba9bf2e972f799c4ab253cea08d850b8433a5499ced8fc7212602f15a4d")

            summary = project_metadata_from_provenance(data, sidecar)
            self.assertEqual(summary["provenance"]["model_id"], "ace-step-v1.5")
            self.assertEqual(summary["provenance"]["artifact_sha256"], data["artifact"]["sha256"])

    def test_sfx_demo_generation_writes_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SFXEngine()
            engine._output_dir = tmp

            result = engine.generate(SFXParams(
                prompt="soft chime",
                duration=0.2,
                seed=77,
                allow_demo_output=True,
            ))

            self.assertIsNone(result.error)
            self.assertTrue(os.path.isfile(result.file_path))
            self.assertTrue(os.path.isfile(result.provenance_path))

            data = read_provenance_sidecar(result.file_path)
            self.assertEqual(data["module"], "sfx")
            self.assertEqual(data["operation"], "generate")
            self.assertEqual(data["output_kind"], "demo")
            self.assertEqual(data["seed"], 77)
            self.assertEqual(data["prompt"], "soft chime")
            self.assertEqual(data["model"]["id"], "stable-audio-open")

    def test_project_import_copies_sidecar_and_stores_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "asset.wav"
            source.write_bytes(b"asset audio")
            write_provenance_sidecar(
                source,
                module="sfx",
                operation="generate",
                model_id="stable-audio-open",
                seed=99,
                prompt="rain",
                parameters={"duration": 1.0},
                export_format="wav",
            )

            config_dir = root / "config"
            ProjectManager._instance = None
            try:
                with mock.patch("core.project.get_config_dir", return_value=config_dir):
                    mgr = ProjectManager()
                    mgr._trash = TrashManager(root / "trash")
                    project = mgr.create("Provenance Project")

                    asset_id = mgr.import_asset(str(source), "audio", "sfx")

                    self.assertIsNotNone(asset_id)
                    asset = next(a for a in project.assets if a.id == asset_id)
                    self.assertTrue(asset.provenance_path)
                    self.assertTrue(Path(asset.provenance_path).is_file())
                    self.assertNotEqual(Path(asset.provenance_path).parent, source.parent)
                    self.assertEqual(asset.metadata["provenance"]["model_id"], "stable-audio-open")
                    self.assertEqual(asset.metadata["provenance"]["seed"], 99)
                    self.assertEqual(asset.metadata["provenance"]["prompt"], "rain")
            finally:
                ProjectManager._instance = None


if __name__ == "__main__":
    unittest.main()
