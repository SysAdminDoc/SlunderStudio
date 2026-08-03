import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.disclosure import (
    build_disclosure_report,
    format_human_contributions,
    parse_human_contributions,
    render_disclosure_sheet,
    write_disclosure_report,
)
from core.project import Project, ProjectAsset, ProjectManager, ProjectVersion
from core.provenance import write_provenance_sidecar


class DisclosureReportTests(unittest.TestCase):
    def test_report_separates_generated_processed_unknown_and_declared_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated.wav"
            generated.write_bytes(b"generated")
            processed = root / "processed.wav"
            processed.write_bytes(b"processed")

            model_metadata = {
                "id": "ace-step-v1.5",
                "name": "ACE-Step",
                "source": "local",
                "revision": "rev-1",
                "resolved_revision": "rev-1",
                "hash": "model-hash",
                "license": "MIT",
                "license_url": "https://example.invalid/mit",
                "metadata_status": "known",
            }
            with mock.patch(
                "core.provenance.collect_model_metadata",
                return_value=model_metadata,
            ):
                generated_sidecar = write_provenance_sidecar(
                    generated,
                    module="song_forge",
                    operation="generate",
                    model_id="ace-step-v1.5",
                    seed=42,
                    prompt="a careful test song",
                    export_format="wav",
                    output_kind="model",
                )
                processed_sidecar = write_provenance_sidecar(
                    processed,
                    module="vocal_suite",
                    operation="vocal_autotune",
                    model_id="librosa-pyin-pitch-shift",
                    model_license="ISC",
                    export_format="wav",
                    output_kind="processed",
                )

            project = Project(
                id="proj_report",
                name="Disclosure Demo",
                lyrics_text="A stored lyric draft",
                mixer_state={"master": {"preset": "Balanced"}},
                versions=[ProjectVersion(version=1, description="First edit")],
                human_contributions=[
                    {"category": "lyrics", "description": "wrote the chorus"},
                    {"category": "midi", "description": "drew the bass notes"},
                    {"category": "edit", "description": "chose the final take"},
                ],
            )
            project.assets.extend([
                ProjectAsset(
                    id="asset_generated",
                    name="Generated Song",
                    asset_type="audio",
                    module="song_forge",
                    file_path=str(generated),
                    provenance_path=str(generated_sidecar),
                ),
                ProjectAsset(
                    id="asset_processed",
                    name="Tuned Vocal",
                    asset_type="audio",
                    module="vocal_suite",
                    file_path=str(processed),
                    provenance_path=str(processed_sidecar),
                ),
                ProjectAsset(
                    id="asset_midi",
                    name="Bass MIDI",
                    asset_type="midi",
                    module="midi_studio",
                    file_path=str(root / "bass.mid"),
                ),
            ])

            report = build_disclosure_report(project, generated_at=1.0)

            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(report["report_type"], "ai-disclosure-human-authorship")
            self.assertTrue(report["ddex"]["IsAIGenerated"])
            self.assertIn("full-composition", report["ddex"]["AIComponentType"])
            self.assertIn("vocal", report["ddex"]["AIComponentType"])
            self.assertEqual(report["summary"]["generated_elements"], 1)
            self.assertEqual(report["summary"]["processed_elements"], 1)
            self.assertEqual(report["summary"]["human_authored_elements"], 3)
            self.assertEqual(report["summary"]["unknown_elements"], 1)
            self.assertEqual(report["summary"]["human_declarations"], 3)

            by_id = {element["id"]: element for element in report["elements"]}
            generated_element = by_id["asset_generated"]
            self.assertEqual(generated_element["classification"], "generated")
            self.assertIs(generated_element["ddex"]["IsAIGenerated"], True)
            self.assertEqual(
                generated_element["provenance"]["model"]["revision"],
                "rev-1",
            )
            self.assertEqual(
                generated_element["provenance"]["model"]["license"],
                "MIT",
            )
            self.assertEqual(generated_element["provenance"]["model"]["hash"], "model-hash")

            processed_element = by_id["asset_processed"]
            self.assertEqual(processed_element["classification"], "processed")
            self.assertIsNone(processed_element["ddex"]["IsAIGenerated"])
            self.assertEqual(processed_element["ddex"]["AIComponentType"], "vocal")

            unknown_element = by_id["asset_midi"]
            self.assertEqual(unknown_element["classification"], "unknown")
            self.assertIn("unknown", unknown_element["limitations"][0])
            human_element = next(
                element for element in report["elements"]
                if element["classification"] == "human-authored"
            )
            self.assertIs(human_element["ddex"]["IsAIGenerated"], False)
            self.assertIn("user declaration", human_element["limitations"][0])

            evidence = report["human_authorship_evidence"]
            self.assertEqual(
                [entry["status"] for entry in evidence[:3]],
                ["user-declared", "user-declared", "user-declared"],
            )
            self.assertTrue(any(entry["id"] == "observed-lyrics-field" for entry in evidence))
            self.assertTrue(any(entry["id"] == "observed-midi-assets" for entry in evidence))
            self.assertTrue(any(entry["id"] == "observed-version-history" for entry in evidence))
            self.assertTrue(any(entry["id"] == "observed-mixer-state" for entry in evidence))
            self.assertTrue(any("does not prove" in value for value in report["limitations"]))

    def test_sheet_and_json_exports_are_copy_pasteable_and_stable(self):
        project = SimpleNamespace(
            id="proj_sheet",
            name="Sheet Demo",
            app_version="0.1.31",
            created_at=1.0,
            updated_at=2.0,
            description="",
            assets=[],
            versions=[],
            lyrics_text="",
            mixer_state={},
            human_contributions=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            json_path, tsv_path = write_disclosure_report(
                project,
                tmp,
                generated_at=3.0,
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(tsv_path.is_file())
            self.assertEqual(json_path.suffix, ".json")
            self.assertEqual(tsv_path.suffix, ".tsv")
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            sheet = tsv_path.read_text(encoding="utf-8")
            self.assertEqual(payload["generated_at"], 3.0)
            self.assertIn("IsAIGenerated\tfalse", sheet)
            self.assertIn("AIComponentType\tUnknown", sheet)
            self.assertIn("Field\tValue", sheet)
            self.assertIn("Contributing elements", sheet)
            self.assertEqual(sheet, render_disclosure_sheet(payload))

    def test_contribution_editor_round_trips_categories_without_claiming_more(self):
        text = (
            "lyrics: wrote verse one\n"
            "midi: drew the bass\n"
            "edits: chose take 3\n"
            "kept the original room tone"
        )
        parsed = parse_human_contributions(text)
        self.assertEqual(
            [item["category"] for item in parsed],
            ["lyrics", "midi", "edit", "other"],
        )
        self.assertEqual(
            format_human_contributions(parsed),
            "lyrics: wrote verse one\n"
            "midi: drew the bass\n"
            "edit: chose take 3\n"
            "other: kept the original room tone",
        )
        self.assertTrue(all(item["basis"] == "user-declared" for item in parsed))

    def test_project_persists_human_contribution_declarations(self):
        project = Project(
            id="proj_persist",
            human_contributions=[
                {"category": "lyrics", "description": "wrote the hook"},
            ],
        )
        payload = ProjectManager._serializable(project)
        restored = ProjectManager._project_from_data(payload, project.id)
        self.assertEqual(restored.human_contributions, project.human_contributions)


if __name__ == "__main__":
    unittest.main()
