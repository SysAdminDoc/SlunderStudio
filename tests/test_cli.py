import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import slunder_cli
from core.audio_export import ExportSettings
from core.engine_contract import ArtifactKind, EngineArtifact, EngineRunResult, RunOutcome
from core.job_state import JobStatus, JobStore


class HeadlessCLITests(unittest.TestCase):
    def test_global_and_subcommand_output_flags_are_preserved(self):
        parser = slunder_cli.build_parser()

        for argv in (
            ("--json", "lyrics", "write a chorus"),
            ("lyrics", "write a chorus", "--json"),
        ):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertTrue(args.json)

        for argv in (("--quiet", "jobs"), ("jobs", "--quiet")):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                self.assertTrue(args.quiet)

    def test_json_validation_errors_are_machine_readable(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = slunder_cli.main(["--json", "lyrics", ""])

        self.assertEqual(slunder_cli.EXIT_USAGE, code)
        payload = json.loads(stdout.getvalue())
        self.assertEqual("failed", payload["status"])
        self.assertIn("prompt", payload["error"])
        self.assertEqual("", stderr.getvalue())

    def test_result_payload_keeps_typed_artifact_metadata_without_payload(self):
        result = EngineRunResult(
            capability_id="midi.generate",
            outcome=RunOutcome.MODEL,
            artifacts=[
                EngineArtifact(
                    ArtifactKind.MIDI,
                    path="C:/renders/take.mid",
                    payload=object(),
                    metadata={"notes": 12},
                )
            ],
            model_id="midi-llm-1b",
        )

        payload = slunder_cli.result_payload(result)

        self.assertEqual("model", payload["outcome"])
        self.assertEqual("C:/renders/take.mid", payload["artifacts"][0]["path"])
        self.assertEqual({"notes": 12}, payload["artifacts"][0]["metadata"])
        self.assertNotIn("payload", payload["artifacts"][0])

    def test_export_runs_through_worker_and_records_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            output = root / "renders" / "take.wav"
            sf.write(source, np.zeros((800, 2), dtype=np.float32), 8000)

            args = slunder_cli.build_parser().parse_args(
                ["export", str(source), str(output), "--format", "wav"]
            )
            slunder_cli._validate_args(args)
            store = JobStore(root / "jobs", cleanup_roots=[root])
            settings = ExportSettings(
                format="wav",
                sample_rate=8000,
                bit_depth=16,
                c2pa_enabled=False,
            )

            with mock.patch.object(
                slunder_cli,
                "configured_export_settings",
                return_value=settings,
            ):
                runner = slunder_cli.HeadlessRunner(
                    json_output=True,
                    quiet=True,
                    job_store=store,
                )
                execution = runner.run(
                    slunder_cli._build_export_task(args),
                    job_kind="audio_export",
                    job_label="CLI export test",
                    job_inputs={"source": str(source)},
                    job_metadata={"module": "audio_export"},
                )

            self.assertEqual(JobStatus.COMPLETED, execution.status)
            self.assertTrue(output.is_file())
            self.assertTrue(Path(str(output) + ".provenance.json").is_file())
            record = store.get(execution.job_id)
            self.assertIsNotNone(record)
            self.assertEqual(JobStatus.COMPLETED, record.status)
            self.assertIn(str(output), record.outputs["paths"])


if __name__ == "__main__":
    unittest.main()
