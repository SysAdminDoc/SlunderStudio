import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.evaluation import (
    DEFAULT_EVALUATION_CASES,
    EvaluationCase,
    EvaluationOutput,
    run_evaluation,
    write_evaluation_report,
)


class EvaluationHarnessTests(unittest.TestCase):
    def test_fixed_cases_cover_languages_durations_and_structures(self):
        self.assertGreaterEqual(len(DEFAULT_EVALUATION_CASES), 5)
        self.assertIn("en", {case.language for case in DEFAULT_EVALUATION_CASES})
        self.assertIn("es", {case.language for case in DEFAULT_EVALUATION_CASES})
        self.assertTrue(any(case.duration_seconds > 0 for case in DEFAULT_EVALUATION_CASES))
        self.assertTrue(all(case.seed > 0 for case in DEFAULT_EVALUATION_CASES))
        self.assertTrue(all(case.prompt and case.case_id for case in DEFAULT_EVALUATION_CASES))

    def test_runner_report_measures_audio_and_keeps_blind_review_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def runner(case: EvaluationCase, case_dir: Path):
                output = case_dir / "render.wav"
                sf.write(output, np.zeros((800, 2), dtype=np.float32), 8000, subtype="PCM_16")
                return EvaluationOutput(
                    artifacts=[output],
                    model={"id": "fixture-engine", "revision": "fixture-revision"},
                    adherence={"score": 0.9, "method": "fixture"},
                    lyric_timing={"aligned": True, "coverage": 1.0},
                    structure={"expected": list(case.expected_structure), "matched": True},
                )

            report = run_evaluation(
                runner,
                cases=DEFAULT_EVALUATION_CASES[:1],
                artifact_dir=root / "artifacts",
                runtime={"git_revision": "fixture"},
            )
            measurement = report["cases"][0]
            self.assertEqual("completed", measurement["status"])
            self.assertGreaterEqual(measurement["latency_ms"], 0)
            self.assertGreater(measurement["peak_ram_mb"], 0)
            self.assertEqual("fixture-engine", measurement["model"]["id"])
            self.assertEqual(1, len(measurement["artifacts"]))
            self.assertEqual(64, len(measurement["artifacts"][0]["sha256"]))
            self.assertEqual(1, len(measurement["loudness_lufs"]))
            self.assertEqual(1, len(measurement["true_peak_dbtp"]))
            self.assertTrue(report["listener_rubric"]["blinded"])
            self.assertNotIn("fixture-engine", json.dumps(report["listener_rubric"]))
            self.assertFalse(report["release_gate"]["fad_used"])
            self.assertFalse(report["release_gate"]["gated"])

    def test_failures_are_reported_without_aborting_following_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def runner(case: EvaluationCase, _case_dir: Path):
                calls.append(case.case_id)
                if len(calls) == 1:
                    raise RuntimeError("fixture failure")
                return {"status": "skipped", "failure": "not applicable"}

            cases = DEFAULT_EVALUATION_CASES[:2]
            report = run_evaluation(runner, cases=cases, artifact_dir=Path(tmp))
            self.assertEqual([case.case_id for case in cases], calls)
            self.assertEqual("failed", report["cases"][0]["status"])
            self.assertIn("fixture failure", report["cases"][0]["failure"])
            self.assertEqual("skipped", report["cases"][1]["status"])

    def test_report_writer_uses_json_and_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            write_evaluation_report({"schema_version": 1}, path)
            self.assertEqual(1, json.loads(path.read_text(encoding="utf-8"))["schema_version"])


if __name__ == "__main__":
    unittest.main()
