import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from core.project import ProjectManager
from core.pronunciation import (
    PronunciationOverride,
    apply_pronunciation_override,
    parse_phoneme_text,
)
from core.provenance import project_metadata_from_provenance
from engines.diffsinger_engine import DiffSingerEngine, SingParams
from ui.waveform_widget import WaveformWidget


class PronunciationCoreTests(unittest.TestCase):
    def test_override_parsing_and_json_round_trip(self):
        self.assertEqual(("k", "ae", "t"), parse_phoneme_text("k, ae t"))
        override = PronunciationOverride.from_values(
            "cat",
            0.25,
            0.75,
            ("k", "ae", "t"),
        )
        self.assertEqual(override, PronunciationOverride.from_dict(override.to_dict()))

    def test_splice_keeps_outside_samples_bit_identical(self):
        base = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)
        replacement = np.ones(50, dtype=np.float32) * 0.25
        corrected = apply_pronunciation_override(
            base,
            replacement,
            1000,
            0.2,
            0.8,
            crossfade_ms=20,
        )

        np.testing.assert_array_equal(corrected[:200], base[:200])
        np.testing.assert_array_equal(corrected[800:], base[800:])
        self.assertEqual(base.shape, corrected.shape)
        self.assertFalse(np.array_equal(corrected[200:800], base[200:800]))

    def test_splice_matches_channels_and_preserves_stereo_layout(self):
        base = np.column_stack((
            np.linspace(-1, 1, 100, dtype=np.float32),
            np.linspace(1, -1, 100, dtype=np.float32),
        ))
        corrected = apply_pronunciation_override(
            base,
            np.ones(10, dtype=np.float32),
            100,
            0.2,
            0.8,
            crossfade_ms=0,
        )
        self.assertEqual(base.shape, corrected.shape)
        np.testing.assert_array_equal(corrected[:20], base[:20])
        np.testing.assert_array_equal(corrected[80:], base[80:])

    def test_region_synthesis_uses_explicit_phonemes_and_note_timing(self):
        engine = DiffSingerEngine()
        engine._sample_rate = 24000
        engine._hop_size = 240
        engine._model_path = __file__
        engine._phoneme_dictionary = {"k": 1, "ae": 2, "t": 3}
        engine._phonemizer = lambda _lyrics: ["unused"]

        class _Input:
            name = "tokens"

        class _Session:
            def get_inputs(self):
                return [_Input()]

            def run(self, _outputs, inputs):
                self.inputs = inputs
                return [np.zeros(2400, dtype=np.float32)]

        session = _Session()
        engine._session = session
        params = SingParams(
            lyrics="cat dog",
            notes=[
                {"pitch": 60, "start": 0.0, "end": 0.5, "text": "cat"},
                {"pitch": 62, "start": 0.5, "end": 1.0, "text": "dog"},
            ],
        )

        result = engine.synthesize_region(params, 0.0, 0.5, ["k", "ae", "t"])

        self.assertIsNone(result.error)
        np.testing.assert_array_equal(session.inputs["tokens"], [[1, 2, 3]])
        self.assertEqual("diffsinger_pronunciation_region", result.provenance["operation"])
        self.assertEqual(0.0, result.provenance["extra"]["region_start"])
        self.assertEqual(["k", "ae", "t"], result.provenance["extra"]["phoneme_override"])

    def test_project_and_provenance_projection_store_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config"
            ProjectManager._instance = None
            try:
                with patch("core.project.get_config_dir", return_value=config):
                    manager = ProjectManager()
                    project = manager.create("Pronunciation Project")
                    override = PronunciationOverride.from_values(
                        "through",
                        1.0,
                        1.5,
                        ("th", "r", "u", "w"),
                    ).to_dict()
                    self.assertTrue(
                        manager.record_pronunciation_override(
                            override,
                            artifact_path="rendered.wav",
                        )
                    )
                    self.assertEqual(
                        override,
                        {
                            key: project.pronunciation_overrides[0][key]
                            for key in override
                        },
                    )
                    saved = (config / "projects" / project.id / "project.json").read_text()
                    self.assertIn("pronunciation_overrides", saved)
                    projection = project_metadata_from_provenance(
                        {"extra": {"pronunciation_overrides": [override]}}
                    )
                    self.assertEqual(
                        [override],
                        projection["provenance"]["pronunciation_overrides"],
                    )
            finally:
                ProjectManager._instance = None


class PronunciationWaveformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_waveform_region_selection_emits_seconds(self):
        widget = WaveformWidget()
        try:
            seen = []
            widget.region_selected.connect(lambda start, end: seen.append((start, end)))
            self.assertTrue(widget.load_audio(np.zeros(1000, dtype=np.float32), 1000))
            widget.set_selection_enabled(True)
            widget.set_selection(0.2, 0.6)
            self.assertEqual((0.2, 0.6), widget.selected_region)
            self.assertEqual((0.2, 0.6), seen[-1])
            self.assertIn("selected region", widget.accessible_state())
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
