import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.stem_export import (
    STEM_EXPORT_TEMPLATES,
    get_stem_export_template,
    stem_export_filename,
    stem_export_filenames,
)
from ui.vocal_suite_view import _vocal_stem_export_task


class StemExportNamingTests(unittest.TestCase):
    def test_every_target_template_renders_portable_names(self):
        for template in STEM_EXPORT_TEMPLATES:
            with self.subTest(template=template.id):
                name = stem_export_filename(
                    template.id,
                    "My Song / Take 01",
                    "Lead Vocals",
                    3,
                )
                self.assertEqual("wav", name.rsplit(".", 1)[-1])
                self.assertNotRegex(name, r"[\\/:*?\"<>| ]")
                self.assertIn("My-Song-Take-01", name)
                self.assertIn("Lead-Vocals", name)

    def test_unknown_selection_falls_back_to_generic(self):
        self.assertEqual(
            get_stem_export_template("missing").id,
            "generic",
        )
        self.assertEqual(
            stem_export_filename("missing", "Project", "Vocals", 1),
            "Project_Vocals.wav",
        )

    def test_duplicate_stems_get_unique_case_insensitive_names(self):
        names = stem_export_filenames(
            "generic",
            "Project",
            ["Vocals", "vocals", "Vocals"],
        )
        self.assertEqual(
            ["Project_Vocals.wav", "Project_vocals-2.wav", "Project_Vocals-3.wav"],
            names,
        )

    def test_invalid_extension_and_index_fail_safe(self):
        self.assertEqual(
            stem_export_filename("pro_tools", "", "", 0, "../WAV"),
            "slunder-project_stem-1_Stem.wav",
        )

    def test_worker_exports_each_stem_at_native_rate_with_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = _vocal_stem_export_task(
                [
                    {
                        "name": "Vocals",
                        "audio": np.zeros((240, 2), dtype=np.float32),
                        "sample_rate": 24000,
                        "source_path": os.path.join(tmp, "source.wav"),
                    },
                    {
                        "name": "Bass",
                        "audio": np.zeros((480, 2), dtype=np.float32),
                        "sample_rate": 48000,
                    },
                ],
                tmp,
                "pro_tools",
                "Demo Song",
            )

            self.assertEqual("stems", result["kind"])
            self.assertEqual(
                ["Demo-Song_Vocals_Stem.wav", "Demo-Song_Bass_Stem.wav"],
                [os.path.basename(path) for path in result["paths"]],
            )
            self.assertEqual(24000, sf.info(result["paths"][0]).samplerate)
            self.assertEqual(48000, sf.info(result["paths"][1]).samplerate)
            sidecar = result["paths"][0] + ".provenance.json"
            data = json.loads(Path(sidecar).read_text(encoding="utf-8"))
            self.assertEqual("pro_tools", data["extra"]["stem_export_template"])
            self.assertEqual("Vocals", data["extra"]["stem_name"])


if __name__ == "__main__":
    unittest.main()
