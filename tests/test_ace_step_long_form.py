import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - dependency is declared for the app
    sf = None

from engines.ace_step_engine import plan_long_form_sections, stitch_audio_files


class LongFormPlanningTests(unittest.TestCase):
    def test_structured_lyrics_keep_song_sections(self):
        lyrics = """[Intro]
Smoke in the room

[Verse 1]
First verse line

[Chorus]
Gang vocals rise

[Bridge]
Half-time drop

[Outro]
Static fades"""

        plan = plan_long_form_sections(lyrics, 180.0)

        self.assertEqual(
            [section.label for section in plan],
            ["Intro", "Verse 1", "Chorus", "Bridge", "Outro"],
        )
        self.assertAlmostEqual(sum(section.duration for section in plan), 180.0, places=1)
        self.assertTrue(all(section.lyrics.startswith("[") for section in plan))

    def test_unstructured_long_lyrics_are_split_for_long_duration(self):
        lyrics = "\n".join(f"line {i}" for i in range(24))

        plan = plan_long_form_sections(lyrics, 260.0, max_section_duration=90.0)

        self.assertGreaterEqual(len(plan), 3)
        self.assertAlmostEqual(sum(section.duration for section in plan), 260.0, places=1)
        self.assertTrue(all(section.duration <= 90.0 for section in plan))


@unittest.skipIf(sf is None, "soundfile is not installed")
class AudioStitchingTests(unittest.TestCase):
    def test_stitch_audio_files_crossfades_sections(self):
        sample_rate = 1000
        first = np.ones((sample_rate, 2), dtype=np.float32) * 0.25
        second = np.ones((sample_rate, 2), dtype=np.float32) * -0.25

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            first_path = tmp_path / "first.wav"
            second_path = tmp_path / "second.wav"
            output_path = tmp_path / "stitched.wav"
            sf.write(first_path, first, sample_rate)
            sf.write(second_path, second, sample_rate)

            path, duration = stitch_audio_files(
                [str(first_path), str(second_path)],
                str(output_path),
                target_sample_rate=sample_rate,
                crossfade_seconds=0.1,
            )

            stitched, sr = sf.read(path, always_2d=True)

        self.assertEqual(sr, sample_rate)
        self.assertEqual(stitched.shape, (1900, 2))
        self.assertAlmostEqual(duration, 1.9, places=2)


if __name__ == "__main__":
    unittest.main()
