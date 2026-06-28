import unittest

from engines.audio_analyzer import AudioAnalysis, infer_clap_style_tags


class AudioClapTagTests(unittest.TestCase):
    def test_infer_clap_style_tags_returns_ranked_tags(self):
        analysis = AudioAnalysis(
            bpm=145.0,
            key="F# minor",
            energy_mean=0.08,
            energy_std=0.05,
            brightness_mean=3600.0,
            onset_density=5.8,
        )

        tags, backend, scores, embedding = infer_clap_style_tags(analysis, limit=4)

        self.assertEqual(backend, "audio-clap-lite")
        self.assertEqual(len(tags), 4)
        self.assertEqual(len(scores), 4)
        self.assertEqual(len(embedding), 8)
        self.assertTrue({"trap", "energetic", "metal"} & set(tags))

    def test_ace_step_tags_include_clap_tags_without_duplicates(self):
        analysis = AudioAnalysis(
            suggested_tags=["hip hop", "dark"],
            suggested_tempo_tag="fast",
            clap_style_tags=["dark", "trap", "bass-heavy"],
        )

        self.assertEqual(
            analysis.to_ace_step_tags(),
            "hip hop, dark, trap, bass-heavy, fast",
        )


if __name__ == "__main__":
    unittest.main()
