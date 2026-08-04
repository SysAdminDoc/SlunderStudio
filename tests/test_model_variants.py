import tempfile
import unittest
from pathlib import Path

from core.model_manager import BUILTIN_MODELS, model_variants
from core.model_variants import (
    VariantBenchmarkCase,
    compare_variants,
    measure_variant,
    write_variant_measurement,
)


class ModelVariantTests(unittest.TestCase):
    def test_registry_exposes_paired_q4_and_q8_variants(self):
        for family in (
            "llama-3.1-8b-instruct",
            "llama-3.2-3b-instruct",
            "qwen-2.5-14b-instruct",
        ):
            variants = model_variants(family, BUILTIN_MODELS)
            self.assertEqual([4, 8], [info.quantization_bits for info in variants])
            q4, q8 = variants
            self.assertEqual(q4.source, q8.source)
            self.assertEqual(q4.revision, q8.revision)
            self.assertGreater(q8.disk_gb, q4.disk_gb)
            self.assertGreater(q8.vram_gb, q4.vram_gb)
            self.assertIn("Q8_0", q8.allow_patterns[0])
            self.assertTrue(q8.quality_label)
            self.assertFalse(q8.has_local_benchmark)

    def test_measurement_records_quality_latency_disk_ram_and_vram(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.gguf"
            path.write_bytes(b"fixture-model" * 128)
            cases = (
                VariantBenchmarkCase(
                    "one", "write verse and chorus", ("verse", "chorus"), 12
                ),
                VariantBenchmarkCase(
                    "two", "write another verse and chorus", ("verse", "chorus"), 12
                ),
            )

            def runner(case):
                return {
                    "text": "[Verse]\nA line\n[Chorus]\nA hook",
                    "token_count": 12,
                    "peak_vram_mb": 321.0 if case.case_id == "one" else 456.0,
                }

            measurement = measure_variant(
                "fixture-q4",
                "Q4_K_M",
                path,
                runner,
                cases=cases,
                quality_scorer=lambda _case, _text: 0.75,
                hardware="fixture CPU/GPU",
            )
            self.assertEqual("completed", measurement.status)
            self.assertTrue(measurement.complete)
            self.assertEqual(0.75, measurement.quality_score)
            self.assertGreater(measurement.latency_tokens_per_second, 0)
            self.assertEqual(path.stat().st_size, measurement.disk_bytes)
            self.assertGreater(measurement.peak_ram_mb, 0)
            self.assertEqual(456.0, measurement.peak_vram_mb)
            self.assertEqual("fixture CPU/GPU", measurement.hardware)

            output = Path(tmp) / "measurement.json"
            write_variant_measurement(measurement, output)
            self.assertTrue(output.is_file())
            self.assertTrue(output.read_text(encoding="utf-8").endswith("\n"))

    def test_comparison_prefers_quality_then_speed_then_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.gguf"
            path.write_bytes(b"x")
            cases = (VariantBenchmarkCase("one", "prompt"),)

            def measurement(model_id, quantization, quality, tokens):
                return measure_variant(
                    model_id,
                    quantization,
                    path,
                    lambda _case: {"text": "output", "token_count": tokens},
                    cases=cases,
                    quality_scorer=lambda _case, _text: quality,
                )

            rows = compare_variants(
                [
                    measurement("slow-high", "Q8_0", 0.9, 4),
                    measurement("fast-same", "Q4_K_M", 0.9, 8),
                    measurement("low", "Q4_K_M", 0.8, 20),
                ]
            )
            self.assertEqual(["fast-same", "slow-high", "low"], [row["model_id"] for row in rows])


if __name__ == "__main__":
    unittest.main()
