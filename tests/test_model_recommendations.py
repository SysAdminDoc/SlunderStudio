import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.model_manager import (
    BUILTIN_MODELS,
    model_hardware_fit,
    model_supports_task,
    model_tasks,
    recommend_model_for_task,
    vram_tier_for_gb,
)
from ui.model_hub import ModelCard, ModelHubView


class ModelRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_every_builtin_model_has_dated_task_measurement_and_vram_tier(self):
        self.assertGreaterEqual(len(model_tasks(BUILTIN_MODELS)), 6)
        for info in BUILTIN_MODELS.values():
            with self.subTest(model=info.model_id):
                self.assertTrue(info.task_labels)
                self.assertTrue(info.measurement_basis)
                self.assertTrue(info.measurement_source)
                self.assertRegex(info.measurement_date, r"^2026-08-03$")
                self.assertTrue(info.advertised_vram_tier)

    def test_vram_tiers_match_registry_boundaries(self):
        self.assertEqual("≤4 GB", vram_tier_for_gb(4))
        self.assertEqual("4–6 GB", vram_tier_for_gb(6))
        self.assertEqual("8–12 GB", vram_tier_for_gb(12))
        self.assertEqual("≥24 GB", vram_tier_for_gb(24.1))

    def test_task_support_is_safe_when_a_card_has_no_registry_info(self):
        self.assertFalse(model_supports_task(None, "best vocal isolation"))

    def test_recommendation_uses_actual_gpu_fit(self):
        hardware = {
            "available": True,
            "backend": "cuda",
            "name": "Fixture RTX",
            "total_gb": 8.0,
            "used_gb": 0.0,
        }
        vocal = recommend_model_for_task(
            "best vocal isolation", BUILTIN_MODELS, hardware
        )
        self.assertEqual("audio-separator", vocal.model_id)
        self.assertEqual("cuda", model_hardware_fit(vocal, hardware).status)

        song = BUILTIN_MODELS["ace-step-v1.5"]
        fit = model_hardware_fit(song, hardware)
        self.assertEqual("cpu-fallback", fit.status)
        self.assertFalse(fit.fits)
        self.assertIn("16.0 GB", fit.reason)

    def test_model_card_surfaces_task_measurement_and_hardware_basis(self):
        card = ModelCard(BUILTIN_MODELS["audio-separator"])
        try:
            self.assertIn("best vocal isolation", card._task_label.text())
            self.assertIn("12.9", card._measurement_label.text())
            card.update_hardware_status({
                "available": True,
                "backend": "cuda",
                "name": "Fixture RTX",
                "total_gb": 8.0,
                "used_gb": 0.0,
            })
            self.assertIn("Fits", card._hardware_label.text())
        finally:
            card.deleteLater()

    def test_model_card_labels_core_status_without_relying_on_color(self):
        core_info = next(info for info in BUILTIN_MODELS.values() if info.is_core)
        card = ModelCard(core_info)
        try:
            self.assertEqual("Core", card._core_badge.text())
            self.assertFalse(card._core_badge.isHidden())
            self.assertIn("core model badge", card._core_badge.accessibleName())
        finally:
            card.deleteLater()

    def test_model_hub_task_and_hardware_filters_show_fit_recommendation(self):
        view = ModelHubView()
        try:
            self.assertGreater(view._task_filter.findData("best vocal isolation"), 0)
            view._update_gpu_display({
                "available": True,
                "backend": "cuda",
                "name": "Fixture RTX",
                "total_gb": 8.0,
                "used_gb": 0.0,
                "current_model_name": "",
            })
            view._task_filter.setCurrentIndex(
                view._task_filter.findData("best vocal isolation")
            )
            view._filter_cards()
            self.assertIn("Audio Separator", view._recommendation_label.text())
            self.assertEqual("fit", view._hardware_filter.currentData())
        finally:
            view.deleteLater()


if __name__ == "__main__":
    unittest.main()
