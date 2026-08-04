import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from core.engine_contract import CAP_PRODUCER_RUN, CAP_SONG_GENERATE, ENGINE_CAPABILITIES
from core.model_manager import (
    COMMERCIAL_USE_ALLOWED,
    ModelCategory,
    ModelInfo,
    ModelManager,
    ModelSecurityError,
)
from core.song_generator_registry import (
    ACE_STEP_GENERATOR,
    HEARTMULA_GENERATOR,
    SongGeneratorConfig,
    SongGeneratorRegistry,
    SongGeneratorRegistryError,
    active_song_generator_model_ids,
    default_local_mirror,
    validate_song_generator_config,
)
from engines.song_generator_adapter import (
    SongGeneratorAdapterError,
    resolve_song_generator,
)


class SongGeneratorRegistryTests(unittest.TestCase):
    def test_builtin_registry_is_pinned_and_only_advertises_ready_generators(self):
        self.assertEqual(("ace-step-v1.5",), active_song_generator_model_ids())
        self.assertTrue(ACE_STEP_GENERATOR.is_pinned)
        self.assertTrue(HEARTMULA_GENERATOR.is_pinned)
        self.assertFalse(HEARTMULA_GENERATOR.enabled)
        self.assertIn("separate torch", HEARTMULA_GENERATOR.availability_reason)
        self.assertEqual(
            ENGINE_CAPABILITIES[CAP_SONG_GENERATE].model_ids,
            active_song_generator_model_ids(),
        )
        self.assertEqual(
            ENGINE_CAPABILITIES[CAP_PRODUCER_RUN].model_ids,
            active_song_generator_model_ids(),
        )

    def test_new_generator_is_a_registry_entry_not_a_capability_rewrite(self):
        config = SongGeneratorConfig(
            generator_id="fixture-generator",
            model_id="fixture-song-model",
            display_name="Fixture Generator",
            adapter_module="engines.fixture_engine",
            loader_fn="load_model",
            generation_fn="generate_song",
            source="fixture/model",
            revision="a" * 40,
            license="Apache-2.0",
            license_url="https://example.invalid/license",
            commercial_use=COMMERCIAL_USE_ALLOWED,
            local_mirror=default_local_mirror("fixture-song-model", "a" * 40),
            enabled=True,
        )
        registry = SongGeneratorRegistry((ACE_STEP_GENERATOR,))
        registry.register(config)
        self.assertEqual(
            ("ace-step-v1.5", "fixture-song-model"),
            registry.active_model_ids(),
        )

    def test_adapter_resolution_uses_the_registry_entry(self):
        config = SongGeneratorConfig(
            generator_id="fixture-generator",
            model_id="fixture-song-model",
            display_name="Fixture Generator",
            adapter_module="engines.fixture_engine",
            loader_fn="load_model",
            generation_fn="render",
            source="fixture/model",
            revision="f" * 40,
            license="Apache-2.0",
            license_url="https://example.invalid/license",
            commercial_use=COMMERCIAL_USE_ALLOWED,
            local_mirror=default_local_mirror("fixture-song-model", "f" * 40),
        )
        registry = SongGeneratorRegistry((config,))
        render = lambda **_kwargs: None
        with patch(
            "engines.song_generator_adapter.importlib.import_module",
            return_value=SimpleNamespace(render=render),
        ) as import_module:
            selected, function = resolve_song_generator(
                config.model_id,
                registry=registry,
            )
        self.assertIs(config, selected)
        self.assertIs(render, function)
        import_module.assert_called_once_with(config.adapter_module)

    def test_disabled_generator_cannot_be_resolved(self):
        with self.assertRaises(SongGeneratorAdapterError):
            resolve_song_generator(HEARTMULA_GENERATOR.model_id)

    def test_noncommercial_or_unlicensed_generator_requires_explicit_acceptance(self):
        config = SongGeneratorConfig(
            generator_id="restricted-generator",
            model_id="restricted-model",
            display_name="Restricted Generator",
            adapter_module="engines.restricted_engine",
            loader_fn="load_model",
            generation_fn="generate_song",
            source="fixture/restricted",
            revision="b" * 40,
            license="CC-BY-NC",
            license_url="https://example.invalid/license",
            commercial_use="non_commercial",
            local_mirror=default_local_mirror("restricted-model", "b" * 40),
        )
        with self.assertRaises(SongGeneratorRegistryError):
            validate_song_generator_config(config)
        validate_song_generator_config(config, license_accepted=True)

        unlicensed = config.__class__(
            **{
                **config.__dict__,
                "generator_id": "unlicensed-generator",
                "model_id": "unlicensed-model",
                "license": "",
                "commercial_use": "unknown",
                "local_mirror": default_local_mirror("unlicensed-model", "b" * 40),
            }
        )
        with self.assertRaises(SongGeneratorRegistryError):
            validate_song_generator_config(unlicensed)

    def test_model_entries_get_revision_pinned_local_mirrors(self):
        from core.model_manager import BUILTIN_MODELS

        for info in BUILTIN_MODELS.values():
            with self.subTest(model_id=info.model_id):
                self.assertTrue(info.local_mirror)
                self.assertIn(info.revision or "package-managed", info.local_mirror)

    def test_download_marker_records_and_verifies_local_mirror_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ModelManager()
            old_settings = manager._settings
            old_registry = manager._registry

            class SettingsStub:
                def get(self, key, default=None):
                    return tmp if key == "model_hub.cache_dir" else default

            try:
                manager._settings = SettingsStub()
                info = ModelInfo(
                    model_id="mirror-model",
                    name="Mirror Model",
                    description="test",
                    category=ModelCategory.EXTRAS,
                    vram_gb=1,
                    disk_gb=0.1,
                    license="MIT",
                    source="fixture/mirror",
                    revision="c" * 40,
                    loader_module="engines.sfx_engine",
                    loader_fn="load_model",
                    commercial_use=COMMERCIAL_USE_ALLOWED,
                    license_url="https://example.invalid/license",
                )
                manager._registry = {info.model_id: info}
                cache = manager.get_cache_dir(info.model_id)
                cache.mkdir(parents=True)
                (cache / "weights.safetensors").write_bytes(b"weights")
                manager._write_complete_marker(
                    info.model_id,
                    cache,
                    resolved_revision=info.revision,
                )
                self.assertTrue(manager.verify_download(info.model_id)[0])
                marker = manager.get_download_manifest(info.model_id)
                self.assertEqual(info.local_mirror, marker["local_mirror"])

                marker["local_mirror"] = "models/other-revision/" + info.revision
                (cache / manager.COMPLETE_MARKER).write_text(
                    json.dumps(marker), encoding="utf-8"
                )
                ok, reason = manager.verify_download(info.model_id)
                self.assertFalse(ok)
                self.assertIn("local_mirror mismatch", reason)
            finally:
                manager._settings = old_settings
                manager._registry = old_registry

    def test_model_manager_license_acceptance_is_revision_scoped(self):
        manager = ModelManager()
        old_settings = manager._settings
        old_registry = manager._registry

        class SettingsStub:
            def __init__(self):
                self.values = {"model_hub.license_consents": {}}

            def get(self, key, default=None):
                return self.values.get(key, default)

            def set(self, key, value):
                self.values[key] = value

        try:
            settings = SettingsStub()
            manager._settings = settings
            info = ModelInfo(
                model_id="restricted-model",
                name="Restricted Model",
                description="test",
                category=ModelCategory.EXTRAS,
                vram_gb=1,
                disk_gb=0.1,
                license="CC-BY-NC",
                source="fixture/restricted",
                revision="d" * 40,
                loader_module="engines.sfx_engine",
                loader_fn="load_model",
                commercial_use="non_commercial",
                license_url="https://example.invalid/license",
            )
            manager._registry = {info.model_id: info}
            with self.assertRaises(ModelSecurityError):
                manager.require_model_license_acceptance(info.model_id)
            manager.record_license_acceptance(info.model_id)
            manager.require_model_license_acceptance(info.model_id)
            info.revision = "e" * 40
            self.assertFalse(manager.has_license_acceptance(info.model_id))
        finally:
            manager._settings = old_settings
            manager._registry = old_registry


if __name__ == "__main__":
    unittest.main()
