import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.credentials import (
    HF_TOKEN_ACCOUNT,
    CredentialError,
    MemoryCredentialStore,
)
from core.settings import SECRET_SETTING_KEYS, Settings

TOKEN = "hf_UNITTESTTOKEN1234567890"


class CredentialStorageTests(unittest.TestCase):
    """Secrets belong in the OS credential service, never in config JSON."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.config_dir = self.root / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / "config.json"
        self.store = MemoryCredentialStore()

        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(
            mock.patch("core.settings.get_config_dir", return_value=self.config_dir))
        self._stack.enter_context(
            mock.patch("core.settings.get_default_output_dir",
                       return_value=self.root / "out"))
        self._stack.enter_context(
            mock.patch("core.settings.get_default_cache_dir",
                       return_value=self.root / "models"))
        self._stack.enter_context(
            mock.patch("core.credentials.get_credential_store",
                       return_value=self.store))
        Settings._instance = None
        self.addCleanup(setattr, Settings, "_instance", None)

    def _write_config(self, data: dict):
        self.config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _config_json(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_setting_a_token_never_reaches_config_json(self):
        settings = Settings()
        settings.set("model_hub.hf_token", TOKEN)

        self.assertEqual(self.store.get_secret(HF_TOKEN_ACCOUNT), TOKEN)
        self.assertNotIn(TOKEN, self.config_path.read_text(encoding="utf-8"))
        self.assertNotIn("hf_token", self._config_json().get("model_hub", {}))
        self.assertEqual(settings.get("model_hub.hf_token", ""), TOKEN)

    def test_clearing_the_token_deletes_the_stored_secret(self):
        settings = Settings()
        settings.set("model_hub.hf_token", TOKEN)
        settings.set("model_hub.hf_token", "")
        self.assertIsNone(self.store.get_secret(HF_TOKEN_ACCOUNT))
        self.assertEqual(settings.get("model_hub.hf_token", ""), "")

    def test_legacy_plaintext_token_is_migrated_and_removed(self):
        self._write_config({
            "schema_version": 3,
            "model_hub": {"cache_dir": str(self.root / "models"), "hf_token": TOKEN},
        })

        settings = Settings()

        self.assertEqual(self.store.get_secret(HF_TOKEN_ACCOUNT), TOKEN)
        self.assertNotIn("hf_token", self._config_json().get("model_hub", {}))
        self.assertNotIn(TOKEN, self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(settings.get("model_hub.hf_token", ""), TOKEN)
        messages = " ".join(settings.repair_status["messages"])
        self.assertIn("model_hub.hf_token", messages)

    def test_migration_scrubs_existing_timestamped_backups(self):
        backups = self.config_dir / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        legacy = {"schema_version": 2, "model_hub": {"hf_token": TOKEN}}
        for name in ("config.json.20260101_010101.pre-save.bak",
                     "config.json.20260102_020202.pre-migration.bak"):
            (backups / name).write_text(json.dumps(legacy), encoding="utf-8")
        self._write_config({
            "schema_version": 3,
            "model_hub": {"cache_dir": str(self.root / "models"), "hf_token": TOKEN},
        })

        settings = Settings()

        for path in backups.glob("config.json.*"):
            self.assertNotIn(TOKEN, path.read_text(encoding="utf-8"), path.name)
        messages = " ".join(settings.repair_status["messages"])
        self.assertIn("backup", messages.lower())

    def test_unreadable_backup_is_reported_not_silently_ignored(self):
        backups = self.config_dir / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        (backups / "config.json.20260103_030303.corrupt.bak").write_text(
            "{ not json", encoding="utf-8")
        self._write_config({
            "schema_version": 3,
            "model_hub": {"cache_dir": str(self.root / "models"), "hf_token": TOKEN},
        })

        settings = Settings()
        messages = " ".join(settings.repair_status["messages"])
        self.assertIn("could not be read", messages)

    def test_plaintext_is_kept_and_flagged_when_no_backend_exists(self):
        offline_store = MemoryCredentialStore(available=False)
        with mock.patch("core.credentials.get_credential_store",
                        return_value=offline_store):
            self._write_config({
                "schema_version": 3,
                "model_hub": {"cache_dir": str(self.root / "models"), "hf_token": TOKEN},
            })
            Settings._instance = None
            settings = Settings()

            status = settings.repair_status
            self.assertEqual(status["status"], "error")
            self.assertIn("plaintext", " ".join(status["messages"]))
            # The user's token is not destroyed just because storage failed.
            self.assertEqual(self._config_json()["model_hub"]["hf_token"], TOKEN)

    def test_saving_a_secret_without_a_backend_raises(self):
        offline_store = MemoryCredentialStore(available=False)
        with mock.patch("core.credentials.get_credential_store",
                        return_value=offline_store):
            Settings._instance = None
            settings = Settings()
            with self.assertRaises(CredentialError):
                settings.set("model_hub.hf_token", TOKEN)
            self.assertNotIn(TOKEN, self.config_path.read_text(encoding="utf-8"))

    def test_save_strips_any_secret_that_reached_the_data_dict(self):
        settings = Settings()
        # Simulate a stray write bypassing the secret gate.
        settings._data["model_hub"]["hf_token"] = TOKEN
        settings.save(create_backup=False)
        self.assertNotIn(TOKEN, self.config_path.read_text(encoding="utf-8"))

    def test_reset_all_clears_stored_secrets(self):
        settings = Settings()
        settings.set("model_hub.hf_token", TOKEN)
        settings.reset_all()
        self.assertIsNone(self.store.get_secret(HF_TOKEN_ACCOUNT))

    def test_registered_secret_keys_are_declared(self):
        self.assertIn("model_hub.hf_token", SECRET_SETTING_KEYS)
        self.assertEqual(SECRET_SETTING_KEYS["model_hub.hf_token"], HF_TOKEN_ACCOUNT)

    def test_diagnostics_report_names_backend_without_the_secret(self):
        from core.diagnostics import _credential_service_info

        settings = Settings()
        settings.set("model_hub.hf_token", TOKEN)
        info = _credential_service_info(settings)

        self.assertTrue(info["available"])
        self.assertTrue(info["secrets_present"]["model_hub.hf_token"])
        self.assertNotIn(TOKEN, json.dumps(info))


if __name__ == "__main__":
    unittest.main()
