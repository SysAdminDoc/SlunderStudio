import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.lyrics_db import LyricsDB
from core.settings import Settings
from engines.lyrics_engine import LyricsLLM, generate_lyrics
from ui.lyrics_view import LyricsView


class LyricsProPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def tearDown(self):
        if LyricsDB._instance is not None:
            LyricsDB._instance.close()
        LyricsDB._instance = None
        Settings._instance = None

    def test_system_prompt_override_reaches_loaded_llm(self):
        manager = mock.Mock()
        llm = LyricsLLM()
        llm._model = object()
        llm._backend = "stub"
        manager.load_model.return_value = llm
        settings = mock.Mock()
        settings.get.side_effect = lambda _key, default=None: default
        custom_system = "Write only vivid, radio-ready lyrics with no clichés."

        with (
            mock.patch("engines.lyrics_engine.ModelManager", return_value=manager),
            mock.patch("engines.lyrics_engine.Settings", return_value=settings),
            mock.patch.object(
                llm, "generate", return_value="[Verse 1]\nA lyric"
            ) as generate,
        ):
            result = generate_lyrics(
                "A midnight train",
                system_prompt_override=custom_system,
            )

        self.assertEqual("[Verse 1]\nA lyric", result["lyrics"])
        self.assertEqual(custom_system, generate.call_args.kwargs["system_prompt"])
        self.assertEqual("A midnight train", generate.call_args.kwargs["user_prompt"])

    def test_pro_mode_passes_editor_prompt_to_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = Path(tmp)
            with (
                mock.patch("core.settings.get_config_dir", return_value=config_dir),
                mock.patch("core.lyrics_db.get_config_dir", return_value=config_dir),
            ):
                view = LyricsView()
                try:
                    custom_system = "Use internal rhyme and a hopeful final chorus."
                    view._system_prompt.setPlainText(custom_system)
                    view._user_prompt.setPlainText("A sunrise after a storm")

                    with mock.patch.object(view, "_run_generation") as run_generation:
                        view._generate_pro()

                    self.assertEqual(
                        custom_system,
                        run_generation.call_args.kwargs["system_prompt_override"],
                    )
                    self.assertEqual(
                        "A sunrise after a storm", run_generation.call_args.args[1]
                    )
                finally:
                    view.deleteLater()
                    self._app.processEvents()
                    if LyricsDB._instance is not None:
                        LyricsDB._instance.close()
                    LyricsDB._instance = None
                    Settings._instance = None


if __name__ == "__main__":
    unittest.main()
