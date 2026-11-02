import os
from pathlib import Path
import threading
from concurrent.futures import Future
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("KIVY_HOME", str(Path(__file__).resolve().parents[1] / "results" / "kivy-smoke"))
os.environ.setdefault("KIVY_NO_ARGS", "1")

from controller.sessionController import SessionController
from ocr.model_service import ModelService


class ModelLifecycleTests(unittest.TestCase):
    def test_cleanup_waits_for_active_inference_and_runs_once(self):
        with patch("ocr.model_service.VLReader") as reader, patch("ocr.model_service.TextGate") as gate:
            service = ModelService()
            release = threading.Event()
            entered = threading.Event()
            def inference():
                entered.set()
                release.wait(5)
            service.worker.submit(inference)
            try:
                self.assertTrue(entered.wait(2))
                service.close()
                service.close()
                reader.return_value.close.assert_not_called()
                gate.return_value.close.assert_not_called()
            finally:
                release.set()
                service.worker.shutdown(wait=True)
            reader.return_value.close.assert_called_once()
            gate.return_value.close.assert_called_once()

    def test_loading_error_blocks_session_and_allows_retry(self):
        with patch("controller.sessionController.ModelService"):
            screen = Mock()
            controller = SessionController(screen, Mock())
            controller.startSession()
            self.assertFalse(controller.session_active)
            failed = Future()
            failed.set_exception(RuntimeError("No GPU memory"))
            controller.modelsLoaded(failed)
            self.assertFalse(controller.ready)
            screen.showLoadingError.assert_called_once_with("No GPU memory")
            with patch.object(controller, "initialize") as retry:
                controller.startSession()
                retry.assert_called_once()
            success = Future()
            success.set_result(None)
            controller.modelsLoaded(success)
            controller.startSession()
            self.assertTrue(controller.ready)
            self.assertTrue(controller.session_active)

    def test_late_loading_callback_does_not_update_closed_app(self):
        with patch("controller.sessionController.ModelService"):
            screen = Mock()
            controller = SessionController(screen, Mock())
            controller.stop()
            success = Future()
            success.set_result(None)
            controller.modelsLoaded(success)
            screen.showHome.assert_not_called()
            self.assertFalse(controller.ready)
