"""Each live-gate choice is an independent alternative to the other methods."""
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("KIVY_HOME", str(Path(__file__).resolve().parents[1] / "results/kivy-smoke"))
os.environ.setdefault("KIVY_NO_ARGS", "1")

from controller.sessionController import SessionController
from controller.snapPrintController import SnapPrintController
from ocr.model_service import ModelService


class GateChoiceTests(unittest.TestCase):
    def makeService(self):
        reader = Mock()
        detector = Mock()
        cnn = Mock()
        hog = Mock()
        modules = {
            "cnn.textClassifier": SimpleNamespace(CNNTextGate=Mock(return_value=cnn)),
            "hog.hogTextGate": SimpleNamespace(HogTextGate=Mock(return_value=hog)),
        }
        patches = [patch.dict("sys.modules", modules),
                   patch("ocr.model_service.VLReader", return_value=reader),
                   patch("ocr.model_service.TextGate", return_value=detector)]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        service = ModelService()
        def close():
            service.close()
            service.worker.shutdown(wait=True)
        self.addCleanup(close)
        return service, reader, {"detector": detector, "cnn": cnn, "hog": hog}, modules

    def test_startup_loads_recognition_without_loading_any_gate(self):
        service, reader, gates, modules = self.makeService()
        service.load(Mock())
        reader.load.assert_called_once()
        for gate in gates.values():
            gate.load.assert_not_called()
            gate.detect.assert_not_called()
        modules["cnn.textClassifier"].CNNTextGate.assert_not_called()
        modules["hog.hogTextGate"].HogTextGate.assert_not_called()

    def test_each_choice_calls_only_its_gate_without_transcribing(self):
        service, reader, gates, _ = self.makeService()
        frame = object()
        for name, selected in gates.items():
            with self.subTest(gate=name):
                for gate in gates.values():
                    gate.reset_mock()
                capture = SnapPrintController(Mock(), service, gate_type=name)
                capture.running = True
                capture.scheduleUi = lambda callback: callback()
                capture.acceptDetection = Mock()
                selected.detect.return_value = True
                capture.checkText(frame, 7)
                selected.detect.assert_called_once_with(frame)
                capture.acceptDetection.assert_called_once_with(frame, 7, True)
                for other_name, other in gates.items():
                    if other_name != name:
                        other.detect.assert_not_called()
                        other.load.assert_not_called()
                reader.read.assert_not_called()

    def test_photos_mode_does_not_run_any_gate(self):
        service, reader, gates, _ = self.makeService()
        capture = SnapPrintController(Mock(), service, gate_type="hog")
        capture.start("photos")
        for gate in gates.values():
            gate.detect.assert_not_called()
            gate.load.assert_not_called()
        reader.read.assert_not_called()

    def test_choice_is_explicit_and_locked_during_capture(self):
        with patch("controller.sessionController.ModelService"), patch(
                "controller.sessionController.SnapPrintController") as capture_factory:
            screen = Mock()
            controller = SessionController(screen, Mock())
            for name in ("detector", "cnn", "hog"):
                controller.selectGate(name)
                self.assertEqual(controller.gate_type, name)
            self.assertIn("HOG + SVM", screen.gate_button.text)
            controller.ready = True
            controller.session_active = True
            controller.chooseMode("live")
            self.assertEqual(capture_factory.call_args.kwargs["gate_type"], "hog")
            controller.selectGate("cnn")
            self.assertEqual(controller.gate_type, "hog")

    def test_unknown_choice_fails_instead_of_falling_back(self):
        service, _, _, _ = self.makeService()
        with self.assertRaises(ValueError):
            service.getGate("all")
        with patch("controller.sessionController.ModelService"):
            controller = SessionController(Mock(), Mock())
            with self.assertRaises(ValueError):
                controller.selectGate("all")


if __name__ == "__main__":
    unittest.main()
