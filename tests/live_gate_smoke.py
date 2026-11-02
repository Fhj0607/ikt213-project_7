"""Exercise automatic live capture with the real CPU detector and mocked VL/camera.

Run manually: python tests/live_gate_smoke.py
Uses cached detector weights, or downloads them on first use.
"""
import os
from pathlib import Path
import sys
import time
from unittest.mock import Mock, patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ["KIVY_HOME"] = str(root / "results" / "kivy-smoke")
os.environ["KIVY_NO_ARGS"] = "1"

import cv2
from kivy.clock import Clock
from kivy.core.window import Window
from main import SnapPrintApp

frame = cv2.imread(str(root / "results" / "printed-test.png"))
assert frame is not None, "Missing printed test image"
camera = Mock()
camera.captureFrame.side_effect = lambda: frame.copy()
reader = Mock()
reader.read.return_value = ("Automatic live recognition", 0)
voice = Mock()
voice.Status.RunningState = 1


class GateSmoke(SnapPrintApp):
    passed = False

    def on_start(self):
        super().on_start()
        Window.hide()
        self.started = time.monotonic()
        Clock.schedule_interval(self.check, 0.1)

    def check(self, dt):
        if self.controller.ready and not self.controller.session_active:
            self.controller.startSession()
            self.controller.chooseMode("live")
        if self.main_screen.ocr_text.text == "Automatic live recognition":
            try:
                reader.read.assert_called_once()
                image = reader.read.call_args.args[0]
                assert image.shape == frame.shape
                assert self.controller.capture.state == self.controller.capture.PAUSED
                self.passed = True
                print(f"Real detector automatically triggered VL in {time.monotonic() - self.started:.1f}s", flush=True)
            finally:
                self.stop()
            return False
        if time.monotonic() - self.started > 120:
            status = self.main_screen.status.text
            self.stop()
            raise AssertionError("Automatic capture timed out: " + status)


with patch("controller.snapPrintController.Camera", return_value=camera), patch("ocr.model_service.VLReader", return_value=reader), patch("win32com.client.Dispatch", return_value=voice):
    app = GateSmoke()
    app.run()
    assert app.passed
