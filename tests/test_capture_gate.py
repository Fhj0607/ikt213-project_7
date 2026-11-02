"""Regression checks for delayed text detection and changing camera stability."""
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock

os.environ.setdefault("KIVY_HOME", str(Path(__file__).resolve().parents[1] / "results" / "kivy-smoke"))
os.environ.setdefault("KIVY_NO_ARGS", "1")

import numpy as np
from controller.snapPrintController import SnapPrintController


class CaptureGateTests(unittest.TestCase):
    def setUp(self):
        self.controller = c = SnapPrintController.__new__(SnapPrintController)
        c.main_screen = Mock()
        c.speech = Mock()
        c.quality_assessment = Mock()
        c.running = True
        c.state = c.SCANNING
        c.processing = threading.Event()
        c.gate_pending = threading.Event()
        c.gate_failed = False
        c.gate_generation = 4
        c.gate_hits = 0
        c.latest_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        c.latest_assessment = {"acceptable": True}
        c.submitFrame = Mock()
        c.scheduleUi = lambda callback: callback()

    def test_transient_motion_during_slow_detection_does_not_invalidate_result(self):
        c = self.controller
        c.gate_pending.set()
        c.quality_assessment.assessFrame.return_value = {"acceptable": False}
        c.handleScanning(c.latest_frame)
        self.assertEqual(c.gate_generation, 4)
        c.quality_assessment.assessFrame.return_value = {"acceptable": True}
        c.handleScanning(c.latest_frame)
        c.acceptDetection(c.latest_frame, 4, True)
        c.submitFrame.assert_not_called()
        c.acceptDetection(c.latest_frame, 4, True)
        c.submitFrame.assert_called_once_with(c.latest_frame)
        self.assertEqual(c.state, c.PAUSED)

    def test_changed_page_rejects_pending_result(self):
        c = self.controller
        c.gate_hits = 1
        old_frame = np.full_like(c.latest_frame, 255)
        c.acceptDetection(old_frame, 4, True)
        c.submitFrame.assert_not_called()
        self.assertEqual(c.gate_hits, 0)

    def test_result_waits_for_current_stability(self):
        c = self.controller
        c.latest_assessment = {"acceptable": False}
        c.gate_hits = 1
        c.acceptDetection(c.latest_frame, 4, True)
        c.submitFrame.assert_not_called()
        self.assertEqual(c.gate_hits, 0)

    def test_detection_error_remains_visible_after_camera_checks(self):
        c = self.controller
        c.gateError("Detector failed")
        message = c.main_screen.status.text
        c.quality_assessment.assessFrame.return_value = {"acceptable": False}
        c.handleScanning(c.latest_frame)
        self.assertEqual(c.main_screen.status.text, message)
