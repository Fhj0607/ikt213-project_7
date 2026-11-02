import os
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
os.environ['KIVY_HOME'] = str(root / 'results' / 'kivy-smoke')
os.environ['KIVY_NO_ARGS'] = '1'

import numpy as np
from kivy.clock import Clock
from kivy.core.window import Window
from main import SnapPrintApp

Window.hide()
frame = np.random.default_rng(42).integers(0, 256, (240, 320, 3), dtype=np.uint8)
camera = Mock()
voice = Mock()
voice.Status.RunningState = 1
reader = Mock()
reader.read.return_value = ('UI smoke result', 0.1)
camera.captureFrame.side_effect = lambda: frame.copy() if not reader.read.called else 255 - frame


class SmokeApp(SnapPrintApp):
    passed = False
    stage = 0
    rescan_sent = False

    def on_start(self):
        super().on_start()
        Window.hide()
        self.deadline = time.monotonic() + 15
        Clock.schedule_interval(self.begin_session, 0.1)

    def begin_session(self, dt):
        if not self.controller.ready:
            assert time.monotonic() < self.deadline, self.main_screen.status.text
            return
        assert self.controller.capture is None, 'Camera must not start before choosing a mode'
        self.main_screen.start_button.dispatch('on_release')
        assert self.controller.session_active
        self.main_screen.live_button.dispatch('on_release')
        # Exercise confirmation policy before the capture loop gets a UI tick.
        controller = self.controller.capture
        with patch.object(controller, 'submitFrame') as submit:
            generation = controller.gate_generation
            controller.acceptDetection(frame, generation, True)
            submit.assert_not_called()
            controller.acceptDetection(frame, generation, False)
            controller.acceptDetection(frame, generation, True)
            submit.assert_not_called()
            controller.resetGate()
            controller.acceptDetection(frame, generation, True)
            submit.assert_not_called()  # stale result after movement/reset
            generation = controller.gate_generation
            controller.acceptDetection(frame, generation, True)
            controller.acceptDetection(frame, generation, True)
            submit.assert_called_once()
            assert submit.call_args.args[0] is frame
            controller.latest_frame = frame
            submit.reset_mock()
            controller.forceCapture()
            submit.assert_called_once()  # override bypasses confirmation
        controller.state = controller.SCANNING
        controller.resetGate()
        self.deadline = time.monotonic() + 15
        Clock.schedule_interval(self.check_result, 0.5)
        return False

    def check_result(self, dt):
        if self.stage == 0 and self.main_screen.ocr_text.text != 'UI smoke result' and time.monotonic() < self.deadline:
            return
        if self.stage == 0:
            assert not hasattr(self.main_screen, 'read_button')
            assert not self.onSpeechKey(Window, 114, 0, 'r', [])
            assert Window.dispatch('on_key_down', 32, 0, None, [])
            self.pause_until = time.monotonic() + 3
            self.stage = 3
            return
        document_calls = [call for call in voice.Speak.call_args_list if call.args[0] == 'UI smoke result']
        stopped = any(call.args == ('', 3) for call in voice.Speak.call_args_list)
        if (self.stage != 3 or not stopped) and time.monotonic() < self.deadline:
            return
        if time.monotonic() < self.pause_until:
            return
        if not self.rescan_sent:
            assert reader.read.call_count == 1, 'Scene movement unexpectedly triggered another scan'
            assert self.controller.capture.state == self.controller.capture.PAUSED
            assert not self.main_screen.scan_again_button.disabled
            self.main_screen.scan_again_button.dispatch('on_release')
            self.rescan_sent = True
            return
        if reader.read.call_count < 2 and time.monotonic() < self.deadline:
            return
        try:
            assert reader.read.call_count == 2, 'Scan again did not start a second capture'
            assert self.main_screen.ocr_text.text == 'UI smoke result\n\nUI smoke result', self.main_screen.ocr_text.text
            assert not document_calls, voice.Speak.call_args_list
            assert stopped
            assert self.controller.capture.speech.error is None
            self.main_screen.end_button.dispatch('on_release')
            assert self.controller.capture is None
            assert len(self.controller.completed_session['pages']) == 2
            self.main_screen.start_button.dispatch('on_release')
            voice.Speak.reset_mock()
            camera.start.reset_mock()
            self.main_screen.photos_button.dispatch('on_release')
            assert self.controller.capture.mode == 'photos'
            self.controller.togglePages()
            assert 'speech' not in self.main_screen.footer.text.lower()
            camera.start.assert_not_called()
            with patch('controller.snapPrintController.cv2.imread', return_value=frame):
                self.controller.uploadImage(self.controller.capture, 'test.png')
            self.controller.capture.processing_future.result(timeout=5)
            Clock.tick()
            assert self.controller.capture.pages == ['UI smoke result']
            self.controller.capture.takePhoto()
            assert camera.start.called
            deadline = time.monotonic() + 2
            while self.controller.capture.latest_frame is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.controller.capture.takePhoto()
            self.controller.capture.processing_future.result(timeout=5)
            assert len(self.controller.capture.pages) == 2
            previous_capture = self.controller.capture
            self.main_screen.end_button.dispatch('on_release')
            Clock.tick()
            assert self.controller.completed_session['text'] == 'UI smoke result\n\nUI smoke result'
            assert 'Session ended' in self.main_screen.status.text
            voice.Speak.assert_not_called()
            reader.load.assert_called_once()
            reader.close.assert_not_called()
            gate.return_value.load.assert_called_once()
            gate.return_value.close.assert_not_called()
            self.main_screen.start_button.dispatch('on_release')
            self.main_screen.end_button.dispatch('on_release')
            assert not self.controller.session_active
            assert not self.controller.canUpload()
            with patch.object(previous_capture, 'uploadImage') as stale_upload:
                self.controller.uploadImage(previous_capture, 'old-session.png')
                stale_upload.assert_not_called()
            self.passed = True
        finally:
            self.stop()
        return False


with patch('ocr.model_service.TextGate') as gate, patch('controller.snapPrintController.Camera', return_value=camera), patch('win32com.client.Dispatch', return_value=voice), patch('ocr.model_service.VLReader', return_value=reader):
    gate.return_value.detect.return_value = True
    app = SmokeApp()
    app.run()
    assert app.passed
    app.controller.models.worker.shutdown(wait=True)
    reader.close.assert_called_once()
    gate.return_value.close.assert_called_once()
    print('UI and speech controls passed. Scene movement stays paused; Scan again starts exactly one new capture.')
