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
        Clock.schedule_interval(self.check_result, 0.5)

    def check_result(self, dt):
        if self.main_screen.ocr_text.text != 'UI smoke result' and time.monotonic() < self.deadline:
            return
        if self.stage == 0:
            assert Window.dispatch('on_key_down', 114, 0, '', [])
            self.pause_until = time.monotonic() + 3
            self.stage = 1
            return
        document_calls = [call for call in voice.Speak.call_args_list if call.args[0] == 'UI smoke result']
        if self.stage == 1 and document_calls:
            self.main_screen.read_button.dispatch('on_release')
            self.stage = 2
            return
        if self.stage == 2 and len(document_calls) >= 2:
            assert Window.dispatch('on_key_down', 32, 0, None, [])
            self.stage = 3
            return
        stopped = any(call.args == ('', 3) for call in voice.Speak.call_args_list)
        if (self.stage != 3 or not stopped) and time.monotonic() < self.deadline:
            return
        if time.monotonic() < self.pause_until:
            return
        if not self.rescan_sent:
            assert reader.read.call_count == 1, 'Scene movement unexpectedly triggered another scan'
            assert self.controller.state == self.controller.PAUSED
            assert not self.main_screen.scan_again_button.disabled
            self.main_screen.scan_again_button.dispatch('on_release')
            self.rescan_sent = True
            return
        if reader.read.call_count < 2 and time.monotonic() < self.deadline:
            return
        try:
            assert reader.read.call_count == 2, 'Scan again did not start a second capture'
            assert self.main_screen.ocr_text.text == 'UI smoke result', self.main_screen.ocr_text.text
            assert len(document_calls) >= 2, voice.Speak.call_args_list
            assert stopped
            assert self.controller.speech.error is None
            self.passed = True
        finally:
            self.stop()
        return False


with patch('controller.snapPrintController.Camera', return_value=camera), patch('win32com.client.Dispatch', return_value=voice), patch('controller.snapPrintController.VLReader', return_value=reader):
    app = SmokeApp()
    app.run()
    assert app.passed
    print('UI and speech controls passed. Scene movement stays paused; Scan again starts exactly one new capture.')
