import queue
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from output.speech import Speech
from processing.qualityAssessment import QualityAssessment


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        # Test announcement policy without producing audio.
        self.speech = Speech.__new__(Speech)
        self.speech.cooldown = 8
        self.speech._last_guidance = float('-inf')
        self.speech._last_result = ''
        self.speech.speak = Mock()

    def test_dark_unstable_frame_has_guidance_before_capture(self):
        result = QualityAssessment().assessFrame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertFalse(result['acceptable'])
        self.speech.announceQuality(result)
        message = self.speech.speak.call_args.args[0]
        self.assertIn('Too dark', message)
        self.assertIn('Too blurry', message)
        self.assertIn('Low contrast', message)
        self.assertIn('Hold steady', message)

    def test_guidance_cooldown(self):
        result = {'stable': True, 'issues': {'too_bright': True}}
        with patch('output.speech.time.monotonic', side_effect=[0, 1, 9]):
            for _ in range(3):
                self.speech.announceQuality(result)
        self.assertEqual(self.speech.speak.call_count, 2)
        self.assertIn('Too bright', self.speech.speak.call_args.args[0])

    def test_confidence_repeat_and_missing_output(self):
        self.speech.announceResult('Hello', 0.923)
        self.speech.speak.assert_called_with('Average OCR confidence: 92 percent.', interrupt=True)
        self.speech.repeat()
        self.speech.speak.assert_called_with('Average OCR confidence: 92 percent.', interrupt=True)
        self.speech.announceResult('', None)
        self.assertIn('No readable text', self.speech.speak.call_args.args[0])
        self.speech.announceResult('Hello', None)
        self.assertIn('unavailable', self.speech.speak.call_args.args[0])

    def test_worker_interrupts_and_closes(self):
        voice = Mock()
        voice.Status.RunningState = 1
        speech = Speech.__new__(Speech)
        speech._commands = queue.Queue()
        speech._closed = threading.Event()
        speech._commands.put(('speak', 'Confidence test', True))
        speech._commands.put(('stop', '', True))
        def pump():
            if speech._commands.empty():
                speech._closed.set()
        with patch('win32com.client.Dispatch', return_value=voice), patch('pythoncom.PumpWaitingMessages', side_effect=pump):
            speech._run()
        self.assertEqual(voice.Speak.call_args_list[0].args, ('Confidence test', 19))
        self.assertEqual(voice.Speak.call_args_list[1].args, ('', 3))

    def test_repeat_without_result_is_audible(self):
        self.speech.repeat()
        self.assertIn('No text to read', self.speech.speak.call_args.args[0])

    def test_repeat_reads_document_instead_of_status(self):
        self.speech.rememberResult('<Note> Read this handwritten sentence.')
        self.speech.speak('Hold steady.')
        self.speech.repeat()
        self.speech.speak.assert_called_with('<Note> Read this handwritten sentence.', interrupt=True)


if __name__ == '__main__':
    unittest.main()
