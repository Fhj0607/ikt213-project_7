import unittest
from unittest.mock import Mock

import numpy as np
from ocr.text_gate import TextGate


class TextGateTests(unittest.TestCase):
    def test_resize_and_filter_without_recognition(self):
        gate = TextGate()
        gate.model = Mock()
        valid = np.array([[0, 0], [40, 0], [40, 10], [0, 10]])
        tiny = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        gate.model.predict.return_value = [{"dt_polys": [valid, valid, tiny],
                                           "dt_scores": [0.8, 0.2, 0.9]}]
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertTrue(gate.detect(frame))
        image = gate.model.predict.call_args.args[0]
        self.assertEqual(image.shape, (540, 960, 3))
        self.assertEqual(frame.shape, (1080, 1920, 3))
        gate.min_regions = 2
        self.assertFalse(gate.detect(frame))

    def test_blank_result_does_not_pass(self):
        gate = TextGate()
        gate.model = Mock()
        gate.model.predict.return_value = [{"dt_polys": [], "dt_scores": []}]
        self.assertFalse(gate.detect(np.zeros((200, 300, 3), dtype=np.uint8)))
