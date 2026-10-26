import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import numpy as np

from ocr.vl_reader import VLReader, readable_text


class VLReaderTests(unittest.TestCase):
    def test_two_pages_are_read_left_then_right(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        image[:, 20:] = 255
        reader = VLReader()
        reader.pipeline = Mock()
        left = {'parsing_res_list': [SimpleNamespace(content='Left page')]}
        right = {'parsing_res_list': [SimpleNamespace(content='Right page')]}
        reader.pipeline.predict.side_effect = [[left], [right]]
        text, elapsed = reader.read(image, two_pages=True)
        self.assertEqual(text, 'Left page\n\nRight page')
        calls = reader.pipeline.predict.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertTrue(np.all(calls[0].args[0] == 0))
        self.assertTrue(np.all(calls[1].args[0] == 255))
        self.assertEqual(calls[0].kwargs['max_pixels'], 1003520)
        self.assertFalse(calls[0].kwargs['use_queues'])
        self.assertTrue(calls[0].kwargs['use_ocr_for_image_block'])

    def test_image_placeholder_is_not_a_transcription(self):
        placeholder = '<div style="text-align: center;"><img src="imgs/image.jpg" alt="Image" width"84%" /></div>'
        self.assertEqual(readable_text(placeholder), '')
        self.assertEqual(readable_text('![Image](imgs/image.jpg)'), '')

    def test_raw_image_region_ocr_is_kept(self):
        reader = VLReader()
        reader.pipeline = Mock()
        reader.pipeline.predict.return_value = [{
            'parsing_res_list': [SimpleNamespace(content='My handwritten note')],
        }]
        text, _ = reader.read(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertEqual(text, 'My handwritten note')

    def test_table_markup_keeps_words_without_attributes(self):
        text = readable_text('<table><tr><td>Price</td><td>12 &amp; 15</td></tr></table>')
        self.assertEqual(text, 'Price 12 & 15')

    def test_empty_output(self):
        reader = VLReader()
        reader.pipeline = Mock()
        reader.pipeline.predict.return_value = []
        text, _ = reader.read(np.zeros((20, 20, 3), dtype=np.uint8))
        self.assertEqual(text, '')


if __name__ == '__main__':
    unittest.main()
