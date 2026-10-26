"""Local full-page PaddleOCR-VL reader, loaded only on first recognition."""

import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path


class _TextContent(HTMLParser):
    """Keep readable content, excluding image markup and HTML attributes."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('style', 'script'):
            self.hidden += 1
        if tag in ('br', 'p', 'div', 'tr', 'li'):
            self.parts.append('\n')
        elif tag in ('td', 'th'):
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('style', 'script'):
            self.hidden = max(0, self.hidden - 1)
        if tag in ('p', 'div', 'tr', 'li'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def readable_text(content):
    content = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', content)
    parser = _TextContent()
    parser.feed(content)
    return '\n'.join(line.strip() for line in ''.join(parser.parts).splitlines() if line.strip())


class VLReader:
    def __init__(self, device='gpu:0', max_pixels=1003520, max_tokens=2048):
        self.device = device
        self.max_pixels = max_pixels
        self.max_tokens = max_tokens
        self.pipeline = None

    def load(self):
        if self.pipeline is not None:
            return
        # Keep downloaded models in the project; recognition is entirely local.
        root = Path(__file__).resolve().parents[1]
        os.environ.setdefault('PADDLE_PDX_CACHE_HOME', str(root / '.models' / 'paddlex'))
        os.environ.setdefault('HF_HOME', str(root / '.models' / 'huggingface'))
        os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
        from paddleocr import PaddleOCRVL
        from paddlex.inference import load_pipeline_config

        config = load_pipeline_config('PaddleOCR-VL-1.6')
        config['batch_size'] = 1
        config['use_queues'] = False
        config['markdown_ignore_labels'] = []
        for module in config['SubModules'].values():
            module['batch_size'] = 1
        layout_dir = root / '.models' / 'PP-DocLayoutV3'
        vl_dir = root / '.models' / 'PaddleOCR-VL-1.6'
        if (layout_dir / 'model.safetensors').exists():
            config['SubModules']['LayoutDetection']['model_dir'] = str(layout_dir)
        recognition = config['SubModules']['VLRecognition']
        if (vl_dir / 'model.safetensors').exists():
            recognition['model_dir'] = str(vl_dir)
        recognition.pop('genai_config', None)
        recognition['engine_config'] = {
            'dtype': 'bfloat16' if self.device.startswith('gpu') else 'float32',
            'attn_implementation': 'sdpa',
        }
        self.pipeline = PaddleOCRVL(
            pipeline_version='v1.6',
            paddlex_config=config,
            engine='transformers',
            device=self.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=True,
            use_ocr_for_image_block=True,
        )

    def read(self, image, two_pages=False):
        self.load()
        start = time.perf_counter()
        # Explicit two-page mode assumes a centered, vertical page divide.
        pages = [image]
        if two_pages:
            middle = image.shape[1] // 2
            pages = [image[:, :middle].copy(), image[:, middle:].copy()]
        texts = []
        for page in pages:
            results = self.pipeline.predict(
                page, max_pixels=self.max_pixels,
                max_new_tokens=self.max_tokens, use_queues=False,
                use_ocr_for_image_block=True,
            )
            # Markdown exports render image regions as <img> placeholders, even
            # when their raw block content contains a useful OCR transcription.
            blocks = [
                readable_text(block.content)
                for result in results
                for block in result['parsing_res_list']
            ]
            texts.append('\n\n'.join(text for text in blocks if text))
        return '\n\n'.join(texts).strip(), time.perf_counter() - start

    def close(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
