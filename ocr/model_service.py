"""App-lifetime models and a shared serial inference worker."""
from concurrent.futures import ThreadPoolExecutor

from ocr.text_gate import TextGate
from ocr.vl_reader import VLReader


class ModelService:
    def __init__(self):
        self.reader = VLReader()
        self.text_gate = TextGate()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapprint-models")
        self.closed = False

    def load(self, progress):
        progress("Loading text detector (1/2)...")
        self.text_gate.load()
        progress("Loading document recognition models (2/2)...")
        self.reader.load()

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Cleanup is queued behind any active inference; never unload a model in use.
        self.worker.submit(self._closeModels)
        self.worker.shutdown(wait=False)

    def _closeModels(self):
        try:
            self.reader.close()
        finally:
            self.text_gate.close()
