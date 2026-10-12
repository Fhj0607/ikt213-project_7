class Pipeline:

    def __init__(self):
        self.ocr = None

    def process(self, frame):

        from ocr.paddle_ocr import ExtractText

        processed = frame.copy()

        text = ExtractText(processed)

        return text