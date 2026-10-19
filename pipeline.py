class Pipeline:

    def __init__(self):
        self.ocr = None

    def process(self, frame):

        from ocr.paddle_ocr import extractText

        processed = frame.copy()

        text = extractText(processed)

        return text