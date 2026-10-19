from ocr.paddle_ocr import OCR

class Pipeline:

    def __init__(self):
        self.ocr = OCR()

    def process(self, frame):

        image = frame.copy()

        return self.ocr.extract(image)