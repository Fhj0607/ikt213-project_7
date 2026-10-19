import cv2

from paddleocr import PaddleOCR

# lazy ocr implementation; loads only when extract is called
class OCR:

    def __init__(self):
        self.ocr = None

    def load(self):
        if self.ocr is None:
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )

    def prepare(self, image):
        if len(image.shape) == 2:
            return cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2BGR
            )

        return image

    def extract(self, image):

        # loads only when extract is called.
        self.load()

        image = self.prepare(image)
        results = self.ocr.predict(image)

        texts = []

        for result in results:
            texts.extend(result["rec_texts"])

        return "\n".join(texts)
