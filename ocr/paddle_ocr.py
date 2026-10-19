import cv2
from paddleocr import PaddleOCR



ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)



def prepareForOCR(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return image



def extractText(image):
    image = prepareForOCR(image)

    results = ocr.predict(image)

    texts = []

    for result in results:
        texts.extend(result["rec_texts"])

    return "\n".join(texts)