import cv2
from paddleocr import PaddleOCR



ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)



def prepareForOcr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    return image



def extractResult(image):
    image = prepareForOcr(image)

    results = ocr.predict(image)

    texts = []
    scores = []

    for result in results:
        texts.extend(result["rec_texts"])
        scores.extend(float(score) for score in result.get("rec_scores", []))

    return "\n".join(texts), sum(scores) / len(scores) if scores else None


def extractText(image):
    return extractResult(image)[0]
