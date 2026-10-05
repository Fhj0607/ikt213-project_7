from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)


def extract_text(image_path):
    results = ocr.predict(image_path)

    texts = []

    for result in results:
        texts.extend(result["rec_texts"])

    return "\n".join(texts)