from processing.pipeline import ProcessImage
from ocr.paddleOcr import ExtractText



def Process(image):
    processedImage, report = ProcessImage(image)

    text = ExtractText(processedImage)

    return processedImage, report, text