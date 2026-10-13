import cv2

from processing.preProcessing import ToGrayscale



def IsBlurry(image, threshold=100.0):
    score = cv2.Laplacian(image, cv2.CV_64F).var()

    return score < threshold, score



def IsLowContrast(image, threshold=20.0):
    score = image.std()

    return score < threshold, score



def IsTooDark(image, threshold=80.0):
    score = image.mean()

    return score < threshold, score



def IsTooBright(image, threshold=200.0):
    score = image.mean()

    return score > threshold, score



def IsAcceptable(image):

    gray = ToGrayscale(image)

    blurry, blurScore = IsBlurry(gray)
    tooDark, brightnessScore = IsTooDark(gray)

    if blurry:
        return False

    if tooDark:
        return False

    return True