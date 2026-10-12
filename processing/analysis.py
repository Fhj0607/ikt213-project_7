import cv2



def IsBlurry(image, threshold=100.0):
    score = cv2.Laplacian(image, cv2.CV_64F).var()

    return score < threshold, score



def ContrastScore(image):
    return image.std()