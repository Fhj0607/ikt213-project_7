import cv2



def BlurScore(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()