import cv2
import numpy as np



def ToGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def Sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)



def ApplyClahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return clahe.apply(image)