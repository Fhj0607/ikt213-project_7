import cv2
import numpy as np



def ToGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def ResizeImage(image, width, height):
    return cv2.resize(image,(width, height))



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



def Brighten(image, amount=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = cv2.add(v, amount)

    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



def Darken(image, amount=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = cv2.subtract(v, amount)

    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



