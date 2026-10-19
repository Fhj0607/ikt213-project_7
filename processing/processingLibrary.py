import cv2
import numpy as np



def toGrayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def resizeImage(image, width, height):
    return cv2.resize(image,(width, height))



def sharpen(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(image, -1, kernel)



def applyClahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    return clahe.apply(image)



def brighten(image, amount=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = cv2.add(v, amount)

    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



def darken(image, amount=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    v = cv2.subtract(v, amount)

    hsv = cv2.merge((h, s, v))

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)



