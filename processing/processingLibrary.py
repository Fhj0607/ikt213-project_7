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



def orderPoints(points):
    points = points.reshape(4, 2).astype(np.float32)

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()  # y - x

    return np.array([
        points[np.argmin(sums)],
        points[np.argmin(differences)],
        points[np.argmax(sums)],
        points[np.argmax(differences)]
    ], dtype=np.float32)



def findPage(image):
    ratio = image.shape[0] / 800.0

    small = resizeImage(image, int(image.shape[1] / ratio), 800)

    gray = toGrayscale(small)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    min_area = 0.2 * small.shape[0] * small.shape[1]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2) * ratio

    return None



def warpPage(image, corners):
    tl, tr, br, bl = orderPoints(corners)

    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), destination)

    return cv2.warpPerspective(image, matrix, (width, height))



def deskew(image):
    gray = toGrayscale(image)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    h, w = image.shape[:2]

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)



def alignImage(image):
    corners = findPage(image)

    page_found = corners is not None

    if page_found:
        image = warpPage(image, corners)

    return deskew(image), page_found

