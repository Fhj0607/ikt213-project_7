import cv2

from processing.preprocessing import ToGrayscale, Sharpen
from processing.analysis import BlurScore


image_path = "images/demo.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

gray = ToGrayscale(image)

blurred = cv2.GaussianBlur(gray, (15, 15), 0)

original_score = BlurScore(gray)
blurred_score = BlurScore(blurred)

print(f"Original blur score: {original_score:.2f}")
print(f"Blurred blur score:  {blurred_score:.2f}")

sharpened = Sharpen(blurred)

sharpened_score = BlurScore(sharpened)

print(f"Sharpened score:     {sharpened_score:.2f}")

cv2.imwrite("results/original_gray.png", gray)
cv2.imwrite("results/blurred.png", blurred)
cv2.imwrite("results/sharpened.png", sharpened)