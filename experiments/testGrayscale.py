import cv2

from processing.preprocessing import ToGrayscale


image_path = "images/test1.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

gray = ToGrayscale(image)

cv2.imwrite("results/grayscale.png", gray)

print("Original shape:", image.shape)
print("Grayscale shape:", gray.shape)
print("Saved grayscale image to results/grayscale.png")