import cv2

from processing.preprocessing import ToGrayscale, ApplyClahe
from processing.analysis import ContrastScore


image_path = "images/demo.png"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )

gray = ToGrayscale(image)

low_contrast = cv2.convertScaleAbs(
    gray,
    alpha=0.25,
    beta=95
)

enhanced = ApplyClahe(low_contrast)

original_score = ContrastScore(gray)
low_score = ContrastScore(low_contrast)
enhanced_score = ContrastScore(enhanced)

print(f"Original contrast score: {original_score:.2f}")
print(f"Low contrast score:      {low_score:.2f}")
print(f"CLAHE score:             {enhanced_score:.2f}")

cv2.imwrite("results/original_contrast.png", gray)
cv2.imwrite("results/low_contrast.png", low_contrast)
cv2.imwrite("results/clahe.png", enhanced)