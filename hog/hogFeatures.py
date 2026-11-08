import cv2
import numpy as np

PATCH_SIZE = 32
WINDOW_STRIDE = 16
MAX_SIDE = 960

# 32x32 window, 16x16 blocks (2x2 cells), block stride 8, 8x8 cells, 9 orientation bins
# -> 3x3 block positions * 4 cells * 9 bins = 324 features per patch
HOG = cv2.HOGDescriptor(
    (PATCH_SIZE, PATCH_SIZE),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)
def prepareFrame(frame):
    """Grayscale, scaled so the longest side is MAX_SIDE pixels."""
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = frame.shape

    scale = MAX_SIDE / max(height, width)

    size = (
        max(PATCH_SIZE, round(width * scale)),
        max(PATCH_SIZE, round(height * scale))
    )

    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

def slidingWindows(image):
    """View of every window as a (rows, cols, PATCH_SIZE, PATCH_SIZE) array."""
    windows = np.lib.stride_tricks.sliding_window_view(
        image,
        (PATCH_SIZE, PATCH_SIZE)
    )

    return windows[::WINDOW_STRIDE, ::WINDOW_STRIDE]

def hogFeatures(patches):
    """HOG vector for each patch in an (N, PATCH_SIZE, PATCH_SIZE) uint8 array."""
    return np.stack([
        HOG.compute(np.ascontiguousarray(patch)).ravel()
        for patch in patches
    ]).astype(np.float32)