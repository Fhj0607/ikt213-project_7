import cv2

from processing.preprocessing import (
    ToGrayscale,
    ResizeImage
)

from processing.analysis import IsBlurry


def LiveCapture(cameraIndex=0, threshold=100.0):

    camera = cv2.VideoCapture(cameraIndex)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open camera"
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)

    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    try:
        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame")
                break

            smallFrame = ResizeImage(frame,320,240)

            grayFrame = ToGrayscale(smallFrame)

            blurry, score = IsBlurry(grayFrame, threshold)

            if blurry:
                status = "BLURRY"
                color = (0, 0, 255)

            else:
                status = "SHARP"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                f"{status} | Score: {score:.1f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.imshow(
                "SnapPrint Camera",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()

        cv2.destroyAllWindows()