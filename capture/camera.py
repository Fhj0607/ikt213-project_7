import cv2

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None

    def start(self):

        if self.cap is not None and self.cap.isOpened():
            return

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.stop()
            raise RuntimeError('Could not start camera.')

    def readFrame(self):
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Camera is not initialized.')

        success, frame = self.cap.read()

        if not success or frame is None:
            raise RuntimeError('Could not read frame.')

        return frame

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None