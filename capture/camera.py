import cv2



class Camera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None



    def start(self):
        self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Could not start camera")



    def captureFrame(self):
        if self.camera is None:
            raise RuntimeError("Camera is not started")

        success, frame = self.camera.read()

        if not success:
            raise RuntimeError("Could not capture frame")

        return frame



    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None