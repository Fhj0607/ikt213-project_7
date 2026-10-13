import cv2
import threading
import time



class Camera:

    def __init__(
        self,
        cameraIndex=0,
        width=640,
        height=480
    ):
        self.cameraIndex = cameraIndex
        self.width = width
        self.height = height

        self.latestFrame = None

        self.frameLock = threading.Lock()

        self.running = False
        self.captureThread = None



    def Start(self):

        self.running = True

        self.captureThread = threading.Thread(target=self.CaptureLoop, daemon=True)

        self.captureThread.start()



    def CaptureLoop(self):

        camera = cv2.VideoCapture(self.cameraIndex)

        if not camera.isOpened():
            self.running = False
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        while self.running:

            success, frame = camera.read()

            if not success:
                time.sleep(0.01)
                continue

            with self.frameLock:
                self.latestFrame = frame.copy()

        camera.release()



    def GetLatestFrame(self):

        with self.frameLock:

            if self.latestFrame is None:
                return None

            return self.latestFrame.copy()



    def Stop(self):

        self.running = False

        if self.captureThread is not None:
            self.captureThread.join(timeout=1)

            self.captureThread = None