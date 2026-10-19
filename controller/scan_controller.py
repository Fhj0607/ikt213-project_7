import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from kivy.clock import Clock

from capture.camera import Camera
from quality.assessment import QualityAssessment
from pipeline import Pipeline

class ScanController:

    def __init__(self, on_frame, on_status, on_capture, on_processing):
        self.on_frame = on_frame
        self.on_status = on_status
        self.on_capture = on_capture
        self.on_processing = on_processing

        self.camera = Camera()
        self.quality = QualityAssessment()
        self.pipeline = Pipeline()
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.state = "SCANNING"
        self.captured_frame = None
        self.current_frame = None
        self.scene_change_count = 0
        self.shutting_down = False

        self.camera_event = None
        self.quality_event = None

    def start(self):

        self.camera.start()

        self.camera_event = Clock.schedule_interval(
            self.updateFrame, 1/30
        )

        self.quality_event = Clock.schedule_interval(
            self.checkQuality, 0.5
        )

    def updateFrame(self, dt):
        frame = self.camera.readFrame()

        self.current_frame = frame

        if self.state == "PAUSED":
            return

        self.on_frame(frame)

    def sceneDiff(self, frame1, frame2):

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        gray1 = cv2.resize(gray1, (320,240))
        gray2 = cv2.resize(gray2, (320,240))

        diff = cv2.absdiff(gray1, gray2)

        return float(np.mean(diff))

    def checkQuality(self, dt):

        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        if self.state in ("PAUSED", "PROCESSING"):
            return

        if self.state == "WAITING":

            diff = self.sceneDiff(
                frame,
                self.captured_frame
            )

            if diff > 15.0:
                self.scene_change_count += 1
            else:
                self.scene_change_count = 0

            if self.scene_change_count >= 3:
                self.state = "SCANNING"
                self.quality.reset()
                self.scene_change_count = 0

                print("New scene detected. Resumming scanning.")

            return

        result = self.quality.assess(frame)

        self.on_status(
            f"Sharpness: {result['sharpness']}\n"
            f"Brightness: {result['brightness']}\n"
            f"Contrast: {result['contrast']}\n"
            f"Stable frames: {result['stable_count']}\n"
            f"Status: {result['reason']}"
        )

        if not result["passed"]:
            return

        self.captured_frame = frame.copy()
        self.state = "PAUSED"

        self.on_frame(self.captured_frame)
        self.on_status("Stable frame captured. Paused for inspection. (for debugging, remove later).")
        self.on_capture()

    def runOCR(self):

        if self.shutting_down or self.state != "PAUSED":
            return

        self.state = "PROCESSING"

        self.on_processing()
        self.on_status("Processing image...")

        future = self.executor.submit(
            self.pipeline.process,
            self.captured_frame
        )

        future.add_done_callback(self.processingFinished)

    def processingFinished(self, future):

        try:
            text = future.result()
            error = None

        except Exception as e:
            text = ""
            error = str(e)

        if self.shutting_down:
            return

        Clock.schedule_once(
            lambda dt: self.displayResult(text,error)
        )

    def displayResult(self, text, error=None):

        if self.shutting_down:
            return

        if error:
            print("Pipeline error:", error)
            self.on_status("Processing failed.")

        elif text.strip():
            print("Recognized text:")
            print(text)

            self.on_status("Text recognized!")

        else:
            print("No readable text detected.")
            self.on_status("No readable text found.")

        self.scene_change_count = 0
        self.state = "WAITING"

    def stop(self):

        self.shutting_down = True

        if self.camera_event is not None:
            self.camera_event.cancel()
            self.camera_event = None

        if self.quality_event is not None:
            self.quality_event.cancel()
            self.quality_event = None

        self.camera.stop()

        self.executor.shutdown(
            wait=False,
            cancel_futures=True
        )
