import cv2
import numpy as np

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from concurrent.futures import ThreadPoolExecutor

from capture.camera import Camera
from quality.assessment import QualityAssessment
from pipeline import Pipeline


class MainScreen(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.camera = Camera()
        self.quality = QualityAssessment()
        self.pipeline = Pipeline()
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.state = "SCANNING"
        self.captured_frame = None
        self.scene_change_count = 0
        self.shutting_down = False
        self.camera_event = None
        self.quality_event = None
        self.current_frame = None

        self.preview = Image()
        self.add_widget(self.preview)

        self.quality_label = Label(
            text="Waiting for camera...",
            size_hint_y=None,
            height=100
        )

        self.add_widget(self.quality_label)

        self.continue_button = Button(
            text="Run OCR", size_hint_y=None, height=48, disabled=True
        )
        self.continue_button.bind(on_release=self.runOCR)
        self.add_widget(self.continue_button)

    def startCamera(self):
        self.camera.start()

        self.camera_event = Clock.schedule_interval(
            self.updateFrame, 1 / 30
        )

        self.quality_event = Clock.schedule_interval(
            self.checkQuality, 0.5
        )

    def updateFrame(self, dt):
        frame = self.camera.readFrame()

        # store original frame
        self.current_frame = frame

        if self.state == "PAUSED":
            return

        self.showFrame(frame)

    def showFrame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        texture = Texture.create(
            size=(rgb.shape[1], rgb.shape[0]),
            colorfmt="rgb"
        )

        texture.blit_buffer(
            rgb.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        texture.flip_vertical()

        self.preview.texture = texture

    def sceneDifference(self, frame1, frame2):

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        gray1 = cv2.resize(gray1, (320,240))
        gray2 = cv2.resize(gray2, (320,240))

        difference= cv2.absdiff(gray1, gray2)

        return float(np.mean(difference))

    def checkQuality(self, dt):

        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        # OCR is already running
        if self.state in ("PAUSED", "PROCESSING"):
            return

        # wait until the scene changes
        if self.state == "WAITING":

            difference = self.sceneDifference(
                frame,
                self.captured_frame
            )

            if difference > 15.0:
                self.scene_change_count += 1
            else:
                self.scene_change_count = 0

            if self.scene_change_count >= 3:
                self.state = "SCANNING"
                self.quality.reset()
                self.scene_change_count = 0

                print("New scene detected. Resuming scanning.")

            return

        # assess image quality
        result = self.quality.assess(frame)

        self.quality_label.text = (
            f"Sharpness: {result['sharpness']}\n"
            f"Brightness: {result['brightness']}\n"
            f"Contrast: {result['contrast']}\n"
            f"Stable frames: {result['stable_count']}\n"
            f"Status: {result['reason']}"
        )

        if not result["passed"]:
            return

        # suitable frame captured
        self.captured_frame = frame.copy()
        self.state = "PAUSED"
        self.showFrame(self.captured_frame)
        self.quality_label.text = "Stable frame captured. Paused for inspection."
        self.continue_button.disabled = False

    def runOCR(self, *_):
        if self.shutting_down or self.state != "PAUSED":
            return

        self.continue_button.disabled = True
        self.state = "PROCESSING"

        self.quality_label.text = "Processing image..."

        print("Image captured. Starting pipeline.")

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
            print("Pipeline error: ", error)
            self.quality_label.text = "Processing failed."
        elif text.strip():
            print("Recognized text:")
            print(text)

            self.quality_label.text = "Text recognized!"

        else:
            print("No readable text detected.")
            self.quality_label.text = "no readable text found."

        self.scene_change_count = 0
        self.state = "WAITING"

    def stopCamera(self):

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
