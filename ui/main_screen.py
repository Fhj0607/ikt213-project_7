import cv2
import numpy as np

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
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

    def start_camera(self):
        self.camera.start()

        self.camera_event = Clock.schedule_interval(
            self.update_frame, 1 / 30
        )

        self.quality_event = Clock.schedule_interval(
            self.check_quality, 0.5
        )

    def update_frame(self, dt):
        frame = self.camera.read_frame()

        # store original frame
        self.current_frame = frame

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

    def scene_difference(self, frame1, frame2):

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        gray1 = cv2.resize(gray1, (320,240))
        gray2 = cv2.resize(gray2, (320,240))

        difference= cv2.absdiff(gray1, gray2)

        return float(np.mean(difference))

    def check_quality(self, dt):

        if self.current_frame is None:
            return

        frame = self.current_frame.copy()

        # OCR is already running
        if self.state == "PROCESSING":
            return

        # wait until the scene changes
        if self.state == "WAITING":

            difference = self.scene_difference(
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
        self.state = "PROCESSING"

        self.quality_label.text = "Processing image..."

        print("Image captured. Starting pipeline.")

        future = self.executor.submit(
            self.pipeline.process,
            self.captured_frame
        )

        future.add_done_callback(self.processing_finished)

    def processing_finished(self, future):

        try:
            text = future.result()
            error = None

        except Exception as e:
            text = ""
            error = str(e)

        if self.shutting_down:
            return

        Clock.schedule_once(
            lambda dt: self.display_result(text,error)
        )

    def display_result(self, text, error=None):

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

    def stop_camera(self):

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