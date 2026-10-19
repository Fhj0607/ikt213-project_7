import cv2

from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button

from controller.scan_controller import ScanController


class MainScreen(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.preview = Image()
        self.add_widget(self.preview)

        self.quality_label = Label(
            text="Waiting for camera...",
            size_hint_y=None,
            height=100
        )

        self.add_widget(self.quality_label)

        self.continue_button = Button(
            text="Run OCR",
            size_hint_y=None,
            height=48,
            disabled=True
        )

        self.continue_button.bind(on_release=self.runOCR)
        self.add_widget(self.continue_button)

        self.controller = ScanController(
            on_frame=self.showFrame,
            on_status=self.updateStatus,
            on_capture=self.enableOCR,
            on_processing=self.disableOCR
        )

    def startCamera(self):
        self.controller.start()

    def stopCamera(self):
        self.controller.stop()

    def runOCR(self, *_):
        self.continue_button.disabled = True
        self.controller.runOCR()

    def updateStatus(self, text):
        self.quality_label.text = text

    def enableOCR(self):
        self.continue_button.disabled = False

    def disableOCR(self):
        self.continue_button.disabled = True

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

        self.preview.texture= texture
