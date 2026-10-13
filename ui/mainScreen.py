import cv2
import threading

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from capture.camera import Camera
from worker.imageWorker import Process



class MainScreen(BoxLayout):

    LIVE = "LIVE"
    FROZEN = "FROZEN"

    def __init__(self, **kwargs):

        super().__init__(orientation="vertical", **kwargs)

        self.camera = Camera()
        self.camera.Start()

        self.state = self.LIVE

        self.imageDisplay = Image(size_hint_y=0.65)

        self.captureButton = Button(text="Capture", size_hint_y=0.1)

        self.captureButton.bind(on_press=self.OnCapturePressed)

        textScroll = ScrollView(size_hint_y=0.25)

        self.ocrText = Label(
            text="Point the camera at some text",
            size_hint_y=None,
            halign="left",
            valign="top"
        )

        self.ocrText.bind(
            width=lambda instance, width:
            setattr(
                instance,
                "text_size",
                (width, None)
            )
        )

        self.ocrText.bind(
            texture_size=lambda instance, size:
            setattr(
                instance,
                "height",
                size[1]
            )
        )

        textScroll.add_widget(self.ocrText)

        self.add_widget(self.imageDisplay)

        self.add_widget(self.captureButton)

        self.add_widget(textScroll)

        Clock.schedule_interval(self.UpdatePreview, 1 / 30)



    def UpdatePreview(self, dt):

        frame = self.camera.GetLatestFrame()

        if frame is None:
            return

        if self.state == self.LIVE:
            self.DisplayFrame(frame)



    def DisplayFrame(self, frame):

        displayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        displayFrame = cv2.flip(displayFrame, 0)

        height, width, channels = displayFrame.shape

        texture = Texture.create(size=(width, height), colorfmt="rgb")

        texture.blit_buffer(
            displayFrame.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        self.imageDisplay.texture = texture



    def OnCapturePressed(self, instance):

        if self.state == self.FROZEN:

            self.state = self.LIVE

            self.captureButton.text = "Capture"

            self.ocrText.text = ("Point the camera at some text")

            return

        if self.state != self.LIVE:
            return

        frame = self.camera.GetLatestFrame()

        if frame is None:
            self.ocrText.text = "No camera frame available yet"
            return

        self.state = self.FROZEN

        self.captureButton.disabled = True

        self.DisplayFrame(frame)
        self.ocrText.text = "Processing..."

        self.StartProcessing(frame)



    def StartProcessing(self, frame):

        processingThread = threading.Thread(
            target=self.ProcessInBackground,
            args=(frame,),
            daemon=True
        )

        processingThread.start()



    def ProcessInBackground(self, frame):

        try:

            processedImage, report, text = Process(frame)

            Clock.schedule_once(
                lambda dt:
                self.OnProcessingFinished(
                    report,
                    text
                )
            )

        except Exception as error:
            errorMessage = str(error)

            Clock.schedule_once(
                lambda dt:
                self.OnProcessingFailed(
                    errorMessage
                )
            )



    def OnProcessingFinished(self, report, text):

        reportText = "\n".join(report)

        if reportText:

            self.ocrText.text = (reportText + "\n\nOCR RESULT:\n\n" + text)

        else:

            self.ocrText.text = ("OCR RESULT:\n\n" + text)

        self.captureButton.disabled = False
        self.captureButton.text = "Scan Again"



    def OnProcessingFailed(self, error):

        self.ocrText.text = ("Processing failed:\n" + error)

        self.captureButton.disabled = False
        self.captureButton.text = "Scan Again"



    def StopCamera(self):
        self.camera.Stop()