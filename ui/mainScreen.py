import cv2

from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView



class MainScreen(BoxLayout):

    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", **kwargs)

        self.image_display = Image(size_hint_y=0.7)

        text_scroll = ScrollView(size_hint_y=0.3)

        self.ocr_text = Label(
            text="Waiting for text...",
            size_hint_y=None,
            halign="left",
            valign="top"
        )

        self.ocr_text.bind(
            width=lambda instance, width:
            setattr(
                instance,
                "text_size",
                (width, None)
            )
        )

        self.ocr_text.bind(
            texture_size=lambda instance, size:
            setattr(
                instance,
                "height",
                size[1]
            )
        )

        text_scroll.add_widget(self.ocr_text)

        self.add_widget(self.image_display)

        self.add_widget(text_scroll)



    def displayFrame(self, frame):

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        display_frame = cv2.flip(display_frame, 0)

        height, width, channels = (display_frame.shape)

        texture = Texture.create(size=(width, height), colorfmt="rgb")

        texture.blit_buffer(
            display_frame.tobytes(),
            colorfmt="rgb",
            bufferfmt="ubyte"
        )

        self.image_display.texture = texture



    def displayOcrOutput(self, text):
        self.ocr_text.text = text