import os

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

from ocr.paddle_ocr import extract_text


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

IMAGE_FOLDER = os.path.join(BASE_DIR, "images")


class SnapPrintApp(App):

    def build(self):
        main_layout = BoxLayout(
            orientation="horizontal"
        )

        menu = BoxLayout(
            orientation="vertical",
            size_hint_x=0.25,
            size_hint_y=None
        )

        menu.bind(
            minimum_height=menu.setter("height")
        )

        for filename in os.listdir(IMAGE_FOLDER):

            if filename.lower().endswith(
                (".jpg", ".png", ".jpeg")
            ):
                button = Button(
                    text=filename,
                    size_hint_y=None,
                    height=50
                )

                button.bind(
                    on_press=lambda instance, name=filename:
                    self.select_image(name)
                )

                menu.add_widget(button)

        scroll = ScrollView()
        scroll.add_widget(menu)

        right_layout = BoxLayout(
            orientation="vertical"
        )

        self.image_display = Image(
            size_hint_y=0.6
        )

        text_scroll = ScrollView(
            size_hint_y=0.4
        )

        self.ocr_text = Label(
            text="Select an image",
            size_hint_y=None,
            text_size=(None, None),
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

        text_scroll.add_widget(
            self.ocr_text
        )

        right_layout.add_widget(
            self.image_display
        )

        right_layout.add_widget(
            text_scroll
        )

        main_layout.add_widget(scroll)
        main_layout.add_widget(right_layout)

        return main_layout


    def select_image(self, filename):

        image_path = os.path.join(
            IMAGE_FOLDER,
            filename
        )

        self.image_display.source = image_path
        self.image_display.reload()

        self.ocr_text.text = "Processing..."

        text = extract_text(image_path)

        self.ocr_text.text = text