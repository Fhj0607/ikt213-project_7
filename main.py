from kivy.app import App
from kivy.uix.label import Label


class SnapPrintApp(App):
    def build(self):
        return Label(text="SnapPrint is running!")


if __name__ == "__main__":
    SnapPrintApp().run()
    