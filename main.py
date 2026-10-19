
from kivy.app import App
from ui.main_screen import MainScreen

class SnapPrintApp(App):

    def build(self):
        self.screen = MainScreen()
        return self.screen

    def on_start(self):
        self.screen.startCamera()

    def on_stop(self):
        self.screen.stopCamera()


if __name__ == "__main__":
    SnapPrintApp().run()
