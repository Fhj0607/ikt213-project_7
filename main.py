from kivy.app import App

from controller.snapPrintController import SnapPrintController
from ui.mainScreen import MainScreen



class SnapPrintApp(App):

    def build(self):
        self.main_screen = MainScreen()

        self.controller = SnapPrintController(
            self.main_screen
        )

        return self.main_screen

    def on_start(self):
        self.controller.start()

    def on_stop(self):
        self.controller.stop()