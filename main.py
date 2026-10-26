from kivy.app import App
from kivy.core.window import Window

from controller.snapPrintController import SnapPrintController
from ui.mainScreen import MainScreen

class SnapPrintApp(App):

    def build(self):
        self.main_screen = MainScreen()

        self.controller = SnapPrintController(
            self.main_screen
        )
        self.main_screen.read_button.bind(on_release=self.controller.readResult)
        self.main_screen.scan_again_button.bind(on_release=self.controller.requestScan)

        return self.main_screen

    def on_start(self):
        Window.bind(on_key_down=self.onSpeechKey)
        self.controller.start()

    def on_stop(self):
        Window.unbind(on_key_down=self.onSpeechKey)
        self.controller.stop()

    def onSpeechKey(self, window, key, scancode, codepoint, modifiers):
        character = (codepoint or '').lower()
        if key == 32:
            self.controller.speech.stop()
            return True
        if key == 114 or character == 'r':
            self.controller.readResult()
            return True
        if key == 50 or character == '2':
            self.controller.two_pages = not self.controller.two_pages
            self.controller.speech.speak(
                'Two page mode.' if self.controller.two_pages else 'Single page mode.',
                interrupt=True,
            )
            return True
        if key == 99 or character == 'c':
            self.controller.requestScan()
            return True
        return False

if __name__ == "__main__":
    SnapPrintApp().run()
