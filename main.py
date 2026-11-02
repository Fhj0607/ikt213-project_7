from kivy.app import App
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup

from controller.sessionController import SessionController
from ui.mainScreen import MainScreen


class SnapPrintApp(App):
    def build(self):
        self.upload_popup = None
        self.main_screen = MainScreen()
        self.controller = SessionController(self.main_screen, self.chooseUpload)
        for button, callback in (
            (self.main_screen.start_button, self.controller.startSession),
            (self.main_screen.live_button, lambda *_: self.controller.chooseMode("live")),
            (self.main_screen.photos_button, lambda *_: self.controller.chooseMode("photos")),
            (self.main_screen.end_button, self.endSession),
            (self.main_screen.capture_button, lambda *_: self.controller.perform("takePhoto")),
            (self.main_screen.scan_again_button, lambda *_: self.controller.perform("requestScan")),
            (self.main_screen.upload_button, self.chooseUpload),
            (self.main_screen.force_capture_button, lambda *_: self.controller.perform("forceCapture")),
            (self.main_screen.pages_button, self.controller.togglePages),
        ):
            button.bind(on_release=callback)
        return self.main_screen

    def on_start(self):
        Window.bind(on_key_down=self.onSpeechKey)
        self.controller.initialize()

    def endSession(self, *_):
        if self.upload_popup:
            self.upload_popup.dismiss()
        self.controller.endSession()

    def chooseUpload(self, *_):
        if not self.controller.canUpload():
            return
        owner = self.controller.capture
        layout = BoxLayout(orientation="vertical", spacing=8)
        chooser = FileChooserListView(filters=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"])
        layout.add_widget(chooser)
        buttons = BoxLayout(size_hint_y=None, height=52)
        upload = Button(text="Upload selected image")
        cancel = Button(text="Cancel")
        buttons.add_widget(upload)
        buttons.add_widget(cancel)
        layout.add_widget(buttons)
        popup = Popup(title="Choose an image for OCR", content=layout, size_hint=(0.9, 0.9))
        self.upload_popup = popup
        def selected(*_):
            if chooser.selection:
                popup.dismiss()
                self.controller.uploadImage(owner, chooser.selection[0])
        upload.bind(on_release=selected)
        cancel.bind(on_release=popup.dismiss)
        popup.bind(on_dismiss=lambda *_: setattr(self, "upload_popup", None))
        popup.open()

    def on_stop(self):
        Window.unbind(on_key_down=self.onSpeechKey)
        self.controller.stop()

    def onSpeechKey(self, window, key, scancode, codepoint, modifiers):
        if self.main_screen.ocr_text.focus:
            return False
        character = (codepoint or chr(key) if 0 <= key < 128 else "").lower()
        if key == 27 and self.controller.session_active:
            self.endSession()
            return True
        if self.upload_popup:
            return False
        return self.controller.handleKey(key, character)

if __name__ == "__main__":
    SnapPrintApp().run()
