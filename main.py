from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup

from controller.sessionController import SessionController
from ui.mainScreen import MainScreen


class SnapPrintApp(App):
    def build(self):
        self.upload_popup = None
        self.gate_popup = None
        self.main_screen = MainScreen()
        self.controller = SessionController(self.main_screen, self.chooseUpload)
        for button, callback in (
            (self.main_screen.start_button, self.controller.startSession),
            (self.main_screen.create_document_button, self.controller.createDocument),
            (self.main_screen.live_button, lambda *_: self.controller.chooseMode("live")),
            (self.main_screen.photos_button, lambda *_: self.controller.chooseMode("photos")),
            (self.main_screen.gate_button, self.chooseGate),
            (self.main_screen.enhance_button, self.controller.toggleEnhancement),
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
        if self.gate_popup:
            self.gate_popup.dismiss()
        if self.upload_popup:
            self.upload_popup.dismiss()
        self.controller.endSession()

    def chooseGate(self, *_):
        if (self.gate_popup or not self.controller.session_active
                or self.controller.capture or self.controller.closed):
            return
        layout = BoxLayout(orientation="vertical", spacing=dp(10), padding=dp(12))
        layout.add_widget(self.main_screen.label(
            "Choose one live text gate. Photos and uploads bypass this step.",
            font_size=15, size_hint_y=None, height=dp(48)))
        popup = Popup(title="Choose a text-detection method", content=layout,
                      size_hint=(.85, None), height=dp(400))
        for gate_type, name in (("detector", "OCR detector"), ("cnn", "CNN"), ("hog", "HOG + SVM")):
            selected = self.controller.gate_type == gate_type
            button = self.main_screen.button(
                ("Selected: " if selected else "") + name, emphasis=selected)
            def choose(*_, gate_type=gate_type):
                self.controller.selectGate(gate_type)
                popup.dismiss()
            button.bind(on_release=choose)
            layout.add_widget(button)
        cancel = self.main_screen.button("Cancel")
        cancel.bind(on_release=popup.dismiss)
        layout.add_widget(cancel)
        popup.bind(on_dismiss=lambda *_: setattr(self, "gate_popup", None))
        self.gate_popup = popup
        popup.open()

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
        if self.gate_popup:
            self.gate_popup.dismiss()
        Window.unbind(on_key_down=self.onSpeechKey)
        self.controller.stop()

    def onSpeechKey(self, window, key, scancode, codepoint, modifiers):
        if self.main_screen.ocr_text.focus:
            return False
        character = (codepoint or chr(key) if 0 <= key < 128 else "").lower()
        if key == 27 and self.controller.session_active:
            self.endSession()
            return True
        if self.upload_popup or self.gate_popup:
            return False
        return self.controller.handleKey(key, character)

if __name__ == "__main__":
    SnapPrintApp().run()
