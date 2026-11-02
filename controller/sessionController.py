"""Application session lifecycle and user-action policy."""
from kivy.clock import Clock

from controller.snapPrintController import SnapPrintController
from ocr.model_service import ModelService


class SessionController:
    def __init__(self, main_screen, request_upload):
        self.main_screen = main_screen
        self.request_upload = request_upload
        self.capture = None
        self.session_active = False
        self.completed_session = None
        self.models = ModelService()
        self.ready = False
        self.closed = False
        self.loading = False
        self.load_error = None

    def initialize(self):
        if self.loading or self.ready or self.closed:
            return
        self.loading = True
        self.load_error = None
        self.main_screen.showLoading("Preparing local recognition models...")
        future = self.models.worker.submit(self.models.load, self.loadingProgress)
        future.add_done_callback(lambda result: Clock.schedule_once(lambda dt: self.modelsLoaded(result)))

    def loadingProgress(self, message):
        Clock.schedule_once(lambda dt: self.main_screen.showLoading(message) if not self.closed else None)

    def modelsLoaded(self, future):
        if self.closed:
            return
        self.loading = False
        try:
            future.result()
        except Exception as error:
            self.load_error = str(error)
            self.main_screen.showLoadingError(self.load_error)
            return
        self.ready = True
        self.main_screen.start_button.disabled = False
        self.main_screen.start_button.text = "Start session (S)"
        self.main_screen.showHome("Ready. Start a session to collect pages for a document.")

    def startSession(self, *_):
        if self.closed or self.session_active:
            return
        if not self.ready:
            if self.load_error:
                self.initialize()
            return
        self.session_active = True
        self.main_screen.showModes()

    def chooseMode(self, mode):
        if not self.ready or self.closed or mode not in ("live", "photos") or not self.session_active or self.capture:
            return
        capture = SnapPrintController(self.main_screen, self.models)
        self.capture = capture
        self.main_screen.showCapture(mode)
        capture.setBusy(False)
        self.main_screen.pages_button.text = "Single page (2 to toggle)"
        try:
            capture.start(mode)
        except Exception as error:
            capture.stop()
            self.capture = None
            self.main_screen.showModes()
            self.main_screen.status.text = str(error) + ". Choose a mode to retry."

    def perform(self, action):
        if self.capture:
            try:
                getattr(self.capture, action)()
            except Exception as error:
                self.main_screen.status.text = str(error)

    def endSession(self, *_):
        pages = []
        if self.capture:
            capture = self.capture
            self.capture = None
            capture.stop()
            pages = list(capture.pages)
        self.session_active = False
        # Integration boundary for the future local document formatter.
        self.completed_session = {"pages": pages, "text": "\n\n".join(pages),
                                  "formatting_status": "not_implemented"}
        self.main_screen.showHome(
            f"Session ended. {len(pages)} page(s) collected.\n"
            "Document creation and formatting are not yet available.")
        self.main_screen.displayOcrOutput(self.completed_session["text"])

    def canUpload(self):
        return bool(self.capture and self.capture.mode == "photos"
                    and not self.capture.processing.is_set())

    def uploadImage(self, owner, path):
        # A dialog belonging to an old session must never upload to a new one.
        if owner is self.capture and self.canUpload():
            try:
                self.capture.uploadImage(path)
            except Exception as error:
                self.main_screen.status.text = str(error)

    def togglePages(self, *_):
        if self.capture:
            self.capture.togglePages()

    def stop(self):
        if self.closed:
            return
        self.closed = True
        if self.capture:
            self.capture.stop()
            self.capture = None
        self.models.close()

    def handleKey(self, key, character):
        if character == "s" and not self.session_active:
            self.startSession()
        elif character in ("l", "p") and self.session_active and not self.capture:
            self.chooseMode("live" if character == "l" else "photos")
        elif self.capture:
            if key == 32:
                self.capture.stopSpeech()
            elif character == "f" and self.capture.mode == "live":
                self.perform("forceCapture")
            elif character == "c":
                self.perform("captureNext")
            elif character == "u" and self.canUpload():
                self.request_upload()
            elif character == "2":
                self.togglePages()
            else:
                return False
        else:
            return False
        return True
