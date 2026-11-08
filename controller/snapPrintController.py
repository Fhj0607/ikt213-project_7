import threading
import time

import cv2

from concurrent.futures import ThreadPoolExecutor

from kivy.clock import Clock

from capture.camera import Camera
from ocr.vl_reader import VLReader
from ocr.text_gate import TextGate
from output.speech import SilentSpeech, Speech
from processing.qualityAssessment import QualityAssessment, STABILITY_THRESHOLD



QUALITY_CHECK_INTERVAL = 0.5




class SnapPrintController:

    SCANNING = "SCANNING"
    PAUSED = "PAUSED"

    def __init__(self, main_screen, models=None, gate_type="detector"):
        self.main_screen = main_screen

        self.camera = Camera()
        self.quality_assessment = QualityAssessment()
        self.speech = SilentSpeech()
        self.owns_models = models is None
        self.reader = VLReader() if models is None else models.reader
        if models is None:
            if gate_type == "cnn":
                from cnn.textClassifier import CNNTextGate
                self.text_gate = CNNTextGate()
            elif gate_type == "hog":
                from hog.hogTextGate import HogTextGate
                self.text_gate = HogTextGate()
            else:
                self.text_gate = TextGate()
        else:
            self.text_gate = models.getGate(gate_type)
        self.gate_type = gate_type
        self.gate_future = None
        self.gate_hits = 0
        self.gate_generation = 0
        self.gate_failed = False
        self.gate_pending = threading.Event()
        self.latest_assessment = None
        self.processing = threading.Event()
        self.two_pages = False
        self.rescan_requested = threading.Event()
        self.processing_future = None

        self.processing_worker = (ThreadPoolExecutor(max_workers=1) if models is None else models.worker)

        self.mode = "live"
        self.pages = []
        self.latest_frame = None
        self.capture_thread = None
        self.running = False
        self.state = self.SCANNING

        self.captured_frame = None

        self.last_quality_check = 0.0



    def start(self, mode="live"):

        if self.running:
            return

        self.mode = mode
        self.running = True
        if mode == "photos":
            return
        self.speech = Speech()
        self.startCamera()

    def startCamera(self):
        try:
            self.camera.start()
        except Exception:
            self.camera.stop()
            raise

        capture_thread = threading.Thread(target=self.captureLoop, daemon=True)

        self.capture_thread = capture_thread
        capture_thread.start()
        self.speech.speak('Ready. Hold the page steady. Press space to stop spoken guidance.')



    def stop(self):

        self.running = False

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2)
        self.camera.stop()

        if self.owns_models:
            self.processing_worker.submit(self.closeModels)
            self.processing_worker.shutdown(wait=False)
        else:
            for future in (self.gate_future, self.processing_future):
                if future is not None:
                    future.cancel()
        self.speech.close()



    def closeModels(self):
        try:
            self.reader.close()
        finally:
            self.text_gate.close()

    def captureLoop(self):

        while self.running:

            try:
                frame = self.camera.captureFrame()
            except Exception as error:
                self.scheduleUi(lambda message=str(error): setattr(self.main_screen.status, "text", message))
                return
            self.latest_frame = frame.copy()

            self.sendFrameToUi(frame)

            if self.mode == "photos":
                time.sleep(0.05)
                continue

            if self.processing.is_set():
                time.sleep(0.05)
                continue

            if self.rescan_requested.is_set():
                self.rescan_requested.clear()
                self.quality_assessment.reset()
                self.resetGate()
                self.state = self.SCANNING

            if self.state == self.PAUSED:
                time.sleep(0.05)
                continue

            current_time = time.monotonic()

            if current_time - self.last_quality_check >= QUALITY_CHECK_INTERVAL:
                self.last_quality_check = current_time

                self.handleScanning(frame)

            time.sleep(0.05)



    def handleScanning(self, frame):

        assessment = self.quality_assessment.assessFrame(frame)
        self.latest_assessment = assessment
        if self.gate_failed:
            return
        # An inference may take several camera checks. Do not discard its result
        # for every transient unstable sample; validate the candidate on return.
        if self.gate_pending.is_set():
            return
        self.speech.announceQuality(assessment)
        if not assessment["acceptable"]:
            self.gate_hits = 0
            self.scheduleUi(lambda: setattr(self.main_screen.status, "text", "Hold your page steady. Waiting for a stable image."))
            return
        # Keep inference off both the camera thread and the Kivy UI thread.
        generation = self.gate_generation
        candidate = frame.copy()
        self.gate_pending.set()
        message = ("Loading text detector and checking your page. First use may take a moment."
                   if self.text_gate.model is None else "Checking the page for text...")
        self.scheduleUi(lambda: setattr(self.main_screen.status, "text", message))
        if self.text_gate.model is None:
            self.speech.speak("Loading text detector. Please wait.", interrupt=True)
        self.gate_future = self.processing_worker.submit(self.checkText, candidate, generation)

    def resetGate(self):
        self.gate_generation += 1
        self.gate_hits = 0

    def checkText(self, frame, generation):
        if not self.running:
            return
        try:
            found = self.text_gate.detect(frame)
        except Exception as error:
            self.scheduleUi(lambda message=str(error): self.gateError(message))
            return
        self.scheduleUi(lambda: self.acceptDetection(frame, generation, found))

    def gateError(self, message):
        self.gate_pending.clear()
        self.gate_failed = True
        self.resetGate()
        self.main_screen.status.text = "Text detection unavailable. Use Capture now (F) or restart the session. " + message
        self.speech.speak("Text detection unavailable. Press F to capture manually.", interrupt=True)

    def acceptDetection(self, frame, generation, found):
        self.gate_pending.clear()
        if (not self.running or generation != self.gate_generation
                or self.state != self.SCANNING or self.processing.is_set()):
            return
        if self.latest_frame is not None:
            # Validate the image actually examined, rather than requiring a
            # perfectly uninterrupted stable run throughout slow CPU inference.
            candidate = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240))
            current = cv2.resize(cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2GRAY), (320, 240))
            moved = float(cv2.absdiff(candidate, current).mean()) >= STABILITY_THRESHOLD
            stable = self.latest_assessment is None or self.latest_assessment["acceptable"]
            if moved or not stable:
                self.gate_hits = 0
                self.main_screen.status.text = "Page moved while checking text. Hold steady to retry."
                return
        self.gate_hits = self.gate_hits + 1 if found else 0
        if self.gate_hits < 2:
            self.main_screen.status.text = ("Text detected. Hold steady to capture." if found else
                                            "Looking for text. Hold your page steady, or press F to capture manually.")
            return
        self.state = self.PAUSED
        self.resetGate()
        self.submitFrame(frame)

    def forceCapture(self, *_):
        if not self.running or self.processing.is_set() or self.latest_frame is None:
            return
        self.state = self.PAUSED
        self.resetGate()
        self.submitFrame(self.latest_frame)

    def submitFrame(self, frame):
        if not self.running or self.processing.is_set():
            return
        issues = {}
        self.processing.set()
        self.speech.rememberResult('')
        self.speech.speak('Page captured. Reading document.', interrupt=True)
        self.scheduleUi(lambda: setattr(self.main_screen.status, "text", "Page captured. Reading document..."))
        self.scheduleUi(lambda: self.setBusy(True))

        self.processing_future = self.processing_worker.submit(
            self.processFrame,
            frame.copy(),
            issues
        )



    def processFrame(self, frame, issues):

        try:
            if not self.running:
                return

            # Preserve color and fine handwriting strokes for the VLM.
            started = time.monotonic()
            text, _ = self.reader.read(frame, self.two_pages)
            elapsed = time.monotonic() - started
            if not self.running:
                return
            text = text.strip()
            self.speech.speak(
                f'Text recognized in {elapsed:.0f} seconds.'
                if text else 'No readable text found. Please scan again.',
                interrupt=True,
            )

            self.scheduleUi(lambda: self.finishRecognition(text))

        except Exception as error:

            error_message = str(error)
            if not self.running:
                return
            self.speech.speak('Processing failed. Please scan again.', interrupt=True)

            self.scheduleUi(lambda message=error_message: setattr(
                self.main_screen.status, "text", "Processing failed: " + message))
        finally:
            self.processing.clear()
            if self.running:
                self.scheduleUi(lambda: self.setBusy(False))

    def finishRecognition(self, text):
        if text:
            self.pages.append(text)
            self.main_screen.appendOcrText(text)
            self.main_screen.status.text = "Page added. Click the notes preview to edit, or capture another page."
        else:
            self.main_screen.status.text = "No readable text found. Try another photo or scan again."

    def togglePages(self, *_):
        if not self.running or self.processing.is_set():
            return
        self.two_pages = not self.two_pages
        label = "Two pages" if self.two_pages else "Single page"
        self.main_screen.pages_button.text = label + " (2 to toggle)"
        self.speech.speak(label, interrupt=True)

    def stopSpeech(self):
        self.speech.stop()

    def captureNext(self):
        if self.mode == "live":
            self.requestScan()
        else:
            self.takePhoto()

    def requestScan(self, *_):
        if not self.running:
            return
        if self.processing.is_set():
            self.speech.speak('Still reading. Please wait.', interrupt=True)
            return
        self.rescan_requested.set()
        self.speech.speak('Ready to scan again.', interrupt=True)

    def scheduleUi(self, callback):
        Clock.schedule_once(lambda dt: callback() if self.running else None)

    def setBusy(self, busy):
        for button in (self.main_screen.scan_again_button, self.main_screen.capture_button,
                       self.main_screen.upload_button, self.main_screen.pages_button,
                       self.main_screen.force_capture_button):
            button.disabled = busy

    def takePhoto(self, *_):
        if self.processing.is_set():
            return
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.startCamera()
            self.main_screen.status.text = "Camera ready. Position your page, then press Take photo again."
        elif self.latest_frame is not None:
            self.submitFrame(self.latest_frame)

    def uploadImage(self, path):
        if self.processing.is_set():
            return
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError("Could not open this image. Choose a PNG or JPEG image.")
        self.sendFrameToUi(frame)
        self.submitFrame(frame)

    def sendFrameToUi(self, frame):
        display_frame = frame.copy()
        self.scheduleUi(lambda: self.main_screen.displayFrame(display_frame))
