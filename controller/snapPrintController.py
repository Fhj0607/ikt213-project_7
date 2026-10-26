import threading
import time

from concurrent.futures import ThreadPoolExecutor

from kivy.clock import Clock

from capture.camera import Camera
from ocr.vl_reader import VLReader
from output.speech import Speech
from processing.qualityAssessment import QualityAssessment



QUALITY_CHECK_INTERVAL = 0.5




class SnapPrintController:

    SCANNING = "SCANNING"
    PAUSED = "PAUSED"

    def __init__(self, main_screen):
        self.main_screen = main_screen

        self.camera = Camera()
        self.quality_assessment = QualityAssessment()
        self.speech = Speech()
        self.reader = VLReader()
        self.processing = threading.Event()
        self.two_pages = False
        self.rescan_requested = threading.Event()
        self.processing_future = None

        self.processing_worker = ThreadPoolExecutor(
            max_workers=1
        )

        self.running = False
        self.state = self.SCANNING

        self.captured_frame = None

        self.last_quality_check = 0.0



    def start(self):

        if self.running:
            return

        self.camera.start()
        self.running = True

        capture_thread = threading.Thread(target=self.captureLoop, daemon=True)

        capture_thread.start()
        self.speech.speak('Ready. Hold the page steady. Press R to read the result, and space to stop speech.')



    def stop(self):

        self.running = False

        self.camera.stop()

        self.processing_worker.shutdown(wait=False, cancel_futures=True)
        if self.processing_future is not None:
            self.processing_future.add_done_callback(lambda future: self.reader.close())
        else:
            self.reader.close()
        self.speech.close()



    def captureLoop(self):

        while self.running:

            frame = self.camera.captureFrame()

            self.sendFrameToUi(frame)

            if self.processing.is_set():
                time.sleep(0.05)
                continue

            if self.rescan_requested.is_set():
                self.rescan_requested.clear()
                self.quality_assessment.reset()
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
        self.speech.announceQuality(assessment)

        if not assessment["acceptable"]:
            return

        self.captured_frame = frame.copy()

        self.state = self.PAUSED

        issues = assessment["issues"]
        self.processing.set()
        self.speech.rememberResult('')
        self.speech.speak('Page captured. Reading document.', interrupt=True)
        Clock.schedule_once(lambda dt: self.main_screen.displayOcrOutput('Reading document...'))
        Clock.schedule_once(lambda dt: setattr(self.main_screen.scan_again_button, 'disabled', True))

        self.processing_future = self.processing_worker.submit(
            self.processFrame,
            self.captured_frame.copy(),
            issues
        )



    def processFrame(self, frame, issues):

        try:

            # Preserve color and fine handwriting strokes for the VLM.
            started = time.monotonic()
            text, _ = self.reader.read(frame, self.two_pages)
            elapsed = time.monotonic() - started
            if not self.running:
                return
            text = text.strip()
            self.speech.rememberResult(text)
            self.speech.speak(
                f'Text recognized in {elapsed:.0f} seconds. Press R to read it aloud.'
                if text else 'No readable text found. Please scan again.',
                interrupt=True,
            )

            Clock.schedule_once(
                lambda dt, output=text or 'No readable text found. Press C to scan again.':
                self.main_screen
                .displayOcrOutput(output)
            )

        except Exception as error:

            error_message = str(error)
            if not self.running:
                return
            self.speech.speak('Processing failed. Please scan again.', interrupt=True)

            Clock.schedule_once(
                lambda dt, message=error_message:
                self.main_screen
                .displayOcrOutput(
                    "Processing failed:\n"
                    + message
                )
            )
        finally:
            self.processing.clear()
            if self.running:
                Clock.schedule_once(lambda dt: setattr(self.main_screen.scan_again_button, 'disabled', False))

    def requestScan(self, *_):
        if self.processing.is_set():
            self.speech.speak('Still reading. Please wait.', interrupt=True)
            return
        self.rescan_requested.set()
        self.speech.speak('Ready to scan again.', interrupt=True)

    def readResult(self, *_):
        if self.processing.is_set():
            self.speech.speak('Still reading the document. Please wait.', interrupt=True)
        else:
            self.speech.repeat()



    def sendFrameToUi(self, frame):

        display_frame = frame.copy()

        Clock.schedule_once(
            lambda dt, image=display_frame:
            self.main_screen
            .displayFrame(image)
        )
