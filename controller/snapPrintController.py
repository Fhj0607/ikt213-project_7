import cv2
import numpy as np
import threading
import time

from concurrent.futures import ThreadPoolExecutor

from kivy.clock import Clock

from capture.camera import Camera
from ocr.paddleOcr import extractText
from processing.processingPipeline import processImage
from processing.qualityAssessment import QualityAssessment
from processing.processingLibrary import (toGrayscale, resizeImage)



QUALITY_CHECK_INTERVAL = 0.5

SCENE_CHANGE_THRESHOLD = 15.0
REQUIRED_CHANGED_FRAMES = 3



class SnapPrintController:

    SCANNING = "SCANNING"
    WAITING_FOR_CHANGE = "WAITING_FOR_CHANGE"

    def __init__(self, main_screen):
        self.main_screen = main_screen

        self.camera = Camera()
        self.quality_assessment = QualityAssessment()

        self.processing_worker = ThreadPoolExecutor(
            max_workers=1
        )

        self.running = False
        self.state = self.SCANNING

        self.captured_frame = None
        self.change_count = 0

        self.last_quality_check = 0.0



    def start(self):

        if self.running:
            return

        self.camera.start()
        self.running = True

        capture_thread = threading.Thread(target=self.captureLoop, daemon=True)

        capture_thread.start()



    def stop(self):

        self.running = False

        self.camera.stop()

        self.processing_worker.shutdown(wait=False, cancel_futures=True)



    def captureLoop(self):

        while self.running:

            frame = self.camera.captureFrame()

            self.sendFrameToUi(frame)

            current_time = time.monotonic()

            if current_time - self.last_quality_check < QUALITY_CHECK_INTERVAL:
                time.sleep(0.05)
                continue

            self.last_quality_check = current_time

            if self.state == self.SCANNING:
                self.handleScanning(frame)

            elif self.state == self.WAITING_FOR_CHANGE:
                self.handleWaitingForChange(frame)

            time.sleep(0.05)



    def handleScanning(self, frame):

        assessment = self.quality_assessment.assessFrame(frame)

        if not assessment["acceptable"]:
            return

        self.captured_frame = frame.copy()

        self.state = self.WAITING_FOR_CHANGE

        issues = assessment["issues"]

        self.processing_worker.submit(
            self.processFrame,
            self.captured_frame.copy(),
            issues
        )



    def handleWaitingForChange(self, frame):

        difference_score = self.sceneDifference(frame, self.captured_frame)

        if difference_score >= SCENE_CHANGE_THRESHOLD:
            self.change_count += 1

        else:
            self.change_count = 0

        if self.change_count < REQUIRED_CHANGED_FRAMES:
            return

        self.quality_assessment.reset()

        self.change_count = 0
        self.captured_frame = None

        self.state = self.SCANNING



    def sceneDifference(self, frame, captured_frame):

        gray_frame = toGrayscale(frame)

        gray_captured = toGrayscale(captured_frame)

        gray_frame = resizeImage(gray_frame, 320, 240)

        gray_captured = resizeImage(gray_captured, 320, 240)

        difference = cv2.absdiff(gray_frame, gray_captured)

        return float(np.mean(difference))



    def processFrame(self, frame, issues):

        try:

            processed_image, report = processImage(frame, issues)

            text = extractText(processed_image)

            Clock.schedule_once(
                lambda dt, output=text:
                self.main_screen
                .displayOcrOutput(output)
            )

        except Exception as error:

            error_message = str(error)

            Clock.schedule_once(
                lambda dt, message=error_message:
                self.main_screen
                .displayOcrOutput(
                    "Processing failed:\n"
                    + message
                )
            )



    def sendFrameToUi(self, frame):

        display_frame = frame.copy()

        Clock.schedule_once(
            lambda dt, image=display_frame:
            self.main_screen
            .displayFrame(image)
        )