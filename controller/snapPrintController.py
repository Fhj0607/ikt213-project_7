import threading

from kivy.clock import Clock

from capture.camera import Camera
from ocr.paddleOcr import extractText
from processing.processingPipeline import processImage
from processing.qualityAssessment import (
    QualityAssessment,
    STABILITY_THRESHOLD
)



class SnapPrintController:

    SCANNING = "SCANNING"
    WAITING_FOR_CHANGE = "WAITING_FOR_CHANGE"

    def __init__(self, main_screen):
        self.main_screen = main_screen

        self.camera = Camera()
        self.quality_assessment = QualityAssessment()

        self.running = False
        self.state = self.SCANNING

        self.change_count = 0



    def start(self):

        if self.running:
            return

        self.camera.start()
        self.running = True

        capture_thread = threading.Thread(
            target=self.captureLoop,
            daemon=True
        )

        capture_thread.start()



    def stop(self):
        self.running = False
        self.camera.stop()



    def captureLoop(self):

        while self.running:

            frame = self.camera.captureFrame()

            self.sendFrameToUi(frame)

            assessment = self.quality_assessment.assessFrame(frame)

            if self.state == self.SCANNING:
                self.handleScanning(frame, assessment)

            elif self.state == self.WAITING_FOR_CHANGE:
                self.handleWaitingForChange(assessment)



    def handleScanning(self, frame, assessment):

        if not assessment["acceptable"]:
            return

        self.state = self.WAITING_FOR_CHANGE

        issues = assessment["issues"]

        processing_thread = threading.Thread(
            target=self.processFrame,
            args=(frame.copy(), issues),
            daemon=True
        )

        processing_thread.start()



    def handleWaitingForChange(self, assessment):

        difference_score = assessment["scores"]["difference_score"]

        if difference_score >= STABILITY_THRESHOLD:
            self.change_count += 1
        else:
            self.change_count = 0

        if self.change_count < 3:
            return

        self.quality_assessment.reset()

        self.change_count = 0
        self.state = self.SCANNING



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