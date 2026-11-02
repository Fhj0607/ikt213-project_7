import queue
import threading
import pyttsx3



class TextToSpeech:

    def __init__(self, rate=160):
        self.rate = rate
        self.queue = queue.Queue()
        self.thread = None



    def start(self):
        self.thread = threading.Thread(target=self.speechLoop, daemon=True)
        self.thread.start()



    def speak(self, text):
        # Only the newest text matters, drop anything still waiting
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

        self.queue.put(text)



    def stop(self):
        self.queue.put(None)



    def speechLoop(self):
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)

        while True:
            text = self.queue.get()

            if text is None:
                break

            engine.say(text)
            engine.runAndWait()