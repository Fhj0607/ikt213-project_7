"""Nonblocking Windows speech and short camera/OCR announcements."""

import logging
import queue
import threading
import time


class Speech:
    def __init__(self, cooldown=8.0):
        self.cooldown = cooldown
        self._last_guidance = float('-inf')
        self._last_result = ''
        self.error = None
        self._commands = queue.Queue()
        self._closed = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def speak(self, text, interrupt=False):
        if text and not self._closed.is_set():
            self._commands.put(('speak', text, interrupt))

    def announceQuality(self, assessment):
        now = time.monotonic()
        if now - self._last_guidance < self.cooldown:
            return
        issues = assessment['issues']
        messages = [message for key, message in (
            ('too_dark', 'Too dark. More light needed.'),
            ('too_bright', 'Too bright.'),
            ('blurry', 'Too blurry.'),
            ('low_contrast', 'Low contrast.'),
        ) if issues.get(key)]
        if not assessment['stable']:
            messages.append('Hold steady.')
        if messages:
            self._last_guidance = now
            self.speak(' '.join(messages))

    def announceResult(self, text, confidence):
        if not text.strip():
            message = 'No readable text found. Please scan again.'
        elif confidence is None:
            message = 'Text detected. OCR confidence is unavailable.'
        else:
            message = f'Average OCR confidence: {confidence * 100:.0f} percent.'
        self._last_result = message
        self.speak(message, interrupt=True)

    def repeat(self):
        self.speak(self._last_result or 'No text to read. Press C to scan again.', interrupt=True)

    def rememberResult(self, text):
        """Store document text for repeat without automatically reading it."""
        self._last_result = text

    def stop(self):
        if not self._closed.is_set():
            self._commands.put(('stop', '', True))

    def close(self):
        self._closed.set()
        self._worker.join(timeout=2)

    def _run(self):
        # COM objects must be created and used on their owning thread.
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        try:
            voice = win32com.client.Dispatch('SAPI.SpVoice')
            while not self._closed.is_set():
                try:
                    action, text, interrupt = self._commands.get(timeout=0.02)
                except queue.Empty:
                    pythoncom.PumpWaitingMessages()
                    continue
                if action == 'stop':
                    voice.Speak('', 3)  # asynchronous + purge queued speech
                elif interrupt or voice.Status.RunningState != 2:
                    # Drop guidance while speaking; never build a stale backlog.
                    # OCR may contain angle brackets; always speak literal text.
                    voice.Speak(text, 19 if interrupt else 17)
                pythoncom.PumpWaitingMessages()
            voice.Speak('', 3)
        except Exception as error:
            self.error = str(error)
            logging.exception('Speech output failed')
        finally:
            # Release the COM voice before its apartment is uninitialized.
            voice = None
            pythoncom.CoUninitialize()


class SilentSpeech:
    """Speech interface for visual capture modes, without an audio worker."""
    error = None

    def speak(self, text, interrupt=False):
        pass

    def announceQuality(self, assessment):
        pass

    def rememberResult(self, text):
        pass

    def stop(self):
        pass

    def close(self):
        pass
