# MetroPaper/utils/speaker.py

import time
import threading
import queue
import pyttsx3
from MetroPaper.config import GLOBAL_SPEECH_COOLDOWN_SEC


class Speaker:
    """
    Threaded TTS speaker using pyttsx3.
    - Non-blocking for the main CV loop (speech runs in its own thread).
    - Respects a global cooldown (GLOBAL_SPEECH_COOLDOWN_SEC).
    """

    def __init__(self):
        self._q = queue.Queue()
        self._last_speak_time = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str, force: bool = False):
        """
        Enqueue a sentence to speak.
        - force=True: ignore global cooldown (for very important alerts).
        """
        if not text:
            return

        now = time.time()
        if (not force) and (now - self._last_speak_time < GLOBAL_SPEECH_COOLDOWN_SEC):
            # Too soon since last message -> drop to avoid overlap / spam
            return

        self._last_speak_time = now
        self._q.put(text)

    def _run(self):
        """
        Runs in a background thread:
        - Creates its own pyttsx3 engine (must live in this thread).
        - Loops on the queue and speaks each text.
        """

        engine = pyttsx3.init()

        # ---------- OPTIONAL: TUNE TTS PARAMETERS ----------
        # <<< TUNE: speaking rate (words per minute) >>>
        engine.setProperty("rate", 170)

        # <<< TUNE: volume 0.0 -> 1.0 >>>
        engine.setProperty("volume", 1.0)

        # <<< OPTIONAL: choose a specific voice >>>
        # voices = engine.getProperty("voices")
        # for v in voices:
        #     print(v.id)  # run once to inspect
        # engine.setProperty("voice", voices[0].id)
        # ---------------------------------------------------

        while True:
            text = self._q.get()
            if text is None:
                break

            print(f"[SPEAK] {text}")  # keep console log for debugging
            engine.say(text)
            engine.runAndWait()

        engine.stop()

    def stop(self):
        """
        Stop the speaker thread gracefully.
        Call this when your program ends.
        """
        self._q.put(None)
        self._thread.join()
