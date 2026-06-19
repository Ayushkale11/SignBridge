"""
============================================
SignBridge - Text-to-Speech Module
============================================
Converts text to spoken audio using pyttsx3.

Key features:
  - Works 100% OFFLINE (no internet needed)
  - Runs in a separate thread (doesn't freeze GUI)
  - Has a cooldown to prevent repeating words

pyttsx3 uses the OS's built-in speech engine:
  - Windows: SAPI5
  - macOS: NSSpeechSynthesizer
  - Linux: espeak
============================================
"""

import threading
import time
import pyttsx3

from src.utils.config import TTS_RATE, TTS_VOLUME, SPEECH_COOLDOWN


class TextToSpeech:
    """
    Thread-safe text-to-speech engine.
    
    Usage:
        tts = TextToSpeech()
        tts.speak("Hello")    # Non-blocking
        tts.speak("World")    # Queues if still speaking
        tts.stop()            # Stop current speech
    """

    def __init__(self):
        # Initialize the speech engine
        self.engine = pyttsx3.init()

        # Configure speech properties
        self.engine.setProperty("rate", TTS_RATE)       # Speed (words per minute)
        self.engine.setProperty("volume", TTS_VOLUME)   # Volume (0.0 to 1.0)

        # Track what was last spoken and when
        self.last_spoken = ""
        self.last_spoken_time = 0

        # Thread lock to prevent concurrent access
        self._lock = threading.Lock()

    def speak(self, text):
        """
        Speak the given text in a background thread.
        
        Won't repeat the same text within SPEECH_COOLDOWN seconds.
        This prevents the system from saying "A A A A" rapidly.
        
        Args:
            text: string to speak (e.g., "Hello", "A")
        """
        if not text or not text.strip():
            return

        # Check cooldown - don't repeat same word too quickly
        current_time = time.time()
        if (text == self.last_spoken and
                current_time - self.last_spoken_time < SPEECH_COOLDOWN):
            return  # Skip - too soon to repeat

        # Update tracking
        self.last_spoken = text
        self.last_spoken_time = current_time

        # Run speech in a separate thread so GUI doesn't freeze
        thread = threading.Thread(
            target=self._speak_thread,
            args=(text,),
            daemon=True,  # Thread dies when main program exits
        )
        thread.start()

    def _speak_thread(self, text):
        """Internal: runs speech in background thread."""
        with self._lock:  # Only one speech at a time
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except RuntimeError:
                # Engine might be busy - create a fresh one
                try:
                    self.engine = pyttsx3.init()
                    self.engine.setProperty("rate", TTS_RATE)
                    self.engine.setProperty("volume", TTS_VOLUME)
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print(f"  TTS Error: {e}")

    def set_rate(self, rate):
        """Change speech speed (words per minute). Default: 150"""
        self.engine.setProperty("rate", rate)

    def set_volume(self, volume):
        """Change volume (0.0 to 1.0). Default: 1.0"""
        self.engine.setProperty("volume", volume)

    def stop(self):
        """Stop speaking immediately."""
        try:
            self.engine.stop()
        except Exception:
            pass
