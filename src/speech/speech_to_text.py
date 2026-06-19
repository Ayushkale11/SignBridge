"""
============================================
SignBridge - Speech-to-Text Module
============================================
Converts microphone audio to text using the
SpeechRecognition library with Google API.

HOW IT WORKS:
  1. Listens to the microphone
  2. Detects when you start/stop speaking
  3. Sends audio to Google's speech API
  4. Returns the recognized text

NOTE: Requires internet for Google API.
For offline use, you can switch to Vosk
(see comments in the code).
============================================
"""

import threading
import speech_recognition as sr


class SpeechToText:
    """
    Background speech-to-text listener.
    
    Usage:
        stt = SpeechToText()
        stt.start_listening(callback=my_function)
        # my_function("hello world") is called when speech is detected
        stt.stop_listening()
    """

    def __init__(self):
        # Create a speech recognizer instance
        self.recognizer = sr.Recognizer()

        # Adjust for ambient noise sensitivity
        # energy_threshold: how loud speech needs to be to be detected
        # Lower = more sensitive (picks up quieter speech)
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

        # Background listener control
        self._stop_listening = None  # Function to stop the listener
        self._is_listening = False

    def start_listening(self, callback):
        """
        Start listening to the microphone in the background.
        
        When speech is detected, calls callback(text) with
        the recognized text as a lowercase string.
        
        Args:
            callback: function that takes one string argument
                      e.g., def on_speech(text): print(text)
        """
        if self._is_listening:
            print("  Already listening!")
            return

        try:
            # Get the default microphone
            mic = sr.Microphone()

            # Adjust for ambient noise (takes ~1 second)
            print("  Adjusting for ambient noise...")
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("  Ready! Start speaking...")

            # Define what happens when speech is detected
            def on_audio(recognizer, audio):
                """Called automatically when speech is detected."""
                try:
                    # Send audio to Google's speech API
                    text = recognizer.recognize_google(audio)
                    # Call the callback with lowercase text
                    callback(text.lower())
                except sr.UnknownValueError:
                    # Google couldn't understand the audio
                    # This is normal - happens with background noise
                    pass
                except sr.RequestError as e:
                    # Google API error (usually no internet)
                    print(f"  Speech API Error: {e}")
                    print("  Check your internet connection.")

            # Start background listening
            # This runs in a separate thread automatically
            self._stop_listening = self.recognizer.listen_in_background(
                mic, on_audio
            )
            self._is_listening = True

        except OSError as e:
            print(f"  Microphone Error: {e}")
            print("  Make sure a microphone is connected.")
            print("  On Windows, you may need to install PyAudio:")
            print("    pip install PyAudio")

    def stop_listening(self):
        """Stop the background listener."""
        if self._stop_listening is not None:
            self._stop_listening(wait_for_stop=False)
            self._stop_listening = None
            self._is_listening = False
            print("  Stopped listening.")

    def listen_once(self):
        """
        Listen for a single phrase (blocking).
        
        Returns:
            text: recognized text as lowercase string
                  or None if nothing understood
        """
        try:
            mic = sr.Microphone()
            with mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("  Listening... (speak now)")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

            text = self.recognizer.recognize_google(audio)
            return text.lower()

        except sr.WaitTimeoutError:
            print("  No speech detected (timeout).")
            return None
        except sr.UnknownValueError:
            print("  Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"  API Error: {e}")
            return None
        except OSError as e:
            print(f"  Microphone Error: {e}")
            return None

    @property
    def is_listening(self):
        """Check if currently listening."""
        return self._is_listening
