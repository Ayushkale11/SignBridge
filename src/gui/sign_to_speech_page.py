"""
============================================
SignBridge - Sign to Speech Page
============================================
GUI page that:
  1. Shows live webcam feed with hand landmarks
  2. Detects hand gestures using the trained model
  3. Displays recognized text
  4. Speaks the text aloud

Threading strategy:
  - Webcam capture: runs in a background thread
  - GUI updates: scheduled via after() on the main thread
  - Speech: runs in its own thread (handled by TTS module)
============================================
"""

import cv2
import threading
import time
import customtkinter as ctk
from PIL import Image, ImageTk

from src.vision.hand_detector import HandDetector
from src.vision.feature_extractor import FeatureExtractor
from src.ml.predictor import Predictor
from src.speech.text_to_speech import TextToSpeech
from src.utils.config import (
    WEBCAM_DISPLAY_WIDTH,
    WEBCAM_DISPLAY_HEIGHT,
    GUI_UPDATE_INTERVAL,
)


class SignToSpeechPage(ctk.CTkFrame):
    """
    Sign Language → Text → Speech page.
    
    Uses webcam to detect signs, shows prediction
    with confidence, builds sentences, and speaks.
    """

    def __init__(self, parent, go_back):
        super().__init__(parent, fg_color="transparent")
        self.go_back = go_back  # Function to return to home

        # ── State ──
        self._is_active = False
        self._cap = None
        self._detector = None
        self._extractor = FeatureExtractor()
        self._predictor = None
        self._tts = None

        # Current frame for display (thread-safe)
        self._current_frame = None
        self._frame_lock = threading.Lock()

        # Text accumulation
        self._current_sign = ""
        self._confidence = 0.0
        self._sentence = ""
        self._last_sign = ""
        self._last_sign_time = 0
        self._sign_hold_threshold = 1.5  # Seconds to hold sign to add letter

        # ── Build GUI ──
        self._build_ui()

    def _build_ui(self):
        """Build all the UI widgets."""
        # Configure grid
        self.grid_columnconfigure(0, weight=3)  # Left: webcam
        self.grid_columnconfigure(1, weight=2)  # Right: info panel
        self.grid_rowconfigure(1, weight=1)

        # ────────────────────────────────────
        # TOP BAR
        # ────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=50, fg_color="#1a1f2e")
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_bar.grid_columnconfigure(1, weight=1)

        # Back button
        back_btn = ctk.CTkButton(
            top_bar,
            text="← Back",
            width=80,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            hover_color="#333344",
            command=self._go_back,
        )
        back_btn.grid(row=0, column=0, padx=10, pady=8)

        # Page title
        title = ctk.CTkLabel(
            top_bar,
            text="✋ Sign to Speech",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00C8B4",
        )
        title.grid(row=0, column=1, pady=8)

        # Status indicator
        self._status_label = ctk.CTkLabel(
            top_bar,
            text="⏸ Paused",
            font=ctk.CTkFont(size=12),
            text_color="#888888",
        )
        self._status_label.grid(row=0, column=2, padx=15, pady=8)

        # ────────────────────────────────────
        # LEFT PANEL: Webcam Feed
        # ────────────────────────────────────
        left_panel = ctk.CTkFrame(self, fg_color="#1a1f2e", corner_radius=15)
        left_panel.grid(row=1, column=0, padx=(15, 8), pady=15, sticky="nsew")
        left_panel.grid_rowconfigure(0, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        # Webcam display label
        self._video_label = ctk.CTkLabel(
            left_panel,
            text="📷 Camera will appear here\nClick 'Start' to begin",
            font=ctk.CTkFont(size=16),
            text_color="#666677",
        )
        self._video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Confidence bar under webcam
        self._conf_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        self._conf_frame.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(
            self._conf_frame,
            text="Confidence:",
            font=ctk.CTkFont(size=12),
            text_color="#8899AA",
        ).grid(row=0, column=0, padx=(0, 10))

        self._conf_bar = ctk.CTkProgressBar(
            self._conf_frame,
            width=200,
            height=12,
            progress_color="#00C8B4",
        )
        self._conf_bar.grid(row=0, column=1, sticky="ew")
        self._conf_bar.set(0)

        self._conf_text = ctk.CTkLabel(
            self._conf_frame,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#00C8B4",
            width=50,
        )
        self._conf_text.grid(row=0, column=2, padx=(10, 0))
        self._conf_frame.grid_columnconfigure(1, weight=1)

        # ────────────────────────────────────
        # RIGHT PANEL: Info & Controls
        # ────────────────────────────────────
        right_panel = ctk.CTkFrame(self, fg_color="#1a1f2e", corner_radius=15)
        right_panel.grid(row=1, column=1, padx=(8, 15), pady=15, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)

        # Detected sign (big display)
        ctk.CTkLabel(
            right_panel,
            text="Detected Sign",
            font=ctk.CTkFont(size=13),
            text_color="#8899AA",
        ).grid(row=0, column=0, padx=15, pady=(20, 5), sticky="w")

        self._sign_display = ctk.CTkLabel(
            right_panel,
            text="-",
            font=ctk.CTkFont(size=64, weight="bold"),
            text_color="#FFFFFF",
        )
        self._sign_display.grid(row=1, column=0, padx=15, pady=5)

        # Divider
        divider1 = ctk.CTkFrame(right_panel, height=2, fg_color="#2a3040")
        divider1.grid(row=2, column=0, padx=15, pady=10, sticky="ew")

        # Sentence display
        ctk.CTkLabel(
            right_panel,
            text="Sentence",
            font=ctk.CTkFont(size=13),
            text_color="#8899AA",
        ).grid(row=3, column=0, padx=15, pady=(5, 5), sticky="w")

        self._sentence_display = ctk.CTkTextbox(
            right_panel,
            height=100,
            font=ctk.CTkFont(size=18),
            fg_color="#0d1117",
            text_color="#FFFFFF",
            corner_radius=10,
            wrap="word",
        )
        self._sentence_display.grid(row=4, column=0, padx=15, pady=5, sticky="ew")

        # Divider
        divider2 = ctk.CTkFrame(right_panel, height=2, fg_color="#2a3040")
        divider2.grid(row=5, column=0, padx=15, pady=10, sticky="ew")

        # Control buttons
        btn_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        btn_frame.grid(row=6, column=0, padx=15, pady=5, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)

        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="▶ Start",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#00AA44",
            hover_color="#008833",
            command=self._toggle_camera,
        )
        self._start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        speak_btn = ctk.CTkButton(
            btn_frame,
            text="🔊 Speak",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#0088CC",
            hover_color="#006699",
            command=self._speak_sentence,
        )
        speak_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Second row of buttons
        space_btn = ctk.CTkButton(
            btn_frame,
            text="⎵ Space",
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#444455",
            hover_color="#555566",
            command=self._add_space,
        )
        space_btn.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        clear_btn = ctk.CTkButton(
            btn_frame,
            text="🗑 Clear",
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#444455",
            hover_color="#CC4444",
            command=self._clear_sentence,
        )
        clear_btn.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Instructions
        instructions = ctk.CTkLabel(
            right_panel,
            text="Hold a sign steady for 1.5s to add it.\n"
                 "Use Space to separate words.\n"
                 "Click Speak to hear the sentence.",
            font=ctk.CTkFont(size=11),
            text_color="#556677",
            justify="left",
        )
        instructions.grid(row=7, column=0, padx=15, pady=(10, 15), sticky="w")

    # ────────────────────────────────────
    # CAMERA & PREDICTION LOGIC
    # ────────────────────────────────────

    def _toggle_camera(self):
        """Start or stop the webcam."""
        if self._is_active:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        """Initialize and start the webcam capture."""
        try:
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                self._sign_display.configure(text="❌")
                self._status_label.configure(text="Camera Error", text_color="#FF4444")
                return

            # Force lower resolution to fix lag and boost FPS
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self._detector = HandDetector()
            self._predictor = Predictor()
            self._tts = TextToSpeech()

            self._is_active = True
            self._start_btn.configure(text="⏹ Stop", fg_color="#CC4444", hover_color="#AA2222")
            self._status_label.configure(text="🟢 Active", text_color="#00CC66")

            # Start webcam thread
            self._cam_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._cam_thread.start()

            # Start GUI update loop
            self._update_gui()

        except Exception as e:
            print(f"  Camera start error: {e}")
            self._status_label.configure(text=f"Error: {e}", text_color="#FF4444")

    def _stop_camera(self):
        """Stop the webcam and release resources."""
        self._is_active = False

        if self._cap:
            self._cap.release()
            self._cap = None

        if self._detector:
            self._detector.release()
            self._detector = None

        self._start_btn.configure(text="▶ Start", fg_color="#00AA44", hover_color="#008833")
        self._status_label.configure(text="⏸ Paused", text_color="#888888")
        self._video_label.configure(
            image=None,
            text="📷 Camera stopped\nClick 'Start' to resume",
        )

    def _capture_loop(self):
        """
        Background thread: continuously captures frames
        and runs prediction. Updates shared state that
        the GUI thread reads.
        """
        while self._is_active and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                continue

            # Mirror the frame
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks (also draws on frame)
            landmarks = self._detector.detect(frame)

            if landmarks is not None:
                # Normalize landmarks
                features = self._extractor.normalize(landmarks)

                # Predict sign
                sign, confidence = self._predictor.predict(features)

                if sign is not None:
                    self._current_sign = sign
                    self._confidence = confidence

                    # Check if this is a new sign or same sign held
                    current_time = time.time()
                    if sign != self._last_sign:
                        self._last_sign = sign
                        self._last_sign_time = current_time
                    elif current_time - self._last_sign_time >= self._sign_hold_threshold:
                        # Sign held long enough - add to sentence
                        self._sentence += sign
                        self._last_sign_time = current_time  # Reset timer

                    # Draw prediction on frame
                    cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
                    cv2.putText(
                        frame,
                        f"Sign: {sign}  ({confidence * 100:.0f}%)",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 200, 180),
                        2,
                    )
                else:
                    self._current_sign = "?"
                    self._confidence = confidence
            else:
                self._current_sign = "-"
                self._confidence = 0.0

            # Store frame for GUI (thread-safe)
            with self._frame_lock:
                self._current_frame = frame.copy()

            # Small delay to limit CPU usage
            time.sleep(0.01)

    def _update_gui(self):
        """
        Runs on the main thread via after().
        Reads the latest frame and updates widgets.
        """
        if not self._is_active:
            return

        # Update webcam display
        with self._frame_lock:
            frame = self._current_frame

        if frame is not None:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Use CTkImage for native scaling support in CustomTkinter
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(WEBCAM_DISPLAY_WIDTH, WEBCAM_DISPLAY_HEIGHT))
            
            self._video_label.configure(image=imgtk, text="")
            self._video_label.imgtk = imgtk  # Keep reference!

        # Update sign display
        self._sign_display.configure(text=self._current_sign)

        # Update confidence bar
        self._conf_bar.set(self._confidence)
        self._conf_text.configure(text=f"{self._confidence * 100:.0f}%")

        # Update sentence display
        self._sentence_display.delete("1.0", "end")
        self._sentence_display.insert("1.0", self._sentence)

        # Schedule next update
        self.after(GUI_UPDATE_INTERVAL, self._update_gui)

    # ────────────────────────────────────
    # BUTTON ACTIONS
    # ────────────────────────────────────

    def _speak_sentence(self):
        """Speak the current sentence."""
        if self._sentence.strip() and self._tts:
            self._tts.speak(self._sentence.strip())

    def _add_space(self):
        """Add a space to the sentence (word separator)."""
        self._sentence += " "

    def _clear_sentence(self):
        """Clear the entire sentence."""
        self._sentence = ""
        self._sentence_display.delete("1.0", "end")
        if self._predictor:
            self._predictor.reset()

    def _go_back(self):
        """Stop camera and go back to home."""
        self._stop_camera()
        self.go_back()

    # ────────────────────────────────────
    # LIFECYCLE
    # ────────────────────────────────────

    def on_show(self):
        """Called when page becomes visible."""
        pass  # Don't auto-start camera — let user click Start

    def on_hide(self):
        """Called when page is hidden."""
        self._stop_camera()
