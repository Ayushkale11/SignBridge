"""
============================================
SignBridge - Speech to Sign Page
============================================
GUI page that:
  1. Listens to microphone input
  2. Converts speech to text
  3. Matches words with sign images
  4. Displays the corresponding sign image

If a word has a direct sign image, shows it.
If not, it spells out the word letter by letter.
============================================
"""

import os
import customtkinter as ctk
from PIL import Image, ImageTk

from src.speech.speech_to_text import SpeechToText
from src.utils.config import SIGNS_DIR


class SpeechToSignPage(ctk.CTkFrame):
    """
    Speech → Text → Sign page.
    
    Listens to speech, shows matching sign images.
    """

    def __init__(self, parent, go_back):
        super().__init__(parent, fg_color="transparent")
        self.go_back = go_back

        # ── State ──
        self._stt = SpeechToText()
        self._is_listening = False

        # Sign image navigation
        self._current_signs = []    # List of image paths to show
        self._current_index = 0     # Which image is currently shown
        self._recognized_text = ""

        # ── Build GUI ──
        self._build_ui()

    def _build_ui(self):
        """Build all the UI widgets."""
        self.grid_columnconfigure(0, weight=2)  # Left: controls
        self.grid_columnconfigure(1, weight=3)  # Right: sign display
        self.grid_rowconfigure(1, weight=1)

        # ────────────────────────────────────
        # TOP BAR
        # ────────────────────────────────────
        top_bar = ctk.CTkFrame(self, height=50, fg_color="#1a1f2e")
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_bar.grid_columnconfigure(1, weight=1)

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

        title = ctk.CTkLabel(
            top_bar,
            text="🎤 Speech to Sign",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#00C8B4",
        )
        title.grid(row=0, column=1, pady=8)

        self._status_label = ctk.CTkLabel(
            top_bar,
            text="⏸ Ready",
            font=ctk.CTkFont(size=12),
            text_color="#888888",
        )
        self._status_label.grid(row=0, column=2, padx=15, pady=8)

        # ────────────────────────────────────
        # LEFT PANEL: Controls
        # ────────────────────────────────────
        left_panel = ctk.CTkFrame(self, fg_color="#1a1f2e", corner_radius=15)
        left_panel.grid(row=1, column=0, padx=(15, 8), pady=15, sticky="nsew")
        left_panel.grid_columnconfigure(0, weight=1)

        # Microphone icon
        mic_label = ctk.CTkLabel(
            left_panel,
            text="🎙",
            font=ctk.CTkFont(size=56),
        )
        mic_label.grid(row=0, column=0, pady=(30, 10))

        # Listen button
        self._listen_btn = ctk.CTkButton(
            left_panel,
            text="🎤 Start Listening",
            width=220,
            height=50,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#00AA88",
            hover_color="#008866",
            command=self._toggle_listening,
        )
        self._listen_btn.grid(row=1, column=0, padx=20, pady=10)

        # Divider
        ctk.CTkFrame(left_panel, height=2, fg_color="#2a3040").grid(
            row=2, column=0, padx=20, pady=15, sticky="ew"
        )

        # Recognized text label
        ctk.CTkLabel(
            left_panel,
            text="Recognized Text",
            font=ctk.CTkFont(size=13),
            text_color="#8899AA",
        ).grid(row=3, column=0, padx=20, pady=(5, 5), sticky="w")

        self._text_display = ctk.CTkTextbox(
            left_panel,
            height=120,
            font=ctk.CTkFont(size=16),
            fg_color="#0d1117",
            text_color="#FFFFFF",
            corner_radius=10,
            wrap="word",
        )
        self._text_display.grid(row=4, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Manual text input
        ctk.CTkLabel(
            left_panel,
            text="Or type manually:",
            font=ctk.CTkFont(size=12),
            text_color="#556677",
        ).grid(row=5, column=0, padx=20, pady=(5, 3), sticky="w")

        input_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        input_frame.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self._text_input = ctk.CTkEntry(
            input_frame,
            height=38,
            font=ctk.CTkFont(size=14),
            placeholder_text="Type a word...",
            fg_color="#0d1117",
            corner_radius=10,
        )
        self._text_input.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        self._text_input.bind("<Return>", lambda e: self._search_typed_text())

        search_btn = ctk.CTkButton(
            input_frame,
            text="Show",
            width=60,
            height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0088CC",
            hover_color="#006699",
            command=self._search_typed_text,
        )
        search_btn.grid(row=0, column=1)

        # Instructions
        ctk.CTkLabel(
            left_panel,
            text="💡 Speak a word like 'Hello' or\n"
                 "   type it and click Show.\n\n"
                 "   Words with sign images show directly.\n"
                 "   Unknown words are spelled letter by letter.",
            font=ctk.CTkFont(size=11),
            text_color="#556677",
            justify="left",
        ).grid(row=7, column=0, padx=20, pady=(5, 20), sticky="w")

        # ────────────────────────────────────
        # RIGHT PANEL: Sign Display
        # ────────────────────────────────────
        right_panel = ctk.CTkFrame(self, fg_color="#1a1f2e", corner_radius=15)
        right_panel.grid(row=1, column=1, padx=(8, 15), pady=15, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)

        # Sign display header
        self._sign_header = ctk.CTkLabel(
            right_panel,
            text="Sign Display",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#8899AA",
        )
        self._sign_header.grid(row=0, column=0, padx=15, pady=(15, 5))

        # Sign image
        self._sign_image_label = ctk.CTkLabel(
            right_panel,
            text="🖼️\n\nSign images will appear here\nafter speaking or typing a word",
            font=ctk.CTkFont(size=14),
            text_color="#556677",
        )
        self._sign_image_label.grid(row=1, column=0, padx=15, pady=10, sticky="nsew")

        # Navigation buttons (Previous / Next)
        nav_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        nav_frame.grid(row=2, column=0, padx=15, pady=(5, 10), sticky="ew")
        nav_frame.grid_columnconfigure(1, weight=1)

        self._prev_btn = ctk.CTkButton(
            nav_frame,
            text="◄ Previous",
            width=110,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#444455",
            hover_color="#555566",
            command=self._prev_sign,
            state="disabled",
        )
        self._prev_btn.grid(row=0, column=0, padx=5)

        self._sign_counter = ctk.CTkLabel(
            nav_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#8899AA",
        )
        self._sign_counter.grid(row=0, column=1)

        self._next_btn = ctk.CTkButton(
            nav_frame,
            text="Next ►",
            width=110,
            height=35,
            font=ctk.CTkFont(size=13),
            fg_color="#444455",
            hover_color="#555566",
            command=self._next_sign,
            state="disabled",
        )
        self._next_btn.grid(row=0, column=2, padx=5)

    # ────────────────────────────────────
    # LISTENING LOGIC
    # ────────────────────────────────────

    def _toggle_listening(self):
        """Start or stop microphone listening."""
        if self._is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        """Start background speech recognition."""
        self._is_listening = True
        self._listen_btn.configure(
            text="🔴 Stop Listening",
            fg_color="#CC4444",
            hover_color="#AA2222",
        )
        self._status_label.configure(text="🟢 Listening...", text_color="#00CC66")

        # Start speech recognition with callback
        self._stt.start_listening(callback=self._on_speech_detected)

    def _stop_listening(self):
        """Stop speech recognition."""
        self._stt.stop_listening()
        self._is_listening = False
        self._listen_btn.configure(
            text="🎤 Start Listening",
            fg_color="#00AA88",
            hover_color="#008866",
        )
        self._status_label.configure(text="⏸ Stopped", text_color="#888888")

    def _on_speech_detected(self, text):
        """
        Called when speech is recognized.
        Runs on the STT background thread, so we use
        after() to safely update GUI on the main thread.
        """
        self.after(0, lambda: self._process_text(text))

    def _search_typed_text(self):
        """Process manually typed text."""
        text = self._text_input.get().strip()
        if text:
            self._process_text(text.lower())
            self._text_input.delete(0, "end")

    def _process_text(self, text):
        """
        Process recognized/typed text:
        1. Display the text
        2. Find matching sign images
        3. Show the first image
        """
        self._recognized_text = text

        # Update text display
        self._text_display.delete("1.0", "end")
        self._text_display.insert("1.0", text)

        # Find sign images for each word/letter
        self._current_signs = []
        words = text.split()

        for word in words:
            # Check if we have a direct image for this word
            word_upper = word.upper()
            word_filename = word_upper.replace(" ", "_") + ".png"
            word_path = os.path.join(SIGNS_DIR, word_filename)

            if os.path.exists(word_path):
                # Direct word match found
                self._current_signs.append((word_upper, word_path))
            else:
                # No direct match - spell it letter by letter
                for letter in word_upper:
                    if letter.isalpha() or letter.isdigit():
                        letter_path = os.path.join(SIGNS_DIR, f"{letter}.png")
                        if os.path.exists(letter_path):
                            self._current_signs.append((letter, letter_path))

        # Show first sign
        if self._current_signs:
            self._current_index = 0
            self._show_current_sign()
        else:
            self._sign_image_label.configure(
                image=None,
                text=f"No sign images found for:\n'{text}'",
            )
            self._sign_counter.configure(text="")
            self._prev_btn.configure(state="disabled")
            self._next_btn.configure(state="disabled")

    def _show_current_sign(self):
        """Display the sign image at the current index."""
        if not self._current_signs:
            return

        label, path = self._current_signs[self._current_index]

        try:
            # Load and resize image using CTkImage for compatibility
            img = Image.open(path)
            imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(350, 350))

            self._sign_image_label.configure(image=imgtk, text="")
            self._sign_image_label.imgtk = imgtk  # Keep reference

            # Update header
            self._sign_header.configure(text=f"Sign: {label}")

        except Exception as e:
            self._sign_image_label.configure(
                image=None,
                text=f"Error loading image:\n{e}",
            )

        # Update counter and nav buttons
        total = len(self._current_signs)
        self._sign_counter.configure(
            text=f"{self._current_index + 1} / {total}"
        )

        self._prev_btn.configure(
            state="normal" if self._current_index > 0 else "disabled"
        )
        self._next_btn.configure(
            state="normal" if self._current_index < total - 1 else "disabled"
        )

    def _prev_sign(self):
        """Show previous sign image."""
        if self._current_index > 0:
            self._current_index -= 1
            self._show_current_sign()

    def _next_sign(self):
        """Show next sign image."""
        if self._current_index < len(self._current_signs) - 1:
            self._current_index += 1
            self._show_current_sign()

    def _go_back(self):
        """Stop listening and go back to home."""
        self._stop_listening()
        self.go_back()

    # ────────────────────────────────────
    # LIFECYCLE
    # ────────────────────────────────────

    def on_show(self):
        """Called when page becomes visible."""
        pass

    def on_hide(self):
        """Called when page is hidden."""
        self._stop_listening()
