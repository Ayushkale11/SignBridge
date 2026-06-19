"""
============================================
SignBridge - Main Window (Home Screen)
============================================
The entry point of the GUI. Shows the app
title and two main buttons:
  - Sign to Speech
  - Speech to Sign

Uses CustomTkinter for a modern dark look.
============================================
"""

import customtkinter as ctk

from src.gui.sign_to_speech_page import SignToSpeechPage
from src.gui.speech_to_sign_page import SpeechToSignPage
from src.utils.config import (
    WINDOW_TITLE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)


class MainWindow(ctk.CTk):
    """
    Main application window with page navigation.
    
    Pages:
        - Home: two buttons (Sign→Speech, Speech→Sign)
        - SignToSpeech: webcam + gesture recognition
        - SpeechToSign: microphone + sign display
    """

    def __init__(self):
        super().__init__()

        # ── Window Configuration ──
        self.title(WINDOW_TITLE)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(900, 600)
        self.resizable(True, True)

        # Set theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Page container ──
        # This frame holds all pages. Only one is visible at a time.
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # ── Create pages ──
        self.pages = {}
        self._current_page = None

        # Home page (created inline below)
        self._create_home_page()

        # Show home page first
        self.show_page("home")

        # ── Handle window close ──
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _create_home_page(self):
        """Build the home page with title and navigation buttons."""
        home = ctk.CTkFrame(self.container, fg_color="transparent")
        home.grid(row=0, column=0, sticky="nsew")
        self.pages["home"] = home

        # Center everything vertically
        home.grid_rowconfigure(0, weight=1)
        home.grid_rowconfigure(6, weight=1)
        home.grid_columnconfigure(0, weight=1)

        # ── App Icon / Emoji ──
        icon_label = ctk.CTkLabel(
            home,
            text="🤟",
            font=ctk.CTkFont(size=72),
        )
        icon_label.grid(row=1, column=0, pady=(0, 5))

        # ── Title ──
        title_label = ctk.CTkLabel(
            home,
            text="SIGNBRIDGE",
            font=ctk.CTkFont(size=42, weight="bold"),
            text_color="#00C8B4",  # Teal
        )
        title_label.grid(row=2, column=0, pady=(0, 5))

        # ── Subtitle ──
        subtitle_label = ctk.CTkLabel(
            home,
            text="Two-Way Sign Language Translator",
            font=ctk.CTkFont(size=16),
            text_color="#8899AA",
        )
        subtitle_label.grid(row=3, column=0, pady=(0, 40))

        # ── Button container ──
        btn_frame = ctk.CTkFrame(home, fg_color="transparent")
        btn_frame.grid(row=4, column=0)

        # Button styling
        btn_width = 260
        btn_height = 60
        btn_font = ctk.CTkFont(size=16, weight="bold")
        btn_corner = 15

        # Sign to Speech button
        sign_btn = ctk.CTkButton(
            btn_frame,
            text="✋  Sign to Speech",
            width=btn_width,
            height=btn_height,
            font=btn_font,
            corner_radius=btn_corner,
            fg_color="#0088CC",
            hover_color="#006699",
            command=lambda: self._open_sign_to_speech(),
        )
        sign_btn.grid(row=0, column=0, padx=15, pady=10)

        # Speech to Sign button
        speech_btn = ctk.CTkButton(
            btn_frame,
            text="🎤  Speech to Sign",
            width=btn_width,
            height=btn_height,
            font=btn_font,
            corner_radius=btn_corner,
            fg_color="#00AA88",
            hover_color="#008866",
            command=lambda: self._open_speech_to_sign(),
        )
        speech_btn.grid(row=0, column=1, padx=15, pady=10)

        # Exit button
        exit_btn = ctk.CTkButton(
            btn_frame,
            text="🚪  Exit",
            width=180,
            height=45,
            font=ctk.CTkFont(size=14),
            corner_radius=btn_corner,
            fg_color="#444455",
            hover_color="#CC4444",
            command=self.on_exit,
        )
        exit_btn.grid(row=1, column=0, columnspan=2, pady=(20, 0))

        # ── Footer ──
        footer = ctk.CTkLabel(
            home,
            text="Built with Python • OpenCV • MediaPipe • Scikit-Learn",
            font=ctk.CTkFont(size=11),
            text_color="#556677",
        )
        footer.grid(row=5, column=0, pady=(30, 0))

    def _open_sign_to_speech(self):
        """Create and show the Sign to Speech page."""
        if "sign_to_speech" not in self.pages:
            page = SignToSpeechPage(
                self.container,
                go_back=lambda: self.show_page("home"),
            )
            page.grid(row=0, column=0, sticky="nsew")
            self.pages["sign_to_speech"] = page
        self.show_page("sign_to_speech")

    def _open_speech_to_sign(self):
        """Create and show the Speech to Sign page."""
        if "speech_to_sign" not in self.pages:
            page = SpeechToSignPage(
                self.container,
                go_back=lambda: self.show_page("home"),
            )
            page.grid(row=0, column=0, sticky="nsew")
            self.pages["speech_to_sign"] = page
        self.show_page("speech_to_sign")

    def show_page(self, page_name):
        """Bring the named page to the front."""
        # Stop any active processes on the previous page
        if self._current_page and self._current_page != page_name:
            old_page = self.pages.get(self._current_page)
            if old_page and hasattr(old_page, "on_hide"):
                old_page.on_hide()

        page = self.pages[page_name]
        page.tkraise()
        self._current_page = page_name

        # Start processes on the new page
        if hasattr(page, "on_show"):
            page.on_show()

    def on_exit(self):
        """Clean up and close the application."""
        print("\n  Shutting down SignBridge...")

        # Stop all pages
        for name, page in self.pages.items():
            if hasattr(page, "on_hide"):
                try:
                    page.on_hide()
                except Exception:
                    pass

        self.destroy()
        print("  Goodbye! 👋\n")
