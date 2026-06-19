"""
============================================
SignBridge - Main Entry Point
============================================
This is the file you run to start the app:

    python app.py

It launches the CustomTkinter GUI with all
the Sign Language translation features.
============================================
"""

import sys
import os

# Add the project root to Python's path
# This allows imports like "from src.vision import ..."
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Launch the SignBridge application."""
    print("\n" + "=" * 50)
    print("  🤟 SignBridge - Two-Way Sign Language Translator")
    print("=" * 50)
    print("  Starting application...\n")

    # Import here to avoid issues if dependencies aren't installed
    try:
        from src.gui.main_window import MainWindow
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency - {e}")
        print("\n  Install all required libraries:")
        print("    pip install -r requirements.txt")
        print("\n  If PyAudio fails, try:")
        print("    pip install pipwin")
        print("    pipwin install pyaudio")
        sys.exit(1)

    # Create and run the app
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
