"""
============================================
SignBridge - Central Configuration
============================================
All paths, constants, and settings live here.
Every other module imports from this file
instead of hardcoding values.
============================================
"""

import os

# ──────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────
# Get the project root directory (two levels up from this file)
# This file is at: SignBridge/src/utils/config.py
# Project root is: SignBridge/
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Dataset paths
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_CSV = os.path.join(DATASET_DIR, "landmark_data.csv")

# Model paths
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.pkl")

# Asset paths (sign images and GIFs)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
SIGNS_DIR = os.path.join(ASSETS_DIR, "signs")
GIFS_DIR = os.path.join(ASSETS_DIR, "gifs")

# ──────────────────────────────────────────
# SIGN LABELS
# ──────────────────────────────────────────
# All the signs our model will recognize.
# 26 letters + 10 digits + 7 common words = 43 classes

ALPHABET_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

NUMBER_LABELS = [str(i) for i in range(10)]  # ["0", "1", ..., "9"]

WORD_LABELS = [
    "HELLO",
    "THANK YOU",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "FOOD",
]

# Combined list of all labels
ALL_LABELS = ALPHABET_LABELS + NUMBER_LABELS + WORD_LABELS

# ──────────────────────────────────────────
# MEDIAPIPE SETTINGS
# ──────────────────────────────────────────
# How confident MediaPipe must be to say "I found a hand"
DETECTION_CONFIDENCE = 0.7

# How confident MediaPipe must be to keep tracking a hand
TRACKING_CONFIDENCE = 0.7

# Maximum number of hands to detect (1 = simpler for beginners)
MAX_HANDS = 1

# ──────────────────────────────────────────
# DATA COLLECTION SETTINGS
# ──────────────────────────────────────────
# How many landmark snapshots to collect per sign
SAMPLES_PER_SIGN = 200

# ──────────────────────────────────────────
# PREDICTION SETTINGS
# ──────────────────────────────────────────
# Rolling window size for smoothing predictions
# (takes the most common prediction from the last N frames)
PREDICTION_WINDOW = 15

# Minimum confidence to accept a prediction (0.0 to 1.0)
MIN_PREDICTION_CONFIDENCE = 0.6

# Cooldown in seconds before repeating the same spoken word
SPEECH_COOLDOWN = 3.0

# ──────────────────────────────────────────
# GUI SETTINGS
# ──────────────────────────────────────────
WINDOW_TITLE = "SignBridge - Two-Way Sign Language Translator"
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700

# Webcam display size inside the GUI
WEBCAM_DISPLAY_WIDTH = 580
WEBCAM_DISPLAY_HEIGHT = 435

# GUI update interval in milliseconds (50fps = 20ms)
GUI_UPDATE_INTERVAL = 20

# ──────────────────────────────────────────
# TEXT-TO-SPEECH SETTINGS
# ──────────────────────────────────────────
TTS_RATE = 150      # Words per minute (default ~200, slower = clearer)
TTS_VOLUME = 1.0    # Volume: 0.0 to 1.0

# ──────────────────────────────────────────
# CREATE DIRECTORIES IF THEY DON'T EXIST
# ──────────────────────────────────────────
for directory in [DATASET_DIR, MODELS_DIR, SIGNS_DIR, GIFS_DIR]:
    os.makedirs(directory, exist_ok=True)
