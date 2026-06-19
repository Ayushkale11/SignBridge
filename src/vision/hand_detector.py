"""
============================================
SignBridge - Hand Detector Module
============================================
Uses MediaPipe Hand Landmarker (Tasks API) to
detect a hand in a webcam frame and extract
21 landmark points.

Each landmark has (x, y, z) coordinates:
  - x, y: position on screen (0.0 to 1.0)
  - z: depth relative to wrist

Total: 21 landmarks × 3 coords = 63 numbers

NOTE: MediaPipe 0.10.14+ removed the legacy
mp.solutions API. This module uses the new
Tasks API (mediapipe.tasks.python.vision).
============================================
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils.config import (
    DETECTION_CONFIDENCE,
    TRACKING_CONFIDENCE,
    MAX_HANDS,
    MODELS_DIR,
)
import os


# ── Hand landmark connections for drawing ──
# These define which landmarks to connect with lines
# (same as the old mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


class HandDetector:
    """
    Wraps MediaPipe Hand Landmarker (Tasks API) to
    provide a simple API for detecting hands and
    extracting landmarks.

    Usage:
        detector = HandDetector()
        landmarks = detector.detect(frame)
        # landmarks is a list of 63 floats, or None if no hand found
        detector.release()
    """

    def __init__(self):
        # Path to the downloaded hand landmarker model
        model_path = os.path.join(MODELS_DIR, "hand_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/"
                "hand_landmarker.task', "
                f"r'{model_path}')\""
            )

        # ── Configure the Hand Landmarker ──
        # BaseOptions: tells MediaPipe which model file to use
        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        # HandLandmarkerOptions: configure detection behavior
        # running_mode=IMAGE → we process one frame at a time
        #   (VIDEO mode requires timestamps, IMAGE is simpler)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_hand_presence_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
        )

        # Create the landmarker
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame):
        """
        Detect a hand in the given frame.

        Args:
            frame: BGR image from OpenCV (numpy array)

        Returns:
            landmarks: list of 63 floats [x0,y0,z0, x1,y1,z1, ...]
                        or None if no hand detected

        Side effect:
            Draws landmarks directly on the input frame
            (the frame is modified in-place)
        """
        # Step 1: Convert BGR (OpenCV) → RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Step 2: Create a MediaPipe Image from the numpy array
        # The new API requires mp.Image objects, not raw arrays
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # Step 3: Run hand detection
        result = self.landmarker.detect(mp_image)

        # Step 4: Check if any hand was found
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            # Get the first hand's landmarks
            hand = result.hand_landmarks[0]

            # Step 5: Draw landmarks on the frame
            self._draw_landmarks(frame, hand)

            # Step 6: Extract landmark coordinates as a flat list
            landmarks = []
            for lm in hand:
                landmarks.extend([lm.x, lm.y, lm.z])

            return landmarks

        # No hand detected
        return None

    def _draw_landmarks(self, frame, hand_landmarks):
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to draw on (modified in-place)
            hand_landmarks: list of NormalizedLandmark objects
        """
        h, w, _ = frame.shape

        # Convert normalized coordinates to pixel coordinates
        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))

        # Draw connections (lines between landmarks)
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(
                    frame,
                    points[start_idx],
                    points[end_idx],
                    (0, 200, 0),  # Green lines
                    2,
                )

        # Draw landmark points (circles)
        for i, (px, py) in enumerate(points):
            # Fingertips get larger red circles
            if i in [4, 8, 12, 16, 20]:
                cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)  # Red
            else:
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)  # Green

    def release(self):
        """Clean up MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()
            self.landmarker = None
