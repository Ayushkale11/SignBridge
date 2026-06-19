"""
============================================
SignBridge - Feature Extractor Module
============================================
Normalizes raw MediaPipe landmarks so the
model works regardless of:
  - Where the hand is on screen (position)
  - How far the hand is from the camera (scale)

Normalization steps:
  1. Subtract the wrist position from all points
     → Hand is now centered at the origin (0,0,0)
  2. Divide by the maximum distance from the wrist
     → Hand always has the same "size"
============================================
"""

import numpy as np


class FeatureExtractor:
    """
    Converts raw MediaPipe landmarks into
    normalized features suitable for ML training.
    
    Usage:
        extractor = FeatureExtractor()
        features = extractor.normalize(raw_landmarks)
        # features is a list of 63 normalized floats
    """

    def normalize(self, landmarks):
        """
        Normalize a list of 63 landmark values.

        Args:
            landmarks: list of 63 floats [x0,y0,z0, x1,y1,z1, ...]
                       from HandDetector.detect()

        Returns:
            normalized: list of 63 floats, position/scale invariant
                        Returns None if input is invalid
        """
        if landmarks is None or len(landmarks) != 63:
            return None

        # Convert to numpy array and reshape to (21, 3)
        # Each row = one landmark with [x, y, z]
        coords = np.array(landmarks).reshape(21, 3)

        # Step 1: Get the wrist position (landmark 0)
        # The wrist is the "anchor" of the hand
        wrist = coords[0]

        # Step 2: Subtract wrist from all landmarks
        # This makes the wrist the origin (0, 0, 0)
        # Now all other points are relative to the wrist
        centered = coords - wrist

        # Step 3: Find the maximum distance from the wrist
        # np.linalg.norm calculates the distance of each point from origin
        # We take the maximum distance to use as our scale factor
        max_distance = np.max(np.linalg.norm(centered, axis=1))

        # Avoid division by zero (happens if all points overlap)
        if max_distance == 0:
            return landmarks  # Return raw if can't normalize

        # Step 4: Divide all coordinates by max_distance
        # Now the farthest point is at distance 1.0
        # All other points are between 0.0 and 1.0
        normalized = centered / max_distance

        # Flatten back to a list of 63 numbers
        return normalized.flatten().tolist()
