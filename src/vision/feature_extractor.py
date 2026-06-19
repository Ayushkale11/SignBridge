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
        centered = coords - wrist

        # Step 3: Align hand to point UP (Rotation Invariance)
        # Find vector from wrist (0) to middle finger MCP (9)
        v = centered[9]
        
        # Calculate current angle in the X-Y plane
        angle = np.arctan2(v[1], v[0])
        
        # Target angle is straight UP (which is -pi/2 in image coordinates)
        target_angle = -np.pi / 2.0
        
        # Rotation difference
        theta = target_angle - angle
        
        # Rotate all points around the Z-axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rot_matrix = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])
        
        rotated = np.dot(centered, rot_matrix.T)

        # Step 4: Scale Invariance
        max_distance = np.max(np.linalg.norm(rotated, axis=1))

        # Avoid division by zero (happens if all points overlap)
        if max_distance == 0:
            return landmarks  # Return raw if can't normalize

        # Step 5: Divide all coordinates by max_distance
        normalized = rotated / max_distance

        # Flatten back to a list of 63 numbers
        return normalized.flatten().tolist()
