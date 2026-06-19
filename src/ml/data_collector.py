"""
============================================
SignBridge - Dataset Collection Tool
============================================
Interactive tool to collect hand gesture data
using your webcam. It guides you through each
sign and saves landmark data to a CSV file.

HOW TO USE:
  1. Run this file: python -m src.ml.data_collector
  2. A webcam window opens showing which sign to perform
  3. Press 'S' to start recording (hold the sign steady)
  4. It captures 200 samples automatically
  5. Press 'N' to skip to the next sign
  6. Press 'Q' to quit and save

The data is saved to: dataset/landmark_data.csv
============================================
"""

import cv2
import csv
import os
import time
from src.vision.hand_detector import HandDetector
from src.vision.feature_extractor import FeatureExtractor
from src.utils.config import (
    ALL_LABELS,
    DATASET_CSV,
    DATASET_DIR,
    SAMPLES_PER_SIGN,
)


def collect_data():
    """
    Main function to collect hand gesture dataset.
    Opens webcam and guides user through data collection.
    """

    # ── Initialize components ──
    detector = HandDetector()
    extractor = FeatureExtractor()
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("Make sure your webcam is connected and not used by another app.")
        return

    # ── Prepare CSV file ──
    # Create header: x0, y0, z0, x1, y1, z1, ..., x20, y20, z20, label
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    # Check if CSV already exists (to append data)
    file_exists = os.path.exists(DATASET_CSV)

    # Open CSV in append mode so we don't lose previous data
    csv_file = open(DATASET_CSV, mode="a", newline="")
    writer = csv.writer(csv_file)

    # Write header only if file is new
    if not file_exists:
        writer.writerow(header)

    # ── Collection loop ──
    current_label_index = 0  # Which sign we're currently collecting
    samples_collected = 0     # How many samples collected for current sign
    is_recording = False       # Are we currently recording?

    print("\n" + "=" * 50)
    print("  SignBridge - Data Collection Tool")
    print("=" * 50)
    print(f"\n  Total signs to collect: {len(ALL_LABELS)}")
    print(f"  Samples per sign: {SAMPLES_PER_SIGN}")
    print("\n  Controls:")
    print("    S = Start recording")
    print("    N = Next sign (skip)")
    print("    Q = Quit and save")
    print("=" * 50)

    while cap.isOpened() and current_label_index < len(ALL_LABELS):
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from webcam!")
            break

        # Flip frame horizontally (mirror effect - more natural)
        frame = cv2.flip(frame, 1)

        # Get current label
        current_label = ALL_LABELS[current_label_index]

        # ── Detect hand and extract landmarks ──
        landmarks = detector.detect(frame)

        # ── Record data if recording ──
        if is_recording and landmarks is not None:
            # Normalize the landmarks
            features = extractor.normalize(landmarks)

            if features is not None:
                # Write to CSV: 63 features + label
                row = features + [current_label]
                writer.writerow(row)
                samples_collected += 1

        # ── Draw info on the frame ──
        # Background bar at top
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (50, 50, 50), -1)

        # Current sign label
        cv2.putText(
            frame,
            f"Sign: {current_label}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),  # Yellow
            2,
        )

        # Progress counter
        cv2.putText(
            frame,
            f"Samples: {samples_collected}/{SAMPLES_PER_SIGN}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),  # White
            2,
        )

        # Progress through all signs
        progress_text = f"Sign {current_label_index + 1}/{len(ALL_LABELS)}"
        cv2.putText(
            frame,
            progress_text,
            (frame.shape[1] - 200, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        # Recording indicator
        if is_recording:
            # Red circle = recording
            cv2.circle(frame, (frame.shape[1] - 30, 60), 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                "REC",
                (frame.shape[1] - 70, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        else:
            # Instruction
            cv2.putText(
                frame,
                "Press 'S' to start recording",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),  # Green
                2,
            )

        # Hand detection status
        if landmarks is None:
            cv2.putText(
                frame,
                "No hand detected - show your hand!",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),  # Red
                2,
            )

        # ── Check if enough samples collected ──
        if samples_collected >= SAMPLES_PER_SIGN:
            print(f"  ✓ Collected {SAMPLES_PER_SIGN} samples for '{current_label}'")
            current_label_index += 1
            samples_collected = 0
            is_recording = False
            time.sleep(0.5)  # Brief pause between signs
            continue

        # ── Show the frame ──
        cv2.imshow("SignBridge - Data Collection", frame)

        # ── Handle keyboard input ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") or key == ord("S"):
            # Start recording
            is_recording = True
            print(f"  ● Recording sign: '{current_label}'...")

        elif key == ord("n") or key == ord("N"):
            # Skip to next sign
            print(f"  → Skipped '{current_label}' ({samples_collected} samples)")
            current_label_index += 1
            samples_collected = 0
            is_recording = False

        elif key == ord("q") or key == ord("Q"):
            # Quit
            print("\n  Data collection stopped by user.")
            break

    # ── Cleanup ──
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    print(f"\n  Dataset saved to: {DATASET_CSV}")
    print("  Data collection complete!")


# ── Run when executed directly ──
# Command: python -m src.ml.data_collector
if __name__ == "__main__":
    collect_data()
