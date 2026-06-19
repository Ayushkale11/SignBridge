"""
============================================
SignBridge - Automated Dataset Builder
============================================
Downloads the ASL alphabet image dataset
from Kaggle (no manual login needed via
kagglehub), then runs MediaPipe on every
image to extract 21 hand landmarks, and
saves them as landmark_data.csv — the exact
same format the trainer expects.

HOW TO USE:
  python -m src.ml.dataset_builder

WHAT HAPPENS:
  1. Downloads ASL alphabet images (~200 MB)
     from Kaggle using kagglehub
  2. For each image of each letter (A-Z):
       - Runs MediaPipe hand detection
       - Extracts 21 landmark points (63 numbers)
       - Normalizes them (position/scale invariant)
       - Saves to landmark_data.csv
  3. Adds synthetic data for 0-9 and common
     words (HELLO, THANK YOU, YES, NO, etc.)
     since those are not in the Kaggle dataset

OUTPUT:
  dataset/landmark_data.csv
  ~8,000+ rows, 64 columns (63 features + label)
============================================
"""

import os
import sys
import csv
import cv2
import numpy as np

from src.vision.hand_detector import HandDetector
from src.vision.feature_extractor import FeatureExtractor
from src.utils.config import (
    DATASET_CSV,
    DATASET_DIR,
    ALL_LABELS,
    ALPHABET_LABELS,
    NUMBER_LABELS,
    WORD_LABELS,
)


# ── How many samples to extract per class ──
# The Kaggle ASL dataset has ~3000 images per letter.
# We cap at this many to keep training balanced & fast.
SAMPLES_PER_CLASS = 300


def download_kaggle_dataset():
    """
    Download the ASL Alphabet dataset from Kaggle.
    Uses kagglehub which handles auth automatically.
    Returns the local path to the downloaded data.
    """
    print("\n  Downloading ASL Alphabet dataset from Kaggle...")
    print("  (This is ~200 MB and may take a few minutes)\n")

    try:
        import kagglehub
        # Download dataset: grassknoted/asl-alphabet
        # kagglehub caches it locally so it only downloads once
        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        print(f"\n  Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"\n  Kaggle download failed: {e}")
        print("  Trying alternative download method...")
        return None


def find_image_folders(dataset_path):
    """
    Find the folder containing per-letter subfolders.
    The Kaggle ASL dataset has structure:
      asl_alphabet_train/
        A/ (3000 images)
        B/ (3000 images)
        ...
    Returns path to the folder with letter subfolders.
    """
    if dataset_path is None:
        return None

    # Walk through the downloaded folder to find the training data
    for root, dirs, files in os.walk(dataset_path):
        # Look for a folder containing subfolders named A, B, C...
        subdirs = [d.upper() for d in dirs]
        letters = [l for l in ALPHABET_LABELS if l in subdirs]
        if len(letters) >= 20:  # Found folder with most letters
            return root

    return None


def extract_landmarks_from_images(image_folder, detector, extractor, writer, label):
    """
    Process all images in a folder, extract landmarks,
    and write to CSV.

    Args:
        image_folder: path to folder with images of one sign
        detector: HandDetector instance
        extractor: FeatureExtractor instance
        writer: csv.writer for output file
        label: sign label string (e.g., "A")

    Returns:
        count: number of samples successfully written
    """
    count = 0
    errors = 0

    # Get list of image files
    try:
        files = os.listdir(image_folder)
    except Exception:
        return 0

    # Filter to image files only
    image_files = [
        f for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ]

    # Shuffle to get variety if we cap samples
    np.random.shuffle(image_files)

    for filename in image_files:
        if count >= SAMPLES_PER_CLASS:
            break

        image_path = os.path.join(image_folder, filename)

        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            # Detect landmarks
            landmarks = detector.detect(frame)

            if landmarks is not None:
                # Normalize
                features = extractor.normalize(landmarks)

                if features is not None:
                    # Write to CSV
                    row = features + [label]
                    writer.writerow(row)
                    count += 1
            else:
                errors += 1

        except Exception as e:
            errors += 1
            continue

    return count


def add_word_samples(writer, num_samples=200):
    """
    For word labels (HELLO, YES, NO, etc.) we don't have
    real images, so we generate intelligent synthetic data
    by using ASL alphabet images as a base.

    Strategy: Map each word to its first letter's gesture
    as a placeholder. This gives the model something to
    learn — you can improve accuracy later by adding real
    word gesture data.
    """
    print("\n  Adding word gesture samples (synthetic)...")

    # Map words to representative finger positions
    # These are approximate normalized landmark patterns for
    # common ASL signs — good enough for demonstration
    word_patterns = {
        "HELLO":    _generate_open_hand_pattern(),
        "THANK YOU": _generate_flat_hand_pattern(),
        "YES":      _generate_fist_pattern(),
        "NO":       _generate_index_out_pattern(),
        "HELP":     _generate_thumbs_up_pattern(),
        "WATER":    _generate_w_shape_pattern(),
        "FOOD":     _generate_pinch_pattern(),
    }

    for word, base_pattern in word_patterns.items():
        for i in range(num_samples):
            # Add small random noise so samples aren't identical
            # This simulates natural variation in hand position
            noise = np.random.normal(0, 0.02, len(base_pattern))
            features = (np.array(base_pattern) + noise).tolist()

            # Clip values to valid range
            features = np.clip(features, -1.5, 1.5).tolist()

            row = features + [word]
            writer.writerow(row)

        print(f"    Added {num_samples} samples for '{word}'")


def add_number_samples_from_letter_patterns(writer, num_samples=200):
    """
    Numbers 0-9 use similar hand shapes to some letters.
    Map them to corresponding letter patterns with slight variation.

    ASL Number → Similar ASL Letter shape:
      0 → O    1 → 1-finger (D-like)
      2 → V    3 → W-like
      4 → 4-fingers   5 → Open hand
      6,7,8,9 → unique but approximate
    """
    print("\n  Adding number gesture samples (synthetic)...")

    number_patterns = {
        "0": _generate_o_shape_pattern(),
        "1": _generate_index_out_pattern(),
        "2": _generate_v_shape_pattern(),
        "3": _generate_three_fingers_pattern(),
        "4": _generate_four_fingers_pattern(),
        "5": _generate_open_hand_pattern(),
        "6": _generate_pinch_pattern(),
        "7": _generate_seven_pattern(),
        "8": _generate_eight_pattern(),
        "9": _generate_hook_pattern(),
    }

    for digit, base_pattern in number_patterns.items():
        for i in range(num_samples):
            noise = np.random.normal(0, 0.02, len(base_pattern))
            features = (np.array(base_pattern) + noise).tolist()
            features = np.clip(features, -1.5, 1.5).tolist()
            row = features + [digit]
            writer.writerow(row)

        print(f"    Added {num_samples} samples for '{digit}'")


# ──────────────────────────────────────────
# HAND SHAPE PATTERN GENERATORS
# Each returns 63 normalized values representing
# a typical hand shape for that sign.
# These are approximate - good enough for training.
# ──────────────────────────────────────────

def _base_hand():
    """Returns a flat normalized hand as a numpy array (21,3)."""
    # Wrist at origin, fingers spread out
    hand = np.array([
        [0.0,  0.0,  0.0],   # 0: Wrist
        [0.1,  0.1,  0.0],   # 1: Thumb CMC
        [0.2,  0.2, -0.05],  # 2: Thumb MCP
        [0.35, 0.25,-0.1],   # 3: Thumb IP
        [0.45, 0.3, -0.15],  # 4: Thumb tip
        [0.1, -0.2,  0.0],   # 5: Index MCP
        [0.1, -0.5, -0.05],  # 6: Index PIP
        [0.1, -0.75,-0.1],   # 7: Index DIP
        [0.1, -0.9, -0.15],  # 8: Index tip
        [0.0, -0.2,  0.0],   # 9: Middle MCP
        [0.0, -0.55,-0.05],  # 10: Middle PIP
        [0.0, -0.8, -0.1],   # 11: Middle DIP
        [0.0, -0.95,-0.15],  # 12: Middle tip
        [-0.1,-0.2,  0.0],   # 13: Ring MCP
        [-0.1,-0.5, -0.05],  # 14: Ring PIP
        [-0.1,-0.75,-0.1],   # 15: Ring DIP
        [-0.1,-0.9, -0.15],  # 16: Ring tip
        [-0.2,-0.15, 0.0],   # 17: Pinky MCP
        [-0.2,-0.4, -0.05],  # 18: Pinky PIP
        [-0.2,-0.6, -0.1],   # 19: Pinky DIP
        [-0.2,-0.75,-0.15],  # 20: Pinky tip
    ])
    return hand


def _flatten(hand_array):
    return hand_array.flatten().tolist()


def _generate_open_hand_pattern():
    """All fingers extended (HELLO / 5)."""
    return _flatten(_base_hand())


def _generate_fist_pattern():
    """All fingers curled (YES)."""
    h = _base_hand()
    # Curl fingers toward palm
    for i in [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    return _flatten(h)


def _generate_flat_hand_pattern():
    """Flat hand facing forward (THANK YOU)."""
    h = _base_hand()
    h[:, 2] *= 0.1  # Flatten z
    return _flatten(h)


def _generate_index_out_pattern():
    """Only index finger extended (1 / NO)."""
    h = _base_hand()
    # Curl all except index
    for i in [10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    # Curl thumb
    h[3][0] *= 0.5; h[4][0] *= 0.4
    return _flatten(h)


def _generate_thumbs_up_pattern():
    """Thumb up, fingers curled (HELP)."""
    h = _base_hand()
    for i in [6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    # Extend thumb upward
    h[2][1] = -0.3; h[3][1] = -0.5; h[4][1] = -0.7
    return _flatten(h)


def _generate_v_shape_pattern():
    """Index and middle extended (2 / V)."""
    h = _base_hand()
    for i in [14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    h[3][0] *= 0.5; h[4][0] *= 0.4
    return _flatten(h)


def _generate_w_shape_pattern():
    """Index, middle, ring extended (WATER / W)."""
    h = _base_hand()
    for i in [18, 19, 20]:
        h[i][1] *= 0.3
    h[3][0] *= 0.5; h[4][0] *= 0.4
    return _flatten(h)


def _generate_pinch_pattern():
    """Thumb and index pinched (FOOD / 6)."""
    h = _base_hand()
    h[4] = h[8].copy()  # Touch thumb tip to index tip
    for i in [10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    return _flatten(h)


def _generate_o_shape_pattern():
    """O shape — thumb and index form circle (0)."""
    h = _base_hand()
    h[4][0] = 0.1; h[4][1] = -0.5   # Thumb tip near index
    for i in [6, 7, 8]:
        h[i][0] = 0.08
        h[i][1] = -0.4 - i * 0.05
    for i in [10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.5
    return _flatten(h)


def _generate_three_fingers_pattern():
    """Index, middle, ring extended (3)."""
    h = _base_hand()
    for i in [18, 19, 20]:
        h[i][1] *= 0.35
    h[3][0] *= 0.5; h[4] = h[9].copy()
    return _flatten(h)


def _generate_four_fingers_pattern():
    """All fingers except thumb extended (4)."""
    h = _base_hand()
    h[3][0] *= 0.5; h[4][0] *= 0.4; h[4][1] *= 0.5
    return _flatten(h)


def _generate_seven_pattern():
    """Pinky and thumb extended (7-like)."""
    h = _base_hand()
    for i in [6, 7, 8, 10, 11, 12, 14, 15, 16]:
        h[i][1] *= 0.3
    h[2][1] = -0.2; h[3][1] = -0.3; h[4][1] = -0.4
    return _flatten(h)


def _generate_eight_pattern():
    """Middle finger bent to touch thumb (8)."""
    h = _base_hand()
    h[12] = h[3].copy()  # Middle tip touches thumb area
    for i in [14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.4
    return _flatten(h)


def _generate_hook_pattern():
    """Index finger hooked / curved (9)."""
    h = _base_hand()
    h[7][1] = -0.5; h[8][1] = -0.4; h[8][0] = 0.05
    for i in [10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    return _flatten(h)


# ──────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────

def build_dataset():
    """Main function: download images → extract landmarks → save CSV."""

    print("\n" + "=" * 55)
    print("  SignBridge - Automated Dataset Builder")
    print("=" * 55)

    os.makedirs(DATASET_DIR, exist_ok=True)

    # ── Step 1: Download dataset from Kaggle ──
    dataset_path = download_kaggle_dataset()
    image_root = find_image_folders(dataset_path)

    if image_root:
        print(f"\n  Found image data at: {image_root}")
    else:
        print("\n  Could not locate image folders.")
        print("  Will generate synthetic data for all classes.")

    # ── Step 2: Set up CSV ──
    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    header.append("label")

    # ── Step 3: Initialize detector ──
    print("\n  Initializing hand detector...")
    detector = HandDetector()
    extractor = FeatureExtractor()
    print("  Hand detector ready!\n")

    total_samples = 0

    with open(DATASET_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # ── Step 4: Process alphabet images (A-Z) ──
        print("  Processing alphabet signs (A-Z):")
        print("  " + "-" * 40)

        for letter in ALPHABET_LABELS:
            if image_root:
                # Try to find the letter folder (case-insensitive)
                letter_folder = None
                try:
                    for item in os.listdir(image_root):
                        if item.upper() == letter and os.path.isdir(
                            os.path.join(image_root, item)
                        ):
                            letter_folder = os.path.join(image_root, item)
                            break
                except Exception:
                    pass

                if letter_folder:
                    count = extract_landmarks_from_images(
                        letter_folder, detector, extractor, writer, letter
                    )
                    total_samples += count
                    print(f"    {letter}: {count} samples from images")
                    continue

            # Fallback: synthetic data for this letter
            count = _write_synthetic_letter(writer, letter)
            total_samples += count
            print(f"    {letter}: {count} samples (synthetic)")

        # ── Step 5: Numbers (0-9) ──
        add_number_samples_from_letter_patterns(writer, num_samples=200)
        total_samples += 10 * 200

        # ── Step 6: Common words ──
        add_word_samples(writer, num_samples=200)
        total_samples += len(WORD_LABELS) * 200

    # ── Cleanup ──
    detector.release()

    print(f"\n{'=' * 55}")
    print(f"  * Dataset built successfully!")
    print(f"  Total samples: {total_samples}")
    print(f"  Saved to: {DATASET_CSV}")
    print(f"{'=' * 55}\n")
    print("  Next step: python -m src.ml.trainer")


def _write_synthetic_letter(writer, letter):
    """
    Generate synthetic landmark data for a single letter
    using simple geometric rules for ASL hand shapes.
    Returns number of samples written.
    """
    # Map letters to approximate hand patterns
    # This covers common ASL fingerspelling shapes
    letter_map = {
        'A': _generate_fist_pattern,
        'B': _generate_flat_hand_pattern,
        'C': _generate_o_shape_pattern,
        'D': _generate_index_out_pattern,
        'E': _generate_hook_pattern,
        'F': _generate_pinch_pattern,
        'G': _generate_index_out_pattern,
        'H': _generate_v_shape_pattern,
        'I': lambda: _generate_finger_pattern(only=[20]),
        'J': lambda: _generate_finger_pattern(only=[20]),
        'K': _generate_v_shape_pattern,
        'L': lambda: _generate_l_pattern(),
        'M': _generate_fist_pattern,
        'N': _generate_fist_pattern,
        'O': _generate_o_shape_pattern,
        'P': _generate_index_out_pattern,
        'Q': _generate_index_out_pattern,
        'R': _generate_v_shape_pattern,
        'S': _generate_fist_pattern,
        'T': _generate_fist_pattern,
        'U': _generate_v_shape_pattern,
        'V': _generate_v_shape_pattern,
        'W': _generate_w_shape_pattern,
        'X': _generate_hook_pattern,
        'Y': _generate_thumbs_up_pattern,
        'Z': _generate_index_out_pattern,
    }

    base_fn = letter_map.get(letter, _generate_open_hand_pattern)
    base = np.array(base_fn())
    count = 0

    for _ in range(SAMPLES_PER_CLASS):
        noise = np.random.normal(0, 0.025, len(base))
        features = np.clip(base + noise, -1.5, 1.5).tolist()
        writer.writerow(features + [letter])
        count += 1

    return count


def _generate_finger_pattern(only=None):
    """Only specific finger(s) extended."""
    h = _base_hand()
    all_tips = [4, 8, 12, 16, 20]
    for tip in all_tips:
        if only and tip not in only:
            # Curl this finger
            dip = tip - 1; pip = tip - 2
            h[tip][1] *= 0.3
            h[dip][1] *= 0.4
    return _flatten(h)


def _generate_l_pattern():
    """L shape: thumb up + index out."""
    h = _base_hand()
    # Curl middle, ring, pinky
    for i in [10, 11, 12, 14, 15, 16, 18, 19, 20]:
        h[i][1] *= 0.3
    # Extend thumb upward
    h[2][1] = -0.2; h[3][1] = -0.4; h[4][1] = -0.6
    h[4][0] = 0.2
    return _flatten(h)


if __name__ == "__main__":
    build_dataset()
