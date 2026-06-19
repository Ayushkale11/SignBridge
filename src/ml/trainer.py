"""
============================================
SignBridge - Model Trainer Module
============================================
Loads the collected CSV dataset, trains a
Random Forest classifier, evaluates it, and
saves the trained model.

HOW TO USE:
  1. First collect data: python -m src.ml.data_collector
  2. Then train: python -m src.ml.trainer

WHAT IS RANDOM FOREST?
  Imagine 100 decision trees, each looking at
  different parts of the data. They all "vote"
  on what sign they think it is. The sign with
  the most votes wins. This is why it's called
  a "forest" - many trees working together!
============================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

from src.utils.config import DATASET_CSV, MODEL_PATH, MODELS_DIR


def train_model():
    """
    Complete training pipeline:
    1. Load CSV data
    2. Split into train/test
    3. Train Random Forest
    4. Evaluate accuracy
    5. Save model
    """

    print("\n" + "=" * 50)
    print("  SignBridge - Model Training")
    print("=" * 50)

    # ──────────────────────────────────────
    # STEP 1: Load the dataset
    # ──────────────────────────────────────
    if not os.path.exists(DATASET_CSV):
        print("\n  ERROR: Dataset not found!")
        print(f"  Expected at: {DATASET_CSV}")
        print("  Run data collection first: python -m src.ml.data_collector")
        return

    print("\n  Step 1: Loading dataset...")
    data = pd.read_csv(DATASET_CSV)

    print(f"    Total samples: {len(data)}")
    print(f"    Number of features: {data.shape[1] - 1}")  # -1 for label column
    print(f"    Number of classes: {data['label'].nunique()}")
    print(f"    Classes: {sorted(data['label'].unique())}")

    # ──────────────────────────────────────
    # STEP 2: Separate features and labels
    # ──────────────────────────────────────
    print("\n  Step 2: Preparing data...")

    # X = all columns EXCEPT 'label' (the 63 landmark features)
    X = data.drop("label", axis=1).values

    # y = only the 'label' column (what sign it is)
    y = data["label"].values

    # ──────────────────────────────────────
    # STEP 3: Split into training and testing sets
    # ──────────────────────────────────────
    # 80% for training, 20% for testing
    # stratify=y → ensures each sign has proportional representation
    # random_state=42 → same split every time (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples:  {len(X_test)}")

    # ──────────────────────────────────────
    # STEP 4: Train the Random Forest
    # ──────────────────────────────────────
    print("\n  Step 3: Training Random Forest...")
    print("    (This usually takes 5-15 seconds)")

    # Create the classifier
    # n_estimators=100 → 100 decision trees in the forest
    # random_state=42 → reproducible results
    # n_jobs=-1 → use all CPU cores for faster training
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    # Train the model (this is where the learning happens!)
    model.fit(X_train, y_train)

    print("    Training complete!")

    # ──────────────────────────────────────
    # STEP 5: Evaluate the model
    # ──────────────────────────────────────
    print("\n  Step 4: Evaluating model...")

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n    * Overall Accuracy: {accuracy * 100:.2f}%")

    # Detailed report per class
    print("\n    Per-class Performance:")
    print("    " + "-" * 46)
    report = classification_report(y_test, y_pred)
    # Indent each line of the report
    for line in report.split("\n"):
        print(f"    {line}")

    # ──────────────────────────────────────
    # STEP 6: Save the trained model
    # ──────────────────────────────────────
    print(f"\n  Step 5: Saving model to {MODEL_PATH}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("    Model saved successfully!")

    print("\n" + "=" * 50)
    print(f"  Training complete! Accuracy: {accuracy * 100:.2f}%")
    print(f"  Model saved to: {MODEL_PATH}")
    print("=" * 50 + "\n")

    return model, accuracy


# ── Run when executed directly ──
# Command: python -m src.ml.trainer
if __name__ == "__main__":
    train_model()
