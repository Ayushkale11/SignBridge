# 🤟 SignBridge

**Two-Way Sign Language Translator**

SignBridge is a beginner-friendly Machine Learning project that bridges the communication gap between sign language users and non-signers. It provides two-way translation in real-time.

1. **Sign to Speech:** Uses your webcam and MediaPipe to detect hand gestures, translates them to text using a Random Forest model, and speaks the sentence out loud using Text-to-Speech (TTS).
2. **Speech to Sign:** Listens to your voice using Google's Speech Recognition, converts it to text, and displays the corresponding American Sign Language (ASL) images.

---

## ✨ Features

- **Real-Time Hand Tracking:** Powered by Google's MediaPipe Tasks API.
- **Rotation-Invariant Features:** The model automatically calculates angles and distances to recognize hands regardless of how you tilt your camera.
- **Auto Dataset Builder:** No need to painstakingly record yourself for hours! The project automatically downloads 87,000 professional ASL images from Kaggle and builds a robust dataset.
- **Beautiful UI:** Built with `customtkinter` for a modern, sleek, dark-mode desktop experience.
- **Background Processing:** Multi-threaded architecture guarantees a smooth, lag-free UI experience.

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/SignBridge.git
cd SignBridge
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
*(Dependencies: `opencv-python`, `mediapipe`, `scikit-learn`, `numpy`, `pandas`, `customtkinter`, `pyttsx3`, `SpeechRecognition`, `Pillow`, `kagglehub`)*

---

## 🚀 Quick Start Guide

### Step 1: Generate Sign Images
We need images to display when someone speaks. Run the asset generator to create placeholder sign images for A-Z, 0-9, and common words:
```bash
python -m src.utils.asset_generator
```

### Step 2: Build the Dataset
The app needs to learn what signs look like. This script will automatically download the Kaggle ASL Alphabet dataset, process it through MediaPipe to extract robust features, and save it to a CSV:
```bash
python -m src.ml.dataset_builder
```
*(Note: This might take 3-5 minutes depending on your internet speed and CPU).*

### Step 3: Train the Model
Once the dataset is built, train the Machine Learning model (Random Forest). This takes less than 10 seconds:
```bash
python -m src.ml.trainer
```

### Step 4: Launch the App
You are ready to go! Launch the main graphical interface:
```bash
python app.py
```

---

## 📂 Project Structure

```text
SignBridge/
│
├── app.py                            # Main application entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
├── dataset/                          # Where the CSV data is saved
├── models/                           # Where the trained .pkl model and .task are saved
├── assets/                           # Placeholder for sign images
│
└── src/
    ├── gui/                          # CustomTkinter interface
    │   ├── main_window.py            
    │   ├── sign_to_speech_page.py
    │   └── speech_to_sign_page.py
    │
    ├── ml/                           # Machine Learning logic
    │   ├── dataset_builder.py        # Kaggle auto-downloader & processor
    │   ├── data_collector.py         # Record your own custom gestures
    │   ├── trainer.py                # Train the Random Forest
    │   └── predictor.py              # Real-time smoothing & prediction
    │
    ├── vision/                       # Computer Vision
    │   ├── hand_detector.py          # MediaPipe Tasks wrapper
    │   └── feature_extractor.py      # Rotation/Scale normalizer
    │
    ├── speech/                       # Audio modules
    │   ├── text_to_speech.py         # Pyttsx3 wrapper
    │   └── speech_to_text.py         # SpeechRecognition listener
    │
    └── utils/
        ├── config.py                 # Global constants & settings
        └── asset_generator.py        # Generates basic sign images
```

---

## 💡 Troubleshooting

- **The camera is lagging:** Ensure you are running the app in a bright room. Low light drops webcam framerates. The resolution is capped at 640x480 by default to guarantee high FPS.
- **Model isn't recognizing my hand:** The Kaggle dataset is excellent, but it might not perfectly match your room's lighting or camera setup. You can easily train the AI on your *own* hand by running `python -m src.ml.data_collector`.
- **Speech API Errors:** The Speech-to-Sign module uses Google's free API, which requires an active internet connection.

---

**Developed for a 1st-Year Computer Engineering Project.** 🎓
