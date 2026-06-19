
🤟 SignBridge — Two-Way Sign Language Translator

A real-time bidirectional sign language translation system built with Python, OpenCV, MediaPipe, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange?style=flat-square)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-yellow?style=flat-square&logo=scikit-learn)

---

## 📌 About

**SignBridge** bridges the communication gap between sign language users and non-signers through two modes:

| Mode | Input → Output |
|------|----------------|
| **Sign to Speech** | 📷 Webcam → ✋ Hand Detection → 🤖 ML Prediction → 🔊 Speech |
| **Speech to Sign** | 🎤 Microphone → 📝 Text → 🖼️ Sign Image Display |

---

## ✨ Features

- 🖐️ Real-time hand tracking with 21 landmark points
- 🧠 Random Forest classifier (95%+ accuracy)
- 🔊 Offline text-to-speech (pyttsx3)
- 🎤 Speech recognition (Google API)
- 🖥️ Modern dark-themed GUI (CustomTkinter)
- 📊 Prediction smoothing with rolling window
- 🔤 Supports A-Z, 0-9, and 7 common words
- 🌐 Works offline (except speech-to-text)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Computer Vision | OpenCV, MediaPipe |
| Machine Learning | Scikit-Learn (Random Forest) |
| Speech Output | pyttsx3 |
| Speech Input | SpeechRecognition |
| GUI | CustomTkinter |
| Data | NumPy, Pandas |

---

## 📁 Project Structure

```
SignBridge/
├── dataset/                    # Collected landmark data
│   └── landmark_data.csv
├── models/                     # Trained ML models
│   └── sign_model.pkl
├── assets/                     # Sign images
│   ├── signs/                  # PNG images for each sign
│   └── gifs/                   # Animated GIFs
├── src/
│   ├── vision/                 # Hand detection & features
│   │   ├── hand_detector.py
│   │   └── feature_extractor.py
│   ├── ml/                     # ML pipeline
│   │   ├── data_collector.py
│   │   ├── trainer.py
│   │   └── predictor.py
│   ├── speech/                 # Speech modules
│   │   ├── text_to_speech.py
│   │   └── speech_to_text.py
│   ├── gui/                    # GUI pages
│   │   ├── main_window.py
│   │   ├── sign_to_speech_page.py
│   │   └── speech_to_sign_page.py
│   └── utils/                  # Configuration
│       ├── config.py
│       └── asset_generator.py
├── app.py                      # Main entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/SignBridge.git
cd SignBridge
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If PyAudio fails to install on Windows:
> ```bash
> pip install pipwin
> pipwin install pyaudio
> ```

### 4. Generate Sign Images

```bash
python -m src.utils.asset_generator
```

### 5. Collect Training Data

```bash
python -m src.ml.data_collector
```

Follow the on-screen instructions:
- Press **S** to start recording each sign
- Hold the sign steady — it captures 200 samples automatically
- Press **N** to skip a sign
- Press **Q** to quit

### 6. Train the Model

```bash
python -m src.ml.trainer
```

### 7. Launch the App

```bash
python app.py
```

---

## 📖 How It Works

### Sign to Speech Mode

1. **Hand Detection**: MediaPipe processes each webcam frame and identifies 21 hand landmark points (fingertips, knuckles, wrist).

2. **Feature Extraction**: Raw landmarks are normalized — centered on the wrist and scaled to unit size — so gestures look the same regardless of hand position.

3. **Classification**: A Random Forest model (100 decision trees) votes on the most likely sign from 43 classes.

4. **Smoothing**: A rolling window of 15 frames prevents flickering by picking the most consistent prediction.

5. **Speech**: pyttsx3 converts the recognized text to spoken audio in a background thread.

### Speech to Sign Mode

1. **Speech Recognition**: The SpeechRecognition library captures audio from the microphone and sends it to Google's API for transcription.

2. **Word Matching**: Each word is checked against our sign image library. Known words show their sign directly; unknown words are spelled letter by letter.

3. **Display**: Sign images are shown in a navigable gallery with Previous/Next controls.

---

## 🎯 Supported Signs

| Category | Signs |
|----------|-------|
| Alphabets | A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z |
| Numbers | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Words | HELLO, THANK YOU, YES, NO, HELP, WATER, FOOD |

---

## 🧪 Testing

```bash
# Verify all imports
python -c "import cv2, mediapipe, sklearn, pyttsx3, speech_recognition, customtkinter; print('All imports OK')"

# Run the full app
python app.py
```

---

## 📄 License

This project is for educational purposes as part of a Computer Engineering curriculum.

---

## 👤 Author

**Your Name** — 1st Year, Computer Engineering

---

*Built with ❤️ using Python, OpenCV, MediaPipe, and Scikit-Learn*
=======
# SignBridge
>>>>>>> 02aeae9fb34194b55a75f0f0a04bb690c8dec69e
