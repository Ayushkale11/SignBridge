# 🤟 SignBridge

### Two-Way Sign Language Translator

SignBridge is an AI-powered desktop application that enables communication between sign language users and non-signers through real-time translation.

The system supports:

* **Sign Language → Text → Speech**
* **Speech → Text → Sign Language**

Built using Python, MediaPipe, OpenCV, and Machine Learning, SignBridge demonstrates how computer vision and speech technologies can be combined to create accessible communication tools.

---

## 📌 Overview

Communication barriers often exist between hearing-impaired individuals and people who do not understand sign language.

SignBridge aims to reduce this gap by providing two translation modes:

### 1. Sign → Speech

The application:

1. Captures hand gestures using a webcam
2. Detects hand landmarks using MediaPipe
3. Extracts geometric features
4. Classifies gestures using a Random Forest model
5. Converts predictions into text
6. Speaks the result using Text-to-Speech

### 2. Speech → Sign

The application:

1. Captures audio from a microphone
2. Converts speech to text
3. Matches recognized words with sign representations
4. Displays corresponding sign images

---

## 🚀 Key Features

### Computer Vision

* Real-time hand tracking
* MediaPipe hand landmark detection
* 21-point hand skeleton visualization
* Rotation and scale-invariant feature extraction

### Machine Learning

* Automated dataset generation
* Random Forest gesture classification
* Real-time prediction smoothing
* Custom gesture training support

### Speech Processing

* Speech-to-Text conversion
* Text-to-Speech output
* Voice-based interaction

### User Interface

* Modern CustomTkinter interface
* Multi-page application design
* Real-time feedback
* Responsive and lightweight

---

## 🏗️ Technology Stack

| Category           | Technology        |
| ------------------ | ----------------- |
| Language           | Python            |
| Computer Vision    | OpenCV, MediaPipe |
| Machine Learning   | Scikit-Learn      |
| Data Processing    | NumPy, Pandas     |
| GUI                | CustomTkinter     |
| Speech Recognition | SpeechRecognition |
| Text-to-Speech     | pyttsx3           |
| Image Processing   | Pillow            |
| Dataset Download   | KaggleHub         |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/Ayushkale11/SignBridge.git
cd SignBridge
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Quick Start

### Step 1: Generate Sign Assets

```bash
python -m src.utils.asset_generator
```

Creates placeholder sign images used by the Speech-to-Sign module.

---

### Step 2: Build Dataset

```bash
python -m src.ml.dataset_builder
```

This script:

* Downloads the ASL dataset
* Processes images using MediaPipe
* Extracts hand landmark features
* Generates a training dataset

---

### Step 3: Train the Model

```bash
python -m src.ml.trainer
```

Trains the Random Forest classifier and saves the model.

---

### Step 4: Launch Application

```bash
python app.py
```

---

## 🔄 System Workflow

### Sign → Speech Pipeline

```text
Webcam
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
Feature Extraction
   │
   ▼
Random Forest Classifier
   │
   ▼
Text Output
   │
   ▼
Text-To-Speech
```

### Speech → Sign Pipeline

```text
Microphone
   │
   ▼
Speech Recognition
   │
   ▼
Text Processing
   │
   ▼
Sign Matching
   │
   ▼
Sign Image Display
```

---

## 📂 Project Structure

```text
SignBridge/

├── app.py
├── README.md
├── requirements.txt

├── dataset/
├── models/
├── assets/

└── src/
    ├── gui/
    ├── ml/
    ├── vision/
    ├── speech/
    └── utils/
```

---

## 🧠 Machine Learning Pipeline

1. Dataset Collection
2. Hand Landmark Extraction
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Real-Time Inference

Current classifier:

* Random Forest

Future models:

* SVM
* XGBoost
* Neural Networks

---

## 🧪 Future Improvements

* Dynamic gesture recognition
* Word-level sign recognition
* Sentence generation
* Indian Sign Language (ISL) support
* Mobile application
* Offline speech recognition
* Improved dataset collection tools

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a Pull Request

---

## 📄 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

**Ayush Kale**

Computer Engineering Student


