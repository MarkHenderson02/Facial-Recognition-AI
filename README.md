# Facial-Recognition-AI

🎭 AI Facial Recognition System – Face ID, Age & Emotion Detector

A real-time AI-powered Facial Recognition system built with Python, OpenCV, and deep learning. It can:
🧠 Recognize known faces (Face ID)
🎂 Estimate a person’s age group (Age Estimation)
😊 Detect emotional expressions (Emotion Recognition)
All in real time using your webcam.

Features:
🔎 Face Detection using OpenCV Haar Cascade
🧬 Face Recognition using face_recognition (optional)
👶 Age Estimation using CNN trained on UTKFace
😄 Emotion Recognition using CNN trained on FER2013
🎥 Live Webcam Interface (OpenCV)
📦 Optional export to TensorFlow Lite for Android/iOS deployment

⚙️ Installation
1. Clone the repo
   git clone https://github.com/your-username/facial-recognition-app.git
   cd facial-recognition-app

2. Install requirements
   pip install -r requirements.txt

3. Download datasets:
   UTKFace: https://susanqq.github.io/UTKFace/
   FER2013: Available via Kaggle or open datasets

   Place them in:
   dataset/age/
   dataset/emotion/

🏋️ Training
# Train age estimator
python training/train_age_model.py

# Train emotion classifier
python training/train_emotion_model.py

🎥 Run Real-Time App
python realtime_app/run_camera_app.py

Make sure your webcam is connected. The app will:
1. Detect your face
2. Show estimated age and emotion
3. Recognize identity (if your face was trained)

📌 Dependencies
Python 3.8+
TensorFlow / Keras
OpenCV
face_recognition (optional, requires dlib)
NumPy, Matplotlib, Scikit-learn

📖 License
MIT License © 2025 Enrique Justine Sun
   
