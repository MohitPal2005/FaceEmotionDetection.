# Face Emotion Detection

A **real-time face emotion detection application** built in **Python** using **OpenCV** and **DeepFace**.  
This project captures live video from your webcam, detects faces, and analyzes the dominant emotion of each detected face. It displays the emotion on the video feed in real time.

---

## Features

- Real-time face detection using OpenCV’s Haar Cascade classifier.
- Emotion recognition with DeepFace’s pre-trained models.
- Overlay of detected faces with emotion labels on webcam video.
- Easy-to-use: press **'s'** key to exit the app.
- Robust error handling during emotion detection.

---

## Technologies Used

- Python 3.8+
- OpenCV (`opencv-python`)
- DeepFace
- NumPy

---

## Install the required dependencies:

pip install -r requirements.txt


## Usage

Run the application with:

python face_emotion_detection.py
The app will open your webcam and start detecting faces and emotions live.

Detected emotions are displayed as labels above each detected face.

Press the 's' key to stop the program and close the webcam window.