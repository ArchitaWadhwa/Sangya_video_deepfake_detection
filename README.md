# AuthentiCheck

## Video and Audio Deepfake Detection Project

## Overview

AuthentiCheck is a robust system designed for detecting deepfake videos and audio using advanced pretrained models. The project offers functionalities for video and audio analysis, including real-time livestream video processing. It leverages Streamlit for the web interface, PyTorch for video deepfake detection, and several other essential libraries.
![i1](https://github.com/ArchitaWadhwa/Sangya_video_deepfake_detection/assets/120119962/d81f446d-f108-4f7b-a813-11b46dd90a86)

## Features

### Deepfake Video Detection

1. **Upload a Video File:**
   - Use the file upload widget to upload a video file (.mp4, .mov).

2. **Analyze Video for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded video for deepfake content using a pretrained ViViT transformer model.

3. **View Results:**
   - Results include the predicted label ('REAL' or 'FAKE') and a confidence score indicating the model's prediction certainty.

### Deepfake Audio Detection

1. **Upload an Audio File:**
   - Use the file upload widget to upload an audio file (.wav, .mp3).

2. **Analyze Audio for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded audio file using the Wav2vec model.

3. **View Analysis Results:**
   - Results provide insights into the authenticity of the audio file, indicating if it contains suspicious content.

### Livestream Video Analysis

1. **Access Webcam for Real-Time Analysis:**
   - Navigate to the "Livestream Video" section and grant permissions to allow the application to access your webcam.

2. **Perform Real-Time Deepfake Detection:**
   - The application starts streaming video from your webcam, performing real-time analysis to detect deepfake content as the video streams.
## Preprocessing Method

- **YOLOv8 Custom Fine-Tuned for Face Detection**:
  - We used YOLOv8, custom fine-tuned for face detection, to preprocess the video frames.
  - **OpenCV Built-In Object Tracking for Face Tracking**:
    - After detecting the face using YOLOv8, we utilized OpenCV's built-in object tracking for continuous face tracking across frames. This approach avoids the need for detecting the face in every frame, saving processing time and improving efficiency.

## Tech Stack

To achieve this, we leveraged cutting-edge technologies:
- **ViViT Transformer Model**: Used to extract intricate features from video data, ensuring high accuracy in deepfake detection.
- **Wav2vec Model**: Employed for analyzing audio features, adding an extra layer of verification and enhancing detection capabilities.
- **Late Fusion Technique**: Used for the fusion of video and audio features, allowing comprehensive and accurate classification.
- **Streamlit**: Utilized for deploying the solution, providing a user-friendly interface for both real-time and uploaded video analysis.
- **PyTorch**: Employed for implementing deep learning models.
![i2](https://github.com/ArchitaWadhwa/Sangya_video_deepfake_detection/assets/120119962/55bc530e-cf75-462a-9cef-4ff6dbb2e1c6)

## Setup Instructions

### Prerequisites
- Python 3.6 or later
- pip package manager

### Notes:
- **requirements.txt:** Ensure your `requirements.txt` file lists all necessary libraries and versions required for the project. You can generate or update this file using `pip freeze > requirements.txt` after installing all dependencies.
  
- **Running the App:** The command `streamlit run app.py` starts the Streamlit application locally. Adjust `app.py` to match your actual Python script filename if different.
  
- **Web Interface:** Streamlit provides an intuitive web interface where users can interact with the application directly in their browser. Make sure to handle errors, exceptions, and user interactions gracefully within your Streamlit application logic.

![image](https://github.com/ArchitaWadhwa/Sangya_video_deepfake_detection/assets/120119962/5f2773c2-e855-426f-b461-83d5cc76f7cb)


