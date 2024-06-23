# #AuthentiCheck

## Video and Audio Deepfake Detection Project

## Overview

This project implements a system for detecting deepfake videos and audio using pretrained models. It provides functionalities for video and audio analysis and livestream video processing. The application utilizes Streamlit for the web interface, PyAV for video handling, PyTorch for video deepfake detection, and other necessary libraries.

## Features

### Deepfake Video Detection

1. **Upload a Video File:**
   - Use the provided file upload widget to upload a video file (.mp4, .mov).

2. **Analyze Video for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded video for deepfake content.
   - The application uses a pretrained VideoMAE model to perform the analysis.

3. **View Results:**
   - Displayed results include the predicted label ('REAL' or 'FAKE') and the confidence score indicating the model's prediction certainty.

### Deepfake Audio Detection

1. **Upload an Audio File:**
   - Use the file upload widget to upload an audio file (.wav, .mp3).

2. **Analyze Audio for Deepfake Content:**
   - Click the "Check" button to analyze the uploaded audio file for deepfake content.
   - The application utilizes an audio-based detection model for this analysis.

3. **View Analysis Results:**
   - Results provide insights into the authenticity of the audio file, indicating if it contains suspicious content.

### Livestream Video Analysis

1. **Access Webcam for Real-Time Analysis:**
   - Navigate to the "Livestream Video" section.
   - Grant permissions to allow the application to access your webcam.

2. **Perform Real-Time Deepfake Detection:**
   - The application starts streaming video from your webcam.
   - Real-time analysis is performed to detect deepfake content as the video streams.
## Setup Instructions

### Prerequisites
- Python 3.6 or later
- pip package manager

### Notes:
- **requirements.txt:** Ensure your `requirements.txt` file lists all necessary libraries and versions required for the project. You can generate or update this file using `pip freeze > requirements.txt` after installing all dependencies.
  
- **Running the App:** The command `streamlit run app.py` starts the Streamlit application locally. Adjust `app.py` to match your actual Python script filename if different.
  
- **Web Interface:** Streamlit provides an intuitive web interface where users can interact with the application directly in their browser. Make sure to handle errors, exceptions, and user interactions gracefully within your Streamlit application logic.


