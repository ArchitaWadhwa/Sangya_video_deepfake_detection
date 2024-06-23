# #AuthentiCheck

## Video and Audio Deepfake Detection Project

### Project Overview
This project implements a system for detecting deepfake videos and audio using pretrained models. It provides functionalities for video and audio analysis and livestream video processing. The application utilizes Streamlit for the web interface, PyAV for video handling, and PyTorch for video deepfake detection.

### Features
- **Deepfake Video Detection:** Upload a video to check for deepfake content using a pretrained VideoMAE model.
- **Deepfake Audio Detection:** Upload an audio file to analyze for deepfake content using an audio-based detection model.
- **Livestream Video Analysis:** Perform real-time video analysis for deepfake detection using a webcam.

## Setup Instructions

### Prerequisites
- Python 3.6 or later
- pip package manager

### Notes:
- **requirements.txt:** Ensure your `requirements.txt` file lists all necessary libraries and versions required for the project. You can generate or update this file using `pip freeze > requirements.txt` after installing all dependencies.
  
- **Running the App:** The command `streamlit run app.py` starts the Streamlit application locally. Adjust `app.py` to match your actual Python script filename if different.
  
- **Web Interface:** Streamlit provides an intuitive web interface where users can interact with the application directly in their browser. Make sure to handle errors, exceptions, and user interactions gracefully within your Streamlit application logic.


