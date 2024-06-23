
import av
import torch
import numpy as np
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from torch.nn.functional import softmax
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode, VideoProcessorBase, ClientSettings
import cv2
from ultralytics import YOLO
import ffmpeg
import pyaudio
import wave
import threading
import tempfile
import streamlit_authenticator as stauth
from pathlib import Path
import pickle
from st_audiorec import st_audiorec

np.random.seed(0)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (int): Total number of frames to sample.
        frame_sample_rate (int): Sample every n-th frame.
        seg_len (int): Maximum allowed index of sample's last frame.
    Returns:
        indices (List[int]): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def classify_video_deepfake(file_path):
    '''
    Classify a video for DeepFake using a pretrained VideoMAE model.
    Args:
        file_path (str): Path to the input video file.
    Returns:
        predicted_label (str): Predicted label for the video (either 'REAL' or 'FAKE').
        confidence (float): Confidence score associated with the predicted label.
    '''
    # Open the video file
    container = av.open(file_path)

    # Sample 16 frames
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
    video = read_video_pyav(container, indices)

    # Load model and processor
    processor = AutoImageProcessor.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")
    model = VideoMAEForVideoClassification.from_pretrained("Ammar2k/videomae-base-finetuned-deepfake-subset")

    # Process video frames
    inputs = processor(list(video), return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Calculate probabilities using softmax
    probs = softmax(logits, dim=-1)

    # Get the predicted label and its confidence
    predicted_label_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_id]
    confidence = probs[0, predicted_label_id].item()

    return predicted_label, confidence


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames = []
        self.audio_frames = []
        self.recording = False
        self.audio_thread = None

    def start_recording(self):
        self.recording = True
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()

    def stop_recording(self):
        self.recording = False
        if self.audio_thread is not None:
            self.audio_thread.join()

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        results = model(frm)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 3)

        self.frames.append(frm)
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

    def record_audio(self):
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100
        chunk = 1024
        audio = pyaudio.PyAudio()

        stream = audio.open(format=audio_format, channels=channels,
                            rate=rate, input=True, frames_per_buffer=chunk)

        while self.recording:
            data = stream.read(chunk)
            self.audio_frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

    def save_audio(self, filepath):
        audio_format = pyaudio.paInt16
        channels = 1
        rate = 44100

        wf = wave.open(filepath, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()

    def save_video(self, filepath):
        if self.frames:
            height, width, _ = self.frames[0].shape
            out = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
            for frame in self.frames:
                out.write(frame)
            out.release()


def audiorec_demo_app():
    st.markdown("Livestream Video")
    st.write('\n\n')
    wav_audio_data = st_audiorec()  # Assuming st_audiorec is properly configured for Streamlit

    if wav_audio_data is not None:
        col_playback, col_space = st.columns([0.58, 0.42])
        with col_playback:
            st.audio(wav_audio_data, format='audio/wav')


def main():
    st.set_page_config(page_title="AuthentiCheck")

    css = """
    <style>
    .main {
        background-color: #4d4855;
        background-image: linear-gradient(147deg, #4d4855 0%, #000000 74%);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    credentials = {
        "usernames": {
            "itsvishnu25": {
                "name": "Vishnu",
                "password": "candycrush"
            },
        }
    }

    authenticator = stauth.Authenticate(credentials, "deepfake", "auth", cookie_expiry_days=30)
    name, authentication_status, username = authenticator.login("main", fields={'Form name': 'Login'})

    if authentication_status == False:
        st.error("Username/password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your username and password")

    if authentication_status:
        model = YOLO("/kaggle/input/yolomodel/yolov8n-face.pt")

        ctx = webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=VideoProcessor,
        )

        audiorec_demo_app()

        st.sidebar.title('AuthentiCheck')
        st.sidebar.text("@ your service")
        st.sidebar.title('Page Selection Menu')

        page = st.sidebar.radio("Select Page:", ("Detect Deepfake Video", "Detect Deepfake Audio", "Livestream Video"))

        if page == "Detect Deepfake Video":
            st.title('AuthentiCheck')
            st.markdown("Deepfake Video Detection")
            st.info("Select Page from Sidebar to the left")
            st.markdown("Upload the video and verify its authenticity")
            st.write("")
            sample_images = st.expander("Sample Videos:", expanded=False)
            col1, col2 = sample_images.columns(2)
            st.write("")
            st.write("")
            video = st.file_uploader("Upload Video:", type=['mp4', 'mov'])
            st.write("")
            st.write("")
            
            if st.button('Check'):
                if video is None:
                    st.error("Please upload a valid video first")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                        temp_file.write(video.read())
                        temp_file_path = temp_file.name
                        predicted_label, confidence = classify_video_deepfake(temp_file_path)
                        capitalized_label = predicted_label.capitalize()
                        st.markdown(f'<h2>Label: {capitalized_label}</h2>', unsafe_allow_html=True)
                        st.markdown(f'<h2>Confidence: {confidence:.3f}</h2>', unsafe_allow_html=True)

        elif page == "Detect Deepfake Audio":
            st.title('AuthentiCheck')
            st.markdown("Audio Detection")
            st.info("Detect audio in the selected video.")
            st.markdown("Upload a video to check for deepfakes")
            st.write("")
            st.write("")
            if st.button('Check'):
                if st.sidebar.checkbox('Upload Video'):
                    if video is None:
                        st.error("Please upload a valid video first")
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                            temp_file.write(video.read())
                            temp_file_path = temp_file.name
                            predicted_label, confidence = classify_video_deepfake(temp_file_path)
                            capitalized_label = predicted_label.capitalize()
                            st.markdown(f'<h2>Label: {capitalized_label}</h2>', unsafe_allow_html=True)
                            st.markdown(f'<h2>Confidence: {confidence:.3f}</h2>', unsafe_allow_html=True)
                            
        elif page == "Livestream Video":
            st.title('AuthentiCheck')
            st.markdown("Livestream Video")
            st.info("Video will stream live")

    return name, authentication_status, username

if __name__ == "__main__":
    main()
