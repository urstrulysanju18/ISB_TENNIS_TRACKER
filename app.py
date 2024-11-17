import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
import os
import time
from pathlib import Path

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
model_path = 'best.pt'  # Replace with your actual .pt file path

try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    # st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e

st.set_page_config(page_title="Tennis Game Tracking", layout="wide")

st.title("Tennis Ball Tracking")

# Streamlit Sidebar for fancy, engaging user instructions
st.sidebar.title("Menu")
selected_option = st.sidebar.radio("Choose an option:", ["Upload and Preview Video", "Process Video"])

st.markdown(
    """
    <style>
    .sidebar .stRadio > label {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
    .sidebar .stRadio div[role='radiogroup'] > label {
        display: block;
        margin-bottom: 10px;
        padding: 10px 20px;
        border-radius: 10px;
        background-color: #e7e7e7;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .sidebar .stRadio div[role='radiogroup'] > label:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# # File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video:
    st.session_state.uploaded_video_name = Path(uploaded_video.name).stem

if selected_option == "Upload and Preview Video":
    if uploaded_video:
        st.session_state.show_input_video = True
        st.subheader("Input Video Preview:")
        st.video(uploaded_video)

# if uploaded_video is not None:
elif selected_option == "Process Video" and uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    processing_message = st.empty()
    processing_message.write("Video Under Processing...")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model (mixed precision if CUDA is available)
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())  # Draw detection boxes on the frame

        # Write frame to output video file
        out.write(frame)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_column_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate in display
        time.sleep(1 / fps)

    # Release video resources
    cap.release()
    out.release()
    
    processing_message.empty()

    st.success("Video processing complete! Click 'Download' button to start download...")

    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name=f"{st.session_state.uploaded_video_name}_output.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)

