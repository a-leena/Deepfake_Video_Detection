import streamlit as st
import cv2
import numpy as np
import tempfile

st.title("Deepfake Video Detection App")

uploaded_video = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mov'])

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    st.video(temp_file.name)

    