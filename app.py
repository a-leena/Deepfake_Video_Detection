import streamlit as st
import tempfile
import model_builder
import os

BEST_MODEL_PATH = os.path.join('artifacts', 'best_mobilenetv2_embeddings_gru.keras')

st.title("Deepfake Video Detection App")

uploaded_video = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mov'])

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    temp_file.flush()
    
    st.video(uploaded_video)

    pred = model_builder.classifier(video_path=temp_file.name, 
                                         trained_classifier_path=BEST_MODEL_PATH)
    
    if pred=='REAL':
        st.success(f"Prediction: {pred}")
    else:
        st.error(f"Prediction: {pred}")

    st.info("Please refresh the app before uploading the next video.")