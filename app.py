import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from huggingface_hub import from_pretrained_keras
import torch

# Set page configuration
st.set_page_config(
    page_title="Deteksi Tuberkulosis",
    page_icon="🩺",
    layout="centered",
)

# Load your trained model
@st.cache_resource
def load_my_model():
    model = from_pretrained_keras('reaim70/tbclassification')  # Ganti dengan path model Anda
    return model

model = load_my_model()

# Fungsi prediksi menggunakan model yang dimuat
def predict(image):
    # Preprocess the image to match the input shape the model expects
    img = image.resize((224, 224))  # Ganti ukuran sesuai dengan input model Anda
    img_array = np.array(img)

    # Ensure the image has 3 channels (RGB)
    if img_array.shape[-1] == 3:  # Already RGB
        img_array = preprocess_input(img_array)
    else:  # Grayscale or other format
        img_array = np.stack((img_array,)*3, axis=-1)  # Convert to RGB
        img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)

    # Assuming model output is probability, choose the class with highest probability
    if prediction[0][0] > prediction[0][1]:
        return "Normal"
    else:
        return "Tuberkulosis"

# Title of the web app
st.title("Deteksi Tuberkulosis")

# Center the icon representation
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 20px;'>
        <i class="fas fa-lungs" style='font-size: 100px; color: grey;'></i>
    </div>
    """, unsafe_allow_html=True
)

# Container for upload and results
with st.container():
    # File uploader
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Columns for image preview and detection button
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Gambar yang diupload', use_column_width=True)

        with col2:
            if st.button("Deteksi"):
                # Display loading message
                with st.spinner('Mendeteksi...'):
                    # Prediction using the loaded model
                    result = predict(image)
                
                # Display result below the button
                st.subheader("Hasil Deteksi")
                if result == "Normal":
                    st.success("Terdeteksi Normal")
                else:
                    st.error("Terdeteksi Tuberkulosis")