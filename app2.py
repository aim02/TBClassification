import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_input_efficientnet
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# Set page configuration
st.set_page_config(
    page_title="Klasifikasi Rontgen Dada",
    page_icon="🩺",
    layout="centered",
)

# Load detection model for chest X-ray vs non-chest X-ray
def load_detection_model():
    # Load EfficientNetV2B0 model for detection
    model_path = '1xray_vs_non_xray_efficientnetv2b0_model.keras'
    model = load_model(model_path)
    return model

detection_model = load_detection_model()

# Load MobileNetV3 model for classification (Normal vs Tuberkulosis)
@st.cache_resource
def load_my_model():
    # Download the MobileNetV3 model from the Hugging Face Hub
    model_path = hf_hub_download(repo_id="reaim70/MobileNetV3.keras", filename="MobileNetV3.keras")
    model = load_model(model_path)
    return model

model = load_my_model()

# Fungsi prediksi menggunakan model klasifikasi
def predict(image):
    # Preprocess the image to match the input shape the EfficientNetV2B0 model expects
    img = image.resize((224, 224))  # Ganti ukuran sesuai dengan input model Anda
    img_array = np.array(img)
    
    # Ensure the image has 3 channels (RGB)
    if img_array.ndim == 2:  # If grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # If RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input_efficientnet(img_array)  # Preprocessing
    
    prediction = model.predict(img_array)
    return 'Normal' if prediction[0][0] > prediction[0][1] else 'Tuberkulosis'

def is_chest_xray(image):
    # Preprocess the image to match the input shape the detection model expects
    img = image.resize((224, 224))  # Resize to match the input size of the detection model
    img_array = np.array(img)

    # Convert grayscale or RGBA to RGB if necessary
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Perform detection
    results = detection_model.predict(img_array)
    
    # Example check, assuming results contain a score or classification
    detection_score = results[0][0]  # Adjust based on your model output
    return detection_score > 0.5  # Example threshold

# Title of the web app
st.title("Klasifikasi Rontgen Dada")

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
    uploaded_file = st.file_uploader("Upload Gambar Rontgen Dada", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Columns for image preview and detection button
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption='Gambar yang diupload', use_column_width=True)

        with col2:
            if st.button("Klasifikasi"):
                # Display loading message
                with st.spinner('Mengklasifikasi...'):
                    # Check if the uploaded image is likely a chest X-ray
                    if not is_chest_xray(image):
                        st.warning("Gambar yang diunggah mungkin bukan rontgen dada. Pastikan Anda mengunggah rontgen dada.")
                    else:
                        # Prediction using the loaded EfficientNetV2B0 model
                        result = predict(image)
                        
                        # Display result below the button
                        st.subheader("Hasil Klasifikasi")
                        if result == "Normal":
                            st.success("Gambar terklasifikasi sebagai Normal")
                        else:
                            st.error("Gambar terklasifikasi sebagai Tuberkulosis")