import streamlit as st
from PIL import Image, ImageChops
import numpy as np
from tensorflow.keras.applications import mobilenet_v3
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

preprocess_input = mobilenet_v3.preprocess_input

# Set page configuration
st.set_page_config(
    page_title="Klasifikasi Tuberkulosis",
    page_icon="🩺",
    layout="centered",
)

# Load your trained model
@st.cache_resource
def load_my_model():
    # Download the MobileNetV3 model from the Hugging Face Hub
    model_path = hf_hub_download(repo_id="reaim70/MobileNetV3.keras", filename="MobileNetV3.keras")
    model = load_model(model_path)
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
        img_array = np.stack((img_array,) * 3, axis=-1)  # Convert to RGB
        img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)

    # Assuming model output is probability, choose the class with highest probability
    if prediction[0][0] > prediction[0][1]:
        return "Normal"
    else:
        return "Tuberkulosis"

def is_chest_xray(image):
    # Check image dimensions and aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    
    # Typical chest X-ray aspect ratio is approximately 0.7 - 1.3
    if not (0.7 < aspect_ratio < 1.3):
        return False
    
    # Convert the image to grayscale and check if it's nearly grayscale
    grayscale_image = image.convert("L")
    diff = ImageChops.difference(image.convert("RGB"), grayscale_image.convert("RGB"))
    
    # Check the difference (whether the image is close to grayscale)
    diff_stat = diff.getbbox()
    
    # If there is no bounding box (None), the image is already grayscale
    if diff_stat is None:
        return True
    
    return False

# Title of the web app
st.title("Klasifikasi Tuberkulosis")

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
                        st.warning("Bukan Citra Rontgen Dada")
                    else:
                        # Prediction using the loaded model
                        result = predict(image)
                        
                        # Display result below the button
                        st.subheader("Hasil Klasifikasi")
                        if result == "Normal":
                            st.success("Terklasifikasi Normal")
                        else:
                            st.error("Terklasifikasi Tuberkulosis")