# First: import streamlit
import streamlit as st

# ✅ First Streamlit command
st.set_page_config(page_title="Pneumonia Detection", layout="centered")

# Other imports (after set_page_config is fine)
import tensorflow as tf
import numpy as np
from PIL import Image

# Title and instruction
st.title("🩻 Pneumonia Detection from Chest X-ray Images")
st.write("Upload a chest X-ray image and the model will predict if it indicates pneumonia or not.")

# Load model with cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_detection_model.h5")
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    class_names = ['Normal', 'Pneumonia']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader(f"🧠 Prediction: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
