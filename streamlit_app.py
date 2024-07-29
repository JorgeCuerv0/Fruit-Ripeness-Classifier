import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load the model
model_path = 'fruit_classifier_model.h5'  # Ensure the correct relative path
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    try:
        model = load_model(model_path)
        st.success("Model loaded successfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")

st.title("Fruit Ripeness Classifier")

def predict(image):
    image = image.resize((64, 64))  # Adjust the resize dimensions according to your model input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    st.write(f"Preprocessed image shape: {image.shape}")
    try:
        prediction = model.predict(image)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    if label is not None:
        st.write(f"Prediction: {label}")
