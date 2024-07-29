import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model_path = 'fruit_classifier_model.h5'  # Ensure the correct relative path
model = load_model(model_path)

st.title("Fruit Ripeness Classifier")

def predict(image):
    image = image.resize((64, 64))  # Resize to 64x64
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    st.write(f"Preprocessed image shape: {np.array(image).shape}")
    label = predict(image)
    st.write(f"Prediction: {label}")
