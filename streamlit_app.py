import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model_path = 'fruit_classifier_model.h5'  # Ensure the correct relative path
model = load_model(model_path)

st.title("Fruit Ripeness Classifier")

def predict(image):
    image = image.resize((224, 224))  # Adjust this size based on your model's expected input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    st.write(f"Preprocessed image shape: {image.shape}")  # Debugging statement
    prediction = model.predict(image)
    return prediction

st.write(f"Model input shape: {model.input_shape}")  # Debugging statement

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    st.write(f"Prediction: {label}")
