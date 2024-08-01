from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import requests

app = Flask(__name__)

# Load the model
model_path = 'fruit_classifier_model.h5'
model = load_model(model_path)

def predict(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    print(f"Raw prediction: {prediction}")  # Debugging statement
    label = int(np.argmax(prediction, axis=1)[0])
    print(f"Predicted label: {label}")  # Debugging statement
    return "ripe" if label == 1 else "unripe"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        image = Image.open(file.stream)
        label = predict(image)
        return jsonify({'prediction': label})
    return jsonify({'error': 'File upload failed'})

@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json()
    image_url = data['imageUrl']
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    label = predict(image)
    return jsonify({'prediction': label})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    correct = data['correct']
    # Handle feedback here
    return jsonify({'message': 'Feedback received'})

if __name__ == '__main__':
    app.run(debug=True)
