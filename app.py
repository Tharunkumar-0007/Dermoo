from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)

# Load the trained model
model = load_model('image_classifier_model.h5')

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the class names with integer keys
class_labels = {
    0: "Acne",
    1: "Eczema",
    2: "Psoriasis",
    3: "Melanoma",
    4: "Basal Cell Carcinoma",
    5: "Squamous Cell Carcinoma",
    6: "Fungal Infections",
    7: "Impetigo",
    8: "Dermatitis",
    9: "Urticaria",
    10: "Tinea",
    11: "Vitiligo",
    12: "Actinic Keratosis",
    13: "Folliculitis",
    14: "Hives",
    15: "Cellulitis",
    16: "Lichen Planus",
    17: "Contact Dermatitis",
    18: "Seborrheic Dermatitis",
    19: "Rosacea",
    20: "Atopic Dermatitis",
    21: "Warts, Molluscum, and some other viral infections",
    22: "Melanocytic Nevus",
    23: "Benign Keratosis",
    24: "Lichen",
    25: "AIDS"
}

# Set up ImageDataGenerator for real-time data augmentation
datagen = ImageDataGenerator(rescale=1./255)

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_probability = np.max(predictions)

        threshold = 0.6

        if predicted_probability < threshold:
            predicted_label = 'Healthy Skin or Not a Valid Disease Image'
        else:
            predicted_label = class_labels.get(predicted_class, 'Unknown')

        return jsonify({'predicted_class': predicted_label}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
