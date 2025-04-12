
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
import gdown
if not os.path.exists("melanoma_model.h5"):
    gdown.download("https://drive.google.com/uc?id=1TfTkLFHPNhw7woNCIwImSqUiUwQa2osF", "melanoma_model.h5", quiet=False)

model = load_model("melanoma_model.h5")

class_names = ['actinic keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus',
               'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion']

@app.route("/")
def home():
    return "Melanoma Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return jsonify({"prediction": predicted_class, "confidence": confidence})
