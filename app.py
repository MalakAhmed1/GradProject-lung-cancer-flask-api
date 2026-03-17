from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
import requests

app = Flask(__name__)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1LuYtsmDUP-e4mO_o334WzUSrcgwcPZfP"
MODEL_PATH = "best_lung_model.keras"

if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

model = load_model(MODEL_PATH)

class_names = [
    "normal",
    "lung_adenocarcinoma",
    "lscc"
]

@app.route("/", methods=["GET"])
def home():
    return '''
    <h2>Upload CT Image</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    temp_path = "temp_image.jpg"
    file.save(temp_path)

    image = load_img(temp_path, target_size=(224, 224), color_mode="rgb")
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]))

    os.remove(temp_path)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=5000)