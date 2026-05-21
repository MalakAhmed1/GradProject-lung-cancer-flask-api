from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = "efficientnet_b1.keras"
class_names = ["normal", "lung_adenocarcinoma", "lscc"]

model = load_model(MODEL_PATH, compile=False)
print("✅ EfficientNet-B1 model loaded")

@app.route("/", methods=["GET"])
def home():
    return '''
    <h2>Lung Disease Classifier (EfficientNet-B1)</h2>
    <form method="POST" action="/predict" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Predict">
    </form>
    '''

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    temp_path = "temp_image.jpg"
    file.save(temp_path)

    image = load_img(temp_path, target_size=(240, 240), color_mode="rgb")
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    prediction = model.predict(image, verbose=0)
    predicted_index = int(np.argmax(prediction[0]))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction[0]))

    os.remove(temp_path)

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "model": "EfficientNet-B1"
    })

if __name__ == "__main__":
    app.run(port=5000)
