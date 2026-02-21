import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from src.classifier import build_classifier
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

IMG_SIZE = (224, 224)

# Load model weights once at startup
classifier = build_classifier(num_classes=10)
classifier.load_weights("models/classifier.weights.h5")
anomaly_model = joblib.load("models/anomaly_model.pkl")

feature_extractor = MobileNet(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

class_labels = [
    "alter",
    "apse",
    "bell-tower",
    "column",
    "dome(inner)",
    "dome(outer)",
    "flying-buttress",
    "gargoyle",
    "stained-glass",
    "vault"
]

def process_image(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Classification
    img_norm = img_array / 255.0
    predictions = classifier.predict(img_norm)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Anomaly detection
    img_pre = preprocess_input(img_array)
    embedding = feature_extractor.predict(img_pre)
    anomaly_score = anomaly_model.predict(embedding)

    CONFIDENCE_THRESHOLD = 0.6

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Uncertain"

    status = "Normal" if anomaly_score[0] == 1 else "Potential Structural Anomaly"

    return predicted_class, confidence, status

@app.route("/")
def home():
    return "European Monument AI System is running."


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid file type"}), 400

    file_path = "temp.jpg"
    file.save(file_path)
    predicted_class, confidence, status = process_image(file_path)


    logger.info(
    f"Prediction: monument={predicted_class}, "
    f"confidence={confidence:.4f}, "
    f"status={status}")
    
    os.remove(file_path)

    return jsonify({
        "monument": predicted_class,
        "confidence": confidence,
        "structural_status": status
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
