import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications import MobileNet

IMG_SIZE = (224, 224)

# Load models
classifier = load_model("models/classifier.h5")
anomaly_model = joblib.load("models/anomaly_model.pkl")

# Load feature extractor
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

def predict_image(img_path):

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Classification
    img_array_norm = img_array / 255.0
    predictions = classifier.predict(img_array_norm)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Anomaly detection
    img_array_pre = preprocess_input(img_array)
    embedding = feature_extractor.predict(img_array_pre)
    anomaly_score = anomaly_model.predict(embedding)

    status = "Normal" if anomaly_score[0] == 1 else "Potential Structural Anomaly"

    return {
        "monument": predicted_class,
        "confidence": confidence,
        "structural_status": status
    }

if __name__ == "__main__":
    result = predict_image("data/test/vault/34670722563_a7efda5e61_m.jpg")
    print(result)
