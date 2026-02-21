import gradio as gr
import numpy as np
import joblib
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from src.classifier import build_classifier

IMG_SIZE = (224, 224)

# Load models
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

CONFIDENCE_THRESHOLD = 0.6


def analyze_image(img):

    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Classification
    img_norm = img_array / 255.0
    predictions = classifier.predict(img_norm)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Uncertain"

    # Anomaly detection
    img_pre = preprocess_input(img_array)
    embedding = feature_extractor.predict(img_pre)
    anomaly_score = anomaly_model.predict(embedding)

    status = "Normal" if anomaly_score[0] == 1 else "Potential Structural Anomaly"

    return predicted_class, confidence, status


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏛 European Monument AI System  
    **Transfer Learning + Anomaly Detection for Structural Assessment**
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Monument Image")
            analyze_btn = gr.Button("Analyze")

        with gr.Column():
            monument_output = gr.Textbox(label="Predicted Monument")
            confidence_output = gr.Slider(
                minimum=0,
                maximum=1,
                step=0.01,
                label="Confidence",
                interactive=False
            )
            status_output = gr.Textbox(label="Structural Status")

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_input,
        outputs=[monument_output, confidence_output, status_output]
    )

    gr.Markdown("""
    ---
    **Model Details**
    - Backbone: MobileNet (Transfer Learning)
    - Classifier Accuracy: ~93–94%
    - Anomaly Detection: Isolation Forest
    - Deployment: Docker + Flask + Gradio
    """)

if __name__ == "__main__":
    demo.launch()