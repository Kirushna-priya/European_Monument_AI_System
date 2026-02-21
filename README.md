# 🏛 European Monument Structural AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**An end-to-end computer vision system for architectural structure classification and embedding-based structural anomaly detection.**

Live Demo:  
https://huggingface.co/spaces/Kirushnapriya/monument-structural-ai

---

## 🔍 Overview

This project implements a modular, production-oriented computer vision pipeline that:

- Classifies 10 European architectural structures  
- Achieves ~94% validation accuracy using transfer learning (`MobileNet`)  
- Extracts deep visual embeddings  
- Detects structural anomalies using `IsolationForest`  
- Is containerized with Docker  
- Provides both API-based and interactive UI deployment  

The anomaly detection component operates as an **out-of-distribution detector over learned embeddings**, making it practical when labeled damage datasets are limited.

---

## 🏗 Architectural Classes

The classifier predicts the following structures:

- Vault  
- Stained Glass  
- Gargoyle  
- Flying Buttress  
- Dome (Outer)  
- Dome (Inner)  
- Column  
- Bell Tower  
- Apse  
- Alter  

---

## 🧠 System Architecture

### High-Level Flow

```text
                ┌──────────────────────────┐
                │        Raw Images        │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  Dataset Cleaning Layer  │
                │ - Remove corrupt files   │
                │ - Resize & normalize     │
                └─────────────┬────────────┘
                              │
                              ▼
                ┌──────────────────────────┐
                │  MobileNet Backbone      │
                │ (Pretrained on ImageNet) │
                └─────────────┬────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
    ┌────────────────────┐       ┌────────────────────────┐
    │ Classification Head │       │  Embedding Extractor   │
    │ Dense + Softmax     │       │  (Global Avg Pooling)  │
    └──────────┬──────────┘       └──────────┬─────────────┘
               │                             │
               ▼                             ▼
    ┌────────────────────┐       ┌────────────────────────┐
    │ Monument Prediction │       │ Isolation Forest Model │
    │ + Confidence Score  │       │  (Anomaly Detection)   │
    └──────────┬──────────┘       └──────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                ┌──────────────────────────┐
                │  Deployment Layer        │
                │  - Flask API             │
                │  - Gradio UI             │
                │  - Docker Container      │
                └──────────────────────────┘

```

The system is divided into clear, modular components:

**1. Data Processing Layer**

- Detects and removes corrupted / truncated image files
- Normalizes input images (224x224)
- Ensures training stability


**2. Feature Backbone**

- MobileNet pretrained on ImageNet
- Base layers frozen during training
- Used both for classification and embedding extraction
- This ensures consistency between prediction and anomaly modeling.


**3. Classification Module**

- Custom dense head
- Softmax output for 10 architectural classes
- Early stopping and checkpointing applied

Output:
- Predicted structure
- Confidence score


**4. Embedding-Based Anomaly Detection**

- Global average pooled embeddings extracted
- Isolation Forest trained only on clean samples
- Detects out-of-distribution embeddings
- This functions as a structural anomaly indicator when labeled damage data is unavailable.


**5. Deployment Layer**

Two deployment interfaces:

- API Layer
- Flask
- Gunicorn
- Dockerized container
- Interactive Demo
- Gradio
- Hosted on Hugging Face Spaces

---
## Design Philosophy

### This system was intentionally built with:

- Clear separation of training and inference
- Modular architecture
- Reproducibility via Docker
- Deployment-first mindset
- Embedding-level reasoning instead of black-box outputs

---
## Repository Structure

```
.
├── src/
│   ├── classifier.py
│   ├── clean_dataset.py
│   └── utils.py
├── models/
│   ├── classifier.weights.h5
│   └── anomaly_model.pkl
├── app/
│   └── app.py
├── gradio_app.py
├── requirements.txt
├── Dockerfile
└── README.md
```
---
## Training Details

- Image Size: 224x224
- Batch Size: 32
- Optimizer: Adam
- Epochs: 20 (with early stopping)
- Validation Accuracy: ~93–94%
- Loss Function: Categorical Crossentropy
- Corrupted and truncated images were detected and removed during preprocessing to ensure training stability.

---
## How to Run Locally

**1. Create Virtual Environment**

```python
python -m venv .venv
source .venv/bin/activate
```

**2. Install Dependencies**
```python
pip install -r requirements.txt
```

**3. Run Gradio UI**
```python
python gradio_app.py
```

**4. Run Flask API**
```python
python app/app.py
```

**Docker Deployment**

**Build image:**
```
docker build -t monument-ai .
```

**Run container:**
```
docker run -p 5000:5000 monument-ai
```
**Health check:**
```
curl http://127.0.0.1:5000/health
```
**Prediction endpoint:**
```
curl -X POST -F "file=@image.jpg" http://127.0.0.1:5000/predict
```

## Design Decisions

- Transfer learning used to leverage pretrained visual representations.
- Base network frozen to reduce overfitting and training time.
- Embedding-based anomaly detection selected due to limited labeled defect data.
- Modular structure to separate training, inference, and UI layers.
- Docker used to ensure reproducibility and deployment consistency.

## Limitations

- Anomaly detection functions as an out-of-distribution detector, not a supervised damage classifier.
- Subtle structural damage may not significantly shift embedding space.
- Performance depends on dataset diversity and image quality.


## Future Work

- Fine-tuning deeper MobileNet layers for better feature adaptation
- Structural damage segmentation (pixel-level localization)
- Grad-CAM visualization for interpretability
- Supervised damage classification with labeled defect data
- MLOps pipeline automation

## Tech Stack

- TensorFlow / Keras
- MobileNet
- Scikit-learn (Isolation Forest)
- Flask
- Gradio
- Docker
- Hugging Face Spaces


## Key Engineering Challenges & Resolutions


### 1. Corrupted & Truncated Image Files

 **Challenge**
 During training, multiple images in the dataset were either corrupted or partially downloaded, causing runtime crashes (e.g., image file is truncated errors).

**Resolution**
- Implemented dataset cleaning pipeline to validate images before training
- Removed corrupted files programmatically
- Enabled truncated image loading using:
```
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```

This stabilized the training process and ensured reproducibility.


### 2. Model Serialization & Version Conflicts

**Challenge**
- Loading saved .h5 models inside Docker resulted in deserialization errors (e.g., Unrecognized keyword arguments passed to Dense: {'quantization_config': None}).
- This was caused by mismatched TensorFlow/Keras versions between training and inference environments.

**Resolution**
- Switched from full model saving to weight-only saving (.weights.h5)
- Reconstructed architecture programmatically during inference
- Aligned TensorFlow and scikit-learn versions explicitly in requirements.txt
- Rebuilt Docker image after dependency normalization
- This improved environment consistency and reduced hidden serialization risk.

### 3. Anomaly Detection Interpretation

**Challenge**
- Isolation Forest does not perform semantic damage detection. It detects statistical outliers in embedding space.
- Subtle structural damage does not always shift embeddings significantly.

**Resolution**
- Trained anomaly model strictly on clean samples
- Clarified that anomaly detection functions as an out-of-distribution detector
- Added confidence thresholding in inference pipeline
- Documented limitations explicitly in README
- This ensures transparency and sets correct expectations for system behavior.

### 4. Dependency Hash & Build Issues in Docker

**Challenge**

Docker builds failed due to:
- Python version incompatibilities
- scikit-learn version constraints
- Hash mismatches in package resolution

**Resolution**
- Removed unnecessary strict hash enforcement
- Explicitly pinned compatible versions
- Upgraded pip during Docker build
- Cleaned Docker cache to remove corrupted layers
- This stabilized containerized deployment.

### 5. Separation of Training and Inference Logic

**Challenge**
- Training code, model logic, and deployment logic were initially tightly coupled.
- This complicates maintenance and reproducibility.

**Resolution**
- Refactored project into modular structure (src/, models/, app/)
- Separated classifier construction from training script
- Ensured inference builds model architecture before loading weights
- This enables clean extension and scalability.

### Author

**Kirushnapriya**
**AI Engineering**
