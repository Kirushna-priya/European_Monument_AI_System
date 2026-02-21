import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

EMBEDDING_PATH = "models/embeddings.npy"

def train_anomaly_detector():

    print("Loading embeddings...")
    embeddings = np.load(EMBEDDING_PATH)

    print(f"Embeddings shape: {embeddings.shape}")

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # assume 5% anomalies
        random_state=42
    )

    print("Training Isolation Forest...")
    iso.fit(embeddings)

    joblib.dump(iso, "models/anomaly_model.pkl")

    print("Anomaly model saved.")

if __name__ == "__main__":
    train_anomaly_detector()
