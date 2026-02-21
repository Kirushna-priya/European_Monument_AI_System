import numpy as np
import os
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tqdm import tqdm

IMG_SIZE = (224, 224)
TRAIN_PATH = "data/train"

def extract_embeddings():

    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    embeddings = []
    labels = []

    for class_name in os.listdir(TRAIN_PATH):

        class_folder = os.path.join(TRAIN_PATH, class_name)

        if not os.path.isdir(class_folder):
            continue

        for img_name in tqdm(os.listdir(class_folder)):

            img_path = os.path.join(class_folder, img_name)

            try:
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                feature = base_model.predict(img_array, verbose=0)
                embeddings.append(feature.flatten())
                labels.append(class_name)

            except Exception:
                continue

    embeddings = np.array(embeddings)

    np.save("models/embeddings.npy", embeddings)

    print("Embeddings saved.")
    return embeddings

if __name__ == "__main__":
    extract_embeddings()
