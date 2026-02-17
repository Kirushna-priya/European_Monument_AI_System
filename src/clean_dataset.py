import os
from PIL import Image

def clean_directory(directory):

    print(f"Scanning directory: {directory}")

    for root, dirs, files in os.walk(directory):
        for file in files:

            file_path = os.path.join(root, file)

            # Remove non-image files
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Removing non-image file: {file_path}")
                os.remove(file_path)
                continue

            # Check corrupted images
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

    print("Cleaning complete.")

if __name__ == "__main__":
    clean_directory("data/train")
    clean_directory("data/test")
