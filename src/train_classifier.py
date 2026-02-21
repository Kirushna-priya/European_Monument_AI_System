from clean_dataset import clean_directory
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from classifier import build_classifier
from utils import setup_logger

# Setup logger
logger = setup_logger()
logger.info("Starting training pipeline...")

TRAIN_PATH = "data/train"
TEST_PATH = "data/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

clean_directory("data/train")
clean_directory("data/test")

logger.info("Dataset cleaned.")

 # Data generators
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_generator.num_classes

logger.info(f"Number of classes detected: {num_classes}")

model = build_classifier(num_classes)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "models/classifier.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True
)

logger.info("Beginning model training...")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

logger.info("Training complete.")
logger.info("Best weights saved to models/classifier.weights.h5")