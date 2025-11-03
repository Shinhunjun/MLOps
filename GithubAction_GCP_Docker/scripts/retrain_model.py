#!/usr/bin/env python3
"""
Retrain MNIST CNN model with new feedback data from GCS
"""
import os
import sys
import argparse
import tempfile
import numpy as np
from google.cloud import storage
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_subset_from_gcs(bucket_name: str, sub_set_id: str, local_dir: str):
    """
    Download a specific subset from GCS

    Args:
        bucket_name: GCS bucket name
        sub_set_id: Subset identifier (e.g., "sub_set_5")
        local_dir: Local directory to download to
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    prefix = f"feedback_data/new_data/{sub_set_id}/"
    blobs = bucket.list_blobs(prefix=prefix)

    images = []
    labels = []

    for blob in blobs:
        if blob.name.endswith('.png'):
            # Download image
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

            # Extract label from filename (format: label_timestamp.png)
            filename = os.path.basename(blob.name)
            label = int(filename.split('_')[0])

            # Load image
            img = Image.open(local_path).convert('L')
            img_array = np.array(img) / 255.0  # Normalize to 0-1

            images.append(img_array)
            labels.append(label)

            logger.info(f"Loaded: {filename} (label: {label})")

    return np.array(images), np.array(labels)

def load_mnist_dataset():
    """Load original MNIST dataset"""
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test)

def build_cnn_model(input_shape=(28, 28, 1)):
    """Build CNN model architecture (same as original)"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def retrain_model(
    bucket_name: str,
    sub_set_id: str,
    output_dir: str,
    epochs: int = 5
):
    """
    Retrain model with new feedback data

    Args:
        bucket_name: GCS bucket name
        sub_set_id: Subset to train with
        output_dir: Directory to save trained model
        epochs: Number of training epochs
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download new data from GCS
        logger.info(f"Downloading {sub_set_id} from GCS...")
        new_images, new_labels = download_subset_from_gcs(
            bucket_name, sub_set_id, temp_dir
        )

        logger.info(f"Downloaded {len(new_images)} new samples")

        # Load original MNIST dataset
        (x_train, y_train), (x_test, y_test) = load_mnist_dataset()

        # Combine with new data
        logger.info("Combining with original dataset...")
        x_train_combined = np.concatenate([x_train, new_images])
        y_train_combined = np.concatenate([y_train, new_labels])

        # Reshape for CNN
        x_train_combined = x_train_combined.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        logger.info(f"Training dataset size: {len(x_train_combined)}")
        logger.info(f"New data contribution: {len(new_images)} samples")

        # Build model
        logger.info("Building model...")
        model = build_cnn_model()

        # Train
        logger.info(f"Training for {epochs} epochs...")
        history = model.fit(
            x_train_combined,
            y_train_combined,
            epochs=epochs,
            batch_size=128,
            validation_data=(x_test, y_test),
            verbose=1
        )

        # Evaluate
        logger.info("Evaluating model...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")

        # Save model as SavedModel format
        logger.info(f"Saving model to {output_dir}...")
        model.save(output_dir)

        # Save metadata
        metadata = {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "training_samples": int(len(x_train_combined)),
            "new_samples": int(len(new_images)),
            "sub_set_id": sub_set_id,
            "epochs": epochs
        }

        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Model training complete!")
        logger.info(f"Metadata: {metadata}")

        return metadata

def main():
    parser = argparse.ArgumentParser(description="Retrain MNIST model with new data")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--subset", required=True, help="Subset ID (e.g., sub_set_5)")
    parser.add_argument("--output-dir", required=True, help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Retrain
    metadata = retrain_model(
        bucket_name=args.bucket,
        sub_set_id=args.subset,
        output_dir=args.output_dir,
        epochs=args.epochs
    )

    logger.info("✅ Retraining complete!")

if __name__ == "__main__":
    main()
