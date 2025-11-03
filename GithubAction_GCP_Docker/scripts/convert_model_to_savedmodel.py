#!/usr/bin/env python3
"""
Convert Keras .h5 models to TensorFlow SavedModel format for Vertex AI
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
import argparse

def convert_h5_to_savedmodel(h5_path, output_dir):
    """
    Convert a Keras .h5 model to SavedModel format

    Args:
        h5_path: Path to the .h5 model file
        output_dir: Directory to save the converted model
    """
    print(f"Loading model from {h5_path}...")
    model = keras.models.load_model(h5_path)

    print(f"Model loaded successfully!")
    print(f"Model summary:")
    model.summary()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save as SavedModel format
    print(f"\nSaving model to {output_dir}...")
    tf.saved_model.save(model, output_dir)

    print(f"✅ Model successfully converted and saved to {output_dir}")

    # Verify the saved model
    print("\nVerifying saved model...")
    loaded_model = tf.saved_model.load(output_dir)
    print("✅ Model verified successfully!")

    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Convert Keras .h5 model to SavedModel format")
    parser.add_argument("--h5-path", required=True, help="Path to the .h5 model file")
    parser.add_argument("--output-dir", required=True, help="Output directory for SavedModel")

    args = parser.parse_args()

    if not os.path.exists(args.h5_path):
        print(f"❌ Error: Model file not found at {args.h5_path}")
        sys.exit(1)

    convert_h5_to_savedmodel(args.h5_path, args.output_dir)

if __name__ == "__main__":
    main()
