#!/usr/bin/env python3
"""
Upload TensorFlow SavedModel to Vertex AI Model Registry
"""
import argparse
import os
from google.cloud import aiplatform
from google.cloud import storage

def upload_model_to_vertex_ai(
    project_id: str,
    region: str,
    model_name: str,
    saved_model_dir: str,
    gcs_bucket: str,
    description: str = "MNIST CNN Model"
):
    """
    Upload a SavedModel to Vertex AI Model Registry

    Args:
        project_id: GCP project ID
        region: GCP region
        model_name: Name for the model in Vertex AI
        saved_model_dir: Local path to SavedModel directory
        gcs_bucket: GCS bucket name (without gs://)
        description: Model description
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Step 1: Upload model to GCS
    print(f"📦 Uploading model to GCS bucket: gs://{gcs_bucket}/models/...")
    gcs_model_path = upload_to_gcs(saved_model_dir, gcs_bucket)
    print(f"✅ Model uploaded to {gcs_model_path}")

    # Step 2: Register model in Vertex AI Model Registry
    print(f"\n🚀 Registering model '{model_name}' in Vertex AI Model Registry...")

    try:
        # Upload model to Vertex AI
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=gcs_model_path,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest",
            description=description,
            sync=True
        )

        print(f"✅ Model registered successfully!")
        print(f"   Model Resource Name: {model.resource_name}")
        print(f"   Model ID: {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   GCS URI: {gcs_model_path}")

        return model

    except Exception as e:
        print(f"❌ Error registering model: {e}")
        raise

def upload_to_gcs(local_dir, bucket_name):
    """
    Upload SavedModel directory to GCS

    Args:
        local_dir: Local directory containing SavedModel
        bucket_name: GCS bucket name

    Returns:
        GCS URI of uploaded model
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Upload all files in the directory
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create relative path for GCS
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = f"models/mnist_cnn/{relative_path}"

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"   Uploaded: {relative_path}")

    # Return the base path (without specific files)
    return f"gs://{bucket_name}/models/mnist_cnn/"

def main():
    parser = argparse.ArgumentParser(description="Upload model to Vertex AI Model Registry")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--model-name", default="mnist-cnn", help="Model name in Vertex AI")
    parser.add_argument("--saved-model-dir", required=True, help="Path to SavedModel directory")
    parser.add_argument("--gcs-bucket", required=True, help="GCS bucket name")
    parser.add_argument("--description", default="MNIST CNN Model for digit recognition", help="Model description")

    args = parser.parse_args()

    if not os.path.exists(args.saved_model_dir):
        print(f"❌ Error: SavedModel directory not found at {args.saved_model_dir}")
        exit(1)

    upload_model_to_vertex_ai(
        project_id=args.project_id,
        region=args.region,
        model_name=args.model_name,
        saved_model_dir=args.saved_model_dir,
        gcs_bucket=args.gcs_bucket,
        description=args.description
    )

if __name__ == "__main__":
    main()
