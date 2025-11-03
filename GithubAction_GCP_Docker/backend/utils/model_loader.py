"""
Vertex AI Model Loader
Loads the latest model from Vertex AI Model Registry
"""
import os
import tempfile
from google.cloud import aiplatform
from google.cloud import storage
import tensorflow as tf
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class VertexAIModelLoader:
    """Load and manage models from Vertex AI Model Registry"""

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        model_name: str = "mnist-cnn"
    ):
        self.project_id = project_id
        self.region = region
        self.model_name = model_name
        self.model = None

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)

    def get_latest_model_uri(self) -> Optional[str]:
        """
        Get the GCS URI of the latest model version from Vertex AI

        Returns:
            GCS URI of the latest model (e.g., gs://bucket/models/mnist_cnn/)
        """
        try:
            # List all models with the given name
            models = aiplatform.Model.list(
                filter=f'display_name="{self.model_name}"',
                order_by="create_time desc"
            )

            if not models:
                logger.error(f"No model found with name: {self.model_name}")
                return None

            # Get the latest model (first in the list due to desc order)
            latest_model = models[0]
            artifact_uri = latest_model.gca_resource.artifact_uri

            logger.info(f"Latest model URI: {artifact_uri}")
            logger.info(f"Model version: {latest_model.version_id}")
            logger.info(f"Created: {latest_model.create_time}")

            return artifact_uri

        except Exception as e:
            logger.error(f"Error getting latest model: {e}")
            return None

    def download_model_from_gcs(self, gcs_uri: str, local_dir: str):
        """
        Download model from GCS to local directory

        Args:
            gcs_uri: GCS URI (e.g., gs://bucket/models/mnist_cnn/)
            local_dir: Local directory to download to
        """
        # Parse GCS URI
        gcs_uri = gcs_uri.rstrip('/')
        parts = gcs_uri.replace('gs://', '').split('/', 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ''

        logger.info(f"Downloading from gs://{bucket_name}/{prefix}")

        # Download files
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            if blob.name.endswith('/'):
                continue

            # Create local path
            relative_path = blob.name[len(prefix):].lstrip('/')
            local_path = os.path.join(local_dir, relative_path)

            # Create directories if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download file
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded: {relative_path}")

    def load_model(self):
        """
        Load the latest model from Vertex AI

        Returns:
            Loaded TensorFlow model
        """
        # Get latest model URI
        gcs_uri = self.get_latest_model_uri()

        if not gcs_uri:
            raise ValueError("Could not find model in Vertex AI")

        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Downloading model to {temp_dir}")

            # Download model from GCS
            self.download_model_from_gcs(gcs_uri, temp_dir)

            # Load model using tf.saved_model.load for compatibility
            logger.info("Loading TensorFlow SavedModel...")
            loaded = tf.saved_model.load(temp_dir)

            # Get the serving function
            self.model = loaded.signatures["serving_default"]
            logger.info("✅ Model loaded successfully!")

        return self.model

    def predict(self, input_data):
        """
        Make predictions with the loaded model

        Args:
            input_data: Input data (numpy array, shape: (batch_size, 28, 28, 1))

        Returns:
            Predictions (numpy array)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Convert to tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Call the serving function
        predictions = self.model(input_tensor)

        # Extract the output (assuming output key exists)
        # The output might be a dictionary, so we need to get the actual prediction tensor
        if isinstance(predictions, dict):
            # Try common output keys
            for key in ['output_0', 'dense_2', 'predictions']:
                if key in predictions:
                    return predictions[key].numpy()
            # If no common key found, use the first value
            return list(predictions.values())[0].numpy()

        return predictions.numpy()
