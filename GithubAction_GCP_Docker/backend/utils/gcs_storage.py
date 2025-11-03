"""
GCS Storage utilities for saving feedback data
"""
import os
import json
import time
from google.cloud import storage
from PIL import Image
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GCSStorage:
    """Handle data storage in Google Cloud Storage"""

    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)

    def get_feedback_count(self) -> Dict[str, int]:
        """
        Get current feedback count from GCS

        Returns:
            Dict with current_count and sub_set_count
        """
        blob = self.bucket.blob("feedback_data/count.json")

        if blob.exists():
            count_data = json.loads(blob.download_as_string())
        else:
            count_data = {"current_count": 0, "sub_set_count": 0}

        return count_data

    def update_feedback_count(self, count_data: Dict[str, int]):
        """Update feedback count in GCS"""
        blob = self.bucket.blob("feedback_data/count.json")
        blob.upload_from_string(
            json.dumps(count_data, indent=2),
            content_type="application/json"
        )

    def save_feedback_image(
        self,
        pixels: list,
        label: int,
        sub_set_count: int
    ) -> str:
        """
        Save feedback image to GCS

        Args:
            pixels: List of 784 pixel values (0-1)
            label: True label (0-9)
            sub_set_count: Current subset number

        Returns:
            GCS path of saved image
        """
        # Generate filename
        timestamp = int(time.time() * 1000)
        filename = f"{label}_{timestamp}.png"
        gcs_path = f"feedback_data/new_data/sub_set_{sub_set_count}/{filename}"

        # Convert pixels to image
        image_array = np.array(pixels).reshape(28, 28)
        image_array = (image_array * 255).astype(np.uint8)
        processed_image = Image.fromarray(image_array, 'L')

        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            processed_image.save(temp_file.name)
            temp_path = temp_file.name

        # Upload to GCS
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        logger.info(f"Saved image to gs://{self.bucket_name}/{gcs_path}")

        return gcs_path

    def save_metadata(
        self,
        sub_set_count: int,
        filename: str,
        label: int,
        timestamp: int
    ):
        """
        Save metadata for a feedback sample

        Args:
            sub_set_count: Current subset number
            filename: Image filename
            label: True label
            timestamp: Timestamp
        """
        metadata_path = f"feedback_data/new_data/sub_set_{sub_set_count}/metadata.json"
        blob = self.bucket.blob(metadata_path)

        # Load existing metadata
        if blob.exists():
            metadata = json.loads(blob.download_as_string())
        else:
            metadata = []

        # Add new entry
        metadata.append({
            "filename": filename,
            "true_label": label,
            "created_at": timestamp,
            "source": "fastapi_feedback"
        })

        # Upload updated metadata
        blob.upload_from_string(
            json.dumps(metadata, indent=2),
            content_type="application/json"
        )

    def count_images_in_subset(self, sub_set_count: int) -> int:
        """
        Count images in a specific subset

        Args:
            sub_set_count: Subset number

        Returns:
            Number of .png images in the subset
        """
        prefix = f"feedback_data/new_data/sub_set_{sub_set_count}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))

        # Count only .png files
        image_count = sum(1 for blob in blobs if blob.name.endswith('.png'))

        return image_count
