"""
FastAPI Backend for MNIST Prediction
Integrated with GCS, Vertex AI Model Registry, and GitHub Actions
"""
import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from typing import List

# Import utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import VertexAIModelLoader
from utils.gcs_storage import GCSStorage
from utils.github_trigger import GitHubActionsTrigger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Prediction API",
    description="MNIST digit recognition with MLOps pipeline",
    version="2.0.0"
)

# CORS middleware (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
PROJECT_ID = os.getenv("GCP_PROJECT_ID", "mlops-compute-lab")
REGION = os.getenv("GCP_REGION", "us-central1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "mlops-mnist-data")
MODEL_NAME = os.getenv("MODEL_NAME", "mnist-cnn")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "Shinhunjun")
GITHUB_REPO = os.getenv("GITHUB_REPO", "MLOps")

# Global instances
model_loader = None
gcs_storage = None
github_trigger = None
model = None

# Pydantic models
class MNISTData(BaseModel):
    pixels: List[float] = Field(..., min_length=784, max_length=784)

    class Config:
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784
            }
        }

class MNISTResponse(BaseModel):
    prediction: int
    confidence: float

class FeedbackData(BaseModel):
    pixels: List[float] = Field(..., min_length=784, max_length=784)
    label: int = Field(..., ge=0, le=9)

class FeedbackResponse(BaseModel):
    status: str
    message: str
    data_count: int
    sub_set: str
    triggered: bool

# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup"""
    global model_loader, gcs_storage, github_trigger, model

    logger.info("🚀 Starting up...")
    logger.info(f"Project ID: {PROJECT_ID}")
    logger.info(f"GCS Bucket: {GCS_BUCKET}")
    logger.info(f"Model Name: {MODEL_NAME}")

    try:
        # Initialize Vertex AI model loader
        logger.info("Initializing Vertex AI model loader...")
        model_loader = VertexAIModelLoader(
            project_id=PROJECT_ID,
            region=REGION,
            model_name=MODEL_NAME
        )

        # Load model
        logger.info("Loading model from Vertex AI...")
        model = model_loader.load_model()
        logger.info("✅ Model loaded successfully!")

        # Initialize GCS storage
        logger.info("Initializing GCS storage...")
        gcs_storage = GCSStorage(
            bucket_name=GCS_BUCKET,
            project_id=PROJECT_ID
        )
        logger.info("✅ GCS storage initialized!")

        # Initialize GitHub Actions trigger
        if GITHUB_TOKEN:
            logger.info("Initializing GitHub Actions trigger...")
            github_trigger = GitHubActionsTrigger(
                github_token=GITHUB_TOKEN,
                owner=GITHUB_OWNER,
                repo=GITHUB_REPO
            )
            logger.info("✅ GitHub Actions trigger initialized!")
        else:
            logger.warning("⚠️ GITHUB_TOKEN not set. GitHub Actions trigger disabled.")

        logger.info("✅ Startup complete!")

    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise

# Health check
@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MNIST Prediction API",
        "version": "2.0.0"
    }

# Model info
@app.get("/model-info")
async def get_model_info():
    """Get current model information"""
    try:
        if model_loader is None:
            return {
                "status": "error",
                "message": "Model loader not initialized"
            }

        gcs_uri = model_loader.get_latest_model_uri()

        return {
            "status": "loaded",
            "model_name": MODEL_NAME,
            "gcs_uri": gcs_uri,
            "project_id": PROJECT_ID,
            "region": REGION
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoint
@app.post("/predict", response_model=MNISTResponse)
async def predict_mnist(data: MNISTData):
    """
    Predict MNIST digit

    Args:
        data: MNIST image data (784 pixels, normalized 0-1)

    Returns:
        Prediction and confidence
    """
    try:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )

        # Validate pixel values
        if not all(0 <= pixel <= 1 for pixel in data.pixels):
            raise HTTPException(
                status_code=400,
                detail="Pixel values must be in range 0-1"
            )

        # Reshape for model input (1, 28, 28, 1)
        input_array = np.array(data.pixels).reshape(1, 28, 28, 1)

        # Predict
        predictions = model.predict(input_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        logger.info(f"Prediction: {predicted_class}, Confidence: {confidence:.4f}")

        return MNISTResponse(
            prediction=predicted_class,
            confidence=confidence
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Save feedback endpoint
@app.post("/save-feedback", response_model=FeedbackResponse)
async def save_feedback(feedback: FeedbackData):
    """
    Save user feedback and trigger retraining if threshold reached

    Args:
        feedback: Feedback data with pixels and correct label

    Returns:
        Status and information about triggering
    """
    try:
        if gcs_storage is None:
            raise HTTPException(
                status_code=503,
                detail="GCS storage not initialized"
            )

        # Get current count
        count_data = gcs_storage.get_feedback_count()
        current_sub_set = count_data['sub_set_count']

        # Save image to GCS
        timestamp = int(__import__('time').time() * 1000)
        filename = f"{feedback.label}_{timestamp}.png"

        gcs_storage.save_feedback_image(
            pixels=feedback.pixels,
            label=feedback.label,
            sub_set_count=current_sub_set
        )

        # Save metadata
        gcs_storage.save_metadata(
            sub_set_count=current_sub_set,
            filename=filename,
            label=feedback.label,
            timestamp=timestamp
        )

        # Count images in current subset
        image_count = gcs_storage.count_images_in_subset(current_sub_set)

        logger.info(f"Feedback saved: {image_count}/10 in sub_set_{current_sub_set}")

        # Check if we need to trigger retraining
        triggered = False
        if image_count >= 10:
            logger.info(f"✅ Threshold reached! Triggering retraining for sub_set_{current_sub_set}")

            # Update count for next subset
            count_data['sub_set_count'] += 1
            count_data['current_count'] = 0
            gcs_storage.update_feedback_count(count_data)

            # Trigger GitHub Actions
            if github_trigger:
                triggered = github_trigger.trigger_retrain(
                    data_count=image_count,
                    sub_set_id=f"sub_set_{current_sub_set}"
                )

                if triggered:
                    message = f"Feedback saved ({image_count} samples). Retraining triggered for sub_set_{current_sub_set}!"
                else:
                    message = f"Feedback saved ({image_count} samples). Failed to trigger retraining."
            else:
                message = f"Feedback saved ({image_count} samples). GitHub Actions trigger not configured."

            return FeedbackResponse(
                status="success",
                message=message,
                data_count=image_count,
                sub_set=f"sub_set_{current_sub_set}",
                triggered=triggered
            )
        else:
            return FeedbackResponse(
                status="success",
                message=f"Feedback saved ({image_count}/10 samples) - sub_set_{current_sub_set}",
                data_count=image_count,
                sub_set=f"sub_set_{current_sub_set}",
                triggered=False
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Reload model endpoint (for manual refresh)
@app.post("/reload-model")
async def reload_model():
    """Reload model from Vertex AI (called after retraining)"""
    global model

    try:
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model loader not initialized"
            )

        logger.info("Reloading model from Vertex AI...")
        model = model_loader.load_model()
        logger.info("✅ Model reloaded successfully!")

        return {
            "status": "success",
            "message": "Model reloaded successfully"
        }

    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
