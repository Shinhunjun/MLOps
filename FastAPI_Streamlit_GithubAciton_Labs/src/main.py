from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
import numpy as np
from predict import predict_with_cnn, reload_models


app = FastAPI()

# Load models on app startup
@app.on_event("startup")
async def startup_event():
    # Execute Git pull on startup
    try:
        import subprocess
        print("🔄 Fetching latest models on startup...")
        result = subprocess.run(['git', 'pull'], capture_output=True, text=True, cwd='..')
        if result.returncode == 0:
            print("✅ Git pull successful")
        else:
            print(f"⚠️ Git pull failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ Error during Git pull: {e}")
    
    # Load models
    from predict import load_models
    load_models()

class MNISTData(BaseModel):
    pixels: list[float]  # 784 pixel values (28x28 = 784)
    
    class Config:
        # 784 pixel validation
        json_schema_extra = {
            "example": {
                "pixels": [0.0] * 784  # Example of 784 0.0 values (normalized black image)
            }
        }

class MNISTResponse(BaseModel):
    prediction: int  # 0-9 digit prediction result
    confidence: float  # Prediction confidence (optional)

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.get("/model-info")
async def get_model_info():
    """Return information about the currently loaded model."""
    try:
        from predict import find_latest_model
        model_path = find_latest_model()
        if model_path:
            import os
            model_name = os.path.basename(model_path)
            return {
                "model_name": model_name,
                "model_path": model_path,
                "status": "loaded"
            }
        else:
            return {
                "model_name": "No model found",
                "model_path": None,
                "status": "not_found"
            }
    except Exception as e:
        return {
            "model_name": "Error",
            "model_path": None,
            "status": "error",
            "error": str(e)
        }

@app.post("/predict", response_model=MNISTResponse)
async def predict_mnist(mnist_data: MNISTData):
    """Prediction using CNN model"""
    try:
        # Convert 784 pixels to 2D array (1, 784)
        features = np.array([mnist_data.pixels])
        
        # Validate pixel count
        if len(mnist_data.pixels) != 784:
            raise HTTPException(
                status_code=400, 
                detail="MNIST image must have exactly 784 pixels (28x28)"
            )
        
        # Validate pixel value range (0-1, normalized values)
        if not all(0 <= pixel <= 1 for pixel in mnist_data.pixels):
            raise HTTPException(
                status_code=400,
                detail="Pixel values must be in range 0-1 (normalized values)"
            )

        # Predict using CNN model
        predictions, confidences = predict_with_cnn(features)
        predicted_class = int(predictions[0])
        confidence_score = float(confidences[0])
        
        return MNISTResponse(
            prediction=predicted_class,
            confidence=confidence_score
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-models")
async def reload_models_endpoint():
    """Endpoint to reload models (called from GitHub Actions)"""
    try:
        reload_models()
        return {"status": "success", "message": "Models have been successfully reloaded."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.post("/auto-update-model")
async def auto_update_model():
    """Automatically update models (called after GitHub Actions completion)."""
    try:
        import subprocess
        
        # 1. Git pull to get latest model
        pull_result = subprocess.run(['git', 'pull'], capture_output=True, text=True, cwd='..')
        
        if pull_result.returncode != 0:
            return {"message": f"Git pull failed: {pull_result.stderr}", "status": "error"}
        
        # 2. Check latest model file
        from predict import find_latest_model
        model_path = find_latest_model()
        if not model_path:
            return {"message": "Model file not found.", "status": "error"}
        
        # 3. Reload models
        reload_models()
        
        return {
            "message": "Models have been automatically updated!",
            "git_output": pull_result.stdout,
            "status": "success"
        }
        
    except Exception as e:
        return {"message": f"Error occurred during auto update: {str(e)}", "status": "error"}

class FeedbackData(BaseModel):
    pixels: list[float]  # 784 pixel values
    label: int  # Correct label (0-9)

@app.post("/save-feedback")
async def save_feedback_endpoint(feedback_data: FeedbackData):
    """Endpoint to save feedback data and automatically check triggers"""
    try:
        import os
        import json
        import time
        from PIL import Image
        import subprocess
        
        # Save data
        save_dir = "../new_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # Load/create count file
        count_file = os.path.join(save_dir, "count.json")
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                count_data = json.load(f)
        else:
            count_data = {"current_count": 0, "sub_set_count": 0}
        
        # Current sub_set folder path
        current_sub_set = f"sub_set_{count_data['sub_set_count']}"
        current_sub_set_path = os.path.join(save_dir, current_sub_set)
        os.makedirs(current_sub_set_path, exist_ok=True)
        
        # Save image
        timestamp = int(time.time() * 1000)
        filename = f"{feedback_data.label}_{timestamp}.png"
        file_path = os.path.join(current_sub_set_path, filename)
        
        # Convert pixel data to image and save
        pixels = feedback_data.pixels
        image_array = np.array(pixels).reshape(28, 28)
        image_array = (image_array * 255).astype(np.uint8)
        processed_image = Image.fromarray(image_array, 'L')
        processed_image.save(file_path)
        
        # Update metadata (in sub_set folder)
        metadata_path = os.path.join(current_sub_set_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = []
        
        metadata.append({
            "filename": filename,
            "true_label": feedback_data.label,
            "created_at": timestamp,
            "source": "streamlit_feedback"
        })
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Load/create count file
        count_file = os.path.join(save_dir, "count.json")
        if os.path.exists(count_file):
            with open(count_file, 'r') as f:
                count_data = json.load(f)
        else:
            count_data = {"current_count": 0, "sub_set_count": 0}
        
        # Check data count in current sub_set folder
        current_sub_set = f"sub_set_{count_data['sub_set_count']}"
        current_sub_set_path = os.path.join(save_dir, current_sub_set)
        
        if not os.path.exists(current_sub_set_path):
            os.makedirs(current_sub_set_path, exist_ok=True)
        
        # Check file count in current sub_set
        current_files = [f for f in os.listdir(current_sub_set_path) if f.endswith('.png')]
        current_count = len(current_files)
        
        if current_count >= 10:
            # Move to next sub_set when 10 samples are collected
            count_data['sub_set_count'] += 1
            count_data['current_count'] = 0
            
            # Update count file
            with open(count_file, 'w') as f:
                json.dump(count_data, f, indent=2)
            
            # Execute auto trigger
            result = subprocess.run(
                ["python", "trigger_retrain.py"], 
                capture_output=True, 
                text=True,
                cwd="../"
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Feedback saved. ({current_count} samples) Retraining has been automatically triggered! sub_set_{count_data['sub_set_count']-1} will be trained.",
                    "data_count": current_count,
                    "sub_set": f"sub_set_{count_data['sub_set_count']-1}",
                    "triggered": True
                }
            else:
                return {
                    "status": "success",
                    "message": f"Feedback saved. ({current_count} samples) but trigger failed.",
                    "data_count": current_count,
                    "triggered": False,
                    "error": result.stderr
                }
        else:
            return {
                "status": "success",
                "message": f"Feedback saved. ({current_count}/10 samples) - sub_set_{count_data['sub_set_count']}",
                "data_count": current_count,
                "sub_set": f"sub_set_{count_data['sub_set_count']}",
                "triggered": False
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred while saving feedback: {str(e)}")



    
