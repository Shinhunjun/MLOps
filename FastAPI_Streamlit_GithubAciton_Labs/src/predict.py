import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

# Global variables for model loading
_cnn_model = None

def load_models():
    """Load CNN models."""
    global _cnn_model
    
    if _cnn_model is None:
        try:
            # Find latest model file
            model_path = find_latest_model()
            if model_path:
                _cnn_model = models.load_model(model_path)
                print(f"CNN model loaded: {os.path.basename(model_path)}")
            else:
                print("❌ Model file not found.")
                _cnn_model = None
        except Exception as e:
            print(f"CNN model load failed: {e}")
            _cnn_model = None

def find_latest_model():
    """Find the latest model file."""
    import glob
    import os
    
    model_dir = "../model"
    if not os.path.exists(model_dir):
        return None
    
    # Find files matching cnn_mnist_model_*.h5 pattern
    model_files = glob.glob(os.path.join(model_dir, "cnn_mnist_model_*.h5"))
    
    if not model_files:
        # Check for old filename as well
        old_model = os.path.join(model_dir, "cnn_mnist_model.h5")
        if os.path.exists(old_model):
            return old_model
        return None
    
    # Sort by timestamp and return the latest file
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    return model_files[0]

def reload_models():
    """Force reload CNN models."""
    global _cnn_model
    
    print("🔄 Reloading CNN models...")
    
    # Pull latest changes from Git (including archived_data)
    try:
        import subprocess
        import os
        
        # Move to MLOps root directory
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Execute Git pull
        result = subprocess.run(
            ["git", "pull"], 
            capture_output=True, 
            text=True,
            cwd=repo_root
        )
        
        if result.returncode == 0:
            print("✅ Git pull completed! (including archived_data)")
            
            # Check if new_data folder is empty (after GitHub Actions completion)
            new_data_path = os.path.join(repo_root, "FastAPI_Labs", "new_data")
            if os.path.exists(new_data_path) and not os.listdir(new_data_path):
                print("📁 new_data folder is empty. (GitHub Actions completed)")
            elif os.path.exists(new_data_path):
                print(f"📁 new_data folder contains {len(os.listdir(new_data_path))} files.")
        else:
            print(f"⚠️ Git pull failed: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Error during Git pull: {e}")
    
    # Initialize existing models
    _cnn_model = None
    
    # Reload models
    load_models()
    print("✅ CNN model reload completed!")

def predict_with_cnn(X):
    """
    Predict using CNN model and return confidence.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made (N, 784).
    Returns:
        tuple: (predictions, confidence_scores)
    """
    if _cnn_model is None:
        load_models()
    
    if _cnn_model is not None:
        # Convert input data to CNN format (N, 784) -> (N, 28, 28, 1)
        X_cnn = X.reshape(-1, 28, 28, 1)
        
        # Calculate prediction probabilities
        probabilities = _cnn_model.predict(X_cnn, verbose=0)
        
        # Use highest probability as confidence
        confidence_scores = np.max(probabilities, axis=1)
        
        # Predicted classes
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, confidence_scores
    else:
        raise Exception("CNN model cannot be loaded.")
