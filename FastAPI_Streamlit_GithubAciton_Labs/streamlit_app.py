import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import os
import time
import glob
import subprocess

def count_new_data():
    """Count and return the number of files in new_data folder"""
    data_dir = "new_data"
    if not os.path.exists(data_dir):
        return 0
    
    # Count only image files
    image_files = glob.glob(os.path.join(data_dir, "*.png"))
    image_files.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    image_files.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
    
    return len(image_files)

def check_and_trigger_retrain():
    """Check data count and trigger retraining if 10 or more samples are available"""
    data_count = count_new_data()
    
    if data_count >= 10:
        st.success(f"🎯 {data_count} data samples have been collected!")
        st.info("🚀 Starting model retraining in GitHub Actions...")
        
        try:
            # Execute trigger_retrain.py (GitHub Actions trigger only)
            result = subprocess.run(
                ["python", "trigger_retrain.py"], 
                capture_output=True, 
                text=True,
                cwd="."
            )
            
            if result.returncode == 0:
                st.success("✅ Retraining trigger executed successfully!")
                st.info("Model retraining is in progress in GitHub Actions.")
                st.info("The model will be automatically updated after retraining is complete.")
                
                # Delete local data after successful trigger (conflict prevention)
                clear_local_data()
                
            else:
                st.error(f"❌ Trigger execution failed: {result.stderr}")
                
        except Exception as e:
            st.error(f"❌ Error during trigger execution: {e}")
    else:
        st.info(f"📊 Current data: {data_count} samples ({10-data_count} more needed to reach 10)")

def clear_local_data():
    """Delete local new_data folder to prevent conflicts"""
    import shutil
    data_dir = "new_data"
    
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
            print("🗑️ Local new_data folder has been deleted. (conflict prevention)")
            st.info("🗑️ Local data has been deleted. (conflict prevention)")
        except Exception as e:
            print(f"⚠️ Data deletion failed: {e}")
    else:
        print("📝 No data to delete.")

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# --- State Management ---
# Initialize session state keys
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False
if 'feedback_mode' not in st.session_state:
    st.session_state.feedback_mode = False
if 'current_file_id' not in st.session_state:
    st.session_state.current_file_id = None

# Title
st.title("🔢 MNIST Digit Recognition")
st.markdown("Upload an image and AI will predict the digit for you!")

# Sidebar
st.sidebar.header("Settings")

api_url = st.sidebar.text_input(
    "API URL", 
    value="http://localhost:8000/predict",
    help="FastAPI CNN server endpoint URL"
)

st.sidebar.markdown("---")
st.sidebar.header("🔄 Model Management")

if st.sidebar.button("🔄 Auto Update Model", use_container_width=True):
    with st.spinner("Fetching latest model..."):
        try:
            response = requests.post("http://localhost:8000/auto-update-model", timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(result["message"])
                st.rerun()
            else:
                st.sidebar.error(f"Update failed: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error during update: {e}")

if st.sidebar.button("🔄 Reload Model", use_container_width=True):
    with st.spinner("Reloading model..."):
        try:
            response = requests.post("http://localhost:8000/reload-models", timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.sidebar.success(result["message"])
            else:
                st.sidebar.error(f"Reload failed: {response.status_code}")
        except Exception as e:
            st.sidebar.error(f"Error during reload: {e}")


# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Image Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a digit image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a 28x28 pixel digit image"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file. If so, clear old state.
        if st.session_state.current_file_id != f"{uploaded_file.name}-{uploaded_file.size}":
            st.session_state.current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
            st.session_state.prediction = None
            st.session_state.confidence = None
            st.session_state.feedback_submitted = False
            st.session_state.feedback_mode = False

        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
        
        # Image preprocessing (runs in background)
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Extract and preprocess pixel values
        try:
            pixels = np.array(image).flatten()
            pixels_2d = pixels.reshape(28, 28)
            pixels_2d = pixels_2d / 255.0 if pixels_2d.max() > 1.0 else pixels_2d
            pixels = pixels_2d.flatten()
            
            mean_pixel = np.mean(pixels)
            if mean_pixel > 0.5:
                pixels = 1.0 - pixels
            
            pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
            pixels = np.clip(pixels, 0.0, 1.0)
            pixels = pixels.tolist()
            
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            pixels = None

        if pixels:
            
            # Prediction button
            if st.button("🔍 Predict Digit", type="primary"):
                with st.spinner("AI is analyzing the digit..."):
                    try:
                        data = {"pixels": pixels}
                        response = requests.post(api_url, json=data)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.prediction = result['prediction']
                            st.session_state.confidence = result['confidence']
                            # Clear previous feedback state for the new prediction
                            st.session_state.feedback_submitted = False
                            st.session_state.feedback_mode = False
                            
                            # Auto trigger check (after prediction)
                            check_and_trigger_retrain()
                        else:
                            st.error(f"API error: {response.status_code}")
                            st.error(response.text)
                            st.session_state.prediction = None
                    
                    except requests.exceptions.ConnectionError:
                        st.error("🚫 Cannot connect to FastAPI server.")
                        st.info("Please check if the server is running: `uvicorn src.main:app --reload`")
                        st.session_state.prediction = None
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.session_state.prediction = None

# Display results and feedback if a prediction exists in the session state
if st.session_state.prediction is not None:
    with col2:
        st.header("🎯 Prediction Result")
        
        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="font-size: 8rem; margin: 0; color: #1f77b4;">{prediction}</h1>
            <p style="font-size: 1.5rem; color: #666;">Predicted Digit</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Confidence", f"{confidence:.2%}")
        
        # Display model information
        try:
            import requests
            response = requests.get("http://localhost:8000/model-info", timeout=5)
            if response.status_code == 200:
                model_info = response.json()
                st.info(f"🤖 Model used: **{model_info.get('model_name', 'CNN (FastAPI)')}**")
            else:
                st.info(f"🤖 Model used: **CNN (FastAPI)**")
        except:
            st.info(f"🤖 Model used: **CNN (FastAPI)**")
        
        st.progress(confidence)
        
        if confidence > 0.8:
            st.success("🎉 Predicted with high confidence!")
        elif confidence > 0.5:
            st.warning("⚠️ Predicted with moderate confidence.")
        else:
            st.error("❌ Low confidence. Try a different image.")

        # --- Feedback Section ---
        st.markdown("---")
        st.subheader("🤔 Add this prediction to the dataset?")

        if not st.session_state.feedback_submitted:
            col_feedback_1, col_feedback_2 = st.columns(2)

            if col_feedback_1.button(f"👍 Yes, '{prediction}' is correct."):
                try:
                    # Send feedback data to FastAPI
                    feedback_data = {
                        "pixels": pixels,
                        "label": prediction
                    }
                    
                    response = requests.post(
                        "http://localhost:8000/save-feedback",
                        json=feedback_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(result["message"])
                        
                        if result.get("triggered"):
                            st.info("🚀 Model retraining is in progress in GitHub Actions!")
                        
                        st.session_state.feedback_submitted = True
                        st.rerun()
                    else:
                        st.error(f"Feedback save failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error occurred while saving feedback: {e}")

            if col_feedback_2.button("👎 No, it's incorrect."):
                st.session_state.feedback_submitted = True
                st.session_state.feedback_mode = True
                st.rerun()

        elif st.session_state.feedback_mode:
            correct_label = st.text_input("Please enter the correct digit:")
            if st.button("Submit Feedback"):
                if correct_label.isdigit() and 0 <= int(correct_label) <= 9:
                    try:
                        # Send feedback data to FastAPI
                        feedback_data = {
                            "pixels": pixels,
                            "label": int(correct_label)
                        }
                        
                        response = requests.post(
                            "http://localhost:8000/save-feedback",
                            json=feedback_data,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(result["message"])
                            
                            if result.get("triggered"):
                                st.info("🚀 Model retraining is in progress in GitHub Actions!")
                            
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_mode = False
                            st.rerun()
                        else:
                            st.error(f"Feedback save failed: {response.status_code}")
                            
                    except Exception as e:
                        st.error(f"Error occurred while saving feedback: {e}")
                else:
                    st.warning("Please enter a digit between 0 and 9.")
        
        else: # Feedback has been submitted
            st.info("Feedback has been recorded. Thank you!")

# Display instructions if no file is uploaded
if uploaded_file is None:
    with col2:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload Image**: Upload a digit image on the left
        2. **Run Prediction**: Click the "Predict Digit" button
        3. **Check Result**: Review the AI's predicted digit and confidence
        4. **Provide Feedback**: Give feedback on whether the prediction is correct
        """)
        
        st.header("💡 Tips")
        st.markdown("""
        - **Image Size**: Images close to 28x28 pixels work best
        - **Background**: White background with black digits are automatically converted
        - **Clarity**: Use clear and sharp digit images
        - **Feedback**: Providing accurate feedback helps improve the model
        """)

# Footer
st.markdown("---")
st.markdown("🤖 Powered by CNN + FastAPI + Streamlit")