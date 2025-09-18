import tensorflow as tf
import keras
from keras import layers, models, callbacks
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create CNN model for MNIST. (M1 GPU optimized)
    
    Args:
        input_shape (tuple): Input image shape (height, width, channels)
        num_classes (int): Number of classes to classify
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Fully Connected Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model



def evaluate_cnn_model(model, X_test, y_test):
    """
    Evaluate CNN model.
    
    Args:
        model (keras.Model): Trained model
        X_test (numpy.ndarray): Test data
        y_test (numpy.ndarray): Test labels
    
    Returns:
        dict: Evaluation results
    """
    # Prediction
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"CNN model test accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'report': report
    }



def save_cnn_model(model, filepath="model/cnn_mnist_model.h5"):
    """
    Save CNN model.
    
    Args:
        model (keras.Model): Model to save
        filepath (str): Save path (timestamp automatically added when using default)
    """
    import time
    import os
    
    # Add timestamp if using default filename
    if filepath == "model/cnn_mnist_model.h5":
        timestamp = int(time.time())
        filepath = f"model/cnn_mnist_model_{timestamp}.h5"
    
    # Create directory
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model.save(filepath)
    print(f"CNN model saved to {filepath}.")

def load_cnn_model(filepath="model/cnn_mnist_model.h5"):
    """
    Load CNN model.
    
    Args:
        filepath (str): Model file path
    
    Returns:
        keras.Model: Loaded model
    """
    model = models.load_model(filepath)
    print(f"CNN model loaded from {filepath}.")
    return model

def predict_with_cnn(model, X):
    """
    Perform prediction using CNN model.
    
    Args:
        model (keras.Model): Trained model
        X (numpy.ndarray): Data to predict
    
    Returns:
        tuple: (predicted classes, prediction probabilities)
    """
    # Calculate prediction probabilities
    probabilities = model.predict(X, verbose=0)
    
    # Predicted classes
    predictions = np.argmax(probabilities, axis=1)
    
    # Confidence (maximum probability)
    confidences = np.max(probabilities, axis=1)
    
    return predictions, confidences
