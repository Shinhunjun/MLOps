# File: src/model_development.py - Wine Quality Prediction
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import requests
from io import StringIO

WORKING_DIR = "/Users/hunjunsin/Desktop/Jun/MLOps/Airflow_Labs/Lab_2_Wine/working_data"
MODEL_DIR = "/Users/hunjunsin/Desktop/Jun/MLOps/Airflow_Labs/Lab_2_Wine/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data() -> str:
    """
    Load wine quality dataset from UCI repository.
    Returns path to saved file.
    """
    print("🍷 Loading wine quality dataset...")
    
    # UCI Wine Quality Dataset URLs
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    try:
        # Load red wine data
        red_wine_response = requests.get(red_wine_url)
        red_wine_data = pd.read_csv(StringIO(red_wine_response.text), sep=';')
        red_wine_data['wine_type'] = 'red'
        
        # Load white wine data
        white_wine_response = requests.get(white_wine_url)
        white_wine_data = pd.read_csv(StringIO(white_wine_response.text), sep=';')
        white_wine_data['wine_type'] = 'white'
        
        # Combine datasets
        df = pd.concat([red_wine_data, white_wine_data], ignore_index=True)
        
        # Encode wine type
        df['wine_type_encoded'] = (df['wine_type'] == 'red').astype(int)
        
        print(f"✅ Loaded wine dataset with {len(df)} samples")
        print(f"📊 Red wines: {len(red_wine_data)} samples")
        print(f"📊 White wines: {len(white_wine_data)} samples")
        print(f"📊 Quality range: {df['quality'].min()} - {df['quality'].max()}")
        print(f"📊 Average quality: {df['quality'].mean():.2f}")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        # Fallback: create synthetic data
        print("🔄 Creating synthetic wine data as fallback...")
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'fixed acidity': np.random.normal(8.3, 1.8, n_samples),
            'volatile acidity': np.random.normal(0.5, 0.2, n_samples),
            'citric acid': np.random.normal(0.3, 0.2, n_samples),
            'residual sugar': np.random.normal(2.5, 1.4, n_samples),
            'chlorides': np.random.normal(0.09, 0.05, n_samples),
            'free sulfur dioxide': np.random.normal(15, 10, n_samples),
            'total sulfur dioxide': np.random.normal(46, 32, n_samples),
            'density': np.random.normal(0.997, 0.002, n_samples),
            'pH': np.random.normal(3.3, 0.15, n_samples),
            'sulphates': np.random.normal(0.66, 0.17, n_samples),
            'alcohol': np.random.normal(10.4, 1.1, n_samples),
            'wine_type_encoded': np.random.choice([0, 1], n_samples),
        })
        
        # Create quality based on wine characteristics
        quality = (
            df['alcohol'] * 0.3 +
            (10 - df['volatile acidity']) * 0.2 +
            df['citric acid'] * 0.15 +
            (10 - df['chlorides']) * 0.1 +
            df['sulphates'] * 0.1 +
            np.random.normal(0, 0.5, n_samples)
        )
        
        df['quality'] = np.clip(np.round(quality), 3, 9).astype(int)
        print(f"✅ Created synthetic wine dataset with {len(df)} samples")
    
    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, preprocess features, and split data.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    
    print("🔄 Preprocessing wine data...")
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['quality', 'wine_type']]
    X = df[feature_columns]
    y = df['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessed data
    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': feature_columns
    }
    
    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"✅ Data preprocessing completed")
    print(f"📊 Training set: {X_train_scaled.shape[0]} samples")
    print(f"📊 Test set: {X_test_scaled.shape[0]} samples")
    print(f"📊 Features: {X_train_scaled.shape[1]} features")
    
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, filename: str) -> str:
    """
    Train ensemble model for wine quality prediction and save to MODEL_DIR/filename.
    Returns model path.
    """
    with open(file_path, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    
    print("🍷 Training wine quality prediction model...")
    
    # Individual models
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    
    lr_model = LinearRegression()
    
    # Ensemble model
    ensemble_model = VotingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('lr', lr_model)
        ]
    )
    
    # Train individual models first (for feature importance)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    
    # Train ensemble model
    ensemble_model.fit(X_train, y_train)
    
    # Evaluate model
    train_pred = ensemble_model.predict(X_train)
    test_pred = ensemble_model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print(f"📊 Training R²: {train_r2:.4f}")
    print(f"📊 Test R²: {test_r2:.4f}")
    print(f"📊 Training RMSE: {train_rmse:.4f}")
    print(f"📊 Test RMSE: {test_rmse:.4f}")
    print(f"📊 Training MAE: {train_mae:.4f}")
    print(f"📊 Test MAE: {test_mae:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(ensemble_model, X_train, y_train, cv=5, scoring='r2')
    print(f"📊 Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (using Random Forest)
    feature_importance = rf_model.feature_importances_
    feature_names = preprocessed_data['feature_names']
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("🔝 Top 10 most important features for wine quality:")
    print(importance_df.head(10))
    
    # Save model and metadata
    model_data = {
        'model': ensemble_model,
        'preprocessed_data': preprocessed_data,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_scores': cv_scores,
        'feature_importance': importance_df
    }
    
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Wine quality model saved to {model_path}")
    return model_path

def load_model(file_path: str, filename: str) -> dict:
    """
    Load saved model and return evaluation metrics.
    """
    with open(file_path, "rb") as f:
        preprocessed_data = pickle.load(f)
    
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    X_test = preprocessed_data['X_test']
    y_test = preprocessed_data['y_test']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate final metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    
    print(f"🍷 Final Wine Quality Prediction Results:")
    print(f"📊 Test R²: {test_r2:.4f}")
    print(f"📊 Test RMSE: {test_rmse:.4f}")
    print(f"📊 Test MAE: {test_mae:.4f}")
    
    # Quality distribution analysis
    quality_dist = pd.Series(y_test).value_counts().sort_index()
    print(f"\n📊 Actual quality distribution:")
    print(quality_dist)
    
    pred_dist = pd.Series(np.round(y_pred)).value_counts().sort_index()
    print(f"\n📊 Predicted quality distribution:")
    print(pred_dist)
    
    return {
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'predictions': y_pred,
        'actual_quality': y_test
    }
