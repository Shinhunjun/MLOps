import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# 모델 로딩을 위한 전역 변수
_cnn_model = None
_xgb_model = None

def load_models():
    """모델들을 로드합니다."""
    global _cnn_model, _xgb_model
    
    if _cnn_model is None:
        try:
            _cnn_model = models.load_model("../model/cnn_mnist_model.h5")
            print("CNN 모델이 로드되었습니다.")
        except Exception as e:
            print(f"CNN 모델 로드 실패: {e}")
            _cnn_model = None
    
    if _xgb_model is None:
        try:
            _xgb_model = joblib.load("../model/mnist_model.pkl")
            print("XGBoost 모델이 로드되었습니다.")
        except Exception as e:
            print(f"XGBoost 모델 로드 실패: {e}")
            _xgb_model = None

def predict_data(X):
    """
    XGBoost 모델로 예측합니다.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels.
    """
    if _xgb_model is None:
        load_models()
    
    if _xgb_model is not None:
        y_pred = _xgb_model.predict(X)
        return y_pred
    else:
        raise Exception("XGBoost 모델을 로드할 수 없습니다.")

def predict_with_confidence(X):
    """
    XGBoost 모델로 예측하고 신뢰도를 반환합니다.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        tuple: (predictions, confidence_scores)
    """
    if _xgb_model is None:
        load_models()
    
    if _xgb_model is not None:
        # 예측 확률 계산 (각 클래스에 대한 확률)
        probabilities = _xgb_model.predict_proba(X)
        
        # 가장 높은 확률을 신뢰도로 사용
        confidence_scores = np.max(probabilities, axis=1)
        
        # 예측 클래스
        predictions = _xgb_model.predict(X)
        
        return predictions, confidence_scores
    else:
        raise Exception("XGBoost 모델을 로드할 수 없습니다.")

def predict_with_cnn(X):
    """
    CNN 모델로 예측하고 신뢰도를 반환합니다.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made (N, 784).
    Returns:
        tuple: (predictions, confidence_scores)
    """
    if _cnn_model is None:
        load_models()
    
    if _cnn_model is not None:
        # 입력 데이터를 CNN 형태로 변환 (N, 784) -> (N, 28, 28, 1)
        X_cnn = X.reshape(-1, 28, 28, 1)
        
        # 예측 확률 계산
        probabilities = _cnn_model.predict(X_cnn, verbose=0)
        
        # 가장 높은 확률을 신뢰도로 사용
        confidence_scores = np.max(probabilities, axis=1)
        
        # 예측 클래스
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, confidence_scores
    else:
        raise Exception("CNN 모델을 로드할 수 없습니다.")
