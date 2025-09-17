import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# 모델 로딩을 위한 전역 변수
_cnn_model = None

def load_models():
    """CNN 모델을 로드합니다."""
    global _cnn_model
    
    if _cnn_model is None:
        try:
            _cnn_model = models.load_model("../model/cnn_mnist_model.h5")
            print("CNN 모델이 로드되었습니다.")
        except Exception as e:
            print(f"CNN 모델 로드 실패: {e}")
            _cnn_model = None

def reload_models():
    """CNN 모델을 강제로 다시 로드합니다."""
    global _cnn_model
    
    print("🔄 CNN 모델을 다시 로드하는 중...")
    
    # Git에서 최신 변경사항 pull (archived_data 포함)
    try:
        import subprocess
        import os
        
        # MLOps 루트 디렉토리로 이동
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        
        # Git pull 실행
        result = subprocess.run(
            ["git", "pull"], 
            capture_output=True, 
            text=True,
            cwd=repo_root
        )
        
        if result.returncode == 0:
            print("✅ Git pull 완료! (archived_data 포함)")
        else:
            print(f"⚠️ Git pull 실패: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️ Git pull 중 오류: {e}")
    
    # 기존 모델 초기화
    _cnn_model = None
    
    # 모델 다시 로드
    load_models()
    print("✅ CNN 모델 리로드 완료!")

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
