import numpy as np
from data import load_data, split_data
from augmentation import augment_dataset
from cnn_model import train_cnn_model, evaluate_cnn_model, save_cnn_model, plot_training_history
import time
import tensorflow as tf
import keras

# M1 GPU 사용 설정
print("TensorFlow 버전:", tf.__version__)
print("Keras 버전:", keras.__version__)
print("사용 가능한 GPU:", tf.config.list_physical_devices('GPU'))
print("M1 GPU 사용 가능:", len(tf.config.list_physical_devices('GPU')) > 0)

def prepare_cnn_data(X, y):
    """
    CNN 훈련을 위해 데이터를 준비합니다.
    
    Args:
        X (numpy.ndarray): 입력 데이터 (N, 784)
        y (numpy.ndarray): 타겟 데이터 (N,)
    
    Returns:
        tuple: (X_reshaped, y) - X는 (N, 28, 28, 1) 형태
    """
    # 784 -> 28x28x1로 reshape
    X_reshaped = X.reshape(-1, 28, 28, 1)
    
    # 데이터 타입 변환
    X_reshaped = X_reshaped.astype('float32')
    y = y.astype('int32')
    
    return X_reshaped, y

def main():
    """
    CNN 모델 훈련 메인 함수
    """
    print("=" * 60)
    print("CNN 모델 훈련 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"원본 훈련 데이터: {X_train.shape[0]}개")
    print(f"원본 테스트 데이터: {X_test.shape[0]}개")
    
    # 2. 데이터 증강 적용
    print("\n2. 데이터 증강 적용 중...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, augmentation_factor=2)  # CNN은 2배로 충분
    
    print(f"증강 후 훈련 데이터: {X_train_aug.shape[0]}개")
    
    # 3. CNN용 데이터 준비
    print("\n3. CNN용 데이터 준비 중...")
    X_train_cnn, y_train_cnn = prepare_cnn_data(X_train_aug, y_train_aug)
    X_test_cnn, y_test_cnn = prepare_cnn_data(X_test, y_test)
    
    print(f"CNN 훈련 데이터 형태: {X_train_cnn.shape}")
    print(f"CNN 테스트 데이터 형태: {X_test_cnn.shape}")
    
    # 4. 검증 데이터 분할 (훈련 데이터의 20%)
    from sklearn.model_selection import train_test_split
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_cnn, y_train_cnn, 
        test_size=0.2, 
        random_state=42,
        stratify=y_train_cnn
    )
    
    print(f"최종 훈련 데이터: {X_train_final.shape[0]}개")
    print(f"검증 데이터: {X_val.shape[0]}개")
    
    # 5. CNN 모델 훈련
    print("\n4. CNN 모델 훈련 중...")
    start_time = time.time()
    
    model, history = train_cnn_model(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=30,
        batch_size=64  # M1 GPU에 최적화된 배치 크기
    )
    
    training_time = time.time() - start_time
    print(f"훈련 완료! 소요 시간: {training_time:.2f}초")
    
    # 6. 모델 평가
    print("\n5. 모델 평가 중...")
    results = evaluate_cnn_model(model, X_test_cnn, y_test_cnn)
    
    # 7. 훈련 히스토리 시각화
    print("\n6. 훈련 히스토리 시각화...")
    plot_training_history(history)
    
    # 8. 모델 저장
    print("\n7. 모델 저장 중...")
    save_cnn_model(model, "model/cnn_mnist_model.h5")
    
    # 9. 결과 요약
    print("\n" + "=" * 60)
    print("훈련 결과 요약")
    print("=" * 60)
    print(f"최종 테스트 정확도: {results['accuracy']:.4f}")
    print(f"훈련 시간: {training_time:.2f}초")
    print(f"모델 저장 위치: model/cnn_mnist_model.h5")
    print("=" * 60)

if __name__ == "__main__":
    main()
