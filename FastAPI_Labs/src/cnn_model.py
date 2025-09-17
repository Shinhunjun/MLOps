import tensorflow as tf
import keras
from keras import layers, models, callbacks
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    MNIST용 CNN 모델을 생성합니다. (M1 GPU 최적화)
    
    Args:
        input_shape (tuple): 입력 이미지 형태 (height, width, channels)
        num_classes (int): 분류할 클래스 수
    
    Returns:
        keras.Model: 컴파일된 CNN 모델
    """
    model = models.Sequential([
        # 첫 번째 Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 두 번째 Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # 세 번째 Convolutional Block
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
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_cnn_model(X_train, y_train, X_val, y_val, epochs=20, batch_size=128):
    """
    CNN 모델을 훈련합니다. (M1 GPU 최적화)
    
    Args:
        X_train (numpy.ndarray): 훈련 데이터
        y_train (numpy.ndarray): 훈련 라벨
        X_val (numpy.ndarray): 검증 데이터
        y_val (numpy.ndarray): 검증 라벨
        epochs (int): 훈련 에포크 수
        batch_size (int): 배치 크기 (M1 GPU에 최적화)
    
    Returns:
        keras.Model: 훈련된 모델
        keras.History: 훈련 히스토리
    """
    # 모델 생성
    model = create_cnn_model()
    
    # 모델 구조 출력
    model.summary()
    
    # 콜백 설정
    callback_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001
        )
    ]
    
    # 모델 훈련
    print("CNN 모델 훈련을 시작합니다...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )
    
    return model, history

def evaluate_cnn_model(model, X_test, y_test):
    """
    CNN 모델을 평가합니다.
    
    Args:
        model (keras.Model): 훈련된 모델
        X_test (numpy.ndarray): 테스트 데이터
        y_test (numpy.ndarray): 테스트 라벨
    
    Returns:
        dict: 평가 결과
    """
    # 예측
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test, y_pred)
    
    # 분류 리포트
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"CNN 모델 테스트 정확도: {accuracy:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'report': report
    }

def plot_training_history(history):
    """
    훈련 히스토리를 시각화합니다.
    
    Args:
        history (keras.History): 훈련 히스토리
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 정확도 그래프
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 손실 그래프
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def save_cnn_model(model, filepath="model/cnn_mnist_model.h5"):
    """
    CNN 모델을 저장합니다.
    
    Args:
        model (keras.Model): 저장할 모델
        filepath (str): 저장 경로
    """
    model.save(filepath)
    print(f"CNN 모델이 {filepath}에 저장되었습니다.")

def load_cnn_model(filepath="model/cnn_mnist_model.h5"):
    """
    CNN 모델을 로드합니다.
    
    Args:
        filepath (str): 모델 파일 경로
    
    Returns:
        keras.Model: 로드된 모델
    """
    model = models.load_model(filepath)
    print(f"CNN 모델이 {filepath}에서 로드되었습니다.")
    return model

def predict_with_cnn(model, X):
    """
    CNN 모델로 예측을 수행합니다.
    
    Args:
        model (keras.Model): 훈련된 모델
        X (numpy.ndarray): 예측할 데이터
    
    Returns:
        tuple: (예측 클래스, 예측 확률)
    """
    # 예측 확률 계산
    probabilities = model.predict(X, verbose=0)
    
    # 예측 클래스
    predictions = np.argmax(probabilities, axis=1)
    
    # 신뢰도 (최대 확률)
    confidences = np.max(probabilities, axis=1)
    
    return predictions, confidences
