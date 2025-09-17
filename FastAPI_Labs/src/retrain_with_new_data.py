#!/usr/bin/env python3
"""
새로운 데이터를 사용하여 모델을 재훈련하는 스크립트
"""
import os
import sys
import json
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models
import time

def load_new_data():
    """new_data 폴더에서 새로운 데이터를 로드"""
    data_dir = "new_data"
    if not os.path.exists(data_dir):
        print("❌ new_data 폴더가 존재하지 않습니다.")
        return None, None
    
    # 메타데이터 파일 찾기
    metadata_files = glob.glob(os.path.join(data_dir, "metadata.json"))
    if not metadata_files:
        print("❌ metadata.json 파일을 찾을 수 없습니다.")
        return None, None
    
    # 메타데이터 로드
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    X_new = []
    y_new = []
    
    # 각 데이터 포인트 처리
    for item in metadata:
        image_path = os.path.join(data_dir, item['filename'])
        if os.path.exists(image_path):
            try:
                # 이미지 로드 및 전처리
                image = Image.open(image_path).convert('L')
                image = image.resize((28, 28))
                pixels = np.array(image).flatten() / 255.0
                
                X_new.append(pixels)
                y_new.append(item['true_label'])
                
            except Exception as e:
                print(f"⚠️ 이미지 처리 실패: {image_path} - {e}")
                continue
    
    if len(X_new) == 0:
        print("❌ 유효한 새로운 데이터가 없습니다.")
        return None, None
    
    print(f"✅ {len(X_new)}개의 새로운 데이터를 로드했습니다.")
    return np.array(X_new), np.array(y_new)

def load_original_data():
    """기존 MNIST 데이터 로드"""
    from data import load_data
    X, y = load_data()
    return X, y


def retrain_cnn(X_original, y_original, X_new, y_new):
    """CNN 모델 재훈련"""
    print("🔄 CNN 모델 재훈련 중...")
    
    # 새로운 데이터에 증강 적용
    print(f"🔄 새로운 데이터 증강 중... (증강 배수: 3)")
    try:
        from augmentation import augment_dataset
        X_new_augmented, y_new_augmented = augment_dataset(X_new, y_new, 3)
        print(f"✅ 데이터 증강 완료: {len(X_new)}개 → {len(X_new_augmented)}개")
    except ImportError:
        print("⚠️ augmentation 모듈을 찾을 수 없습니다. 증강 없이 진행합니다.")
        X_new_augmented, y_new_augmented = X_new, y_new
    except Exception as e:
        print(f"⚠️ 데이터 증강 실패: {e}. 원본 데이터로 진행합니다.")
        X_new_augmented, y_new_augmented = X_new, y_new
    
    # 기존 데이터와 증강된 새로운 데이터 결합
    X_combined = np.vstack([X_original, X_new_augmented])
    y_combined = np.hstack([y_original, y_new_augmented])
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )
    
    # 훈련 데이터를 훈련 세트와 검증 세트로 다시 분할
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # CNN 모델 로드 (기존 모델)
    try:
        cnn_model = models.load_model("model/cnn_mnist_model.h5")
        print("✅ 기존 CNN 모델을 로드했습니다.")
    except:
        print("⚠️ 기존 CNN 모델을 찾을 수 없습니다. 새로 생성합니다.")
        from cnn_model import create_cnn_model
        cnn_model = create_cnn_model()
    
    # 데이터 형태 변환 (28, 28, 1)
    X_train_cnn = X_train_cnn.reshape(-1, 28, 28, 1)
    X_val_cnn = X_val_cnn.reshape(-1, 28, 28, 1)
    X_test_cnn = X_test.reshape(-1, 28, 28, 1)
    
    # 모델 재훈련 (fine-tuning)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    ]
    
    start_time = time.time()
    history = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        validation_data=(X_val_cnn, y_val_cnn),
        epochs=10,  # 적은 에포크로 fine-tuning
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # 정확도 평가
    test_loss, test_accuracy = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
    print(f"✅ CNN 재훈련 완료! 정확도: {test_accuracy:.4f}, 시간: {training_time:.2f}초")
    
    # 모델 저장
    cnn_model.save("model/cnn_mnist_model.h5")
    print("💾 CNN 모델이 저장되었습니다.")
    
    return cnn_model

def archive_used_data():
    """사용된 데이터를 아카이브 폴더로 이동"""
    data_dir = "new_data"
    archive_dir = "archived_data"
    
    os.makedirs(archive_dir, exist_ok=True)
    
    # 현재 타임스탬프로 아카이브 폴더 생성
    timestamp = int(time.time())
    archive_subdir = os.path.join(archive_dir, f"batch_{timestamp}")
    os.makedirs(archive_subdir, exist_ok=True)
    
    # 파일들 이동
    for filename in os.listdir(data_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.json')):
            src = os.path.join(data_dir, filename)
            dst = os.path.join(archive_subdir, filename)
            os.rename(src, dst)
    
    print(f"📁 사용된 데이터가 {archive_subdir}로 아카이브되었습니다.")

def main():
    """메인 함수"""
    print("🚀 모델 재훈련 프로세스를 시작합니다...")
    
    # 새로운 데이터 로드
    X_new, y_new = load_new_data()
    if X_new is None:
        print("❌ 새로운 데이터를 로드할 수 없습니다.")
        return
    
    # 기존 데이터 로드
    print("📊 기존 MNIST 데이터를 로드 중...")
    X_original, y_original = load_original_data()
    
    print(f"📈 데이터 통계:")
    print(f"  - 기존 데이터: {len(X_original)}개")
    print(f"  - 새로운 데이터: {len(X_new)}개")
    print(f"  - 총 데이터: {len(X_original) + len(X_new)}개")
    
    # 모델 재훈련
    try:
        # CNN 재훈련
        retrain_cnn(X_original, y_original, X_new, y_new)
        
        # 사용된 데이터 아카이브
        archive_used_data()
        
        print("🎉 CNN 모델 재훈련이 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 재훈련 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
