import os
import numpy as np
from PIL import Image
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import necessary functions from other project files
from data import load_data, split_data
from augmentation import augment_dataset
from cnn_model import create_cnn_model, load_cnn_model, save_cnn_model, evaluate_cnn_model

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'cnn_mnist_model.h5')
NEW_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'new_data')

# --- Helper Functions ---

def prepare_cnn_data(X, y):
    """Prepares data for CNN training by reshaping and converting type."""
    X_reshaped = X.reshape(-1, 28, 28, 1).astype('float32')
    y_prepared = y.astype('int32')
    return X_reshaped, y_prepared

def load_and_prepare_new_data(data_path):
    """Loads and prepares new image data from the specified directory."""
    X_new, y_new = [], []
    if not os.path.exists(data_path):
        return None, None

    image_files = [f for f in os.listdir(data_path) if f.endswith('.png')]
    if not image_files:
        return None, None

    print(f"{len(image_files)}개의 새로운 이미지 파일을 로드합니다...")
    for filename in image_files:
        try:
            label = int(filename.split('_')[0])
            img_path = os.path.join(data_path, filename)
            with Image.open(img_path) as img:
                img_array = np.array(img.convert('L'))
                img_array = img_array.astype('float32') / 255.0
                X_new.append(img_array.flatten()) # Flatten to (784,)
                y_new.append(label)
        except Exception as e:
            print(f"'{filename}' 파일 처리 중 오류 발생: {e}")

    return np.array(X_new), np.array(y_new) if X_new else (None, None)

# --- Main Logic ---

def main():
    """Main function to run either initial training or fine-tuning."""
    if os.path.exists(MODEL_PATH):
        # --- FINE-TUNING LOGIC ---
        print("=" * 60)
        print("기존 모델 발견됨. 파인튜닝 모드로 전환합니다.")
        print("=" * 60)
        
        # 1. Load new data
        X_new, y_new = load_and_prepare_new_data(NEW_DATA_PATH)
        if X_new is None or X_new.shape[0] == 0:
            print("파인튜닝할 새로운 데이터가 없습니다. 스크립트를 종료합니다.")
            return

        # 2. Augment the NEW data
        print(f"\n1. {X_new.shape[0]}개의 새 데이터에 증강을 적용합니다...")
        X_aug, y_aug = augment_dataset(X_new, y_new, augmentation_factor=3)
        X_aug_cnn, y_aug_cnn = prepare_cnn_data(X_aug, y_aug)
        print(f"증강 후 데이터 수: {X_aug_cnn.shape[0]}개")

        # 3. Load existing model and prepare for fine-tuning
        print(f"\n2. 기존 모델 로드 중: '{MODEL_PATH}'")
        model = load_cnn_model(MODEL_PATH)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # Lower learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 4. Fine-tune on augmented new data
        print("\n3. 새로운 데이터로 모델 파인튜닝 중...")
        start_time = time.time()
        history = model.fit(
            X_aug_cnn, y_aug_cnn,
            epochs=10, # More epochs for fine-tuning on new data
            batch_size=32,
            validation_split=0.15,
            verbose=1
        )
        training_time = time.time() - start_time
        print(f"파인튜닝 완료! 소요 시간: {training_time:.2f}초")

        # 5. Save the fine-tuned model
        print("\n4. 파인튜닝된 모델 저장 중...")
        save_cnn_model(model, MODEL_PATH)
        final_accuracy = history.history['val_accuracy'][-1]
        print(f"\n모델이 업데이트되었습니다. 최종 검증 정확도: {final_accuracy:.4f}")

    else:
        # --- INITIAL TRAINING LOGIC ---
        print("=" * 60)
        print("기존 모델 없음. 처음부터 새로 훈련합니다.")
        print("=" * 60)
        
        # 1. Load original MNIST data
        print("1. 원본 MNIST 데이터 로드 중...")
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)

        # 2. Augment original data
        print("\n2. 데이터 증강 적용 중...")
        X_train_aug, y_train_aug = augment_dataset(X_train, y_train, augmentation_factor=2)

        # 3. Prepare data for CNN
        print("\n3. CNN용 데이터 준비 중...")
        X_train_cnn, y_train_cnn = prepare_cnn_data(X_train_aug, y_train_aug)
        X_test_cnn, y_test_cnn = prepare_cnn_data(X_test, y_test)
        
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_cnn, y_train_cnn, test_size=0.2, random_state=42, stratify=y_train_cnn
        )
        
        # 4. Create and train a new model
        print("\n4. 새로운 CNN 모델 훈련 중...")
        model = create_cnn_model()
        model.summary()
        
        # Callbacks
        callback_list = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        ]

        start_time = time.time()
        history = model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=64,
            callbacks=callback_list,
            verbose=1
        )
        training_time = time.time() - start_time
        print(f"훈련 완료! 소요 시간: {training_time:.2f}초")

        # 5. Evaluate and save
        print("\n5. 모델 평가 및 저장 중...")
        evaluate_cnn_model(model, X_test_cnn, y_test_cnn)
        save_cnn_model(model, MODEL_PATH)

    print("\n" + "=" * 60)
    print("스크립트 실행 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
