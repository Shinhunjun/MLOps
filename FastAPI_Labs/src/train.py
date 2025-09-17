from xgboost import XGBClassifier
import joblib
from data import load_data, split_data
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

def fit_model(X_train, y_train):
    """
    Train an XGBoost Classifier and save the model to a file.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    """
    # XGBoost 분류기 사용
    xgb_classifier = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=12,
        verbosity=1  # 진행상황 출력 (0=무음, 1=경고, 2=정보, 3=디버그)
    )
    
    # 훈련 시작
    print("Starting XGBoost training...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Number of estimators: {xgb_classifier.n_estimators}")
    
    start_time = time.time()
    
    # XGBoost 훈련 (진행상황 표시)
    with tqdm(total=100, desc="Training XGBoost", unit="%") as pbar:
        xgb_classifier.fit(X_train, y_train)
        pbar.update(100)
    
    # 훈련 완료
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # 모델 저장
    print("Saving model...")
    joblib.dump(xgb_classifier, "../model/mnist_model.pkl")
    print("Model saved successfully!")
    
    
def test_model(X_test, y_test):
    """
    Test the model and print the accuracy.
    Args:
        X_test (numpy.ndarray): Testing features.
        y_test (numpy.ndarray): Testing target values.
    """
    model = joblib.load("../model/mnist_model.pkl")
    y_pred = model.predict(X_test)
    print(y_pred)
    print(y_test)
    print(accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    fit_model(X_train, y_train)
    print("Model trained and saved successfully.")
    
    # test the model
    test_model(X_test, y_test)
