from xgboost import XGBClassifier
import joblib
from data import load_data, split_data
from sklearn.metrics import accuracy_score

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
        random_state=12
    )
    xgb_classifier.fit(X_train, y_train)
    joblib.dump(xgb_classifier, "../model/mnist_model.pkl")
    
    
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
