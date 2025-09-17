import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the MNIST dataset and return the features and target values.
    Returns:
        X (numpy.ndarray): The features of the MNIST dataset.
        y (numpy.ndarray): The target values of the MNIST dataset.
    """
    # load MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist.data
    y = mnist.target.astype(int)  # convert target to integer
    
    # MNIST 표준 정규화: 0-255 → 0-1
    X = X / 255.0
    
    return X, y

def split_data(X, y):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    return X_train, X_test, y_train, y_test