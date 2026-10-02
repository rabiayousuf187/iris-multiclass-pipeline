# Module responsible for downloading, preprocessing, and splitting data
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_prep_data(test_size: float = 0.2, random_state: int = 42):
    """
    Loads the Iris dataset, converts features to a DataFrame, 
    and performs a stratified train-test split.
    
    Stratified split ensures all 3 flower species are represented 
    equally in both training (80%) and testing (20%) sets.
    """
    # Load raw dataset from scikit-learn built-in datasets
    iris = load_iris()
    
    # Convert numerical features into a structured pandas DataFrame
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    # Target labels (0: Setosa, 1: Versicolor, 2: Virginica)
    y = pd.Series(iris.target, name="species")

    # Split data: 80% for model training, 20% for final validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Return prepared data splits and target class name strings
    return X_train, X_test, y_train, y_test, iris.target_names