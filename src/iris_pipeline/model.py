# Module responsible for defining and training the machine learning model
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    """
    Instantiates and trains a Random Forest Classifier.
    
    n_estimators=100 creates an ensemble of 100 decision trees to vote on 
    the target class, improving generalization over a single decision tree.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    
    # Train the model on the training features and labels
    model.fit(X_train, y_train)
    
    return model