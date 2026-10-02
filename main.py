# Pipeline orchestrator: connects data, model training, and evaluation
from iris_pipeline.data import load_and_prep_data
from iris_pipeline.model import train_model
from iris_pipeline.evaluate import evaluate_model

def run():
    print("Starting Iris ML Pipeline Execution...")
    
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test, target_names = load_and_prep_data()
    
    # 2. Train classifier model
    model = train_model(X_train, y_train)
    
    # 3. Evaluate model and output results
    evaluate_model(model, X_test, y_test, target_names)
    
    print("Pipeline execution finished successfully.")

if __name__ == "__main__":
    run()