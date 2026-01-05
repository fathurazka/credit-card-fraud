import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, confusion_matrix, classification_report
import random
import numpy as np
import os
import warnings
import sys
import dagshub
import matplotlib.pyplot as plt
#import seaborn as sns

from preprocessing import preprocess_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset.csv')
    
    data = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data=data, target_column='fraud', save_path="preprocessor.joblib", file_path="data_columns.csv")
    
    input_example = X_train[0:5]
    max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    
    # DagsHub configuration
    DAGSHUB_USERNAME = "fathurazka"
    DAGSHUB_REPO = "credit-card-fraud"
    
    # Initialize DagsHub - authenticate with token if available (for CI environments)
    dagshub_token = os.environ.get("DAGSHUB_USER_TOKEN")
    if dagshub_token:
        # Use token-based auth for CI (no OAuth prompt)
        os.environ["DAGSHUB_TOKEN"] = dagshub_token
        dagshub.auth.add_app_token(dagshub_token)
        
        # Configure S3-compatible artifact store for DagsHub
        os.environ["AWS_ACCESS_KEY_ID"] = DAGSHUB_USERNAME
        os.environ["AWS_SECRET_ACCESS_KEY"] = dagshub_token
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.s3"
    
    # Set tracking URI
    tracking_uri = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment with explicit artifact location
    artifact_location = f"s3://dagshub/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}"
    mlflow.set_experiment(
        experiment_name="Default",
        artifact_location=artifact_location
    )
    
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact Location: {artifact_location}")
    
    param_grid = [
        {
            # liblinear supports both l1 and l2
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [500, 1000]
        },
        {
            # lbfgs only supports l2 penalty
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [500, 1000]
        }
    ]
    
    clf = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=max_iter), param_grid=param_grid, cv=5, scoring='f1', n_jobs=2, verbose=3)
    
    with mlflow.start_run():
        
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path='model',
            input_example=input_example
        )
        print(f"Model logged successfully to artifact path: 'model'")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
        
        testing_accuracy_score = accuracy_score(y_test, predictions)
        testing_f1_score = f1_score(y_test, predictions)
        testing_log_loss = log_loss(y_test, clf.predict_proba(X_test))
        testing_precision_score = precision_score(y_test, predictions)
        testing_recall_score = recall_score(y_test, predictions)
        testing_roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        testing_score = clf.score(X_test, y_test)

        mlflow.log_metrics({
            "testing_accuracy_score": testing_accuracy_score,
            "testing_f1_score": testing_f1_score,
            "testing_log_loss": testing_log_loss,
            "testing_precision_score": testing_precision_score,
            "testing_recall_score": testing_recall_score,
            "testing_roc_auc": testing_roc_auc,
            "testing_score": testing_score
        })
        
        #mlflow.log_params(clf.best_params_)
        
        # Output run_id for CI pipeline to capture
        run_id = mlflow.active_run().info.run_id
        print(f"MLFLOW_RUN_ID={run_id}")
