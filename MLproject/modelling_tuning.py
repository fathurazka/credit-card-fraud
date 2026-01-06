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

    
    # Initialize DagsHub for MLflow tracking
    dagshub.init(repo_owner='fathurazka', repo_name='credit-card-fraud', mlflow=True)
    
    # Set experiment name
    mlflow.set_experiment("Fraud_Detection")
    
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
        
        # Log best parameters from GridSearchCV
        mlflow.log_params(clf.best_params_)