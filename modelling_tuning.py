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
import seaborn as sns

from preprocessing import preprocess_data

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    data = pd.read_csv("dataset.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data=data, target_column='fraud', save_path="preprocessor.joblib", file_path="data_columns.csv")
    
    input_example = X_train[0:5]
    max_iter = 1000
    
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    dagshub.init(repo_owner='fathurazka', repo_name='credit-card-fraud', mlflow=True)
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
    
    clf = GridSearchCV(estimator=LogisticRegression(random_state=42, max_iter=1000), param_grid=param_grid, cv=5, scoring='f1', n_jobs=2, verbose=3)
    
    with mlflow.start_run():
        
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path='model',
            input_example=input_example
        )
        
        
        # Get training accuracy, f1_score, log loss, precision, recall, roc_auc, training score
        
        train_predictions = clf.predict(X_train)
        
        training_accuracy_score = accuracy_score(y_train, train_predictions)
        training_f1_score = f1_score(y_train, train_predictions)
        training_log_loss = log_loss(y_train, clf.predict_proba(X_train))
        training_precision_score = precision_score(y_train, train_predictions)
        training_recall_score = recall_score(y_train, train_predictions)
        training_roc_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        training_score = clf.score(X_train, y_train)
        
        # Metrics not covered by autolog
        
        testing_accuracy_score = accuracy_score(y_test, predictions)
        testing_f1_score = f1_score(y_test, predictions)
        testing_log_loss = log_loss(y_test, clf.predict_proba(X_test))
        testing_precision_score = precision_score(y_test, predictions)
        testing_recall_score = recall_score(y_test, predictions)
        testing_roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        testing_score = clf.score(X_test, y_test)

        mlflow.log_metrics({
            "training_accuracy_score": training_accuracy_score,
            "training_f1_score": training_f1_score,
            "training_log_loss": training_log_loss,
            "training_precision_score": training_precision_score,
            "training_recall_score": training_recall_score,
            "training_roc_auc": training_roc_auc,
            "training_score": training_score,
            "testing_accuracy_score": testing_accuracy_score,
            "testing_f1_score": testing_f1_score,
            "testing_log_loss": testing_log_loss,
            "testing_precision_score": testing_precision_score,
            "testing_recall_score": testing_recall_score,
            "testing_roc_auc": testing_roc_auc,
            "testing_score": testing_score
        })
        
        mlflow.log_params(clf.best_params_)
        
        # Confusion Matrix
        
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        # Classification Report
        cls_report = classification_report(y_test, predictions, output_dict=True)
        cls_report_df = pd.DataFrame(cls_report).transpose()
        cls_report_df.to_csv('classification_report.csv', index=True)
        mlflow.log_artifact('classification_report.csv')