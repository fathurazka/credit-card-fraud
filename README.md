# :credit_card: Credit Card Fraud Detection

A complete end-to-end Machine Learning project with MLOps practices, from model training to cloud deployment.

## 🌐 Live Demo

- **Web App**: [Streamlit App](https://cc-fraud-fathurazka.streamlit.app/)

## 📋 Project Overview

This project predicts whether a credit card transaction is fraudulent based on transaction features. It demonstrates a complete MLOps pipeline including:

- **Data Preprocessing** with scikit-learn pipelines
- **Class Imbalance Handling** with SMOTE
- **Hyperparameter Tuning** with GridSearchCV
- **Experiment Tracking** with MLflow
- **Automated CI/CD** with GitHub Actions
- **Containerization** with Docker
- **Cloud Deployment** on Railway

## 🏗️ Architecture

<img width="1103" height="331" alt="image" src="https://github.com/user-attachments/assets/de8d4a00-6a6f-4141-bfbb-df902dfa4192" />


## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **ML Framework** | scikit-learn, imbalanced-learn (SMOTE) |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI |
| **Frontend** | Streamlit |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud Deployment** | Railway, Streamlit Cloud |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/fathurazka/credit-card-fraud.git
   cd credit-card-fraud
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
credit-card-fraud/
├── .github/workflows/     # CI/CD pipeline
│   └── main.yml
├── MLproject/             # MLflow project
│   |── modelling_tuning.py
│   ├── preprocessing.py
│   └── dataset.csv
├── app.py                 # Streamlit frontend
├── server.py              # FastAPI model server
├── Dockerfile             # Container configuration
└── requirements.txt
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow automatically:

1. **Trains** the model using MLflow
2. **Exports** the model to joblib format
3. **Builds** a lightweight Docker image
4. **Pushes** to Docker Hub
5. **Deploys** to Railway

## 📝 Features

- **Distance from Home**: How far the transaction occurred from the cardholder's home
- **Distance from Last Transaction**: Distance from the previous transaction location
- **Ratio to Median Purchase Price**: Transaction amount compared to typical spending
- **Repeat Retailer**: Whether the merchant was used before
- **Used Chip**: Whether chip was used (more secure)
- **Used PIN**: Whether PIN was entered
- **Online Order**: Whether it was an online transaction
