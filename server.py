"""
Lightweight FastAPI server for Credit Card Fraud Detection
Replaces MLflow's heavy serving infrastructure
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(title="Credit Card Fraud API")

# Load model and preprocessor at startup
MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")
PREPROCESSOR_PATH = os.environ.get("PREPROCESSOR_PATH", "preprocessor.joblib")

model = None
preprocessor = None

@app.on_event("startup")
def load_model():
    global model, preprocessor
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Preprocessor loaded from {PREPROCESSOR_PATH}")

class PredictionRequest(BaseModel):
    inputs: List[List[float]]

class PredictionResponse(BaseModel):
    predictions: List[int]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/invocations")
def predict(request: PredictionRequest):
    """
    Prediction endpoint - compatible with MLflow's format
    Expects preprocessed data in 'inputs' format
    """
    inputs = np.array(request.inputs)
    predictions = model.predict(inputs)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
