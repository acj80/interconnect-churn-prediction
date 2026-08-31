from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from api.schemas import CustomerFeatures


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"


app = FastAPI(
    title="Interconnect Churn Prediction API",
    description=(
        "REST API for customer churn prediction "
        "using the official Interconnect CatBoost pipeline."
    ),
    version="1.0.0",
)


# Load the serialized production pipeline once.
churn_pipeline = joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "project": "Interconnect Customer Churn Prediction",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": churn_pipeline is not None,
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "CatBoostClassifier + One-Hot Encoding",
        "artifact": MODEL_PATH.name,
        "threshold": 0.5,
        "cv_auc_roc": 0.850601,
        "test_auc_roc": 0.843972,
        "test_accuracy": 0.807665,
        "features_expected": 23,
    }


def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "low"

    if probability < 0.60:
        return "medium"

    return "high"


@app.post("/predict")
def predict(customer: CustomerFeatures):
    customer_df = pd.DataFrame(
        [customer.model_dump()]
    )

    churn_probability = float(
        churn_pipeline.predict_proba(
            customer_df
        )[:, 1][0]
    )

    prediction = int(
        churn_probability >= 0.5
    )

    risk_level = get_risk_level(
        churn_probability
    )

    return {
        "prediction": prediction,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "threshold": 0.5,
    }
