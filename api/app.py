
import joblib
import pandas as pd

from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intrusion_model.joblib"

# Load trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train the model first."
    )

model = joblib.load(MODEL_PATH)

# Get required features
FEATURE_COLUMNS = list(model.feature_names_in_)

# Initialize API
app = FastAPI(
    title="NetGuard AI",
    description="Network Intrusion Detection and Threat Monitoring API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/")
def home():
    return {
        "project": "NetGuard AI",
        "message": "Network Intrusion Detection API is running",
        "model": "Random Forest",
        "status": "Ready"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "RandomForestClassifier"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "features_required": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": ["Benign", "Attack"]
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        received_features = request.features

        missing_features = [
            column for column in FEATURE_COLUMNS
            if column not in received_features
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Missing required features",
                    "missing_features": missing_features
                }
            )

        input_data = {
            column: received_features[column]
            for column in FEATURE_COLUMNS
        }

        input_df = pd.DataFrame([input_data])

        prediction = int(model.predict(input_df)[0])

        probabilities = model.predict_proba(input_df)[0]
        attack_index = list(model.classes_).index(1)

        attack_probability = float(probabilities[attack_index])

        result = "Attack" if prediction == 1 else "Benign"
        risk_level = "HIGH" if prediction == 1 else "LOW"

        return {
            "prediction": result,
            "label": prediction,
            "attack_probability": round(attack_probability, 4),
            "risk_level": risk_level,
            "status": "success"
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
