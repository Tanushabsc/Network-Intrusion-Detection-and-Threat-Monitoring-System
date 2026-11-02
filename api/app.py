
import os
import joblib
import pandas as pd
import requests

from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intrusion_model.joblib"

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/Tanushabsc/Network-Intrusion-Detection-and-Threat-Monitoring-System/releases/download/v1.0/intrusion_model.joblib"
)

# Download the model if it is not already available
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading NetGuard AI model...")

    response = requests.get(MODEL_URL, stream=True, timeout=300)
    response.raise_for_status()

    with open(MODEL_PATH, "wb") as model_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                model_file.write(chunk)

    print("Model downloaded successfully!")

# Load trained model
model = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = list(model.feature_names_in_)

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


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        received_features = request.features

        missing_features = [
            col for col in FEATURE_COLUMNS
            if col not in received_features
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
            col: received_features[col]
            for col in FEATURE_COLUMNS
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
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "features_required": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "classes": ["Benign", "Attack"]
    }
