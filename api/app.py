from pathlib import Path
from typing import Literal, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware



# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "xgb_model.pkl"
GENDER_ENCODER_PATH = MODELS_DIR / "le_gender.pkl"
HISTORY_ENCODER_PATH = MODELS_DIR / "le_history.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"


# --------------------------------------------------
# Load artifacts at startup
# --------------------------------------------------
def load_artifacts():
    missing_files = [
        str(path.name)
        for path in [
            MODEL_PATH,
            GENDER_ENCODER_PATH,
            HISTORY_ENCODER_PATH,
            FEATURE_COLUMNS_PATH,
        ]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required model files in '{MODELS_DIR}': {', '.join(missing_files)}"
        )

    model = joblib.load(MODEL_PATH)
    le_gender = joblib.load(GENDER_ENCODER_PATH)
    le_history = joblib.load(HISTORY_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    return model, le_gender, le_history, feature_columns


try:
    model, le_gender, le_history, feature_columns = load_artifacts()
except Exception as e:
    model = None
    le_gender = None
    le_history = None
    feature_columns = None
    startup_error = str(e)
else:
    startup_error = None


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="Digital Health Twin Risk Prediction API",
    description="API for predicting patient health risk using a trained XGBoost model.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------
class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: Literal["Male", "Female"] = Field(..., description="Patient gender")
    bmi: float = Field(..., ge=10, le=80, description="Body Mass Index")
    activity_level: float = Field(..., ge=0.0, le=1.0, description="Activity level scaled between 0 and 1")
    medical_history: Literal["none", "hypertension", "diabetes", "cardiac_history"] = Field(
        ...,
        description="Primary medical history category"
    )
    hr_mean: float = Field(..., ge=0, le=250, description="Mean heart rate")
    hr_std: float = Field(..., ge=0, le=100, description="Heart rate standard deviation")
    bp_mean: float = Field(..., ge=0, le=300, description="Mean systolic blood pressure")
    glucose_mean: float = Field(..., ge=0, le=500, description="Mean glucose level")
    sleep_mean: float = Field(..., ge=0, le=24, description="Mean sleep hours")
    hr_trend: float = Field(..., description="Heart rate trend over time")
    glucose_trend: float = Field(..., description="Glucose trend over time")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def safe_encode(value: str, encoder, field_name: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        allowed = list(map(str, encoder.classes_))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value for '{field_name}': '{value}'. Allowed values: {allowed}"
        )


def build_feature_row(patient: PatientInput) -> pd.DataFrame:
    gender_encoded = safe_encode(patient.gender, le_gender, "gender")
    history_encoded = safe_encode(patient.medical_history, le_history, "medical_history")

    input_dict = {
        "age": patient.age,
        "gender": gender_encoded,
        "bmi": patient.bmi,
        "activity_level": patient.activity_level,
        "medical_history": history_encoded,
        "hr_mean": patient.hr_mean,
        "hr_std": patient.hr_std,
        "bp_mean": patient.bp_mean,
        "glucose_mean": patient.glucose_mean,
        "sleep_mean": patient.sleep_mean,
        "hr_trend": patient.hr_trend,
        "glucose_trend": patient.glucose_trend,
    }

    input_df = pd.DataFrame([input_dict])

    # Enforce exact training column order
    missing_cols = [col for col in feature_columns if col not in input_df.columns]
    extra_cols = [col for col in input_df.columns if col not in feature_columns]

    if missing_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch: missing columns required by model: {missing_cols}"
        )

    if extra_cols:
        raise HTTPException(
            status_code=500,
            detail=f"Feature mismatch: unexpected extra columns: {extra_cols}"
        )

    input_df = input_df[feature_columns]
    return input_df


def determine_risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Moderate"
    return "Low"


def generate_health_insight(patient: PatientInput, risk_probability: float) -> str:
    reasons: List[str] = []

    if patient.bmi > 30:
        reasons.append("elevated BMI")
    if patient.activity_level < 0.35:
        reasons.append("low physical activity")
    if patient.glucose_mean > 125:
        reasons.append("high glucose level")
    if patient.bp_mean > 135:
        reasons.append("elevated blood pressure")
    if patient.hr_mean > 82:
        reasons.append("high average heart rate")
    if patient.hr_trend > 0.03:
        reasons.append("rising heart rate trend")
    if patient.glucose_trend > 0.03:
        reasons.append("rising glucose trend")

    risk_level = determine_risk_level(risk_probability).lower()

    if reasons:
        return f"Patient shows {risk_level} health risk driven by " + ", ".join(reasons) + "."
    return f"Patient shows {risk_level} health risk with no major modeled risk drivers."


def generate_recommendation(patient: PatientInput) -> str:
    recommendations: List[str] = []

    if patient.activity_level < 0.35:
        recommendations.append("increase physical activity")
    if patient.bmi > 30:
        recommendations.append("consider weight management")
    if patient.glucose_mean > 125:
        recommendations.append("monitor blood glucose levels")
    if patient.bp_mean > 135:
        recommendations.append("monitor blood pressure regularly")
    if patient.hr_mean > 82:
        recommendations.append("track cardiovascular status more closely")
    if patient.sleep_mean < 6:
        recommendations.append("improve sleep duration and recovery habits")

    if not recommendations:
        return "Maintain current healthy lifestyle and continue regular monitoring."

    return "Recommended actions: " + ", ".join(recommendations) + "."


# --------------------------------------------------
# API routes
# --------------------------------------------------
@app.get("/")
def root():
    if startup_error is not None:
        return {
            "status": "error",
            "message": "API started, but model artifacts failed to load.",
            "detail": startup_error,
        }

    return {
        "status": "ok",
        "message": "Digital Health Twin Risk Prediction API is running."
    }


@app.get("/health")
def health_check():
    if startup_error is not None:
        raise HTTPException(
            status_code=500,
            detail=f"Startup artifact loading failed: {startup_error}"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_count": len(feature_columns),
    }


@app.post("/predict")
def predict(patient: PatientInput):
    if startup_error is not None or model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model is unavailable: {startup_error}"
        )

    try:
        input_df = build_feature_row(patient)

        risk_probability = float(model.predict_proba(input_df)[0, 1])
        predicted_class = int(risk_probability >= 0.5)
        risk_label = "High Risk" if predicted_class == 1 else "Low Risk"
        risk_level = determine_risk_level(risk_probability)

        health_insight = generate_health_insight(patient, risk_probability)
        recommendation = generate_recommendation(patient)

        return {
            "risk_probability": round(risk_probability, 4),
            "predicted_class": predicted_class,
            "risk_label": risk_label,
            "risk_level": risk_level,
            "health_insight": health_insight,
            "recommendation": recommendation,
            "input_features_used": feature_columns,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")