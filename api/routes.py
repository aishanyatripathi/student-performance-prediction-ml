import json
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from config.settings import (
    BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    METRICS_PATH, DATASET_PATH
)
from api.schemas import (
    SinglePredictionInput, PredictionResponse, ModelInfoResponse,
    FeatureImportanceItem, HealthCheckResponse
)
from preprocessing.feature_engineering import add_engineered_features
from utils.risk_analyzer import analyze_student_risk
from training.train_pipeline import train_and_evaluate_all
from utils.xai import compute_shap_explanations

router = APIRouter()

@router.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """Checks REST API health status and artifact availability."""
    model_loaded = BEST_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()
    return HealthCheckResponse(
        status="healthy",
        message="AI Student Performance API is running.",
        model_loaded=model_loaded
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_student_performance(payload: SinglePredictionInput):
    """Predicts student Pass/Fail outcome, expected final score, risk level, and intervention steps."""
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifacts missing. Run /train endpoint first.")
        
    try:
        pipeline_artifact = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        classifier = pipeline_artifact["classifier"]
        regressor = pipeline_artifact["regressor"]
        
        input_dict = payload.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Transform input
        X_trans = preprocessor.transform(input_df)
        
        pass_prob = float(classifier.predict_proba(X_trans)[0, 1])
        pred_class_val = int(classifier.predict(X_trans)[0])
        pred_outcome = "Pass" if pred_class_val == 1 else "Fail"
        pred_score = float(regressor.predict(X_trans)[0])
        
        # Risk & Engineered metric
        eng_df = add_engineered_features(input_df)
        acad_risk_score = float(eng_df["AcademicRiskScore"].iloc[0])
        
        risk_res = analyze_student_risk(
            pass_prob=pass_prob,
            pred_score=pred_score,
            study_hours=payload.StudyHoursPerWeek,
            attendance=payload.AttendancePercentage,
            prev_grade=payload.PreviousGrade,
            absences=payload.Absences,
            sleep_hours=payload.SleepHours,
            academic_risk_score=acad_risk_score
        )
        
        return PredictionResponse(
            predicted_outcome=pred_outcome,
            predicted_exam_score=risk_res["predicted_score"],
            pass_probability_pct=risk_res["pass_probability"],
            fail_probability_pct=risk_res["fail_probability"],
            risk_level=risk_res["risk_level"],
            composite_risk_score=risk_res["composite_risk_score"],
            confidence_pct=risk_res["confidence"],
            recommendations=risk_res["recommendations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model Specs"])
def get_model_info():
    """Returns champion model metrics and dataset metadata."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Please train models first.")
        
    with open(METRICS_PATH, "r") as f:
        data = json.load(f)
        
    return ModelInfoResponse(
        best_model_name=data.get("best_model_name", "Unknown"),
        best_model_f1=data.get("best_model_f1", 0.0),
        best_model_accuracy=data.get("best_model_accuracy", 0.0),
        best_model_roc_auc=data.get("best_model_roc_auc", 0.0),
        total_records_trained=data.get("total_records", 0),
        feature_count=data.get("feature_count", 0)
    )

@router.get("/feature-importance", response_model=List[FeatureImportanceItem], tags=["XAI"])
def get_feature_importance():
    """Returns top features driving model predictions using SHAP analysis."""
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifacts missing.")
        
    pipeline_artifact = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    classifier = pipeline_artifact["classifier"]
    
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        X_raw = df.drop(columns=["PassFail", "ExamScore"], errors="ignore").sample(min(100, len(df)), random_state=42)
        X_trans = preprocessor.transform(X_raw)
    else:
        raise HTTPException(status_code=404, detail="Sample dataset missing.")
        
    shap_res = compute_shap_explanations(classifier, X_trans, feature_names)
    imp_df = shap_res["importance_df"]
    
    return [
        FeatureImportanceItem(feature=row["Feature"], importance=float(row["Importance"]))
        for _, row in imp_df.iterrows()
    ]

@router.post("/train", tags=["Pipeline"])
def retrain_models(background_tasks: BackgroundTasks):
    """Triggers ML model retraining and metric export."""
    try:
        summary = train_and_evaluate_all()
        return {"status": "success", "message": "Model retraining completed successfully.", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")
