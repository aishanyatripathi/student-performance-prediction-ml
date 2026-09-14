import pytest
import pandas as pd
import joblib

from training.train_pipeline import train_and_evaluate_all
from config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH

def test_model_training_pipeline():
    metrics = train_and_evaluate_all()
    
    assert "best_model_name" in metrics
    assert "best_model_f1" in metrics
    assert metrics["best_model_f1"] > 0.5
    assert BEST_MODEL_PATH.exists()
    assert PREPROCESSOR_PATH.exists()

def test_model_prediction():
    pipeline_artifact = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    classifier = pipeline_artifact["classifier"]
    regressor = pipeline_artifact["regressor"]
    
    sample = pd.DataFrame([{
        "Gender": "Male",
        "Age": 19,
        "StudyHoursPerWeek": 20.0,
        "AttendancePercentage": 90.0,
        "PreviousGrade": 85.0,
        "SleepHours": 8.0,
        "Absences": 1,
        "ParentEducation": "Master",
        "FamilyIncome": 85000,
        "ExtracurricularActivities": "Yes",
        "InternetAccess": "Yes"
    }])
    
    X_trans = preprocessor.transform(sample)
    
    prob = classifier.predict_proba(X_trans)[0, 1]
    pred_cls = classifier.predict(X_trans)[0]
    score = regressor.predict(X_trans)[0]
    
    assert 0.0 <= prob <= 1.0
    assert pred_cls in [0, 1]
    assert 0.0 <= score <= 100.0
