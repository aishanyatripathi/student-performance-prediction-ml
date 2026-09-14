import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_endpoint():
    payload = {
        "Gender": "Female",
        "Age": 18,
        "StudyHoursPerWeek": 18.0,
        "AttendancePercentage": 88.0,
        "PreviousGrade": 75.0,
        "SleepHours": 7.5,
        "Absences": 2,
        "ParentEducation": "Bachelor",
        "FamilyIncome": 65000.0,
        "ExtracurricularActivities": "Yes",
        "InternetAccess": "Yes"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["predicted_outcome"] in ["Pass", "Fail"]
    assert 0 <= data["predicted_exam_score"] <= 100
    assert 0 <= data["pass_probability_pct"] <= 100
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk", "Critical Risk"]
    assert len(data["recommendations"]) > 0

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "best_model_name" in data
    assert "best_model_f1" in data
