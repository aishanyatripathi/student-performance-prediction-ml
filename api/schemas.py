from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SinglePredictionInput(BaseModel):
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    Age: int = Field(..., ge=10, le=30, json_schema_extra={"example": 18})
    StudyHoursPerWeek: float = Field(..., ge=0, le=100, json_schema_extra={"example": 16.5})
    AttendancePercentage: float = Field(..., ge=0, le=100, json_schema_extra={"example": 85.0})
    PreviousGrade: float = Field(..., ge=0, le=100, json_schema_extra={"example": 72.0})
    SleepHours: float = Field(..., ge=0, le=24, json_schema_extra={"example": 7.5})
    Absences: int = Field(..., ge=0, le=100, json_schema_extra={"example": 3})
    ParentEducation: str = Field(..., json_schema_extra={"example": "Bachelor"})
    FamilyIncome: float = Field(..., json_schema_extra={"example": 60000.0})
    ExtracurricularActivities: str = Field(..., json_schema_extra={"example": "Yes"})
    InternetAccess: str = Field(..., json_schema_extra={"example": "Yes"})


class RecommendationItem(BaseModel):
    category: str
    priority: str
    action: str

class PredictionResponse(BaseModel):
    predicted_outcome: str
    predicted_exam_score: float
    pass_probability_pct: float
    fail_probability_pct: float
    risk_level: str
    composite_risk_score: float
    confidence_pct: float
    recommendations: List[RecommendationItem]

class ModelInfoResponse(BaseModel):
    best_model_name: str
    best_model_f1: float
    best_model_accuracy: float
    best_model_roc_auc: float
    total_records_trained: int
    feature_count: int

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class HealthCheckResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
