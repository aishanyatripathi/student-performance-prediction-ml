import os
from pathlib import Path

# Try importing dotenv for local .env loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "student_performance.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

ASSETS_DIR = BASE_DIR / "assets"
BANNER_PATH = ASSETS_DIR / "banner.png"

# Environment Configurations
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Model Training Settings
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

# Target Column
TARGET_CLASSIFICATION = "PassFail"
TARGET_REGRESSION = "ExamScore"

# Raw Feature Definitions
NUMERICAL_FEATURES = [
    "Age",
    "StudyHoursPerWeek",
    "AttendancePercentage",
    "PreviousGrade",
    "SleepHours",
    "Absences",
    "FamilyIncome",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "ParentEducation",
    "ExtracurricularActivities",
    "InternetAccess",
]

# Feature Engineering Generated Columns
ENGINEERED_FEATURES = [
    "AttendanceRatio",
    "StudyEfficiency",
    "AcademicRiskScore",
    "HomeworkConsistency",
    "LifestyleScore",
]

# Risk Level Thresholds (AcademicRiskScore based)
RISK_THRESHOLDS = {
    "Low": 30.0,
    "Medium": 55.0,
    "High": 75.0,
}

# Categorical Option Mapping
GENDER_OPTIONS = ["Female", "Male", "Other"]
PARENT_EDU_OPTIONS = ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
BINARY_OPTIONS = ["No", "Yes"]
