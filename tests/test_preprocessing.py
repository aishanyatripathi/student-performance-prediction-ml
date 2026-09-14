import pytest
import pandas as pd
import numpy as np

from preprocessing.feature_engineering import add_engineered_features
from preprocessing.cleaner import StudentDataPreprocessor

@pytest.fixture
def sample_raw_dataframe():
    return pd.DataFrame([{
        "Gender": "Female",
        "Age": 18,
        "StudyHoursPerWeek": 16.0,
        "AttendancePercentage": 85.0,
        "PreviousGrade": 70.0,
        "SleepHours": 7.5,
        "Absences": 3,
        "ParentEducation": "Bachelor",
        "FamilyIncome": 60000,
        "ExtracurricularActivities": "Yes",
        "InternetAccess": "Yes"
    }])

def test_engineered_features(sample_raw_dataframe):
    df_feat = add_engineered_features(sample_raw_dataframe)
    
    expected_cols = ["AttendanceRatio", "StudyEfficiency", "AcademicRiskScore", "HomeworkConsistency", "LifestyleScore"]
    for col in expected_cols:
        assert col in df_feat.columns, f"Missing engineered feature: {col}"
        
    assert 0.0 <= df_feat["AttendanceRatio"].iloc[0] <= 1.0
    assert df_feat["AcademicRiskScore"].iloc[0] >= 0.0

def test_preprocessor_fit_transform(sample_raw_dataframe):
    preprocessor = StudentDataPreprocessor(handle_outliers=True)
    transformed = preprocessor.fit_transform(sample_raw_dataframe)
    
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == len(preprocessor.get_feature_names())
