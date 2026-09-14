import numpy as np
import pandas as pd
from typing import Union, Dict, Any

from utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes domain-specific engineered features for student performance analysis."""
    df_feat = df.copy()
    
    # Fill missing temporary values if needed for calculation
    study_hours = df_feat["StudyHoursPerWeek"].fillna(df_feat["StudyHoursPerWeek"].median() if not df_feat["StudyHoursPerWeek"].isna().all() else 14.0)
    attendance = df_feat["AttendancePercentage"].fillna(df_feat["AttendancePercentage"].median() if not df_feat["AttendancePercentage"].isna().all() else 80.0)
    prev_grade = df_feat["PreviousGrade"].fillna(df_feat["PreviousGrade"].median() if not df_feat["PreviousGrade"].isna().all() else 65.0)
    absences = df_feat["Absences"].fillna(0)
    sleep = df_feat["SleepHours"].fillna(7.0)
    
    # 1. Attendance Ratio (0.0 to 1.0)
    df_feat["AttendanceRatio"] = np.round(attendance / 100.0, 4)
    
    # 2. Study Efficiency
    df_feat["StudyEfficiency"] = np.round((prev_grade * study_hours) / (absences + 1.0), 2)
    
    # 3. Academic Risk Score (0 to 100 index where higher means higher risk of failure)
    # Higher absences increase risk, lower attendance increases risk, lower previous grade increases risk
    absence_factor = np.clip(absences / 20.0, 0, 1.0) * 35.0
    attendance_factor = np.clip((100.0 - attendance) / 60.0, 0, 1.0) * 35.0
    grade_factor = np.clip((100.0 - prev_grade) / 65.0, 0, 1.0) * 30.0
    
    df_feat["AcademicRiskScore"] = np.round(absence_factor + attendance_factor + grade_factor, 2)
    
    # 4. Homework Consistency
    df_feat["HomeworkConsistency"] = np.round((study_hours / 40.0) * (attendance / 100.0) * 100.0, 2)
    
    # 5. Lifestyle Score (0 to 100)
    # Optimal sleep (7-8 hours) gets full points, extracurriculars add points, absences penalize
    sleep_score = np.maximum(0, 30.0 - np.abs(sleep - 7.5) * 6.0)
    extra_act = df_feat["ExtracurricularActivities"].apply(lambda x: 20.0 if str(x).strip().lower() in ["yes", "1", "true"] else 5.0)
    absence_penalty = np.clip(absences * 2.0, 0, 30.0)
    internet_score = df_feat["InternetAccess"].apply(lambda x: 20.0 if str(x).strip().lower() in ["yes", "1", "true"] else 5.0)
    
    df_feat["LifestyleScore"] = np.round(np.clip(sleep_score + extra_act + internet_score + 30.0 - absence_penalty, 0, 100), 2)
    
    logger.debug("Successfully appended 5 engineered features to DataFrame.")
    return df_feat
