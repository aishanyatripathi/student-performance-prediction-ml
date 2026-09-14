import os
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import DATASET_PATH, DATASET_DIR, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger("DataGenerator")

def generate_student_dataset(num_samples: int = 1200, output_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Generates a realistic Kaggle-style Student Performance Dataset with correlated features."""
    np.random.seed(RANDOM_STATE)
    
    # Base demography
    genders = np.random.choice(["Female", "Male", "Other"], size=num_samples, p=[0.49, 0.49, 0.02])
    ages = np.random.randint(15, 23, size=num_samples)
    
    # Academic & Habits
    study_hours = np.round(np.clip(np.random.normal(loc=14, scale=6, size=num_samples), 1, 40), 1)
    attendance = np.round(np.clip(np.random.beta(a=7, b=2, size=num_samples) * 100, 40, 100), 1)
    previous_grade = np.round(np.clip(np.random.normal(loc=68, scale=14, size=num_samples), 35, 100), 1)
    absences = np.random.poisson(lam=4, size=num_samples)
    absences = np.clip(absences, 0, 30)
    sleep_hours = np.round(np.clip(np.random.normal(loc=7.0, scale=1.2, size=num_samples), 4, 10), 1)
    
    # Socioeconomic & Environment
    parent_edu = np.random.choice(
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
        size=num_samples,
        p=[0.30, 0.25, 0.25, 0.15, 0.05]
    )
    family_income = np.random.randint(18000, 150000, size=num_samples)
    extra_activities = np.random.choice(["No", "Yes"], size=num_samples, p=[0.45, 0.55])
    internet_access = np.random.choice(["No", "Yes"], size=num_samples, p=[0.12, 0.88])
    
    # Synthetic target score calculation with noise
    edu_weights = {"High School": 0, "Associate": 2, "Bachelor": 4, "Master": 6, "Doctorate": 8}
    edu_bonus = np.array([edu_weights[e] for e in parent_edu])
    internet_bonus = np.where(internet_access == "Yes", 3, 0)
    extra_bonus = np.where(extra_activities == "Yes", 2, -1)
    
    # Core score formula
    base_score = (
        0.35 * previous_grade +
        0.30 * (attendance * 0.7) +
        0.25 * (study_hours * 2.2) -
        0.80 * absences +
        0.50 * sleep_hours +
        edu_bonus +
        internet_bonus +
        extra_bonus
    )
    
    # Add noise & map to 0-100 scale
    noise = np.random.normal(0, 5, size=num_samples)
    exam_score = np.round(np.clip(base_score + noise, 25, 100), 1)
    
    # Pass / Fail classification (Threshold = 50.0)
    pass_fail = np.where(exam_score >= 50.0, "Pass", "Fail")
    
    df = pd.DataFrame({
        "Gender": genders,
        "Age": ages,
        "StudyHoursPerWeek": study_hours,
        "AttendancePercentage": attendance,
        "PreviousGrade": previous_grade,
        "SleepHours": sleep_hours,
        "Absences": absences,
        "ParentEducation": parent_edu,
        "FamilyIncome": family_income,
        "ExtracurricularActivities": extra_activities,
        "InternetAccess": internet_access,
        "ExamScore": exam_score,
        "PassFail": pass_fail
    })
    
    # Introduce small missing value rate (~1%) to demonstrate EDA & cleaning pipeline capability
    missing_mask_study = np.random.rand(num_samples) < 0.015
    df.loc[missing_mask_study, "StudyHoursPerWeek"] = np.nan
    
    missing_mask_attendance = np.random.rand(num_samples) < 0.01
    df.loc[missing_mask_attendance, "AttendancePercentage"] = np.nan

    # Ensure parent output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully generated dataset with {num_samples} records at {output_path}")
    logger.info(f"Pass count: {(df['PassFail'] == 'Pass').sum()}, Fail count: {(df['PassFail'] == 'Fail').sum()}")
    return df

if __name__ == "__main__":
    generate_student_dataset()
