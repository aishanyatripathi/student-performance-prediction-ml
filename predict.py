import argparse
import joblib
import pandas as pd
from typing import Dict, Any

from config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH
from preprocessing.feature_engineering import add_engineered_features
from utils.risk_analyzer import analyze_student_risk

def run_cli_inference(student_dict: Dict[str, Any]):
    """Executes single inference via CLI."""
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        print("Error: Trained model artifacts missing. Run 'python train.py' first.")
        return
        
    pipeline_artifact = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    classifier = pipeline_artifact["classifier"]
    regressor = pipeline_artifact["regressor"]
    
    input_df = pd.DataFrame([student_dict])
    X_trans = preprocessor.transform(input_df)
    
    pass_prob = float(classifier.predict_proba(X_trans)[0, 1])
    pred_class_val = int(classifier.predict(X_trans)[0])
    pred_label = "Pass" if pred_class_val == 1 else "Fail"
    pred_score = float(regressor.predict(X_trans)[0])
    
    eng_df = add_engineered_features(input_df)
    acad_risk_score = float(eng_df["AcademicRiskScore"].iloc[0])
    
    analysis = analyze_student_risk(
        pass_prob=pass_prob,
        pred_score=pred_score,
        study_hours=float(student_dict["StudyHoursPerWeek"]),
        attendance=float(student_dict["AttendancePercentage"]),
        prev_grade=float(student_dict["PreviousGrade"]),
        absences=int(student_dict["Absences"]),
        sleep_hours=float(student_dict["SleepHours"]),
        academic_risk_score=acad_risk_score
    )
    
    print("\n" + "="*50)
    print("AI STUDENT PERFORMANCE PREDICTION RESULT:")
    print("="*50)
    print(f"Predicted Outcome:     {pred_label}")
    print(f"Predicted Exam Score:  {analysis['predicted_score']} / 100")
    print(f"Pass Probability:      {analysis['pass_probability']}%")
    print(f"Risk Level:            {analysis['risk_level']}")
    print(f"Confidence Level:      {analysis['confidence']}%")
    print("\nRECOMMENDED ACTIONS:")
    for rec in analysis["recommendations"]:
        print(f"  • [{rec['priority']}] {rec['category']}: {rec['action']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    sample_student = {
        "Gender": "Female",
        "Age": 18,
        "StudyHoursPerWeek": 16.5,
        "AttendancePercentage": 85.0,
        "PreviousGrade": 72.0,
        "SleepHours": 7.5,
        "Absences": 3,
        "ParentEducation": "Bachelor",
        "FamilyIncome": 60000,
        "ExtracurricularActivities": "Yes",
        "InternetAccess": "Yes"
    }
    run_cli_inference(sample_student)
