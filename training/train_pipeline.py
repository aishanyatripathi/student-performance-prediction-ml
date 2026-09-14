import json
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    XGBClassifier = None


from config.settings import (
    DATASET_PATH, MODELS_DIR, BEST_MODEL_PATH, PREPROCESSOR_PATH,
    FEATURE_NAMES_PATH, METRICS_PATH, RANDOM_STATE, TEST_SIZE,
    TARGET_CLASSIFICATION, TARGET_REGRESSION
)
from dataset.generate_data import generate_student_dataset
from preprocessing.cleaner import StudentDataPreprocessor
from training.evaluate import evaluate_classifier, evaluate_regressor
from utils.logger import get_logger

logger = get_logger("TrainPipeline")

def load_data() -> pd.DataFrame:
    """Loads dataset from CSV or generates synthetic data if missing."""
    if not DATASET_PATH.exists():
        logger.info(f"Dataset not found at {DATASET_PATH}. Auto-generating Kaggle-style dataset...")
        df = generate_student_dataset(num_samples=1200, output_path=DATASET_PATH)
    else:
        logger.info(f"Loading existing dataset from {DATASET_PATH}...")
        df = pd.read_csv(DATASET_PATH)
    return df

def train_and_evaluate_all() -> Dict[str, Any]:
    """Trains 7 ML models, benchmarks performance, selects best model, and saves joblib artifacts."""
    df = load_data()
    
    # Encode target column: Pass -> 1, Fail -> 0
    y_class = (df[TARGET_CLASSIFICATION].astype(str).str.strip().str.lower() == "pass").astype(int)
    y_reg = df[TARGET_REGRESSION]
    
    X_raw = df.drop(columns=[TARGET_CLASSIFICATION, TARGET_REGRESSION], errors="ignore")
    
    # 1. Fit Preprocessor
    preprocessor = StudentDataPreprocessor(handle_outliers=True)
    X_transformed = preprocessor.fit_transform(X_raw)
    feature_names = preprocessor.get_feature_names()
    
    # 2. Train Test Split
    X_train, X_test, y_train, y_test, y_reg_train, y_reg_test = train_test_split(
        X_transformed, y_class, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )
    
    models_to_train = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }
    
    if HAS_XGBOOST:
        models_to_train["XGBoost"] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE, eval_metric="logloss")

    
    logger.info(f"Starting training pipeline on {len(models_to_train)} machine learning models...")
    
    all_results = []
    fitted_models = {}
    
    for name, model in models_to_train.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        eval_metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test, name)
        all_results.append(eval_metrics)
        fitted_models[name] = model
        
    # Also fit exact score regressor for score prediction
    regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)
    regressor.fit(X_train, y_reg_train)
    reg_metrics = evaluate_regressor(regressor, X_test, y_reg_test, "Exam Score Regressor")
    logger.info(f"Exam Score Regressor MAE: {reg_metrics['mae']}, RMSE: {reg_metrics['rmse']}, R2: {reg_metrics['r2_score']}")

    # 4. Auto-select Best Model based on F1 Score & ROC AUC
    best_result = max(all_results, key=lambda x: (x["f1_score"], x["roc_auc"]))
    best_name = best_result["model_name"]
    best_classifier = fitted_models[best_name]
    
    logger.info(f"*** BEST MODEL SELECTED: {best_name} (F1 Score: {best_result['f1_score']}, ROC AUC: {best_result['roc_auc']})")
    
    # Package pipeline artifact (Classifier + Regressor)
    pipeline_artifact = {
        "best_model_name": best_name,
        "classifier": best_classifier,
        "regressor": regressor,
        "all_classifiers": fitted_models
    }
    
    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline_artifact, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    
    metrics_summary = {
        "best_model_name": best_name,
        "best_model_f1": best_result["f1_score"],
        "best_model_accuracy": best_result["accuracy"],
        "best_model_roc_auc": best_result["roc_auc"],
        "all_models_metrics": all_results,
        "regressor_metrics": reg_metrics,
        "feature_count": len(feature_names),
        "total_records": len(df)
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    logger.info(f"All model artifacts successfully exported to {MODELS_DIR}")
    return metrics_summary

if __name__ == "__main__":
    train_and_evaluate_all()
