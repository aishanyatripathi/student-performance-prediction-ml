import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from config.settings import RANDOM_STATE, CV_FOLDS
from utils.logger import get_logger

logger = get_logger("Evaluator")

def evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics, confusion matrix, and Stratified K-Fold CV score."""
    y_pred = model.predict(X_test)
    
    # Calculate probability for ROC-AUC if model supports predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.5
        
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # Cross Validation
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    
    results = {
        "model_name": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "cv_f1_mean": round(float(np.mean(cv_scores)), 4),
        "cv_f1_std": round(float(np.std(cv_scores)), 4),
        "confusion_matrix": cm
    }
    
    logger.info(f"Model: {model_name:<20} | Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | CV F1: {results['cv_f1_mean']:.4f}")
    return results

def evaluate_regressor(model, X_test, y_test, model_name: str) -> Dict[str, Any]:
    """Calculates regression metrics for Exam Score prediction."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        "model_name": model_name,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4)
    }
