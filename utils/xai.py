import numpy as np
import pandas as pd
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None


from typing import Dict, Any, Tuple, Optional
from utils.logger import get_logger


logger = get_logger("XAI")

def compute_shap_explanations(model, X_sample: np.ndarray, feature_names: list) -> Dict[str, Any]:
    """Computes SHAP values, feature importance rankings, and local attributions."""
    try:
        # Determine appropriate explainer based on model type
        model_type_str = str(type(model)).lower()
        if any(tree_type in model_type_str for tree_type in ["forest", "tree", "boosting", "xgb"]):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, min(20, len(X_sample))))
            
        shap_values = explainer(X_sample)
        
        # Handle multi-class / 2D outputs for binary classification
        if len(shap_values.shape) == 3:  # (samples, features, classes)
            shap_vals_matrix = shap_values.values[:, :, 1]
        else:
            shap_vals_matrix = shap_values.values
            
        # Global feature importance calculation (mean absolute SHAP value)
        mean_abs_shap = np.mean(np.abs(shap_vals_matrix), axis=0)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_abs_shap
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "shap_vals_matrix": shap_vals_matrix,
            "importance_df": importance_df
        }
    except Exception as e:
        logger.error(f"Error computing SHAP values: {str(e)}")
        # Fallback to feature importances attribute if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
            
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        return {
            "explainer": None,
            "shap_values": None,
            "shap_vals_matrix": None,
            "importance_df": importance_df
        }

def get_single_prediction_impact(model, X_single_transformed: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Computes feature attributions for a single student instance."""
    try:
        model_type_str = str(type(model)).lower()
        if any(tree_type in model_type_str for tree_type in ["forest", "tree", "boosting", "xgb"]):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_single_transformed)
            if len(shap_values.shape) == 3:
                val = shap_values.values[0, :, 1]
            else:
                val = shap_values.values[0, :]
        else:
            if hasattr(model, "coef_"):
                val = model.coef_[0] * X_single_transformed[0]
            else:
                val = np.zeros(len(feature_names))
                
        impact_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": val,
            "Direction": np.where(val > 0, "Positive (Increases Pass)", "Negative (Decreases Pass)")
        }).sort_values(by="SHAP Value", key=abs, ascending=False).reset_index(drop=True)
        
        return impact_df
    except Exception as e:
        logger.error(f"Error generating local SHAP impact: {str(e)}")
        return pd.DataFrame({"Feature": feature_names, "SHAP Value": 0.0, "Direction": "Neutral"})
