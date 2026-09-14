import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH, DATASET_PATH
from utils.xai import compute_shap_explanations

def render_xai_tab():
    """Renders Explainable AI (SHAP) Interpretability Dashboard."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 20px !important; margin-bottom: 4px !important;">Explainable AI (SHAP Diagnostics)</h2>
        <p style="color: var(--text-secondary); margin: 0;">Quantify feature attribution weights and model decision drivers using SHAP (SHapley Additive exPlanations).</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not BEST_MODEL_PATH.exists() or not DATASET_PATH.exists():
        st.warning("Model artifacts missing. Please train models from Model Performance first.")
        return
        
    pipeline_artifact = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    classifier = pipeline_artifact["classifier"]
    best_model_name = pipeline_artifact.get("best_model_name", "Classifier")
    
    st.markdown(f"<div class='caption-text' style='margin-bottom: 16px;'>Interpreting Classifier: <b>{best_model_name}</b></div>", unsafe_allow_html=True)
    
    df = pd.read_csv(DATASET_PATH)
    X_raw = df.drop(columns=["PassFail", "ExamScore"], errors="ignore").sample(min(150, len(df)), random_state=42)
    X_sample_trans = preprocessor.transform(X_raw)
    
    with st.spinner("Computing SHAP feature attributions..."):
        shap_res = compute_shap_explanations(classifier, X_sample_trans, feature_names)
        
    importance_df = shap_res["importance_df"].head(12)
    
    st.markdown("<h3 style='font-size: 18px !important; margin-top: 24px !important; margin-bottom: 12px !important;'>Global Feature Importance</h3>", unsafe_allow_html=True)
    
    fig_imp = px.bar(
        importance_df, y="Feature", x="Importance", orientation="h",
        title="Top 12 Features Influencing Pass/Fail Decision Boundary",
        color="Importance", color_continuous_scale="Purples"
    )
    fig_imp.update_layout(
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter",
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 18px !important; margin-bottom: 12px !important;'>Feature Weight Matrix</h3>", unsafe_allow_html=True)
    st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-left: 3px solid var(--accent); border-radius: var(--radius-md); padding: 20px; margin-top: 24px;">
        <div style="font-weight: 600; color: var(--text-primary); font-size: 15px; margin-bottom: 6px;">Interpretability Analysis Notes</div>
        <ul style="color: var(--text-secondary); font-size: 14px; line-height: 1.6; margin: 0; padding-left: 20px;">
            <li>Features near the top (e.g., <b>AttendancePercentage</b>, <b>PreviousGrade</b>, and composite <b>AcademicRiskScore</b>) exert the strongest mathematical force on model predictions.</li>
            <li>Composite engineered features effectively reduce multi-collinearity while providing high explanatory power for academic decision boundaries.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
