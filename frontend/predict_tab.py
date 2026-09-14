import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from config.settings import (
    BEST_MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    GENDER_OPTIONS, PARENT_EDU_OPTIONS, BINARY_OPTIONS
)
from training.train_pipeline import train_and_evaluate_all
from utils.risk_analyzer import analyze_student_risk
from frontend.report_generator import generate_pdf_report

def render_predict_tab():
    """Renders the Single Student Prediction & Risk Assessment Page."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 20px !important; margin-bottom: 4px !important;">Student Performance & Risk Predictor</h2>
        <p style="color: var(--text-secondary); margin: 0;">Specify student demographic, academic, and lifestyle parameters to generate predictive risk analytics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        st.info("No trained model found. Executing initial training pipeline...")
        with st.spinner("Training baseline machine learning models..."):
            train_and_evaluate_all()
            
    # Load Model Artifacts
    pipeline_artifact = joblib.load(BEST_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    
    classifier = pipeline_artifact["classifier"]
    regressor = pipeline_artifact["regressor"]
    best_model_name = pipeline_artifact.get("best_model_name", "Machine Learning Model")
    
    st.markdown(f"<div class='caption-text' style='margin-bottom: 16px;'>Active Inference Model: <b>{best_model_name}</b></div>", unsafe_allow_html=True)
    
    # Input Form Layout
    with st.form("student_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 12px;'>Demographics</div>", unsafe_allow_html=True)
            gender = st.selectbox("Gender", GENDER_OPTIONS)
            age = st.slider("Age", 14, 25, 18)
            parent_edu = st.selectbox("Parent Education Level", PARENT_EDU_OPTIONS)
            family_income = st.number_input("Family Income ($)", min_value=10000, max_value=200000, value=55000, step=5000)
            
        with col2:
            st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 12px;'>Academic History</div>", unsafe_allow_html=True)
            attendance = st.slider("Attendance Rate (%)", 30.0, 100.0, 82.0, step=0.5)
            study_hours = st.slider("Weekly Study Hours", 1.0, 45.0, 16.0, step=0.5)
            prev_grade = st.slider("Previous Grade (%)", 30.0, 100.0, 68.0, step=0.5)
            absences = st.slider("Class Absences", 0, 30, 4)
            
        with col3:
            st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 12px;'>Lifestyle Parameters</div>", unsafe_allow_html=True)
            sleep_hours = st.slider("Nightly Sleep (hrs)", 4.0, 10.0, 7.5, step=0.5)
            extra_activities = st.selectbox("Extracurricular Activities", BINARY_OPTIONS, index=1)
            internet_access = st.selectbox("Home Internet Access", BINARY_OPTIONS, index=1)
            
        submit_btn = st.form_submit_button("Run Diagnostic Inference", use_container_width=True)
        
    if submit_btn:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "StudyHoursPerWeek": study_hours,
            "AttendancePercentage": attendance,
            "PreviousGrade": prev_grade,
            "SleepHours": sleep_hours,
            "Absences": absences,
            "ParentEducation": parent_edu,
            "FamilyIncome": family_income,
            "ExtracurricularActivities": extra_activities,
            "InternetAccess": internet_access
        }])
        
        # Transform input
        X_trans = preprocessor.transform(input_data)
        
        # Class & Score Inference
        pass_prob = classifier.predict_proba(X_trans)[0, 1]
        pred_class_val = classifier.predict(X_trans)[0]
        pred_label = "Pass" if pred_class_val == 1 else "Fail"
        
        pred_score = float(regressor.predict(X_trans)[0])
        
        # Compute engineered Academic Risk Score
        from preprocessing.feature_engineering import add_engineered_features
        eng_df = add_engineered_features(input_data)
        acad_risk_score = float(eng_df["AcademicRiskScore"].iloc[0])
        
        # Risk Evaluation
        risk_res = analyze_student_risk(
            pass_prob=pass_prob,
            pred_score=pred_score,
            study_hours=study_hours,
            attendance=attendance,
            prev_grade=prev_grade,
            absences=absences,
            sleep_hours=sleep_hours,
            academic_risk_score=acad_risk_score
        )
        
        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='font-size: 18px !important; margin-bottom: 16px !important;'>Diagnostic Results & Risk Assessment</h3>", unsafe_allow_html=True)
        
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        
        with res_col1:
            status_color = "#16A34A" if pred_label == "Pass" else "#DC2626"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {status_color};">{pred_label}</div>
                <div class="metric-label">Predicted Outcome</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{risk_res['predicted_score']:.1f}</div>
                <div class="metric-label">Expected Score (/100)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{risk_res['pass_probability']:.1f}%</div>
                <div class="metric-label">Pass Probability</div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {risk_res['badge_color']}; font-size: 22px;">{risk_res['risk_level']}</div>
                <div class="metric-label">Assessed Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        # Confidence Gauge & Recommendations
        g_col, r_col = st.columns([1, 1])
        
        with g_col:
            st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;'>Probability Gauge</div>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_res["pass_probability"],
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Pass Probability (%)", "font": {"color": "#0F172A", "family": "Inter", "size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#0F172A"},
                    "bar": {"color": "#4F46E5"},
                    "steps": [
                        {"range": [0, 40], "color": "#FEE2E2"},
                        {"range": [40, 70], "color": "#FEF9C3"},
                        {"range": [70, 100], "color": "#DCFCE7"}
                    ],
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", height=280, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with r_col:
            st.markdown("<div style='font-size: 15px; font-weight: 600; color: var(--text-primary); margin-bottom: 8px;'>Targeted Recommendations</div>", unsafe_allow_html=True)
            for rec in risk_res["recommendations"]:
                st.markdown(f"""
                <div class="recommendation-item">
                    <div class="rec-category">[{rec['priority']} Priority] {rec['category']}</div>
                    <div class="rec-action">{rec['action']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # PDF Generation
            pdf_bytes = generate_pdf_report(input_data.iloc[0].to_dict(), risk_res)
            st.download_button(
                label="Download PDF Diagnostic Report",
                data=pdf_bytes,
                file_name=f"Student_Performance_Report_{age}yo.pdf",
                mime="application/pdf",
                use_container_width=True
            )
