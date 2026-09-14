import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from config.settings import BEST_MODEL_PATH, PREPROCESSOR_PATH
from preprocessing.feature_engineering import add_engineered_features
from utils.risk_analyzer import analyze_student_risk

def render_batch_tab():
    """Renders Batch CSV Prediction UI."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 20px !important; margin-bottom: 4px !important;">Batch CSV Inference</h2>
        <p style="color: var(--text-secondary); margin: 0;">Upload student cohort CSV files for bulk model execution and automated risk categorization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download sample template
    sample_data = pd.DataFrame([{
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
    }])
    sample_csv = sample_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Sample CSV Template", data=sample_csv, file_name="sample_student_batch.csv", mime="text/csv")
    
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select CSV File for Processing", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} records from CSV file.")
            
            if not BEST_MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
                st.error("Trained model artifacts not found. Please train models in Model Performance first.")
                return
                
            pipeline_artifact = joblib.load(BEST_MODEL_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            
            classifier = pipeline_artifact["classifier"]
            regressor = pipeline_artifact["regressor"]
            
            # Feature transform
            X_trans = preprocessor.transform(batch_df)
            
            probs = classifier.predict_proba(X_trans)[:, 1]
            preds = classifier.predict(X_trans)
            scores = regressor.predict(X_trans)
            
            # Compute Risk Levels
            eng_df = add_engineered_features(batch_df)
            risk_levels = []
            
            for i in range(len(batch_df)):
                p_prob = float(probs[i])
                p_score = float(scores[i])
                s_hrs = float(batch_df.get("StudyHoursPerWeek", pd.Series([15]*len(batch_df))).iloc[i])
                att = float(batch_df.get("AttendancePercentage", pd.Series([80]*len(batch_df))).iloc[i])
                pg = float(batch_df.get("PreviousGrade", pd.Series([65]*len(batch_df))).iloc[i])
                ab = int(batch_df.get("Absences", pd.Series([4]*len(batch_df))).iloc[i])
                slp = float(batch_df.get("SleepHours", pd.Series([7]*len(batch_df))).iloc[i])
                ars = float(eng_df["AcademicRiskScore"].iloc[i])
                
                analysis = analyze_student_risk(p_prob, p_score, s_hrs, att, pg, ab, slp, ars)
                risk_levels.append(analysis["risk_level"])
                
            results_df = batch_df.copy()
            results_df["Predicted_Outcome"] = np.where(preds == 1, "Pass", "Fail")
            results_df["Predicted_ExamScore"] = np.round(scores, 1)
            results_df["Pass_Probability_%"] = np.round(probs * 100, 2)
            results_df["Assessed_Risk_Level"] = risk_levels
            
            st.markdown("<h3 style='font-size: 18px !important; margin-top: 24px !important; margin-bottom: 12px !important;'>Inference Output Table</h3>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)
            
            # Risk Level Bar Chart
            risk_counts = results_df["Assessed_Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Student Count"]
            
            fig_risk = px.bar(
                risk_counts, x="Risk Level", y="Student Count", color="Risk Level",
                title="Batch Cohort Risk Level Breakdown",
                color_discrete_map={
                    "Low Risk": "#166534", "Medium Risk": "#854D0E",
                    "High Risk": "#9A3412", "Critical Risk": "#991B1B"
                }
            )
            fig_risk.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Export CSV Button
            out_csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Export Full Batch Inference Results (CSV)",
                data=out_csv,
                file_name="student_predictions_batch_output.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing batch CSV file: {str(e)}")
