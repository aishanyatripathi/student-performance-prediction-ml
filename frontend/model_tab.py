import json
import streamlit as st
import pandas as pd
import plotly.express as px

from config.settings import METRICS_PATH
from training.train_pipeline import train_and_evaluate_all

def render_model_tab():
    """Renders Model Evaluation & Benchmarking Dashboard."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 20px !important; margin-bottom: 4px !important;">Model Benchmarking & Performance</h2>
        <p style="color: var(--text-secondary); margin: 0;">Evaluation metrics, cross-validation scores, and confusion matrix diagnostics across 7 ML algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_retrain, col_info = st.columns([1, 3])
    with col_retrain:
        if st.button("Retrain Model Suite", use_container_width=True):
            with st.spinner("Executing 5-Fold Stratified Cross Validation pipeline..."):
                train_and_evaluate_all()
                st.success("Retraining pipeline completed successfully.")
                
    if not METRICS_PATH.exists():
        with st.spinner("Initializing metrics data..."):
            train_and_evaluate_all()
            
    with open(METRICS_PATH, "r") as f:
        metrics_data = json.load(f)
        
    best_name = metrics_data.get("best_model_name", "Random Forest")
    st.markdown(f"""
    <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-left: 3px solid var(--accent); border-radius: var(--radius-sm); padding: 14px 20px; margin: 16px 0;">
        <div class="caption-text" style="text-transform: uppercase; letter-spacing: 0.04em;">Active Champion Classifier</div>
        <div style="color: var(--text-primary); font-size: 18px; font-weight: 700; margin-top: 2px;">{best_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    all_metrics = metrics_data.get("all_models_metrics", [])
    df_metrics = pd.DataFrame(all_metrics)
    
    # Format metrics table
    display_df = df_metrics[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_f1_mean", "cv_f1_std"]].copy()
    display_df.columns = ["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "CV F1 (Mean)", "CV F1 (Std)"]
    
    st.markdown("<h3 style='font-size: 18px !important; margin-top: 24px !important; margin-bottom: 12px !important;'>Model Comparison Matrix</h3>", unsafe_allow_html=True)
    st.dataframe(
        display_df.style.highlight_max(axis=0, color="#EEF2FF", subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]),
        use_container_width=True,
        hide_index=True
    )
    
    # Metric Benchmarks Bar Chart
    st.markdown("<h3 style='font-size: 18px !important; margin-top: 28px !important; margin-bottom: 12px !important;'>Metric Benchmark Explorer</h3>", unsafe_allow_html=True)
    selected_metric = st.selectbox("Select Evaluation Metric", ["f1_score", "roc_auc", "accuracy", "cv_f1_mean"], index=0)
    
    fig_comp = px.bar(
        df_metrics, x="model_name", y=selected_metric, color="model_name",
        title=f"Benchmark Comparison: {selected_metric.upper()}",
        text=selected_metric,
        color_discrete_sequence=px.colors.sequential.Purples_r
    )
    fig_comp.update_traces(texttemplate='%{text:.4f}', textposition='outside')
    fig_comp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Confusion Matrix Explorer
    st.markdown("<h3 style='font-size: 18px !important; margin-top: 28px !important; margin-bottom: 12px !important;'>Confusion Matrix Diagnostics</h3>", unsafe_allow_html=True)
    selected_model_cm = st.selectbox("Select Model for Diagnostic Matrix", df_metrics["model_name"].tolist(), index=0)
    
    model_row = df_metrics[df_metrics["model_name"] == selected_model_cm].iloc[0]
    cm = model_row["confusion_matrix"]
    
    fig_cm = px.imshow(
        cm, text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Records"),
        x=["Fail (0)", "Pass (1)"], y=["Fail (0)", "Pass (1)"],
        title=f"Confusion Matrix ({selected_model_cm})",
        color_continuous_scale="Purples"
    )
    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_cm, use_container_width=True)
