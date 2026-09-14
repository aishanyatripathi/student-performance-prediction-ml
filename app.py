import streamlit as st

# Streamlit page configuration MUST be the first command
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from frontend.styles import get_custom_css
from frontend.dashboard_tab import render_dashboard_tab
from frontend.predict_tab import render_predict_tab
from frontend.batch_tab import render_batch_tab
from frontend.model_tab import render_model_tab
from frontend.xai_tab import render_xai_tab

# Inject Enterprise SaaS CSS Design Tokens
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Top Header Layout (Linear / Vercel Enterprise Header)
st.markdown("""
<div class="app-header">
    <div class="app-header-top">
        <span class="header-tag">Production ML Engine</span>
        <span class="caption-text">Hugging Face & REST API Deployable</span>
    </div>
    <h1 style="margin: 0; font-size: 28px !important;">Student Performance Prediction System</h1>
    <p class="header-sub">Automated Machine Learning Diagnostics • SHAP Explainability • Academic Risk Interventions</p>
</div>
""", unsafe_allow_html=True)

# Modern Underline Tab Bar Navigation
tab_dash, tab_pred, tab_batch, tab_model, tab_xai = st.tabs([
    "Overview",
    "Student Predictor",
    "Batch Inference",
    "Model Performance",
    "Explainable AI"
])

with tab_dash:
    render_dashboard_tab()

with tab_pred:
    render_predict_tab()
    
with tab_batch:
    render_batch_tab()
    
with tab_model:
    render_model_tab()
    
with tab_xai:
    render_xai_tab()
    
# Footer
st.markdown("""
<div class="app-footer">
    Student Performance Prediction System • Built with FastAPI, Streamlit, Scikit-learn & SHAP
</div>
""", unsafe_allow_html=True)
