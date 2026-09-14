import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from config.settings import DATASET_PATH
from dataset.generate_data import generate_student_dataset

def render_dashboard_tab():
    """Renders the Dataset Overview & EDA tab."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 20px !important; margin-bottom: 4px !important;">Dataset Overview & Analytics</h2>
        <p style="color: var(--text-secondary); margin: 0;">Comprehensive student cohort metrics, distribution profiles, and data quality diagnostics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not DATASET_PATH.exists():
        st.warning("Dataset file not found. Auto-generating baseline dataset...")
        df = generate_student_dataset(1200, DATASET_PATH)
    else:
        df = pd.read_csv(DATASET_PATH)
        
    # Metric Summary Cards (4-Column Layout)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Cohort Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        pass_rate = (df['PassFail'] == 'Pass').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{pass_rate:.1f}%</div>
            <div class="metric-label">Pass Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_score = df['ExamScore'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.1f}</div>
            <div class="metric-label">Mean Exam Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_attendance = df['AttendancePercentage'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_attendance:.1f}%</div>
            <div class="metric-label">Mean Attendance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    # Data Quality Diagnostic Section
    st.markdown("<h3 style='font-size: 18px !important; margin-bottom: 12px !important;'>Data Integrity Summary</h3>", unsafe_allow_html=True)
    q_col1, q_col2 = st.columns(2)
    
    with q_col1:
        missing_count = df.isnull().sum()
        missing_df = pd.DataFrame({"Feature": missing_count.index, "Missing Values": missing_count.values})
        missing_df["Missing %"] = np.round((missing_df["Missing Values"] / len(df)) * 100, 2)
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
        
    with q_col2:
        dup_count = df.duplicated().sum()
        st.markdown(f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 20px; font-size: 14px; line-height: 1.8;">
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 8px;">Cohort Structure Metadata</div>
            <div>• <b>Total Records:</b> {df.shape[0]}</div>
            <div>• <b>Total Feature Columns:</b> {df.shape[1]}</div>
            <div>• <b>Duplicate Rows Detected:</b> {dup_count}</div>
            <div>• <b>Pass Outcomes:</b> {(df['PassFail']=='Pass').sum()}</div>
            <div>• <b>Fail Outcomes:</b> {(df['PassFail']=='Fail').sum()}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    
    # Visualizations Tabs
    st.markdown("<h3 style='font-size: 18px !important; margin-bottom: 16px !important;'>Exploratory Visualizations</h3>", unsafe_allow_html=True)
    viz_tabs = st.tabs(["Outcome Distribution", "Correlation Matrix", "Feature Histograms", "Outlier Analysis"])
    
    with viz_tabs[0]:
        fig_class = px.pie(
            df, names="PassFail", title="Class Ratio (Pass vs Fail)",
            color="PassFail", color_discrete_map={"Pass": "#4F46E5", "Fail": "#0F172A"},
            hole=0.4
        )
        fig_class.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_class, use_container_width=True)
        
    with viz_tabs[1]:
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", title="Feature Correlation Heatmap",
            color_continuous_scale="Purples", aspect="auto"
        )
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with viz_tabs[2]:
        selected_hist_col = st.selectbox("Select Feature Column", numeric_df.columns, index=0)
        fig_hist = px.histogram(
            df, x=selected_hist_col, color="PassFail", marginal="rug",
            title=f"Distribution of {selected_hist_col} grouped by Outcome",
            color_discrete_map={"Pass": "#4F46E5", "Fail": "#0F172A"},
            opacity=0.8
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with viz_tabs[3]:
        selected_box_col = st.selectbox("Select Feature for Outlier Boxplot", numeric_df.columns, index=1)
        fig_box = px.box(
            df, y=selected_box_col, x="PassFail", color="PassFail",
            title=f"Boxplot of {selected_box_col} across Outcomes",
            color_discrete_map={"Pass": "#4F46E5", "Fail": "#0F172A"}
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#0F172A", font_family="Inter", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)
