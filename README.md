# 🎓 AI-Driven Student Performance Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Machine Learning system and interactive Web Dashboard designed for predicting student academic performance, categorizing risk levels (*Low*, *Medium*, *High*, *Critical Risk*), and providing Explainable AI (SHAP) insights alongside tailored academic intervention plans.

---

## 🌟 Key Features

* **Complete ML Pipeline**: Automated data loading, cleaning, missing value imputation, outlier clipping, one-hot encoding, feature scaling, and 5-fold cross-validation.
* **Domain Feature Engineering**: Automatically computes composite metrics including *Attendance Ratio*, *Study Efficiency*, *Academic Risk Score*, *Homework Consistency*, and *Lifestyle Score*.
* **Multi-Model Benchmarking**: Trains and evaluates 7 machine learning models (*Random Forest*, *Gradient Boosting*, *XGBoost*, *Decision Tree*, *Logistic Regression*, *SVM*, *KNN*) and selects the champion model based on F1 Score and ROC-AUC.
* **Explainable AI (SHAP)**: Global feature importance rankings and local force/waterfall attributions.
* **Risk Categorization & Advice**: Classifies student academic risk and generates dynamic, prioritized action steps (e.g. attendance targets, study hour adjustments, peer mentoring).
* **Dual User Interface**:
  * **Streamlit Interactive UI (`app.py`)**: Glassmorphism dark mode dashboard, live EDA, single/batch prediction, retrain triggers, and PDF report downloads.
  * **FastAPI REST API (`api/main.py`)**: High-performance REST API endpoints with Pydantic validation and auto-generated Swagger UI (`/docs`).
* **Automated PDF Diagnostics**: Generates downloadable diagnostic PDF reports summarizing student metrics, predicted score, risk status, and intervention plan.
* **Hugging Face Spaces Ready**: Native single-file entry points and configuration for zero-setup deployment.

---

## 🏗️ System Architecture

```
Ai_driven_stu_performance_prediction/
├── config/
│   └── settings.py              # Centralized configuration & hyperparams
├── dataset/
│   ├── student_performance.csv  # Kaggle-style realistic dataset
│   └── generate_data.py         # Synthetic dataset generator
├── models/                      # Saved joblib artifacts & metrics json
│   ├── best_model.joblib
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   └── metrics.json
├── preprocessing/
│   ├── cleaner.py               # Imputation, outlier capping, scaling
│   └── feature_engineering.py   # Domain engineered features
├── training/
│   ├── train_pipeline.py        # Model benchmarking & selection
│   └── evaluate.py              # Metrics evaluation & cross-validation
├── api/
│   ├── main.py                  # FastAPI application entry point
│   ├── schemas.py               # Pydantic input/output models
│   └── routes.py                # REST endpoints (/predict, /train, /health)
├── frontend/
│   ├── styles.py                # Modern glassmorphic dark CSS
│   ├── dashboard_tab.py         # EDA visual tab
│   ├── predict_tab.py           # Single student predictor UI
│   ├── batch_tab.py             # CSV batch upload UI
│   ├── model_tab.py             # Model performance benchmarking UI
│   ├── xai_tab.py               # SHAP feature importance tab
│   └── report_generator.py      # Automated PDF report generation
├── utils/
│   ├── logger.py                # Logging utility
│   ├── xai.py                   # SHAP explainer utilities
│   └── risk_analyzer.py         # Risk classification engine
├── tests/
│   ├── test_preprocessing.py    # Preprocessing unit tests
│   ├── test_model.py            # Model training & inference tests
│   └── test_api.py              # FastAPI endpoint tests
├── assets/
│   └── banner.png               # High-res UI banner asset
├── app.py                       # Main Streamlit web application
├── train.py                     # CLI script to execute ML training pipeline
├── predict.py                   # CLI inference tool
├── requirements.txt             # Dependency specification
├── README.md                    # System documentation
└── LICENSE                      # MIT License
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/AI-Student-Performance.git
cd AI-Student-Performance

pip install -r requirements.txt
```

### 2. Train Machine Learning Pipeline

Generate dataset and train all 7 machine learning algorithms:

```bash
python train.py
```

### 3. Launch Streamlit Web Dashboard

Start the interactive Web Application:

```bash
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 4. Run FastAPI Backend

Launch the production REST API server:

```bash
uvicorn api.main:app --reload --port 8000
```

Access interactive API Documentation (Swagger) at `http://localhost:8000/docs`.

---

## 📡 REST API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check and model status |
| `POST` | `/predict` | Single student inference & risk analysis |
| `GET` | `/model-info` | Metadata and metrics of active champion model |
| `GET` | `/feature-importance` | SHAP feature importance breakdown |
| `POST` | `/train` | Trigger model retraining pipeline |

### Example Input Payload (`POST /predict`):

```json
{
  "Gender": "Female",
  "Age": 18,
  "StudyHoursPerWeek": 16.5,
  "AttendancePercentage": 85.0,
  "PreviousGrade": 72.0,
  "SleepHours": 7.5,
  "Absences": 3,
  "ParentEducation": "Bachelor",
  "FamilyIncome": 60000.0,
  "ExtracurricularActivities": "Yes",
  "InternetAccess": "Yes"
}
```

---

## 🧪 Running Automated Unit Tests

Run the complete `pytest` test suite:

```bash
pytest tests/ -v
```

---

## 🤗 Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Commit and push all project files (`app.py`, `requirements.txt`, `config/`, `frontend/`, `models/`, `dataset/`, etc.) to the Space repository.
4. Hugging Face Spaces will automatically install dependencies and launch `app.py`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author & Acknowledgments

* **Developer**: ML & Full Stack Engineering Team
* **Suitability**: Final Year Major Project, College Submissions, GitHub Portfolio, Hugging Face Spaces, Internship Resumes.
