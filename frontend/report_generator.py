import io
from typing import Dict, Any, List
from fpdf import FPDF

class StudentReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, "AI Student Performance & Risk Analysis Report", ln=True, align="C")
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Automated Machine Learning Diagnostics & Interventions", ln=True, align="C")
        self.ln(5)
        self.set_draw_color(203, 213, 225)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Page {self.page_no()} | AI-Driven Student Performance Prediction System", align="C")

def generate_pdf_report(student_inputs: Dict[str, Any], risk_res: Dict[str, Any]) -> bytes:
    """Generates a downloadable PDF student performance diagnostic report."""
    pdf = StudentReportPDF()
    pdf.add_page()
    
    # 1. Executive Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "1. Executive Diagnostic Summary", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 7, f"Predicted Final Exam Score:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, f"{risk_res['predicted_score']:.1f} / 100", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 7, f"Pass Probability:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(40, 7, f"{risk_res['pass_probability']:.1f}%", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 7, f"Assessed Risk Status:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    
    # Color badge text
    if risk_res["risk_level"] == "Critical Risk":
        pdf.set_text_color(239, 68, 68)
    elif risk_res["risk_level"] == "High Risk":
        pdf.set_text_color(249, 115, 22)
    elif risk_res["risk_level"] == "Medium Risk":
        pdf.set_text_color(234, 179, 8)
    else:
        pdf.set_text_color(34, 197, 94)
        
    pdf.cell(40, 7, f"{risk_res['risk_level']}", ln=True)
    pdf.set_text_color(15, 23, 42)
    
    pdf.ln(5)
    
    # 2. Student Input Metrics Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Student Academic Profile & Metrics", ln=True)
    
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(60, 6, "Metric / Feature", border=1, fill=True)
    pdf.cell(40, 6, "Value", border=1, fill=True, ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    metrics_to_show = [
        ("Gender", student_inputs.get("Gender", "N/A")),
        ("Age", student_inputs.get("Age", "N/A")),
        ("Attendance Percentage", f"{student_inputs.get('AttendancePercentage', 0):.1f}%"),
        ("Study Hours / Week", f"{student_inputs.get('StudyHoursPerWeek', 0):.1f} hrs"),
        ("Previous Grade", f"{student_inputs.get('PreviousGrade', 0):.1f}%"),
        ("Absences", student_inputs.get("Absences", 0)),
        ("Sleep Hours", f"{student_inputs.get('SleepHours', 0):.1f} hrs"),
        ("Parent Education", student_inputs.get("ParentEducation", "N/A")),
        ("Extracurricular Activities", student_inputs.get("ExtracurricularActivities", "N/A")),
        ("Internet Access", student_inputs.get("InternetAccess", "N/A")),
    ]
    
    for label, val in metrics_to_show:
        pdf.cell(60, 6, str(label), border=1)
        pdf.cell(40, 6, str(val), border=1, ln=True)
        
    pdf.ln(6)
    
    # 3. Targeted Recommendations & Interventions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Recommended Academic Interventions", ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    for rec in risk_res.get("recommendations", []):
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 5, f"- [{rec['priority']} Priority] {rec['category']}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, f"  Action: {rec['action']}")
        pdf.ln(2)

        
    return bytes(pdf.output())
