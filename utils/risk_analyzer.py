from typing import Dict, Any, List

def analyze_student_risk(
    pass_prob: float,
    pred_score: float,
    study_hours: float,
    attendance: float,
    prev_grade: float,
    absences: int,
    sleep_hours: float,
    academic_risk_score: float
) -> Dict[str, Any]:
    """Analyzes risk profile of a student and generates dynamic academic recommendations."""
    
    # Risk Classification logic combining model probability & academic risk score
    fail_prob = 1.0 - pass_prob
    composite_risk = (fail_prob * 60.0) + (academic_risk_score * 0.40)
    
    if composite_risk >= 70.0 or pass_prob < 0.35:
        risk_level = "Critical Risk"
        badge_color = "#FF4D4D"  # Red
    elif composite_risk >= 45.0 or pass_prob < 0.65:
        risk_level = "High Risk"
        badge_color = "#FF9900"  # Orange
    elif composite_risk >= 25.0 or pass_prob < 0.85:
        risk_level = "Medium Risk"
        badge_color = "#FFD700"  # Yellow
    else:
        risk_level = "Low Risk"
        badge_color = "#00CC66"  # Green

    confidence = round(max(pass_prob, fail_prob) * 100.0, 2)
    
    # Custom Recommendations
    recommendations: List[Dict[str, str]] = []
    
    if attendance < 75.0:
        recommendations.append({
            "category": "Attendance",
            "priority": "High",
            "action": f"Current attendance is low ({attendance:.1f}%). Target a minimum of 85% attendance to prevent academic failure."
        })
        
    if study_hours < 14.0:
        recommendations.append({
            "category": "Study Habits",
            "priority": "High",
            "action": f"Increase study hours from {study_hours:.1f} hrs/week to at least 18-20 hrs/week with structured timetables."
        })
        
    if prev_grade < 60.0:
        recommendations.append({
            "category": "Academic Mentoring",
            "priority": "Critical" if prev_grade < 45 else "Medium",
            "action": f"Enrol in peer tutoring or subject counseling to address weak base concepts from previous grade ({prev_grade:.1f}%)."
        })
        
    if absences > 5:
        recommendations.append({
            "category": "Absence Management",
            "priority": "High",
            "action": f"Student has {absences} recorded absences. Conduct a parent-teacher counseling session to improve class presence."
        })

    if sleep_hours < 6.5 or sleep_hours > 9.5:
        recommendations.append({
            "category": "Lifestyle & Well-being",
            "priority": "Medium",
            "action": f"Adjust daily sleep routine ({sleep_hours:.1f} hrs). Aim for 7 to 8 hours of consistent nightly sleep for optimal cognitive performance."
        })
        
    if not recommendations:
        recommendations.append({
            "category": "Enrichment",
            "priority": "Low",
            "action": "Maintain current high study standards and consider participating in competitive academic challenges or advanced projects."
        })
        
    return {
        "risk_level": risk_level,
        "composite_risk_score": round(composite_risk, 2),
        "pass_probability": round(pass_prob * 100.0, 2),
        "fail_probability": round(fail_prob * 100.0, 2),
        "predicted_score": round(pred_score, 1),
        "confidence": confidence,
        "badge_color": badge_color,
        "recommendations": recommendations
    }
