import joblib
import numpy as np

model  = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)
    pred     = int(model.predict(scaled)[0])
    probs    = model.predict_proba(scaled)[0]
    pass_prob   = float(probs[1])
    confidence  = float(max(probs))

    if pass_prob >= 0.80:
        band = "High"
    elif pass_prob >= 0.50:
        band = "Medium"
    else:
        band = "Low"

    label = "PASS" if pred == 1 else "FAIL"
    print(f"\nResult     : {label}")
    print(f"Pass Prob  : {pass_prob * 100:.1f}%")
    print(f"Confidence : {confidence * 100:.1f}%")
    print(f"Performance: {band}")
    return pred

if __name__ == "__main__":
    print("=== Student Evaluator ===")
    attendance  = float(input("Attendance   (0-100): "))
    assignment  = float(input("Assignment   (0-100): "))
    quiz        = float(input("Quiz         (0-100): "))
    mid         = float(input("Mid-term     (0-100): "))
    study_hours = float(input("Study Hours (per week): "))
    evaluate_student(attendance, assignment, quiz, mid, study_hours)