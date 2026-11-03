from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"

def main():
    if not MODEL_PATH.exists():
        print("Model not found. Run 'python main.py' first.")
        return
    model = joblib.load(MODEL_PATH)
    applicant = pd.DataFrame([{
        "age": 32,
        "annual_income": 72000,
        "employment_years": 6,
        "loan_amount": 12000,
        "loan_term_months": 36,
        "existing_debt": 5000,
        "credit_history_years": 8,
        "missed_payments": 0,
        "home_ownership": "rent",
        "employment_type": "full_time",
        "loan_purpose": "vehicle"
    }])
    prediction = int(model.predict(applicant)[0])
    probability = float(model.predict_proba(applicant)[0, 1])
    label = "Good repayment outcome (demo)" if prediction == 1 else "Bad repayment outcome (demo)"
    print("\\nCUSTOM APPLICANT RESULT")
    print("Prediction:", label)
    print(f"Model probability for good outcome: {probability:.2%}")
    print("Educational synthetic-data example only; not a lending decision.")

if __name__ == "__main__":
    main()
