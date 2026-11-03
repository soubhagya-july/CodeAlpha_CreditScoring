from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
for folder in ["data", "models", "outputs"]:
    (ROOT / folder).mkdir(exist_ok=True)

def create_synthetic_dataset(n=5000, seed=42):
    """Create illustrative synthetic data; not real credit bureau data."""
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 71, n)
    income = np.clip(rng.normal(65000, 22000, n), 15000, 180000).round()
    employment_years = np.clip(rng.normal(7, 5, n), 0, 40).round(1)
    loan_amount = np.clip(rng.normal(16000, 9000, n), 1000, 60000).round()
    loan_term = rng.choice([12, 24, 36, 48, 60], n)
    existing_debt = np.clip(rng.normal(9000, 7000, n), 0, 50000).round()
    credit_history = np.clip(rng.normal(8, 5, n), 0, 35).round(1)
    missed_payments = rng.poisson(0.7, n)
    home = rng.choice(["rent", "own", "mortgage"], n, p=[.35, .25, .40])
    employment = rng.choice(["full_time", "part_time", "self_employed", "unemployed"],
                            n, p=[.62, .12, .20, .06])
    purpose = rng.choice(["debt_consolidation", "home_improvement", "education",
                          "medical", "vehicle", "other"], n)
    debt_ratio = (existing_debt + loan_amount) / (income + 1)
    risk = (-2.2 + 3.0 * debt_ratio + .42 * missed_payments
            + .018 * loan_amount / 1000 - .025 * employment_years
            - .035 * credit_history + .8 * (employment == "unemployed")
            + .25 * (employment == "self_employed") + .25 * (home == "rent")
            - .25 * (home == "own") + rng.normal(0, .55, n))
    probability_bad = 1 / (1 + np.exp(-risk))
    good_credit = 1 - rng.binomial(1, probability_bad)
    return pd.DataFrame({
        "age": age, "annual_income": income, "employment_years": employment_years,
        "loan_amount": loan_amount, "loan_term_months": loan_term,
        "existing_debt": existing_debt, "credit_history_years": credit_history,
        "missed_payments": missed_payments, "home_ownership": home,
        "employment_type": employment, "loan_purpose": purpose,
        "good_credit": good_credit
    })

def build_model():
    numeric = ["age", "annual_income", "employment_years", "loan_amount",
               "loan_term_months", "existing_debt", "credit_history_years",
               "missed_payments"]
    categorical = ["home_ownership", "employment_type", "loan_purpose"]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1))
    ])

def main():
    print("CODEALPHA - CREDIT SCORING (SYNTHETIC EDUCATIONAL DEMO)")
    root = Path(__file__).resolve().parent
    df = create_synthetic_dataset()
    df.to_csv(root / "data" / "credit_data.csv", index=False)
    X, y = df.drop(columns="good_credit"), df["good_credit"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=42, stratify=y)
    model = build_model()
    print("\\nTraining Random Forest...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    print(f"\\nAccuracy: {accuracy:.3%}")
    print(f"ROC-AUC: {auc:.4f}")
    print("\\nClassification report:")
    print(classification_report(y_test, pred,
          target_names=["Bad outcome (0)", "Good outcome (1)"], zero_division=0))
    with open(root / "outputs" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "roc_auc": auc}, f, indent=2)
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Bad", "Good"],
                                             cmap="Blues")
    plt.title("Credit Scoring - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(root / "outputs" / "confusion_matrix.png", dpi=150)
    plt.close()
    RocCurveDisplay.from_predictions(y_test, prob)
    plt.title("Credit Scoring - ROC Curve")
    plt.tight_layout()
    plt.savefig(root / "outputs" / "roc_curve.png", dpi=150)
    plt.close()
    joblib.dump(model, root / "models" / "credit_scoring_model.joblib")
    print("\\nModel saved in models/credit_scoring_model.joblib")
    print("Charts and metrics saved in outputs/. Training complete!")

if __name__ == "__main__":
    main()
