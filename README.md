# CodeAlpha Credit Scoring

A beginner-friendly machine-learning project using Python, scikit-learn and a Random Forest classifier.

**Important:** This project creates synthetic data for learning. It is not based on real credit bureau records and must not be used to approve/reject real loans or make financial decisions.

## Features
- Generates 5,000 synthetic applicant records
- Handles numeric and categorical features
- Trains a Random Forest classification model
- Reports accuracy, precision, recall, F1-score and ROC-AUC
- Saves confusion matrix and ROC curve plots
- Saves the trained model and includes a custom prediction script

## Folder structure
```text
CodeAlpha_CreditScoring/
├── data/       # Generated CSV dataset
├── models/     # Saved trained model
├── outputs/    # Metrics and evaluation charts
├── main.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup (Windows PowerShell)
Open this folder in VS Code and run:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

## Train and evaluate
```powershell
python main.py
```
This creates `data/credit_data.csv`, trains the model, prints evaluation metrics, saves charts under `outputs/`, and saves the model under `models/`.

## Make a sample prediction
```powershell
python predict.py
```
Run training first. You can change the example applicant values in `predict.py`.

## Target label
- `good_credit = 1`: synthetic good repayment outcome
- `good_credit = 0`: synthetic bad repayment outcome

The labels are generated using a made-up formula and random noise. Model metrics are only meaningful for this generated demonstration dataset.

## Responsible use
Real credit scoring requires appropriate lawful data, validation, privacy safeguards, explainability, fairness testing and human oversight. This demo does not provide those safeguards and is not suitable for real lending.
