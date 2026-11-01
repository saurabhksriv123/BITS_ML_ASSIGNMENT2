"""
Train and evaluate five classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

Outputs (written to this `model/` folder and the project root):
  - <model_name>.joblib        : trained scikit-learn pipelines (preprocessing + classifier)
  - feature_names.json         : ordered list of input features the models expect
  - metrics_summary.csv        : evaluation metrics of every model on the held-out test set
  - ../test_data.csv           : held-out test set (features + target) used for evaluation and in the app

Run from the project root:
    python model/train_models.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "diagnosis"          # 1 = malignant, 0 = benign

MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODEL_DIR.parent


def load_data() -> pd.DataFrame:
    """Load the UCI Breast Cancer Wisconsin (Diagnostic) data as a DataFrame.

    scikit-learn encodes 0 = malignant, 1 = benign. We recode so that the
    positive class (1) is malignant, which is the clinically important class.
    """
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df[TARGET] = (df["target"] == 0).astype(int)
    return df.drop(columns=["target"])


def build_models() -> dict:
    """All models are wrapped in a Pipeline so preprocessing travels with the model."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("clf", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
    }


def positive_scores(model, X):
    """Scores for the positive class, used for AUC.

    Uses predicted probabilities when the model provides them; otherwise
    (SVM) uses the decision-function value, which ranks samples the same way.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = positive_scores(model, X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def file_name(model_name: str) -> str:
    return model_name.lower().replace(" ", "_") + ".joblib"


def main():
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Dataset: {X.shape[0]} instances, {X.shape[1]} features")
    print(f"Class balance: malignant={int(y.sum())}, benign={int((1 - y).sum())}")

    # 80/20 stratified split; the 20% test split is never used for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    rows = []
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **metrics})
        joblib.dump(model, MODEL_DIR / file_name(name))
        print(f"Trained and saved: {name}")

    summary = pd.DataFrame(rows)
    summary.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)

    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT_DIR / "test_data.csv", index=False)

    print("\nTest-set metrics:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
