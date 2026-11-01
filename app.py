"""
Streamlit app: evaluate five pre-trained classifiers on uploaded test data.

The models are trained once by `model/train_models.py` and loaded here.
Nothing is retrained in the app: uploaded data is only used for prediction
and evaluation.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
TARGET = "diagnosis"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🩺", layout="wide")


# ------------------------------------------------------------------ loading
def find_file(name):
    """Locate a file in the repo: model/ first, then the app folder, then any subfolder.

    The match ignores upper/lower case, because the deployment server (Linux) is
    case-sensitive while Windows is not (e.g. a folder named 'Model').
    """
    for candidate in (BASE_DIR / "model" / name, BASE_DIR / name):
        if candidate.is_file():
            return candidate
    target = name.lower()
    for path in sorted(BASE_DIR.rglob("*")):
        if path.is_file() and path.name.lower() == target and ".git" not in path.parts:
            return path
    return None


def required_files():
    names = list(MODEL_FILES.values()) + ["feature_names.json"]
    found = {n: find_file(n) for n in names}
    missing = [n for n, p in found.items() if p is None]
    return found, missing


def show_missing_files_error(missing):
    st.error(
        "The app could not find these required file(s) in the GitHub repository:\n\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n\nUpload them into a folder named `model` (all lowercase) next to `app.py`, "
          "commit, then reboot the app."
    )
    repo_files = sorted(
        str(p.relative_to(BASE_DIR)) for p in BASE_DIR.rglob("*")
        if p.is_file() and ".git" not in p.parts
    )
    with st.expander("Files the app can currently see in the repository"):
        st.code("\n".join(repo_files) or "(none)")
    st.stop()


@st.cache_resource
def load_models(paths):
    return {name: joblib.load(paths[fname]) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_feature_names(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------- evaluation
def positive_scores(model, X):
    """Positive-class scores for AUC: probabilities if available, else decision function."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y):
    y_pred = model.predict(X)
    scores = positive_scores(model, X)
    auc = roc_auc_score(y, scores) if len(np.unique(y)) == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------- UI
st.title("🩺 Breast Cancer Diagnosis: Classifier Comparison")
st.caption(
    "UCI Breast Cancer Wisconsin (Diagnostic) dataset · 30 numeric features · "
    "target `diagnosis` (1 = malignant, 0 = benign). Models are pre-trained; the app only evaluates."
)

found, missing = required_files()
if missing:
    show_missing_files_error(missing)
models = load_models({n: str(p) for n, p in found.items()})
features = load_feature_names(str(found["feature_names.json"]))
SAMPLE_TEST_FILE = find_file("test_data.csv")

with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help="Must contain all 30 feature columns and the `diagnosis` column.")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None,
                             help="Held-out 20% test split used in the experiments.")
    if SAMPLE_TEST_FILE is not None:
        st.download_button("Download sample test_data.csv", SAMPLE_TEST_FILE.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))

# choose data source
if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = f"Uploaded file: **{uploaded.name}**"
elif use_sample and SAMPLE_TEST_FILE is not None:
    df = pd.read_csv(SAMPLE_TEST_FILE)
    source = "Bundled file: **test_data.csv**"
else:
    st.info("Upload a test CSV in the sidebar (or tick *Use bundled test_data.csv*) to begin.")
    st.stop()

# validate columns
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The CSV is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

X = df[features]
y = df[TARGET].astype(int)
if not set(y.unique()) <= {0, 1}:
    st.error("The `diagnosis` column must contain only 0 (benign) and 1 (malignant).")
    st.stop()

st.markdown(f"{source} · {len(df)} rows · malignant: {int(y.sum())} · benign: {int((1 - y).sum())}")
with st.expander("Preview data"):
    st.dataframe(df.head(10), width="stretch")

# ------------------------------------------------------- selected model view
st.subheader(f"Evaluation metrics: {model_name}")
metrics, y_pred = evaluate(models[model_name], X, y)

cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    st.pyplot(plot_confusion_matrix(y, y_pred, model_name))
with right:
    st.markdown("**Classification report**")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# --------------------------------------------------------- all-model summary
st.subheader("All models on this test data")
rows = []
for name, model in models.items():
    m, _ = evaluate(model, X, y)
    rows.append({"ML Model Name": name, **m})
summary = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(summary.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
             width="stretch")

fig, ax = plt.subplots(figsize=(10, 4))
summary.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylim(max(0.0, summary.min().min() - 0.05), 1.0)
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Metric comparison across models")
ax.legend(loc="lower right", ncol=3, fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=0)
fig.tight_layout()
st.pyplot(fig)

best = summary["MCC"].idxmax()
st.success(f"Best model on this data by MCC: **{best}** (MCC = {summary.loc[best, 'MCC']:.4f})")
