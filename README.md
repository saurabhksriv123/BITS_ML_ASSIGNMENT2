# Breast Cancer Diagnosis – Comparison of Five ML Classifiers

## a. Problem statement

Breast cancer is one of the most common cancers, and early, accurate diagnosis of a tumour as **malignant** or **benign** is critical for treatment decisions. Using measurements computed from digitised images of fine-needle aspirate (FNA) of breast masses, the goal of this project is to build and compare five classification models that predict whether a tumour is malignant (1) or benign (0), evaluate them with six metrics, and deploy an interactive Streamlit app that evaluates the saved models on uploaded test data.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic (loaded via `sklearn.datasets.load_breast_cancer`) |
| Instances | 569 |
| Features | 30 numeric, real-valued features |
| Target | `diagnosis` – 1 = malignant, 0 = benign (binary classification) |
| Class balance | 212 malignant (37.3%), 357 benign (62.7%) |
| Missing values | None |

**Features.** Ten characteristics of the cell nuclei are measured: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each one, three values are recorded: the **mean**, the **standard error** and the **worst** (mean of the three largest values), giving 10 × 3 = 30 features (e.g. `mean radius`, `radius error`, `worst radius`).

**Train/test split.** 80/20 stratified split with `random_state=42`: 455 training rows and 114 test rows (72 benign, 42 malignant). The test split is saved as `test_data.csv` and is never used for training.

**Target encoding.** scikit-learn encodes 0 = malignant and 1 = benign. This project recodes the target so that **1 = malignant** (the positive class), because detecting malignant tumours is the clinically important task. Precision, Recall and F1 are therefore reported for the malignant class.

## c. GitHub Repository Link

https://github.com/saurabhksriv123/BITS_ML_ASSIGNMENT2/

### Repository structure

```
project-folder/
│-- app.py                      # Streamlit app (loads saved models, no retraining)
│-- requirements.txt            # pinned dependencies for Streamlit Community Cloud
│-- README.md
│-- test_data.csv               # held-out test set (30 features + diagnosis)
│-- model/
    │-- train_models.py         # training + evaluation source code
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- support_vector_machine.joblib
    │-- feature_names.json      # ordered list of input features
    │-- metrics_summary.csv     # test-set metrics
```

Each `.joblib` file is a scikit-learn `Pipeline`, so the preprocessing (standard scaling, where used) is saved together with the classifier.

## d. Models used

All five models were trained on the same 455-row training set and evaluated on the same 114-row test set.

| Model | Configuration |
|---|---|
| Logistic Regression | StandardScaler + LogisticRegression (max_iter=5000) |
| Decision Tree | DecisionTreeClassifier (default settings, random_state=42) |
| K-Nearest Neighbour | StandardScaler + KNeighborsClassifier (k = 5) |
| Naive Bayes | GaussianNB |
| Support Vector Machine | StandardScaler + SVC (RBF kernel) |

AUC is computed from predicted probabilities; for the SVM, which does not output probabilities, it is computed from the decision-function scores (AUC depends only on the ranking of scores).

### Comparison table of evaluation metrics (test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | **0.9737** | 0.9947 | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall (accuracy 0.9649, MCC 0.9245) and the **highest AUC (0.9960)**, so its probability ranking separates the classes almost perfectly. It missed 3 of 42 malignant cases and raised 1 false alarm. The strong result shows the scaled features are close to linearly separable. |
| Decision Tree | Weakest model on every metric (accuracy 0.9298, AUC 0.9246, MCC 0.8492). It made 4 false negatives and 4 false positives. The unpruned tree grew to depth 8 with 24 leaves, which suggests it overfits the training data; its low AUC also reflects coarse, leaf-based probability estimates. |
| KNN | Solid performance (accuracy 0.9561, MCC 0.9058) once features are standardised, with only 1 false positive. It missed 4 malignant cases, so its recall (0.9048) is lower than Logistic Regression and SVM. |
| Naive Bayes | **Perfect precision (1.0000)** with no false positives, and a high AUC (0.9934), but the **lowest recall (0.8333)**: it missed 7 of 42 malignant tumours. Its assumption that features are independent is violated here (radius, perimeter and area are strongly correlated), which hurts its hard predictions more than its ranking. |
| Support Vector Machine | **Best model**: highest accuracy (0.9737), F1 (0.9630) and MCC (0.9442), perfect precision (no benign tumour labelled malignant) and joint-best recall (0.9286, 3 missed malignant cases). The RBF kernel captures the slight non-linearity in the class boundary. |
| Overall, winner for your dataset? | **Support Vector Machine.** It leads on Accuracy, F1 and MCC, ties for best Precision and Recall, and its AUC (0.9947) is within 0.0013 of the best. MCC is the most reliable single metric here because the classes are imbalanced (63% benign), and SVM has the highest MCC. Logistic Regression is a close second. |

## Streamlit app

Live app: https://bitsmlassignment2-pleasefollow.streamlit.app/*

Features:
- **Dataset upload (CSV):** upload test data only; the CSV must contain all 30 feature columns and the `diagnosis` column. The bundled `test_data.csv` can be used or downloaded from the sidebar.
- **Model selection dropdown:** choose any of the five models.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC for the selected model.
- **Confusion matrix and classification report** for the selected model.
- **All-model comparison table and bar chart** on the uploaded data, with the best model by MCC highlighted.

The app loads the saved `.joblib` models and only predicts and evaluates; it never retrains on the uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python model/train_models.py   # optional: retrains models and regenerates test_data.csv
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

1. Go to https://streamlit.io/cloud and sign in with GitHub.
2. Click **New app**, select this repository, branch `main`, and main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.12** (or newer), then click **Deploy**.
