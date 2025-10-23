# Train Logistic Regression on anonymized CC22_3.csv and evaluate on full NHANES_P_merged_HI100K_like.csv

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

base = Path("/mnt/data")

# Load datasets
train_df = pd.read_csv(base / "CC22_3.csv")
test_df  = pd.read_csv(base / "NHANES_P_merged_HI100K_like.csv")

# Common columns
common_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))
if "obesity_flag" in common_cols:
    common_cols.remove("obesity_flag")

# Split numeric vs categorical
train_features = train_df[common_cols].copy()
test_features  = test_df[common_cols].copy()
numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in common_cols if c not in numeric_cols]

# Impute numeric NaN with TRAIN median
train_medians = train_features[numeric_cols].median()
train_features[numeric_cols] = train_features[numeric_cols].fillna(train_medians)
test_features[numeric_cols]  = test_features[numeric_cols].fillna(train_medians)

# One-hot encode categoricals jointly
combined_cat = pd.concat([train_features[categorical_cols], test_features[categorical_cols]], axis=0, ignore_index=True)
combined_cat_enc = pd.get_dummies(combined_cat, drop_first=True, dummy_na=True)

# Rebuild features
X_train = pd.concat([train_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[:len(train_df), :].reset_index(drop=True)], axis=1)
X_test  = pd.concat([test_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[len(train_df):, :].reset_index(drop=True)], axis=1)

# Standardize numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Targets
y_train = train_df["obesity_flag"].astype(int).values
y_test  = test_df["obesity_flag"].astype(int).values

# Train logistic regression (balanced)
clf = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", n_jobs=-1)
clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# Metrics
metrics = {
    "Accuracy": float(round(accuracy_score(y_test, y_pred), 4)),
    "Precision": float(round(precision_score(y_test, y_pred, zero_division=0), 4)),
    "Recall": float(round(recall_score(y_test, y_pred, zero_division=0), 4)),
    "F1": float(round(f1_score(y_test, y_pred, zero_division=0), 4)),
    "ROC_AUC": float(round(roc_auc_score(y_test, y_prob), 4)),
    "n_features_after_encoding": int(X_train.shape[1]),
    "n_train": int(len(train_df)),
    "n_test": int(len(test_df))
}

cm = confusion_matrix(y_test, y_pred)
report_text = classification_report(y_test, y_pred, zero_division=0)

# Save outputs
pred_out = test_df.copy()
pred_out["pred_label_logreg"] = y_pred
pred_out["pred_prob_logreg"]  = y_prob
pred_path = base / "NHANES_full_eval_predictions_logreg.csv"
pred_out.to_csv(pred_path, index=False)

metrics_path = base / "NHANES_full_eval_metrics_logreg.json"
with open(metrics_path, "w") as f:
    json.dump({"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report_text}, f, indent=2)

summary_logreg = {
    "metrics": metrics,
    "confusion_matrix": cm.tolist(),
    "predictions_csv": str(pred_path),
    "metrics_json": str(metrics_path),
    "classification_report_snippet": report_text[:1000]
}
summary_logreg
