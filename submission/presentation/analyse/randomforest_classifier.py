# Train on anonymized CC22_3.csv and evaluate on the full NHANES_P_merged_HI100K_like.csv
# Preprocessing:
# - Identify common predictors
# - Numeric: impute NaN with TRAIN median
# - Categorical: one-hot encode jointly with dummy_na=True to align columns (labels are not used in encoding)
# Model: RandomForestClassifier
# Outputs: metrics JSON + predictions CSV

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import json

base = Path("/mnt/data")

# Load datasets
train_df = pd.read_csv(base / "CC22_3.csv")
test_df  = pd.read_csv(base / "NHANES_P_merged_HI100K_like.csv")

# Ensure label exists
assert "obesity_flag" in train_df.columns and "obesity_flag" in test_df.columns, "Label obesity_flag missing."

# Determine common predictors
common_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))
if "obesity_flag" in common_cols:
    common_cols.remove("obesity_flag")

# Split numeric vs categorical on TRAIN set (robust reference)
train_features = train_df[common_cols].copy()
test_features  = test_df[common_cols].copy()

numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in common_cols if c not in numeric_cols]

# Impute numeric NaNs with TRAIN medians
train_medians = train_features[numeric_cols].median()
train_features[numeric_cols] = train_features[numeric_cols].fillna(train_medians)
test_features[numeric_cols]  = test_features[numeric_cols].fillna(train_medians)

# One-hot encode categoricals jointly (labels not involved)
combined_cat = pd.concat([train_features[categorical_cols], test_features[categorical_cols]], axis=0, ignore_index=True)
combined_cat_enc = pd.get_dummies(combined_cat, drop_first=True, dummy_na=True)

# Rebuild full X matrices
X_train = pd.concat([train_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[:len(train_df), :].reset_index(drop=True)], axis=1)
X_test  = pd.concat([test_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[len(train_df):, :].reset_index(drop=True)], axis=1)

# Targets
y_train = train_df["obesity_flag"].astype(int).values
y_test  = test_df["obesity_flag"].astype(int).values

# Fit model
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    random_state=42,
    class_weight="balanced_subsample",
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

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

# Save predictions & metrics
pred_out = test_df.copy()
pred_out["pred_label"] = y_pred
pred_out["pred_prob"]  = y_prob
pred_path = base / "NHANES_full_eval_predictions.csv"
pred_out.to_csv(pred_path, index=False)

metrics_path = base / "NHANES_full_eval_metrics.json"
with open(metrics_path, "w") as f:
    json.dump({"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report_text}, f, indent=2)

# Return summary
summary = {
    "metrics": metrics,
    "confusion_matrix": cm.tolist(),
    "predictions_csv": str(pred_path),
    "metrics_json": str(metrics_path),
    "classification_report_snippet": report_text[:1000]
}
summary
