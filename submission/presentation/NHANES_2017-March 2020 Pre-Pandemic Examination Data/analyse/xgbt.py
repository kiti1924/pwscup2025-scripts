# Train and evaluate XGBoost model on CC22_3.csv (train) and NHANES_P_merged_HI100K_like.csv (test)

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import json

# base = Path("/mnt/data")
base = Path("")


# Load datasets
train_df = pd.read_csv(base / "CC22_3.csv")
test_df  = pd.read_csv(base / "NHANES_P_merged_HI100K_like.csv")

# Common predictors
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

# Combine numeric + encoded categoricals
X_train = pd.concat([train_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[:len(train_df), :].reset_index(drop=True)], axis=1)
X_test  = pd.concat([test_features[numeric_cols].reset_index(drop=True),
                     combined_cat_enc.iloc[len(train_df):, :].reset_index(drop=True)], axis=1)

y_train = train_df["obesity_flag"].astype(int).values
y_test  = test_df["obesity_flag"].astype(int).values

# Standardize numeric columns to help XGBoost convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# XGBoost model
xgb_clf = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb_clf.fit(X_train_scaled, y_train)

# Predictions
y_pred = xgb_clf.predict(X_test_scaled)
y_prob = xgb_clf.predict_proba(X_test_scaled)[:, 1]

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

# Save results
pred_out = test_df.copy()
pred_out["pred_label_xgb"] = y_pred
pred_out["pred_prob_xgb"]  = y_prob
pred_path = base / "NHANES_full_eval_predictions_xgb.csv"
pred_out.to_csv(pred_path, index=False)

metrics_path = base / "NHANES_full_eval_metrics_xgb.json"
with open(metrics_path, "w") as f:
    json.dump({"metrics": metrics, "confusion_matrix": cm.tolist(), "classification_report": report_text}, f, indent=2)

summary_xgb = {
    "metrics": metrics,
    "confusion_matrix": cm.tolist(),
    "predictions_csv": str(pred_path),
    "metrics_json": str(metrics_path),
    "classification_report_snippet": report_text[:1000]
}
summary_xgb
