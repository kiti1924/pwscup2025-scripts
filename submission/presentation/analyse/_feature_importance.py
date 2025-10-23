import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

base = Path(".")  # CSVがあるフォルダ
train_df = pd.read_csv(base / "CC22_3.csv")
test_df  = pd.read_csv(base / "NHANES_P_merged_HI100K_like.csv")

# ---------- 共通前処理 ----------
common_cols = sorted(set(train_df.columns) & set(test_df.columns))
common_cols.remove("obesity_flag")

num_cols = train_df[common_cols].select_dtypes(include=[np.number]).columns
cat_cols = [c for c in common_cols if c not in num_cols]

# 数値欠損補完
med = train_df[num_cols].median()
for c in num_cols:
    train_df[c] = train_df[c].fillna(med[c])
    test_df[c]  = test_df[c].fillna(med[c])

# One-hot encode
combined = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0)
encoded = pd.get_dummies(combined, drop_first=True, dummy_na=True)

X_train = pd.concat([train_df[num_cols].reset_index(drop=True), encoded.iloc[:len(train_df)]], axis=1)
X_test  = pd.concat([test_df[num_cols].reset_index(drop=True), encoded.iloc[len(train_df):]], axis=1)
y_train = train_df["obesity_flag"].astype(int)
y_test  = test_df["obesity_flag"].astype(int)

# ---------- RandomForest ----------
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
rf_imp.head(20).to_csv("feature_importance_randomforest.csv")

plt.figure(figsize=(8,6))
rf_imp.head(20).plot(kind='barh', color='steelblue')
plt.title("RandomForest Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("feature_importance_randomforest.png", dpi=300)
plt.close()

# ---------- Logistic Regression ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
logr = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
logr.fit(X_train_scaled, y_train)
logr_imp = pd.Series(logr.coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
logr_imp.head(20).to_csv("feature_importance_logreg.csv")

plt.figure(figsize=(8,6))
logr_imp.head(20).plot(kind='barh', color=['crimson' if v>0 else 'navy' for v in logr_imp.head(20)])
plt.title("Logistic Regression Coefficients (Top 20)")
plt.tight_layout()
plt.savefig("feature_importance_logreg.png", dpi=300)
plt.close()

# ---------- XGBoost ----------
xgb = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", use_label_encoder=False
)
xgb.fit(X_train, y_train)
xgb_imp = pd.Series(xgb.get_booster().get_score(importance_type='gain'))
xgb_imp = xgb_imp.sort_values(ascending=False)
xgb_imp.head(20).to_csv("feature_importance_xgb.csv")

plt.figure(figsize=(8,6))
xgb_imp.head(20).plot(kind='barh', color='darkorange')
plt.title("XGBoost Feature Importance (Gain, Top 20)")
plt.tight_layout()
plt.savefig("feature_importance_xgb.png", dpi=300)
plt.close()

print("✅ Feature importance files created for all 3 models.")
