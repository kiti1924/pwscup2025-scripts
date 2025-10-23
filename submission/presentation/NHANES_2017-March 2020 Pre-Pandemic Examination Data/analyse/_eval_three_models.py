# filename: eval_three_models.py
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ========= User paths =========
BASE = Path(".")  # CC22_3.csv と NHANES_P_merged_HI100K_like.csv があるディレクトリ
TRAIN_CSV = BASE / "CC22_3.csv"                          # 匿名化データ（学習）
TEST_CSV  = BASE / "NHANES_P_merged_HI100K_like.csv"     # NHANES（検証）

SAVE_PRED = False   # 予測CSVを保存する場合 True
OUT_DIR   = BASE / "model_eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# =============================

def common_preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    assert "obesity_flag" in train_df.columns and "obesity_flag" in test_df.columns, \
        "Both datasets must have 'obesity_flag'."

    # 共通特徴量（目的変数は除外）
    common_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))
    if "obesity_flag" in common_cols:
        common_cols.remove("obesity_flag")

    # 数値/カテゴリに分割（学習側を基準）
    num_cols = train_df[common_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in common_cols if c not in num_cols]

    # 数値欠損を学習中央値で補完（学習分布基準）
    med = train_df[num_cols].median()
    Xtr_num = train_df[num_cols].copy().fillna(med)
    Xte_num = test_df[num_cols].copy().fillna(med)

    # カテゴリは学習＋テストで結合エンコード（次元合わせのため）
    comb_cat = pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0, ignore_index=True)
    comb_cat_enc = pd.get_dummies(comb_cat, drop_first=True, dummy_na=True)

    ntr = len(train_df)
    Xtr_cat = comb_cat_enc.iloc[:ntr, :].reset_index(drop=True)
    Xte_cat = comb_cat_enc.iloc[ntr:, :].reset_index(drop=True)

    # 数値＋カテゴリを結合
    X_train = pd.concat([Xtr_num.reset_index(drop=True), Xtr_cat], axis=1)
    X_test  = pd.concat([Xte_num.reset_index(drop=True), Xte_cat], axis=1)

    y_train = train_df["obesity_flag"].astype(int).values
    y_test  = test_df["obesity_flag"].astype(int).values

    feature_names = X_train.columns.tolist()
    return X_train, X_test, y_train, y_test, feature_names

def eval_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
    }

def pretty_print(title, metrics, cm):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>10}: {v}")
    print("Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(np.array(cm))

def main():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    X_train, X_test, y_train, y_test, feat_names = common_preprocess(train_df, test_df)

    results = {}

    # 1) RandomForest
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=14,
        class_weight="balanced_subsample",
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = eval_metrics(y_test, rf_pred, rf_prob)
    rf_cm = confusion_matrix(y_test, rf_pred)
    pretty_print("RandomForest", rf_metrics, rf_cm)
    results["RandomForest"] = {"metrics": rf_metrics, "confusion_matrix": rf_cm.tolist()}

    if SAVE_PRED:
        df_out = test_df.copy()
        df_out["pred_label_rf"] = rf_pred
        df_out["pred_prob_rf"]  = rf_prob
        df_out.to_csv(OUT_DIR / "predictions_randomforest.csv", index=False)

    # 2) Logistic Regression（数値＋one-hot後に標準化）
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)

    logr = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", n_jobs=-1)
    logr.fit(Xtr_s, y_train)
    lg_pred = logr.predict(Xte_s)
    lg_prob = logr.predict_proba(Xte_s)[:, 1]
    lg_metrics = eval_metrics(y_test, lg_pred, lg_prob)
    lg_cm = confusion_matrix(y_test, lg_pred)
    pretty_print("LogisticRegression", lg_metrics, lg_cm)
    results["LogisticRegression"] = {"metrics": lg_metrics, "confusion_matrix": lg_cm.tolist()}

    if SAVE_PRED:
        df_out = test_df.copy()
        df_out["pred_label_logreg"] = lg_pred
        df_out["pred_prob_logreg"]  = lg_prob
        df_out.to_csv(OUT_DIR / "predictions_logreg.csv", index=False)

    # 3) XGBoost（インストール済みなら実行）
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42,
            eval_metric="logloss", use_label_encoder=False, n_jobs=-1
        )
        # XGBoostはスケーリング不要だが、上と比較のため未スケール版を使用
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_prob = xgb.predict_proba(X_test)[:, 1]
        xgb_metrics = eval_metrics(y_test, xgb_pred, xgb_prob)
        xgb_cm = confusion_matrix(y_test, xgb_pred)
        pretty_print("XGBoost", xgb_metrics, xgb_cm)
        results["XGBoost"] = {"metrics": xgb_metrics, "confusion_matrix": xgb_cm.tolist()}

        if SAVE_PRED:
            df_out = test_df.copy()
            df_out["pred_label_xgb"] = xgb_pred
            df_out["pred_prob_xgb"]  = xgb_prob
            df_out.to_csv(OUT_DIR / "predictions_xgb.csv", index=False)
    else:
        print("\n[Info] xgboost が見つかりませんでした。`pip install xgboost` を実行してください。")

    # 保存（任意）
    with open(OUT_DIR / "three_models_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Metrics saved to: {OUT_DIR/'three_models_metrics.json'}")

if __name__ == "__main__":
    main()
