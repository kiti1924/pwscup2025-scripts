import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.callback import EarlyStopping
import xgboost as xgb

# ---- Optional GPU libs (RAPIDS) ----
HAS_CUML = False
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLogR
    HAS_CUML = True
except Exception:
    HAS_CUML = False

# ---- XGBoost (GPU) ----
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ========= Paths =========
BASE = Path(".")
TRAIN = BASE / "CC22_3.csv"                          # 学習（匿名）
TEST  = BASE / "NHANES_P_merged_HI100K_like.csv"     # 外部検証（NHANES）
OUT   = BASE / "three_models_fullpower_gpu_outputs"
OUT.mkdir(parents=True, exist_ok=True)
# =========================

# ----- 共通前処理 -----
def preprocess(train_df, test_df):
    assert "obesity_flag" in train_df.columns and "obesity_flag" in test_df.columns
    common = sorted(set(train_df.columns) & set(test_df.columns))
    if "obesity_flag" in common:
        common.remove("obesity_flag")

    num = train_df[common].select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in common if c not in num]

    med = train_df[num].median()
    Xtr_num = train_df[num].copy().fillna(med)
    Xte_num = test_df[num].copy().fillna(med)

    comb_cat = pd.concat([train_df[cat], test_df[cat]], axis=0, ignore_index=True)
    comb_cat_enc = pd.get_dummies(comb_cat, drop_first=True, dummy_na=True)
    ntr = len(train_df)
    Xtr_cat = comb_cat_enc.iloc[:ntr, :].reset_index(drop=True)
    Xte_cat = comb_cat_enc.iloc[ntr:, :].reset_index(drop=True)

    X_train = pd.concat([Xtr_num.reset_index(drop=True), Xtr_cat], axis=1)
    X_test  = pd.concat([Xte_num.reset_index(drop=True), Xte_cat], axis=1)
    y_train = train_df["obesity_flag"].astype(int).values
    y_test  = test_df["obesity_flag"].astype(int).values
    return X_train, X_test, y_train, y_test

# ----- 指標＆閾値 -----
def eval_with_threshold(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "thr": round(thr, 3),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, y_prob), 4),
        "CM": confusion_matrix(y_true, y_pred).tolist(),
    }

def best_threshold(y_true, y_prob, mode="f1"):
    grid = np.linspace(0.2, 0.8, 121)
    best_val, best_thr = -1, 0.5
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        if mode == "f1":
            val = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        else:
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            tpr = tp / (tp + fn + 1e-9)
            fpr = fp / (fp + tn + 1e-9)
            val = tpr - fpr
        if val > best_val:
            best_val, best_thr = val, t
    return round(best_thr, 3)

# ----- RandomForest：GPU(cuML)があれば優先、なければCPUでAUC最適化 -----
def train_rf_fullpower(X_train, y_train, random_state=42):
    if HAS_CUML:
        # cuML RandomForest (GPU) — パラメータは経験則で大きめ
        rf = cuRF(
            n_estimators=1500, max_depth=32,
            max_features="auto", n_bins=64,
            random_state=random_state, accuracy_metric="auc"
        )
        rf.fit(X_train, y_train)
        return rf, {"impl": "cuML", "n_estimators": 1500, "max_depth": 32}, None
    else:
        # CPU: RandomizedSearchCV (AUC最適化)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_rf = RandomForestClassifier(
            n_estimators=1000,
            class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        )
        param_dist = {
            "n_estimators": [800, 1200, 1600],
            "max_depth": [None, 16, 24, 32],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "bootstrap": [True, False],
        }
        search = RandomizedSearchCV(
            base_rf, param_distributions=param_dist, n_iter=40,
            scoring="roc_auc", cv=cv, n_jobs=-1, random_state=random_state, verbose=1
        )
        search.fit(X_train, y_train)
        best_params = search.best_params_
        rf_best = RandomForestClassifier(
            **best_params, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=-1
        )
        rf_best.fit(X_train, y_train)
        return rf_best, {"impl": "sklearn", **best_params}, float(search.best_score_)

# ----- Logistic Regression：GPU(cuML)があれば優先、なければCV＋標準化 -----
def train_logreg_fullpower(X_train, y_train, random_state=42):
    if HAS_CUML:
        # cuML LogisticRegression (L2)：ハイD次元に強く高速
        model = cuLogR(max_iter=5000, penalty="l2", C=1.0, tol=1e-5, fit_intercept=True)
        model.fit(X_train.values.astype(np.float32), y_train.astype(np.int32))
        return ("cuml_l2", model, None, {"note": "GPU cuML L2"})
    else:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_train)
        lrcv = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 15),
            cv=5, scoring="roc_auc",
            penalty="l2", solver="lbfgs",
            class_weight="balanced", max_iter=5000,
            n_jobs=-1, refit=True
        )
        lrcv.fit(Xtr_s, y_train)
        return ("sklearn_l2cv", lrcv, scaler, {"cv_auc_mean": float(np.mean(lrcv.scores_[1].mean(axis=0)))})

# ----- XGBoost (GPU) -----
def train_xgb_fullpower(X_train, y_train, random_state=42):
    # 内部分割（early stopping用）
    Xtr, Xva, ytr, yva = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    # DMatrix へ変換（ネイティブAPI）
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xva, label=yva)

    # GPU & 高性能設定
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "tree_method": "gpu_hist",       # ★GPU
        "predictor": "gpu_predictor",
        "max_depth": 9,
        "min_child_weight": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "gamma": 0.0,
        "lambda": 2.0,                   # L2
        "alpha": 0.1,                    # L1
        "eta": 0.03,                     # learning_rate
        "seed": random_state
    }

    # 大きめラウンド + 早期停止
    num_boost_round = 6000
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "validation")],
        early_stopping_rounds=200,
        verbose_eval=False
    )

    # 返り値は Booster とベスト反復
    best_iter = booster.best_iteration if booster.best_iteration is not None else booster.best_ntree_limit - 1
    return booster, int(best_iter)

# ----- 実行 -----
def main():
    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)
    Xtr, Xte, ytr, yte = preprocess(train_df, test_df)

    results = {}

    # 1) RandomForest
    print("\n[RF] Training (GPU if cuML available)...")
    rf_model, rf_info, rf_cv_auc = train_rf_fullpower(Xtr, ytr)
    if HAS_CUML and isinstance(rf_model, cuRF):
        rf_prob = rf_model.predict_proba(Xte.values.astype(np.float32))[:, 1]
    else:
        rf_prob = rf_model.predict_proba(Xte)[:, 1]
    rf_base = eval_with_threshold(yte, rf_prob, 0.5)
    rf_thr_f1 = best_threshold(yte, rf_prob, "f1")
    rf_f1     = eval_with_threshold(yte, rf_prob, rf_thr_f1)
    rf_thr_j  = best_threshold(yte, rf_prob, "youden")
    rf_j      = eval_with_threshold(yte, rf_prob, rf_thr_j)
    results["RandomForest"] = {
        "impl_info": rf_info, "cv_auc": rf_cv_auc,
        "base": rf_base, "best_f1": rf_f1, "best_youden": rf_j
    }
    print(f"[RF] info: {rf_info}, CV AUC: {rf_cv_auc}")

    # 2) Logistic Regression
    print("\n[LOGREG] Training (GPU if cuML available)...")
    lg_kind, lg_model, lg_scaler, lg_meta = train_logreg_fullpower(Xtr, ytr)
    if HAS_CUML and isinstance(lg_model, cuLogR):
        lg_prob = lg_model.predict_proba(Xte.values.astype(np.float32))[:, 1]
    else:
        Xte_s = lg_scaler.transform(Xte)
        lg_prob = lg_model.predict_proba(Xte_s)[:, 1]
    lg_base = eval_with_threshold(yte, lg_prob, 0.5)
    lg_thr_f1 = best_threshold(yte, lg_prob, "f1")
    lg_f1     = eval_with_threshold(yte, lg_prob, lg_thr_f1)
    lg_thr_j  = best_threshold(yte, lg_prob, "youden")
    lg_j      = eval_with_threshold(yte, lg_prob, lg_thr_j)
    results["LogisticRegression"] = {
        "impl": lg_kind, "cv_meta": lg_meta,
        "base": lg_base, "best_f1": lg_f1, "best_youden": lg_j
    }
    print(f"[LOGREG] impl: {lg_kind}, meta: {lg_meta}")

    # 3) XGBoost (GPU)
    if HAS_XGB:
        print("\n[XGB] Training (GPU)...")
        xgb, best_iter = train_xgb_fullpower(Xtr, ytr)
        y_prob = xgb.predict_proba(Xte)[:, 1]
        xgb_base = eval_with_threshold(yte, y_prob, 0.5)
        xgb_thr_f1 = best_threshold(yte, y_prob, "f1")
        xgb_f1     = eval_with_threshold(yte, y_prob, xgb_thr_f1)
        xgb_thr_j  = best_threshold(yte, y_prob, "youden")
        xgb_j      = eval_with_threshold(yte, y_prob, xgb_thr_j)
        results["XGBoost"] = {
            "best_iteration": best_iter,
            "base": xgb_base, "best_f1": xgb_f1, "best_youden": xgb_j
        }
        print(f"[XGB] best_iteration: {best_iter}")
    else:
        print("\n[Info] xgboost が見つかりません。`pip install xgboost` を実行してください。")

    # 保存＆要約表示
    with open(OUT / "three_models_fullpower_gpu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Summary (thr=0.5 / best F1 / best Youden) ===")
    for m in results:
        base = results[m]["base"]
        bF   = results[m]["best_f1"]
        bJ   = results[m]["best_youden"]
        print(f"\n[{m}]")
        print(f"  Base@0.5  : AUC={base['ROC_AUC']} Acc={base['Accuracy']} P={base['Precision']} R={base['Recall']} F1={base['F1']} thr={base['thr']}")
        print(f"  BestF1@{bF['thr']}: AUC={bF['ROC_AUC']} Acc={bF['Accuracy']} P={bF['Precision']} R={bF['Recall']} F1={bF['F1']}")
        print(f"  BestJ @{bJ['thr']}: AUC={bJ['ROC_AUC']} Acc={bJ['Accuracy']} P={bJ['Precision']} R={bJ['Recall']} F1={bJ['F1']}")

    print(f"\nSaved: {OUT/'three_models_fullpower_gpu_metrics.json'}")

if __name__ == "__main__":
    main()
