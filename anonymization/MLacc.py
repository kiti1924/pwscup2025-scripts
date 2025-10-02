"""Evaluate the prediction accuracy of a trained Di XGBoost model against a Bi dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

# Allow reuse of the feature engineering helper from the analysis package.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "analysis"))
from xgbt_train import build_X  # type: ignore

TARGET = "stroke_flag"


def _load_booster(path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(path)
    return booster


def _get_feature_names(booster: xgb.Booster) -> Sequence[str]:
    if booster.feature_names:
        return booster.feature_names

    attrs = booster.attributes() or {}
    if "feature_names" not in attrs:
        raise ValueError("The provided model JSON does not contain feature metadata.")

    feature_names = json.loads(attrs["feature_names"])
    if not isinstance(feature_names, list) or not all(isinstance(x, str) for x in feature_names):
        raise ValueError("Invalid feature metadata stored in the model JSON.")
    return feature_names


def _prepare_matrix(df: pd.DataFrame, booster: xgb.Booster) -> tuple[pd.DataFrame, np.ndarray]:
    X = build_X(df, TARGET)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any():
        raise ValueError("Target column contains non-binary or missing values.")
    y_array = y.astype(int).to_numpy()

    feature_names = list(_get_feature_names(booster))

    extra_columns = set(X.columns) - set(feature_names)
    if extra_columns:
        X = X.drop(columns=sorted(extra_columns))

    missing_columns = [name for name in feature_names if name not in X.columns]
    for name in missing_columns:
        X[name] = 0

    X = X.reindex(columns=feature_names)
    return X, y_array


def _compute_metrics(y_true: np.ndarray, pred_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred_label = (pred_prob >= threshold).astype(int)
    correct = (pred_label == y_true)

    tp = float(np.sum((pred_label == 1) & (y_true == 1)))
    tn = float(np.sum((pred_label == 0) & (y_true == 0)))
    fp = float(np.sum((pred_label == 1) & (y_true == 0)))
    fn = float(np.sum((pred_label == 0) & (y_true == 1)))

    total = float(y_true.shape[0])
    accuracy = float(correct.mean()) if total else float("nan")
    positive_rate = float(np.mean(y_true)) if total else float("nan")

    return {
        "accuracy": accuracy,
        "total_samples": total,
        "threshold": threshold,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "positive_rate": positive_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report how accurately a trained Di model predicts stroke_flag on a Bi dataset."
        )
    )
    parser.add_argument("model_json", help="Path to the trained Di model JSON (Booster.save_model output).")
    parser.add_argument("bi_csv", help="Path to the Bi CSV that contains stroke_flag ground truth.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to predicted probabilities (default: 0.5).",
    )
    parser.add_argument(
        "--save-pred",
        metavar="CSV",
        help="Optional path to save per-row predictions and ground truth for inspection.",
    )
    args = parser.parse_args()

    booster = _load_booster(args.model_json)
    df = pd.read_csv(args.bi_csv, dtype=str, keep_default_na=False)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from {args.bi_csv}.")

    X, y = _prepare_matrix(df, booster)
    pred_prob = booster.predict(xgb.DMatrix(X))

    metrics = _compute_metrics(y, pred_prob, args.threshold)

    print(f"Total samples   : {int(metrics['total_samples'])}")
    print(f"Accuracy        : {metrics['accuracy']:.6f}")
    print(f"Threshold       : {metrics['threshold']:.3f}")
    print(f"Positive rate   : {metrics['positive_rate']:.6f}")
    print(
        "TP/FP/TN/FN     : "
        f"{int(metrics['true_positive'])}/"
        f"{int(metrics['false_positive'])}/"
        f"{int(metrics['true_negative'])}/"
        f"{int(metrics['false_negative'])}"
    )

    if args.save_pred:
        out_df = pd.DataFrame(
            {
                "pred_prob": pred_prob,
                "pred_label": (pred_prob >= args.threshold).astype(int),
                TARGET: y,
            }
        )
        out_df.to_csv(args.save_pred, index=False)
        print(f"Saved detailed predictions to {args.save_pred}")


if __name__ == "__main__":
    main()
