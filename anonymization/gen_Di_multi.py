"""Utility script to train a Di model from multiple anonymized CSV files.

This script extends the single Bi/Ci training workflow in ``gen_Di_fixed.py`` so
that an arbitrary number of CSV files can be used for training.  Each CSV file
is automatically interpreted as either a Bi or Ci dataset using the helper
classes provided by the competition utilities.  The loaded data is concatenated
and used to train an XGBoost classifier that predicts ``stroke_flag``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from xgboost import XGBClassifier

# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "analysis"))
from xgbt_train import build_X  # type: ignore
sys.path.append(os.path.join(current_dir, "..", "util"))
from pws_data_format import BiDataFrame, CiDataFrame  # type: ignore

TARGET = "stroke_flag"

DIFFERENCE_SUMMARY = """
Differences from gen_Di_fixed.py:
  * Accepts any number of Bi/Ci CSV files instead of a fixed pair.
  * Adds configurable sampling fraction, validation ratio, and split strategy.
  * Provides an optional separate evaluation dataset and automatic class weighting.
  * Persists feature metadata exactly as in the original helper.
"""


def df_to_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a dataframe into the training matrix and target vector."""

    X = build_X(df, TARGET)
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)
    return X, y


class DiGenBase(ABC):
    """Abstract base class for Di generator implementations."""

    def __init__(
        self,
        n_estimators: int = 600,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        # See https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
        self.Di = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names: List[str] | None = None

    @abstractmethod
    def fit(self, csv_paths: Iterable[str], sample_frac: float, val_ratio: float) -> None:
        """Fit the generator model using the provided CSV paths."""

    def save_Di(self, out: str) -> None:
        booster = self.Di.get_booster()
        booster.set_attr(feature_names=json.dumps(self.feature_names or [], ensure_ascii=False))
        booster.set_attr(target=TARGET)
        booster.set_attr(xgboost_version=xgb.__version__)
        booster.save_model(out)

    def eval_accuracy(self, path_to_test_data: str) -> float:
        test_data = pd.read_csv(path_to_test_data, dtype=str, keep_default_na=False)
        X_test, y_test = df_to_Xy(test_data)
        y_pred = self.Di.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        return float(accuracy)


class TrainDiFromMultiple(DiGenBase):
    """Train a Di model from multiple anonymized CSV files."""

    def fit(
        self,
        csv_paths: Iterable[str],
        sample_frac: float = 1.0,
        val_ratio: float = 0.1,
        split_method: str = "auto",
        auto_class_weight: bool = False,
    ) -> None:
        frames: List[pd.DataFrame] = []

        for path in csv_paths:
            try:
                frames.append(BiDataFrame.read_csv(path))
                continue
            except Exception:  # noqa: BLE001 - fall back to Ci format
                pass

            try:
                frames.append(CiDataFrame.read_csv(path))
            except Exception as ci_error:  # noqa: BLE001
                raise ValueError(
                    f"Failed to load '{path}' as Bi or Ci dataset."
                ) from ci_error

        if not frames:
            raise ValueError("No training data could be loaded from the provided CSV paths.")

        raw_data = pd.concat(frames, ignore_index=True)

        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be in the range (0, 1].")
        if not (0 <= val_ratio < 1):
            raise ValueError("val_ratio must be in the range [0, 1).")

        random_state = self.Di.random_state if isinstance(self.Di.random_state, int) else None
        data = raw_data.sample(frac=sample_frac, random_state=random_state, ignore_index=True)

        X, y = df_to_Xy(data)

        if auto_class_weight:
            scale_pos_weight = self._compute_scale_pos_weight(y)
            self.Di.set_params(scale_pos_weight=scale_pos_weight)
        else:
            self.Di.set_params(scale_pos_weight=1.0)

        eval_set = []
        if val_ratio > 0 and len(y) > 1:
            train_idx, val_idx = self._train_val_split(
                X,
                y,
                val_ratio,
                split_method=split_method,
                random_state=random_state,
            )

            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)

            if len(y_val) > 0:
                eval_set.append((X_val, y_val))
        else:
            X_train = X.reset_index(drop=True)
            y_train = y.reset_index(drop=True)

        if not eval_set:
            eval_set = [(X_train, y_train)]

        self.Di.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.feature_names = list(X.columns)

    @staticmethod
    def _compute_scale_pos_weight(y: pd.Series) -> float:
        counts = y.value_counts()
        negative = counts.get(0, 0)
        positive = counts.get(1, 0)
        if positive == 0:
            return 1.0
        return float(negative / positive) if negative > 0 else 1.0

    @staticmethod
    def _train_val_split(
        X: pd.DataFrame,
        y: pd.Series,
        val_ratio: float,
        split_method: str,
        random_state: int | None,
    ) -> Tuple[Sequence[int], Sequence[int]]:
        n_samples = len(y)
        if n_samples < 2 or val_ratio <= 0:
            indices = list(range(n_samples))
            return indices, []

        method = split_method.lower()

        if method in {"auto", "stratified"} and y.nunique() > 1:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_ratio,
                random_state=random_state,
            )
            try:
                train_idx, val_idx = next(splitter.split(X, y))
                return train_idx, val_idx
            except ValueError:
                method = "random" if method == "auto" else method

        if method in {"auto", "random"}:
            train_idx, val_idx = train_test_split(
                list(range(n_samples)),
                test_size=val_ratio,
                random_state=random_state,
                shuffle=True,
            )
            return train_idx, val_idx

        # Fallback to deterministic split matching the legacy implementation.
        val_size = max(1, min(int(round(n_samples * val_ratio)), n_samples - 1))
        split_point = n_samples - val_size
        train_idx = list(range(split_point))
        val_idx = list(range(split_point, n_samples))
        return train_idx, val_idx


def print_difference_summary() -> None:
    print(DIFFERENCE_SUMMARY.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Di model from one or more anonymized CSV files.",
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="Paths to Bi/Ci CSV files used for training.",
    )
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Path to the output model file (JSON).",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of the concatenated rows to sample for training (0 < frac ≤ 1).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of the sampled rows reserved for validation (0 ≤ ratio < 1).",
    )
    parser.add_argument(
        "--split-method",
        choices=["auto", "stratified", "random", "sequential"],
        default="auto",
        help=(
            "Strategy to split the training data. 'auto' prefers a stratified shuffle "
            "and falls back to a random shuffle when stratification is infeasible."
        ),
    )
    parser.add_argument(
        "--auto-class-weight",
        action="store_true",
        help="Automatically set XGBoost's scale_pos_weight from class imbalance.",
    )
    parser.add_argument(
        "--eval-csv",
        help=(
            "Optional path to evaluate the trained model. If omitted, the first training "
            "CSV is used for accuracy reporting."
        ),
    )
    parser.add_argument(
        "--report-diff",
        action="store_true",
        help="Print a summary of how this script differs from gen_Di_fixed.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.report_diff:
        print_difference_summary()

    trainer = TrainDiFromMultiple()
    trainer.fit(
        args.csv_paths,
        sample_frac=args.sample_frac,
        val_ratio=args.val_ratio,
        split_method=args.split_method,
        auto_class_weight=args.auto_class_weight,
    )

    eval_target = args.eval_csv or args.csv_paths[0]
    acc = trainer.eval_accuracy(eval_target)
    print(f"accuracy: {acc}")

    trainer.save_Di(args.out)
    print(f"a Di.json example was saved as {args.out}")
