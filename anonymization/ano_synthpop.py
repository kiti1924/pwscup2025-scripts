"""Generate anonymised CSVs with a synthpop-inspired workflow.

The original R package ``synthpop`` provides a sequence of predictive models
to draw synthetic records column by column.  The Python package available on
PyPI only ships the lower-level census synthesiser utilities, so we implement
the column-wise modelling approach directly with scikit-learn.  The end result
keeps marginal distributions close to the source data while injecting modelled
variation for numeric columns.

Usage
-----
python anonymization/ano_synthpop.py INPUT.csv -o OUTPUT.csv \
    [--seed 42] [--num_tuples N]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


NUM_NOISE_SCALE = 0.1  # relative noise injected for continuous columns
INT_NOISE_SCALE = 0.05  # relative noise injected for integer columns
SENTINEL_MISSING = "__MISSING__"


@dataclass(frozen=True)
class ColumnStats:
    kind: str  # "categorical", "float", "integer"
    min: float | None = None
    max: float | None = None
    std: float | None = None
    categories: List[str] | None = None
    probabilities: np.ndarray | None = None
    missing_rate: float = 0.0


def infer_column_stats(df: pd.DataFrame) -> Dict[str, ColumnStats]:
    stats: Dict[str, ColumnStats] = {}

    for column in df.columns:
        series = df[column]
        missing_rate = float(series.isna().mean())

        if pd.api.types.is_numeric_dtype(series):
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            if finite.empty:
                stats[column] = ColumnStats(
                    kind="float", min=None, max=None, std=None, missing_rate=missing_rate
                )
                continue

            min_val = float(finite.min())
            max_val = float(finite.max())
            std_val = float(finite.std(ddof=0))

            if pd.api.types.is_integer_dtype(series.dropna()):
                stats[column] = ColumnStats(
                    kind="integer",
                    min=min_val,
                    max=max_val,
                    std=std_val,
                    missing_rate=missing_rate,
                )
            else:
                stats[column] = ColumnStats(
                    kind="float",
                    min=min_val,
                    max=max_val,
                    std=std_val,
                    missing_rate=missing_rate,
                )
        else:
            observed = series.dropna().astype(str)
            if observed.empty:
                stats[column] = ColumnStats(
                    kind="categorical",
                    categories=[],
                    probabilities=np.array([], dtype=float),
                    missing_rate=missing_rate,
                )
                continue

            value_counts = observed.value_counts(normalize=True)
            categories = value_counts.index.tolist()
            probabilities = value_counts.to_numpy(dtype=float)

            stats[column] = ColumnStats(
                kind="categorical",
                categories=categories,
                probabilities=probabilities,
                missing_rate=missing_rate,
            )

    return stats


def sample_first_column(
    series: pd.Series, meta: ColumnStats, rng: np.random.Generator, n_rows: int
) -> pd.Series:
    if meta.kind == "categorical":
        if not meta.categories:
            return pd.Series([np.nan] * n_rows, dtype=object)

        sampled = rng.choice(meta.categories, size=n_rows, p=meta.probabilities)
        if meta.missing_rate > 0:
            mask = rng.random(n_rows) < meta.missing_rate
            sampled = sampled.astype(object)
            sampled[mask] = np.nan
        return pd.Series(sampled, dtype=object)

    numeric = series.dropna().to_numpy()
    if numeric.size == 0:
        return pd.Series([np.nan] * n_rows, dtype=float)

    base = rng.choice(numeric, size=n_rows, replace=True)
    std = meta.std or 0.0
    if meta.kind == "float" and std > 0:
        noise = rng.normal(0.0, std * NUM_NOISE_SCALE, size=n_rows)
        base = base + noise
    elif meta.kind == "integer" and std > 0:
        noise_span = max(1, int(round(std * INT_NOISE_SCALE)))
        noise = rng.integers(-noise_span, noise_span + 1, size=n_rows)
        base = base + noise

    if meta.min is not None and meta.max is not None:
        base = np.clip(base, meta.min, meta.max)

    if meta.kind == "integer":
        base = np.round(base).astype(int)
    return pd.Series(base)


def build_feature_transformer(processed: Iterable[str], stats: Dict[str, ColumnStats]) -> ColumnTransformer | None:
    numeric_cols = [col for col in processed if stats[col].kind in {"float", "integer"}]
    categorical_cols = [col for col in processed if stats[col].kind == "categorical"]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value=SENTINEL_MISSING)),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        return None

    return ColumnTransformer(transformers, sparse_threshold=0.0)


def generate_synthetic(
    df: pd.DataFrame, n_rows: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stats = infer_column_stats(df)

    synth = pd.DataFrame(index=range(n_rows))
    processed: List[str] = []

    for column in df.columns:
        meta = stats[column]

        if not processed:
            synth[column] = sample_first_column(df[column], meta, rng, n_rows)
            processed.append(column)
            continue

        features = processed.copy()
        transformer = build_feature_transformer(features, stats)

        X_train = df[features].copy()
        X_synth = synth[features].copy()

        for feature in features:
            if stats[feature].kind == "categorical":
                X_train[feature] = X_train[feature].astype(object)
                X_synth[feature] = X_synth[feature].astype(object)

        if transformer is None:
            synth[column] = sample_first_column(df[column], meta, rng, n_rows)
            processed.append(column)
            continue

        if meta.kind == "categorical":
            y = df[column].astype(object).where(~df[column].isna(), SENTINEL_MISSING)
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)

            classifier = Pipeline(
                [
                    ("prep", transformer),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=12,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            classifier.fit(X_train, y_encoded)

            proba = classifier.predict_proba(X_synth)
            classes = encoder.classes_
            generated: List[str] = []
            for row_probs in proba:
                probs = np.clip(row_probs, a_min=0.0, a_max=None)
                if probs.sum() == 0:
                    probs = np.ones_like(probs) / probs.size
                else:
                    probs = probs / probs.sum()
                sampled = classes[rng.choice(len(classes), p=probs)]
                generated.append(sampled)

            series = pd.Series(generated, dtype=object)
            synth[column] = series.replace(SENTINEL_MISSING, np.nan)

        else:
            y_numeric = df[column].astype(float)
            regressor = Pipeline(
                [
                    ("prep", transformer),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=250,
                            max_depth=14,
                            random_state=seed,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            regressor.fit(X_train, y_numeric)

            predictions = regressor.predict(X_synth)
            residuals = y_numeric - regressor.predict(X_train)
            residual_std = float(np.std(residuals.dropna())) if not residuals.dropna().empty else 0.0

            if meta.std and meta.std > 0:
                residual_std = max(residual_std, meta.std * 0.05)

            if residual_std > 0:
                noise = rng.normal(0.0, residual_std, size=n_rows)
                predictions = predictions + noise

            if meta.min is not None and meta.max is not None:
                predictions = np.clip(predictions, meta.min, meta.max)

            if meta.kind == "integer":
                synth[column] = np.round(predictions).astype(int)
            else:
                synth[column] = predictions

            if meta.missing_rate > 0:
                mask = rng.random(n_rows) < meta.missing_rate
                synth.loc[mask, column] = np.nan

        processed.append(column)

    # Ensure column order matches the original input and dtypes are consistent
    final = synth[df.columns]
    for column, meta in stats.items():
        if meta.kind == "integer":
            final[column] = pd.to_numeric(final[column], errors="coerce").round().astype("Int64")
        elif meta.kind == "float":
            final[column] = pd.to_numeric(final[column], errors="coerce")
        else:
            final[column] = final[column].astype(object)

    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="synthpop風の匿名化CSV生成")
    parser.add_argument("input_csv", help="入力CSVファイル")
    parser.add_argument("-o", "--out", required=True, help="出力CSVファイル")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--num_tuples", type=int, default=None, help="生成レコード数")
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv)
    n_rows = len(source)
    target_rows = args.num_tuples if args.num_tuples is not None else n_rows

    synthetic = generate_synthetic(source, target_rows, args.seed)
    synthetic.to_csv(args.out, index=False)
    print(f"Saved synthpop-anonymized CSV: {args.out}")


if __name__ == "__main__":
    main()
