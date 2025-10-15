"""Compute normalized column-wise distribution differences between two CSV files.

This script builds per-column weights by comparing the empirical distributions
observed in two datasets. The workflow is:

1. Load both CSV files using pandas.
2. For each shared column, derive a probability distribution for each dataset.
   * Categorical columns rely on relative frequencies of distinct values
     (including missing values).
   * Numeric columns are compared via aligned histograms using automatic bin
     selection.
3. Measure distribution dissimilarity with the total variation distance.
4. Min-max normalize the distances across columns so the resulting weights fall
   within [0, 1].
5. Serialize the weight mapping to JSON for downstream use.

Example:
    python attack/distribution_weight.py baseline.csv target.csv -o weights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare column-wise distributions between two CSV files and "
            "export normalized weights."
        )
    )
    parser.add_argument("reference_csv", help="Path to the reference CSV file.")
    parser.add_argument("target_csv", help="Path to the target CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        default="distribution_weights.json",
        help="Path to the JSON file written with the computed weights.",
    )
    parser.add_argument(
        "--numeric-bins",
        type=int,
        default=None,
        help=(
            "Optional fixed number of bins for numeric columns. When omitted, "
            "numpy selects the bin edges automatically based on the data."
        ),
    )
    return parser.parse_args()


def is_numeric_series(values: Iterable[pd.Series]) -> bool:
    """Return True when all non-null values across the series are numeric."""
    concatenated = pd.concat(values, ignore_index=True)
    if concatenated.notna().sum() == 0:
        return True

    converted = pd.to_numeric(concatenated.dropna(), errors="coerce")
    return converted.notna().all()


def histogram_edges(data: pd.Series, bins: int | None = None) -> np.ndarray:
    """Compute stable histogram edges for numeric comparison."""
    numeric = pd.to_numeric(data.dropna(), errors="coerce")
    numeric = numeric[numeric.notna()]
    if numeric.empty:
        return np.asarray([-0.5, 0.5], dtype=float)

    if bins is not None and bins > 0:
        edges = np.histogram_bin_edges(numeric.to_numpy(), bins=bins)
    else:
        edges = np.histogram_bin_edges(numeric.to_numpy(), bins="auto")

    if edges.size < 2 or np.all(edges == edges[0]):
        value = numeric.iloc[0]
        edges = np.asarray([value - 0.5, value + 0.5], dtype=float)
    return edges


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute total variation distance between probability vectors."""
    if p.size == 0 and q.size == 0:
        return 0.0
    return 0.5 * np.abs(p - q).sum()


def categorical_distribution(series: pd.Series) -> Dict[str, float]:
    """Return probability mass function for a categorical series."""
    normalized = series.fillna("<MISSING>").astype(str).value_counts(normalize=True)
    return normalized.to_dict()


def numeric_distribution(series: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Return normalized histogram counts following provided bin edges."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[numeric.notna()]
    counts, _ = np.histogram(numeric.to_numpy(), bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def column_distance(
    column: str,
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    numeric_bins: int | None = None,
) -> Tuple[str, float]:
    """Compute distribution distance for a single column."""
    if column not in df_ref.columns or column not in df_target.columns:
        return column, 0.0

    ref_series = df_ref[column]
    tgt_series = df_target[column]

    if is_numeric_series([ref_series, tgt_series]):
        combined = pd.concat([ref_series, tgt_series], ignore_index=True)
        edges = histogram_edges(combined, bins=numeric_bins)
        ref_hist = numeric_distribution(ref_series, edges)
        tgt_hist = numeric_distribution(tgt_series, edges)
        distance = total_variation_distance(ref_hist, tgt_hist)
    else:
        ref_dist = categorical_distribution(ref_series)
        tgt_dist = categorical_distribution(tgt_series)
        categories = set(ref_dist).union(tgt_dist)
        ref_probs = np.array([ref_dist.get(cat, 0.0) for cat in categories], dtype=float)
        tgt_probs = np.array([tgt_dist.get(cat, 0.0) for cat in categories], dtype=float)
        distance = total_variation_distance(ref_probs, tgt_probs)

    return column, float(distance)


def normalize_distances(distances: Dict[str, float]) -> Dict[str, float]:
    """Scale distance scores to the [0, 1] range via min-max normalization."""
    if not distances:
        return {}

    values = np.array(list(distances.values()), dtype=float)
    min_val = float(values.min())
    max_val = float(values.max())
    if np.isclose(max_val, min_val):
        return {column: 0.0 for column in distances}

    scale = max_val - min_val
    return {column: (value - min_val) / scale for column, value in distances.items()}


def compute_weights(
    df_ref: pd.DataFrame,
    df_target: pd.DataFrame,
    numeric_bins: int | None = None,
) -> Dict[str, float]:
    """Compute normalized differences for all common columns."""
    shared_columns = [col for col in df_ref.columns if col in df_target.columns]
    distances: Dict[str, float] = {}
    for column in shared_columns:
        col_name, distance = column_distance(column, df_ref, df_target, numeric_bins)
        distances[col_name] = distance
    return normalize_distances(distances)


def main() -> None:
    args = parse_args()

    reference_path = Path(args.reference_csv)
    target_path = Path(args.target_csv)
    output_path = Path(args.output)

    df_reference = pd.read_csv(reference_path, dtype=str, keep_default_na=False)
    df_target = pd.read_csv(target_path, dtype=str, keep_default_na=False)

    weights = compute_weights(
        df_reference,
        df_target,
        numeric_bins=args.numeric_bins,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(weights, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(weights)} weights to {output_path}")


if __name__ == "__main__":
    main()
