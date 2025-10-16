import os
import sys
import json
import argparse
from abc import ABC
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

from mia import build_feature_matrices


class AttackCiHungarianWeightedEntropy(ABC):
    def __init__(
        self,
        path_to_Ci_csv: str,
        mode: str = "auto",
        k: int = 300,
        fill_cost: float = 1000.0,
        max_full_mn: int = 30_000_000,
        verbose: bool = False,
        column_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.5,
    ):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
        self.mode = mode
        self.k = int(k)
        self.fill_cost = float(fill_cost)
        self.max_full_mn = int(max_full_mn)
        self.verbose = bool(verbose)
        self.column_weights = column_weights or {}
        self.alpha = float(alpha)

        self.inferred: Optional[pd.DataFrame] = None
        self.match_table_: Optional[pd.DataFrame] = None
        self.feature_names_: List[str] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _calculate_entropy_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute per-feature entropy weights (inverse of entropy, normalized)."""
        if X is None:
            raise RuntimeError("X is None in _calculate_entropy_weights")
        n_features = X.shape[1]
        ent_w = np.empty(n_features, dtype=float)
        for i in range(n_features):
            col = X[:, i]
            vals, counts = np.unique(col, return_counts=True)
            probs = counts / counts.sum()
            ent = float(entropy(probs))
            ent_w[i] = 1.0 / (1.0 + ent)
        sumw = ent_w.sum()
        if sumw == 0:
            return np.ones(n_features) / n_features
        return ent_w / sumw

    def _combine_weights(self, entropy_weights: np.ndarray, column_weights: Dict[str, float], feature_names: List[str]) -> np.ndarray:
        """Combine entropy-based per-feature weights and per-column weights (from JSON)."""
        n = len(feature_names)
        combined = np.zeros(n, dtype=float)
        if column_weights:
            vals = np.array(list(column_weights.values()), dtype=float)
            # use np.ptp for NumPy 2.0 compatibility
            if np.ptp(vals) > 0:
                minv, maxv = vals.min(), vals.max()
                norm_map = {k: (float(v) - float(minv)) / float(maxv - minv) for k, v in column_weights.items()}
            else:
                norm_map = {k: 1.0 for k in column_weights.keys()}
        else:
            norm_map = {}

        for i, fname in enumerate(feature_names):
            base_col = fname.split("_onehot")[0]
            e_w = float(entropy_weights[i])
            c_w = float(norm_map.get(base_col, e_w))
            combined[i] = self.alpha * e_w + (1.0 - self.alpha) * c_w

        s = combined.sum()
        if s == 0:
            return np.ones(n) / n
        return combined / s

    def _features(self, path_to_Ai_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)
        res = build_feature_matrices(Ai_df, self.Ci_df)
        if res is None:
            raise RuntimeError("mia.build_feature_matrices returned None; check mia implementation")
        if len(res) == 3:
            X_ai, X_ci, feature_names = res
        elif len(res) == 2:
            X_ai, X_ci = res
            feature_names = None
        else:
            raise RuntimeError(f"Unexpected return from build_feature_matrices: len={len(res)}")

        if X_ai is None or X_ci is None:
            raise RuntimeError("build_feature_matrices returned None arrays (X_ai or X_ci)")

        X_ai = np.asarray(X_ai)
        X_ci = np.asarray(X_ci)

        if feature_names is None:
            if hasattr(X_ai, "dtype") and getattr(X_ai.dtype, "names", None):
                feature_names = list(X_ai.dtype.names)
                X_ai = np.vstack([X_ai[name].astype(object) for name in feature_names]).T
                X_ci = np.vstack([X_ci[name].astype(object) for name in feature_names]).T
            else:
                feature_names = [f"f{i}" for i in range(X_ai.shape[1])]

        self.feature_names_ = list(feature_names)

        entropy_w = self._calculate_entropy_weights(X_ai)
        weights = self._combine_weights(entropy_w, self.column_weights, self.feature_names_)
        self._log(f"Computed weights sum={weights.sum():.6f}, n_features={len(weights)}")
        return X_ai.astype(float), X_ci.astype(float), weights.astype(float)

    def _solve_full(self, X_ai: np.ndarray, X_ci: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        self._log(f"[full] Ci={m} Ai={n} feat={X_ai.shape[1]}")
        X_ai_w = X_ai * weights
        X_ci_w = X_ci * weights
        cost = cdist(X_ci_w, X_ai_w, metric="cityblock")
        r, c = linear_sum_assignment(cost)
        d = cost[r, c]
        return r, c, d

    def _solve_knn(self, X_ai: np.ndarray, X_ci: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        k = min(max(1, self.k), max(1, n))
        self._log(f"[knn] Ci={m} Ai={n} feat={X_ai.shape[1]} k={k}")
        X_ai_w = X_ai * weights
        X_ci_w = X_ci * weights
        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(X_ai_w)
        dists, inds = nn.kneighbors(X_ci_w, return_distance=True)
        unique_ai = np.unique(inds.ravel())
        ai_map = {ai: i for i, ai in enumerate(unique_ai)}
        reduced = len(unique_ai)
        cost = np.full((m, reduced), self.fill_cost, dtype=float)
        for i in range(m):
            for j_idx, dist in zip(inds[i], dists[i]):
                cost[i, ai_map[j_idx]] = dist
        r, c = linear_sum_assignment(cost)
        d = cost[r, c]
        c = np.array([unique_ai[ci] for ci in c], dtype=int)
        return r, c, d

    def infer(self, path_to_Ai_csv: str):
        X_ai, X_ci, weights = self._features(path_to_Ai_csv)
        m, n = X_ci.shape[0], X_ai.shape[0]
        # choose solver
        if self.mode == "full" or (self.mode == "auto" and m * n <= self.max_full_mn):
            r, c, d = self._solve_full(X_ai, X_ci, weights)
        else:
            r, c, d = self._solve_knn(X_ai, X_ci, weights)

        # build membership vector: 1 for matched Ai indices
        membership = np.zeros(n, dtype=int)
        membership[c] = 1
        self.inferred = pd.DataFrame({"membership": membership})
        # match table for out-map
        self.match_table_ = pd.DataFrame({"ci_idx": r, "ai_idx": c, "distance": d})

    def save_inferred(self, path_to_output: str):
        if self.inferred is None:
            raise RuntimeError("Call infer() before save_inferred()")
        # save as single-column CSV without header to match previous behavior
        self.inferred.to_csv(path_to_output, index=False, header=False)
        self._log(f"Saved inferred membership -> {path_to_output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ci→Ai Hungarian assignment with entropy-weighted columns")
    ap.add_argument("path_to_Ai_csv", help="CSV with header")
    ap.add_argument("path_to_Ci_csv", help="CSV with header")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path")
    ap.add_argument("-m", "--mode", choices=["auto", "full", "knn"], default="auto")
    ap.add_argument("-k", "--k", type=int, default=300)
    ap.add_argument("--fill-cost", type=float, default=1000.0)
    ap.add_argument("--max-full-mn", type=int, default=30_000_000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-map", default=None)
    ap.add_argument("--weights-file", type=str, default=None, help="Path to JSON file with column weights")
    ap.add_argument("--alpha", type=float, default=0.5, help="mixing factor for entropy vs column weights (0..1)")

    args = ap.parse_args()

    column_weights = {}
    if args.weights_file:
        with open(args.weights_file, "r", encoding="utf-8") as f:
            column_weights = json.load(f)
        print(f"Using column weights from {args.weights_file}")

    attacker = AttackCiHungarianWeightedEntropy(
        path_to_Ci_csv=args.path_to_Ci_csv,
        mode=args.mode,
        k=args.k,
        fill_cost=args.fill_cost,
        max_full_mn=args.max_full_mn,
        verbose=args.verbose,
        column_weights=column_weights,
        alpha=args.alpha,
    )

    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_map and attacker.match_table_ is not None:
        attacker.match_table_.to_csv(args.out_map, index=False)
        print(f"match map saved to {args.out_map}")