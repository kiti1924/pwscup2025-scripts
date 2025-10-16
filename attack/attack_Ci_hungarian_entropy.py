import os
import sys
import argparse
from abc import ABC
from typing import Tuple, List
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

from mia import build_feature_matrices

class AttackCiHungarianEntropy(ABC):
    def __init__(
        self,
        path_to_Ci_csv: str,
        mode: str = "auto",
        k: int = 300,
        fill_cost: float = 1000.0,
        max_full_mn: int = 30_000_000,
        verbose: bool = False,
        alpha: float = 1.0,    # Add alpha parameter
    ):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
        self.mode = mode
        self.k = int(k)
        self.fill_cost = float(fill_cost)
        self.max_full_mn = int(max_full_mn)
        self.verbose = bool(verbose)
        self.alpha = float(alpha)  # Store alpha

        self.inferred: pd.DataFrame | None = None
        self.match_table_: pd.DataFrame | None = None
        self.feature_names_: List[str] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _calculate_entropy_weights(self, X: np.ndarray) -> np.ndarray:
        """Compute per-feature entropy weights (proportional to entropy, normalized).
        
        Alpha controls entropy weighting strength:
        - alpha = 1.0: full entropy weighting (original behavior)
        - alpha = 0.0: uniform weighting (no entropy effect)
        - 0 < alpha < 1: partial entropy weighting
        """
        if X is None:
            raise RuntimeError("X is None in _calculate_entropy_weights")
        n_features = X.shape[1]
        
        # If alpha is 0, return uniform weights
        if self.alpha <= 0.0:
            return np.ones(n_features) / n_features
            
        ent_w = np.empty(n_features, dtype=float)
        for i in range(n_features):
            col = X[:, i]
            vals, counts = np.unique(col, return_counts=True)
            probs = counts / counts.sum()
            # Apply alpha to entropy weight
            ent_w[i] = float(entropy(probs)) ** self.alpha
        
        sumw = ent_w.sum()
        if sumw == 0:
            return np.ones(n_features) / n_features
        return ent_w / sumw

    def _features(self, path_to_Ai_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)
        res = build_feature_matrices(Ai_df, self.Ci_df, return_feature_names=True)
        if res is None:
            raise RuntimeError("mia.build_feature_matrices returned None")
        
        if len(res) == 3:
            X_ai, X_ci, feature_names = res
        elif len(res) == 2:
            X_ai, X_ci = res
            feature_names = None
        else:
            raise RuntimeError(f"Unexpected return from build_feature_matrices: len={len(res)}")

        if X_ai is None or X_ci is None:
            raise RuntimeError("build_feature_matrices returned None arrays")

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

        # Calculate entropy-based weights from Ai data
        weights = self._calculate_entropy_weights(X_ai)
        self._log(f"Computed entropy weights sum={weights.sum():.6f}, n_features={len(weights)}")
        if self.verbose:
            print("[entropy_weights first10]:", np.round(weights[:10], 6).tolist())

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
        if self.mode == "full" or (self.mode == "auto" and m * n <= self.max_full_mn):
            r, c, d = self._solve_full(X_ai, X_ci, weights)
        else:
            r, c, d = self._solve_knn(X_ai, X_ci, weights)

        membership = np.zeros(n, dtype=int)
        membership[c] = 1
        self.inferred = pd.DataFrame({"membership": membership})
        self.match_table_ = pd.DataFrame({"ci_idx": r, "ai_idx": c, "distance": d})

    def save_inferred(self, path_to_output: str):
        if self.inferred is None:
            raise RuntimeError("Call infer() before save_inferred()")
        self.inferred.to_csv(path_to_output, index=False, header=False)
        self._log(f"Saved inferred membership -> {path_to_output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ci→Ai Hungarian assignment with entropy-weighted features")
    ap.add_argument("path_to_Ai_csv", help="CSV with header")
    ap.add_argument("path_to_Ci_csv", help="CSV with header")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path")
    ap.add_argument("-m", "--mode", choices=["auto", "full", "knn"], default="auto")
    ap.add_argument("-k", "--k", type=int, default=300)
    ap.add_argument("--fill-cost", type=float, default=1000.0)
    ap.add_argument("--max-full-mn", type=int, default=30_000_000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-map", default=None)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Entropy weight strength (0=uniform, 1=full entropy)")
    
    args = ap.parse_args()

    attacker = AttackCiHungarianEntropy(
        path_to_Ci_csv=args.path_to_Ci_csv,
        mode=args.mode,
        k=args.k,
        fill_cost=args.fill_cost,
        max_full_mn=args.max_full_mn,
        verbose=args.verbose,
        alpha=args.alpha,    # Pass alpha to constructor
    )

    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_map and attacker.match_table_ is not None:
        attacker.match_table_.to_csv(args.out_map, index=False)
        print(f"match map saved to {args.out_map}")