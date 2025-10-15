from abc import ABC
import argparse
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from mia import build_feature_matrices

class AttackCiHungarian(ABC):
    def __init__(self, path_to_Ci_csv: str,
                 mode: str = "auto",
                 k: int = 300,
                 fill_cost: float = 1000.0,
                 max_full_mn: int = 30_000_000,
                 verbose: bool = False,
                 column_weights: dict = None):
        self.Ci_df = pd.read_csv(path_to_Ci_csv, dtype=str, keep_default_na=False)
        self.mode = mode
        self.k = int(k)
        self.fill_cost = float(fill_cost)
        self.max_full_mn = int(max_full_mn)
        self.verbose = bool(verbose)
        self.column_weights = column_weights or {}

        self.inferred: pd.DataFrame | None = None
        self.match_table_: pd.DataFrame | None = None
        self.feature_names_: List[str] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def save_inferred(self, path_to_output: str):
        if self.inferred is None:
            raise RuntimeError("Must call infer() before save_inferred()")
        self.inferred.to_csv(path_to_output, index=False, header=False)
        print(f"inferred membership saved to {path_to_output}")

    def _features(self, path_to_Ai_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)
        X1, X2 = build_feature_matrices(Ai_df, self.Ci_df)
        
        # Get common columns
        common_cols = sorted(set(Ai_df.columns).intersection(set(self.Ci_df.columns)))
        
        # Build feature names list matching the actual feature matrix
        feature_names = []
        feature_idx = 0
        
        for col in common_cols:
            # Try to convert to numeric
            try:
                pd.to_numeric(Ai_df[col], errors='raise')
                # Numeric column -> 1 feature
                feature_names.append(col)
                feature_idx += 1
            except (ValueError, TypeError):
                # Categorical column -> multiple one-hot features
                unique_vals = sorted(set(Ai_df[col].dropna().unique()).union(set(self.Ci_df[col].dropna().unique())))
                for val in unique_vals:
                    feature_names.append(f"{col}_{val}")
                    feature_idx += 1
        
        self.feature_names_ = feature_names
        
        # Ensure we have the right number of features
        actual_features = X1.shape[1]
        if len(feature_names) != actual_features:
            self._log(f"Warning: feature name count mismatch: {len(feature_names)} names vs {actual_features} actual features")
            # Pad or truncate feature names to match
            if len(feature_names) < actual_features:
                feature_names.extend([f"unknown_{i}" for i in range(len(feature_names), actual_features)])
            else:
                feature_names = feature_names[:actual_features]
            self.feature_names_ = feature_names
        
        # Build weight vector
        weights = np.ones(actual_features, dtype=np.float64)
        weighted_count = 0
        weighted_features = []
        
        for i in range(min(len(feature_names), actual_features)):
            fname = feature_names[i]
            base_col = fname.split('_')[0]
            
            if base_col in self.column_weights:
                weights[i] = self.column_weights[base_col]
                weighted_count += 1
                weighted_features.append(f"{fname}={self.column_weights[base_col]}")
            elif fname in self.column_weights:
                weights[i] = self.column_weights[fname]
                weighted_count += 1
                weighted_features.append(f"{fname}={self.column_weights[fname]}")
        
        self._log(f"Features: {actual_features}, Applied weights to {weighted_count} features")
        if weighted_features and self.verbose:
            self._log(f"Weighted features: {', '.join(weighted_features[:10])}")
        
        return X1, X2, weights

    def _solve_full(self, X_ai: np.ndarray, X_ci: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        self._log(f"[full] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}")
        
        X_ai_weighted = X_ai * weights
        X_ci_weighted = X_ci * weights
        
        cost = cdist(X_ci_weighted, X_ai_weighted, metric="cityblock")
        r, c = linear_sum_assignment(cost)
        d = cost[r, c]
        return r, c, d

    def _solve_knn(self, X_ai: np.ndarray, X_ci: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n = X_ci.shape[0], X_ai.shape[0]
        k = min(max(1, self.k), max(1, n))
        self._log(f"[knn] shapes: Ci={m}, Ai={n}, feat={X_ai.shape[1]}, k={k}")

        X_ai_weighted = X_ai * weights
        X_ci_weighted = X_ci * weights

        nn = NearestNeighbors(n_neighbors=k, metric="manhattan")
        nn.fit(X_ai_weighted)
        dists, inds = nn.kneighbors(X_ci_weighted, n_neighbors=k, return_distance=True)

        uniq_cols = np.unique(inds.ravel())
        col_map: Dict[int, int] = {int(j): i for i, j in enumerate(uniq_cols)}
        U = uniq_cols.size
        if U < m:
            self._log(f"[knn] warning: unique candidate Ai columns U={U} < |Ci|={m}")

        fill = float(self.fill_cost)
        C = np.full((m, U), fill, dtype=np.float64)
        for i in range(m):
            for t in range(k):
                jj = int(inds[i, t])
                C[i, col_map[jj]] = float(dists[i, t])

        r, c_small = linear_sum_assignment(C)
        d = C[r, c_small]
        c_orig = np.array([uniq_cols[j] for j in c_small], dtype=int)
        return r, c_orig, d

    def infer(self, path_to_Ai_csv: str):
        X_ai, X_ci, weights = self._features(path_to_Ai_csv)
        m, n = X_ci.shape[0], X_ai.shape[0]

        if X_ai.shape[1] == 0 or X_ci.shape[1] == 0 or m == 0:
            marks = np.zeros(n, dtype=int)
            self.inferred = pd.DataFrame(marks)
            self.match_table_ = pd.DataFrame(columns=["ci_idx", "ai_idx", "distance"])
            print(f"[AttackCiHungarian] no comparable features or empty inputs; selected=0/{n}")
            return self.inferred

        mode = self.mode
        if mode == "auto":
            mode = "full" if (m * n) <= self.max_full_mn else "knn"
            self._log(f"[auto] m*n={m*n} (limit {self.max_full_mn}) -> mode={mode}")

        if mode == "full":
            r, c, d = self._solve_full(X_ai, X_ci, weights)
        elif mode == "knn":
            r, c, d = self._solve_knn(X_ai, X_ci, weights)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        marks = np.zeros(n, dtype=int)
        used_cols = set()
        for rr, cc in zip(r, c):
            if 0 <= cc < n and cc not in used_cols:
                marks[cc] = 1
                used_cols.add(int(cc))

        self.inferred = pd.DataFrame(marks)
        self.match_table_ = pd.DataFrame({
            "ci_idx": r.astype(int),
            "ai_idx": c.astype(int),
            "distance": d.astype(float),
        })

        print(f"[AttackCiHungarian] matched={len(self.match_table_)} of Ci={m}; selected Ai={int(marks.sum())}/{n}")
        return self.inferred


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ci→Ai Hungarian assignment with column weights")
    ap.add_argument("path_to_Ai_csv", help="CSV with header")
    ap.add_argument("path_to_Ci_csv", help="CSV with header")
    ap.add_argument("-o", "--out", default="Fij.csv", help="output CSV path")
    ap.add_argument("-m", "--mode", choices=["auto", "full", "knn"], default="auto")
    ap.add_argument("-k", "--k", type=int, default=300)
    ap.add_argument("--fill-cost", type=float, default=1000.0)
    ap.add_argument("--max-full-mn", type=int, default=30_000_000)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-map", default=None)
    ap.add_argument("--weights-file", type=str, default=None,
                    help='Path to JSON file with column weights, e.g., path/to/weights.json')

    args = ap.parse_args()

    column_weights = {}
    if args.weights_file:
        import json
        with open(args.weights_file, "r", encoding="utf-8") as f:
            column_weights = json.load(f)
        print(f"Using column weights from {args.weights_file}: {column_weights}")

    attacker = AttackCiHungarian(
        path_to_Ci_csv=args.path_to_Ci_csv,
        mode=args.mode,
        k=args.k,
        fill_cost=args.fill_cost,
        max_full_mn=args.max_full_mn,
        verbose=args.verbose,
        column_weights=column_weights,
    )

    attacker.infer(args.path_to_Ai_csv)
    attacker.save_inferred(args.out)
    if args.out_map and attacker.match_table_ is not None:
        attacker.match_table_.to_csv(args.out_map, index=False)
        print(f"match table saved as {args.out_map}")