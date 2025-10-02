# synth_compare_report.py
# 合成データの同質性検証を一括実行し、PDFに出力する
# 依存: pandas, numpy, matplotlib, scikit-learn, scipy

import argparse
import itertools
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from scipy.stats import ks_2samp, chi2_contingency

# ------------------------
# ユーティリティ
# ------------------------

def text_page(pdf, title, lines, font_size=10):
    plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    plt.axis("off")
    plt.title(title, loc="left", fontsize=14, pad=16)
    y = 0.92
    for line in lines:
        plt.text(0.02, y, str(line), fontsize=font_size, va="top", family="monospace")
        y -= 0.032
        if y < 0.05:
            pdf.savefig(bbox_inches="tight")
            plt.close()
            plt.figure(figsize=(11.69, 8.27))
            plt.axis("off")
            y = 0.92
    pdf.savefig(bbox_inches="tight")
    plt.close()

def figure_title(title):
    plt.title(title, loc="left", fontsize=12, pad=8)

def pairwise(iterable):
    arr = list(iterable)
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            yield arr[i], arr[j]

def detect_columns(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols

def mmd_rbf(X1, X2, gamma=1.0):
    # X?: (n, d) 2D ndarray
    XX = np.exp(-gamma * ((X1[:, None, :] - X1[None, :, :]) ** 2).sum(axis=2)).mean()
    YY = np.exp(-gamma * ((X2[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)).mean()
    XY = np.exp(-gamma * ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(axis=2)).mean()
    return XX + YY - 2 * XY

def add_source_column(dfs, names, source_col="__source__"):
    out = []
    for df, name in zip(dfs, names):
        tmp = df.copy()
        tmp[source_col] = name
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

# ------------------------
# メイン処理
# ------------------------

def main(args):
    # 入力読み込み
    file_paths = args.inputs
    names = args.names if args.names else [f"Data{i+1}" for i in range(len(file_paths))]
    dfs = [pd.read_csv(p) for p in file_paths]

    # PDF開始
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        # 表紙
        lines = [
            f"合成データ比較レポート",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"入力: " + ", ".join([os.path.basename(p) for p in file_paths]),
            f"データセット名: " + ", ".join(names),
            "",
            "本レポートの内容:",
            "1) 基本統計・分布/箱ひげ図",
            "2) 相関構造（相関行列のヒートマップ）",
            "3) 次元削減可視化（PCA, t-SNE）",
            "4) 識別可能性テスト（RF, 交差検証）",
            "5) 統計的検定（KS, カイ二乗, MMD）",
        ]
        text_page(pdf, "表紙", lines)

        # 1. 各データ:基本統計・分布
        for name, df in zip(names, dfs):
            num_cols, cat_cols = detect_columns(df)

            # 基本統計(数値)
            if num_cols:
                desc = df[num_cols].describe().T
                lines = [f"=== {name}: 基本統計（数値） ===", str(desc)]
                text_page(pdf, f"{name} - 基本統計（数値）", lines)

                # ヒストグラム（数値）
                n = len(num_cols)
                cols = 3
                rows = math.ceil(n / cols)
                plt.figure(figsize=(11.69, 8.27))
                for idx, c in enumerate(num_cols, 1):
                    ax = plt.subplot(rows, cols, idx)
                    ax.hist(df[c].dropna(), bins=30)
                    ax.set_title(c, fontsize=9)
                figure_title(f"{name} - 数値ヒストグラム")
                plt.tight_layout()
                pdf.savefig(bbox_inches="tight")
                plt.close()

                # 箱ひげ図（数値）
                plt.figure(figsize=(11.69, 8.27))
                plt.boxplot([df[c].dropna().values for c in num_cols], labels=num_cols, vert=True, showfliers=False)
                figure_title(f"{name} - 箱ひげ図（外れ値非表示）")
                plt.xticks(rotation=90, fontsize=8)
                plt.tight_layout()
                pdf.savefig(bbox_inches="tight")
                plt.close()

            # カテゴリ分布（上位カテゴリー）
            for c in cat_cols:
                vc = df[c].astype(str).value_counts().head(20)
                lines = [f"=== {name}: カテゴリ分布 [{c}] Top20 ===", str(vc)]
                text_page(pdf, f"{name} - カテゴリ分布 [{c}] Top20", lines)

        # 2. 相関構造（各データセット）
        for name, df in zip(names, dfs):
            num_cols, _ = detect_columns(df)
            if not num_cols:
                continue
            corr = df[num_cols].corr()
            plt.figure(figsize=(11.69, 8.27))
            im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(len(num_cols)), num_cols, rotation=90, fontsize=7)
            plt.yticks(range(len(num_cols)), num_cols, fontsize=7)
            figure_title(f"{name} - 相関行列")
            plt.tight_layout()
            pdf.savefig(bbox_inches="tight")
            plt.close()

        # 3. 次元削減（PCA / t-SNE）
        combined = add_source_column(dfs, names, source_col="__source__")
        all_num_cols, _ = detect_columns(combined.drop(columns=["__source__"]))
        if all_num_cols:
            X = combined[all_num_cols].copy()
            mask = X.notnull().all(axis=1)
            X = X[mask]
            y = combined.loc[mask, "__source__"].values

            scaler = StandardScaler()
            Xz = scaler.fit_transform(X)

            # PCA
            pca = PCA(n_components=2, random_state=42)
            Xp = pca.fit_transform(Xz)
            plt.figure(figsize=(11.69, 8.27))
            for name in np.unique(y):
                sel = y == name
                plt.scatter(Xp[sel, 0], Xp[sel, 1], label=name, s=10, alpha=0.7)
            plt.legend(markerscale=2)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            exp = pca.explained_variance_ratio_
            figure_title(f"PCA 可視化（累積寄与: {(exp[:2].sum()*100):.1f}%）")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            # t-SNE（計算コスト注意）
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, Xz.shape[0] // 100)))
                Xt = tsne.fit_transform(Xz)
                plt.figure(figsize=(11.69, 8.27))
                for name in np.unique(y):
                    sel = y == name
                    plt.scatter(Xt[sel, 0], Xt[sel, 1], label=name, s=10, alpha=0.7)
                plt.legend(markerscale=2)
                plt.xlabel("t-SNE-1")
                plt.ylabel("t-SNE-2")
                figure_title("t-SNE 可視化")
                pdf.savefig(bbox_inches="tight")
                plt.close()
            except Exception as e:
                text_page(pdf, "t-SNE 実行スキップ", [f"原因: {repr(e)}"])

        # 4. 識別可能性テスト（RF）
        if all_num_cols:
            X = combined[all_num_cols].copy()
            mask = X.notnull().all(axis=1)
            X = X[mask]
            y = combined.loc[mask, "__source__"].values
            Xz = StandardScaler().fit_transform(X)

            clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, Xz, y, cv=cv, scoring="accuracy", n_jobs=-1)
            lines = [
                "=== ランダムフォレストによる識別可能性テスト ===",
                f"平均精度: {scores.mean():.4f}",
                f"標準偏差: {scores.std():.4f}",
                "",
                "解釈メモ:",
                "- 精度が低いほど区別困難（=分布が似ている傾向）。",
                "- 精度が高い場合、生成過程/パラメータ差の可能性。",
            ]
            text_page(pdf, "識別可能性テスト（RF, 5-fold CV）", lines)

        # 5. 統計的検定（KS / カイ二乗 / MMD）
        # 5-1 KS（数値の列ごと・ペアごと）
        ks_lines = ["=== KS検定（数値列）==="]
        for col in all_num_cols:
            series_by_src = {n: df[col].dropna() for n, df in zip(names, dfs) if col in df.columns}
            for (n1, s1), (n2, s2) in pairwise(series_by_src.items()):
                if len(s1) > 0 and len(s2) > 0:
                    stat, p = ks_2samp(s1, s2)
                    ks_lines.append(f"{col}: {n1} vs {n2} -> KS p={p:.4g}, stat={stat:.4g}")
        text_page(pdf, "KS検定の結果", ks_lines)

        # 5-2 カイ二乗（カテゴリ列）
        cat_cols_union = sorted(
            set(itertools.chain.from_iterable([df.select_dtypes(exclude=[np.number]).columns.tolist() for df in dfs]))
        )
        if cat_cols_union:
            chi_lines = ["=== カイ二乗検定（カテゴリ列×データセット）==="]
            for c in cat_cols_union:
                sub = combined[[c, "__source__"]].dropna()
                if sub.empty:
                    continue
                cont = pd.crosstab(sub[c].astype(str), sub["__source__"])
                if cont.shape[0] > 1 and cont.shape[1] > 1:
                    chi2, p, dof, _ = chi2_contingency(cont)
                    chi_lines.append(f"{c}: p={p:.4g}, chi2={chi2:.4g}, dof={dof}")
            text_page(pdf, "カイ二乗検定の結果", chi_lines)

        # 5-3 MMD（数値の全体分布距離）
        if all_num_cols:
            X_by_src = {}
            for n, df in zip(names, dfs):
                if set(all_num_cols).issubset(df.columns):
                    Xn = df[all_num_cols].dropna()
                    if not Xn.empty:
                        X_by_src[n] = StandardScaler().fit_transform(Xn.values)
            if len(X_by_src) >= 2:
                mmd_lines = ["=== MMD（RBF, gamma=1.0）==="]
                for n1, n2 in pairwise(X_by_src.keys()):
                    val = mmd_rbf(X_by_src[n1], X_by_src[n2], gamma=1.0)
                    mmd_lines.append(f"{n1} vs {n2}: MMD={val:.6f}  （小さいほど近い）")
                text_page(pdf, "MMDによる分布距離", mmd_lines)

        # 付録: データ情報
        appendix = ["=== 付録: 列タイプ情報 ==="]
        for name, df in zip(names, dfs):
            num_cols, cat_cols = detect_columns(df)
            appendix += [f"[{name}] 数値: {num_cols}", f"[{name}] カテゴリ: {cat_cols}", ""]
        text_page(pdf, "付録", appendix)

    print(f"[OK] PDF 出力: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成データ比較レポート（PDF出力）")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="比較するCSVファイル群（スペース区切りで複数指定）"
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="各データセットの表示名（省略時は Data1, Data2, ...）"
    )
    parser.add_argument(
        "--out", type=str, default="analysis_report.pdf",
        help="出力PDFファイル名（既定: analysis_report.pdf）"
    )
    args = parser.parse_args()
    main(args)
