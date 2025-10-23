# Retry: NHANES P-series (2021-2022) -> HI_100K-like merger and export
import pandas as pd
import numpy as np
from pathlib import Path

base = Path("/mnt/data")

def read_xpt(fname):
    return pd.read_sas(base / fname, format="xport")

files = {
    "DEMO": "P_DEMO.xpt",
    "BMX": "P_BMX.xpt",
    "BPX": "P_BPXO.xpt",
    "MCQ": "P_MCQ.xpt",
    "DPQ": "P_DPQ.xpt",
    "RXQ_RX": "P_RXQ_RX.xpt",
    "IMQ": "P_IMQ.xpt",
}

dfs = {}
for key, fname in files.items():
    df = read_xpt(fname)
    df.columns = [str(c) for c in df.columns]
    dfs[key] = df

# DEMO
demo_cols = ["SEQN","RIAGENDR","RIDAGEYR"]
if "RIDRETH3" in dfs["DEMO"].columns:
    demo_cols.append("RIDRETH3")
elif "RIDRETH1" in dfs["DEMO"].columns:
    demo_cols.append("RIDRETH1")
demo = dfs["DEMO"][demo_cols].copy()

# BMX
bmx = dfs["BMX"][["SEQN"] + [c for c in ["BMXWT","BMXBMI"] if c in dfs["BMX"].columns]].copy()

# BPX
bpx = dfs["BPX"].filter(regex="^SEQN$|^BPXSY|^BPXDI").copy()
def row_mean(cols, df):
    present = [c for c in cols if c in df.columns]
    return df[present].replace({-1: np.nan, 0: np.nan}).mean(axis=1)

bpx["mean_systolic_bp"] = row_mean([f"BPXSY{i}" for i in range(1,6)], bpx)
bpx["mean_diastolic_bp"] = row_mean([f"BPXDI{i}" for i in range(1,6)], bpx)
bpx = bpx[["SEQN","mean_systolic_bp","mean_diastolic_bp"]]

# MCQ
mcq = dfs["MCQ"][["SEQN"] + [c for c in ["MCQ010","MCQ160F","MCQ190"] if c in dfs["MCQ"].columns]].copy()

# DPQ
dpq = dfs["DPQ"][["SEQN"] + [c for c in ["DPQ020"] if c in dfs["DPQ"].columns]].copy()

# RXQ_RX
rx = dfs["RXQ_RX"].copy()
rx_counts = rx.groupby("SEQN").size().reset_index(name="num_medications")

# IMQ
imq = dfs["IMQ"].copy()
imq_cols = [c for c in ["IMQ020","IMQ030","IMQ040","IMQ050","IMQ060"] if c in imq.columns]
imq_use = imq[["SEQN"] + imq_cols].copy()
for c in imq_cols:
    imq_use[c] = (imq_use[c] == 1).astype(int)
imq_use["num_immunizations"] = imq_use[imq_cols].sum(axis=1)
imq_use = imq_use[["SEQN","num_immunizations"]]

# Merge
merged = demo.merge(bmx, on="SEQN", how="left") \
             .merge(bpx, on="SEQN", how="left") \
             .merge(mcq, on="SEQN", how="left") \
             .merge(dpq, on="SEQN", how="left") \
             .merge(rx_counts, on="SEQN", how="left") \
             .merge(imq_use, on="SEQN", how="left")

# Derivations
merged["GENDER"] = merged["RIAGENDR"].map({1: "M", 2: "F"})
merged["AGE"] = merged["RIDAGEYR"]

def derive_ethnicity(row):
    r3 = row.get("RIDRETH3", np.nan) if isinstance(row, dict) else row.get("RIDRETH3", np.nan)
    r1 = row.get("RIDRETH1", np.nan) if isinstance(row, dict) else row.get("RIDRETH1", np.nan)
    if not pd.isna(r3):
        return "hispanic" if r3 in [1,2] else "nonhispanic"
    if not pd.isna(r1):
        return "hispanic" if r1 in [1,2] else "nonhispanic"
    return np.nan

def derive_race(row):
    r3 = row.get("RIDRETH3", np.nan) if isinstance(row, dict) else row.get("RIDRETH3", np.nan)
    r1 = row.get("RIDRETH1", np.nan) if isinstance(row, dict) else row.get("RIDRETH1", np.nan)
    if not pd.isna(r3):
        if r3 == 3: return "white"
        if r3 == 4: return "black"
        if r3 == 6: return "asian"
        if r3 in [1,2]: return "other"
        return "other"
    if not pd.isna(r1):
        if r1 == 3: return "white"
        if r1 == 4: return "black"
        if r1 in [1,2]: return "other"
        return "other"
    return np.nan

merged["ETHNICITY"] = merged.apply(derive_ethnicity, axis=1)
merged["RACE"] = merged.apply(derive_race, axis=1)

merged["num_allergies"] = (merged.get("MCQ190", pd.Series(index=merged.index)) == 1).astype(int)
merged["asthma_flag"] = (merged.get("MCQ010", pd.Series(index=merged.index)) == 1).astype(int)
merged["stroke_flag"] = (merged.get("MCQ160F", pd.Series(index=merged.index)) == 1).astype(int)
if "DPQ020" in merged.columns:
    merged["depression_flag"] = merged["DPQ020"].apply(lambda x: 1 if pd.notna(x) and x >= 2 else 0)
else:
    merged["depression_flag"] = np.nan

merged["mean_bmi"] = merged.get("BMXBMI")
merged["mean_weight"] = merged.get("BMXWT")
merged["obesity_flag"] = (merged["mean_bmi"] >= 30).astype(int)

for col in ["num_medications","num_immunizations"]:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0).astype(int)

merged["encounter_count"] = np.nan
merged["num_procedures"] = np.nan
merged["num_devices"] = np.nan

final_cols = [
    "GENDER","AGE","RACE","ETHNICITY",
    "encounter_count","num_procedures","num_medications","num_immunizations","num_allergies","num_devices",
    "asthma_flag","stroke_flag","obesity_flag","depression_flag",
    "mean_systolic_bp","mean_diastolic_bp","mean_bmi","mean_weight","SEQN"
]
final = merged[final_cols]

out_path = base / "NHANES_P_merged_HI100K_like.csv"
final.to_csv(out_path, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("NHANES to HI_100K preview (first 200 rows)", final.head(200))

print("Saved:", out_path)
