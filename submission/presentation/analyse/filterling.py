# Re-filter allowing NaN only for 5 specified columns

import pandas as pd
import json
import numpy as np
from pathlib import Path

base = Path("/mnt/data")
merged_path = base / "NHANES_P_merged_HI100K_like.csv"
schema_path = base / "pre_columns_range.json"
filtered_path4 = base / "NHANES_P_filtered_skip5cols.csv"

# Load data and schema
df = pd.read_csv(merged_path)
with open(schema_path, "r") as f:
    schema = json.load(f)["columns"]

# Columns allowed to have NaN (ignored for missing)
skipna_cols = {"encounter_count", "num_procedures", "num_devices", "mean_systolic_bp", "mean_diastolic_bp"}

total_records = len(df)
mask = pd.Series(True, index=df.index)

for col, rules in schema.items():
    if col not in df.columns:
        continue

    allow_na = col in skipna_cols

    if rules["type"] == "category":
        valid_values = set(rules["values"])
        if allow_na:
            mask &= df[col].isin(valid_values) | df[col].isna()
        else:
            mask &= df[col].isin(valid_values)

    elif rules["type"] == "number":
        min_v, max_v = rules.get("min", -np.inf), rules.get("max", np.inf)
        if allow_na:
            valid = df[col].between(min_v, max_v, inclusive="both") | df[col].isna()
        else:
            valid = df[col].between(min_v, max_v, inclusive="both") & (~df[col].isna())
        mask &= valid

# Apply mask
filtered_df = df[mask].copy()
remaining = len(filtered_df)
removed = total_records - remaining

# Save filtered dataset
filtered_df.to_csv(filtered_path4, index=False)

import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Filtered NHANES (Skip 5 specific columns only, first 200 rows)", filtered_df.head(200))

summary4 = {
    "total_records": int(total_records),
    "remaining_records": int(remaining),
    "removed_records": int(removed),
    "retained_percentage": round(remaining / total_records * 100, 2) if total_records > 0 else 0
}
summary4
