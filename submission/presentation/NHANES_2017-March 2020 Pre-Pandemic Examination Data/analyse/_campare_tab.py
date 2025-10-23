import json
import pandas as pd

with open("model_eval_outputs/three_models_metrics.json", encoding="utf-8") as f:
    res = json.load(f)

df = pd.DataFrame({m: v["metrics"] for m, v in res.items()}).T
df = df[["Accuracy","Precision","Recall","F1","ROC_AUC"]]
print(df.round(4).to_markdown())
