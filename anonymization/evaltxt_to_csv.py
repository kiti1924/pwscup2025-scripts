import pandas as pd
import os
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("eval_txt", help="path to eval.txt")
    ap.add_argument("-o", "--output", help="path to eval_summary.csv", default="eval_summary.csv")
    args = ap.parse_args()

    if not os.path.isfile(args.eval_txt):
        raise FileNotFoundError(f"{args.eval_txt} は存在しません。")

    result = []
    if os.path.exists(args.eval_txt):
        result.append(os.path.basename(args.eval_txt))
    with open(args.eval_txt) as f:
        lines = f.readlines()
        for line in lines:
            if ":" in line:
                score_a = line.split(":")[1].strip()
                score_b = score_a.split()[0]
                result.append(float(score_b))

    df1 = pd.DataFrame([result], columns=["anonymization", "stats_diff", "LR_asthma_diff", "KW_IND_diff", "Ci_utility"])
    df1.to_csv(args.output, index=False, mode="a", header=not os.path.exists(args.output))