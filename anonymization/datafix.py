import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="CSV fix script")
    parser.add_argument("input_b", help="original CSV file path")
    parser.add_argument("input_c", help="anonymized CSV file path")
    parser.add_argument("-o", "--out", required=True, help="Output CSV file path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    original = pd.read_csv(args.input_b)
    anonymized = pd.read_csv(args.input_c)
    anonymized_fixed = anonymized.copy()
    for col in original.columns:
        anonymized_fixed[col] = anonymized[col].astype(original[col].dtype)
    anonymized_fixed.to_csv(args.out, index=False)