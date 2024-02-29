import pandas as pd
import numpy as np

PREDS_PATH = ""
TRUTH_PATH = "data/ground_truth.parquet"

truth = pd.read_parquet(TRUTH_PATH)['Cover_Type'].to_numpy()
preds = pd.read_parquet(PREDS_PATH)['Cover_Type'].to_numpy()

accuracy = (preds == truth).mean()

print("Accuracy:", accuracy)