import pandas as pd
import gzip

# Replace 'file.gz' with the path to your gzipped file
file_path = 'data/covtype.data.gz'

test_df = pd.read_csv('../test-full.csv')
pred_df = pd.read_csv('preds.csv')

# Open the gzipped file and read its contents
df = pd.read_csv(gzip.open(file_path), compression='infer', header=None)
cols = list(df.columns)
df['Id'] = df.index + 1
df = df[['Id'] + cols]
df.columns = list(test_df.columns) + ['Cover_Type']

# Check accuracy
predictions = pred_df['Cover_Type']

# Assuming 'Cover_Type' is the column containing the ground truth in df_ground_truth
ground_truth = df['Cover_Type']

print(ground_truth.value_counts())
print(len(ground_truth))

# Compare the predicted values with the ground truth
accuracy = (predictions == ground_truth).mean()

print("Accuracy:", accuracy)

df.to_parquet('data/ground_truth.parquet', index=False)