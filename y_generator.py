from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]
predict_best = pd.read_csv("test_predictions_best.csv")["Cover_Type"]

### 1. OVERSAMPLING CLASS 2 AND 1
ovs_strat = {1: 30_000, 2: 30_000}

# Separating train 
X_train = df_train.drop(columns=['Cover_Type'], axis=1)
y_train = df_train['Cover_Type']

# Oversampling
adasyn = ADASYN(sampling_strategy=ovs_strat)
X_train_synth, y_train_synth = adasyn.fit_resample(X_train, y_train)
X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)

### 2. KMEANS
km_test = KMeans(n_clusters=30, n_init=10, init="k-means++")
km_test.fit_predict(df_test.loc[:, "Id":"Wilderness_Area4"])
df_test["kmean_cluster"] = km_test.labels_
X_train_synth["kmean_cluster"] = km_test.predict(X_train_synth.loc[:, "Id":"Wilderness_Area4"])

### 3. GENERATING
cat_col = ["kmean_cluster"]
cols = X_train_synth.drop(columns=['kmean_cluster']).columns

preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col),
        ("others", "passthrough", cols),
    ])
clf = RandomForestClassifier(n_estimators=150)
pipe = make_pipeline(preprocessor, clf)

pipe.fit(X_train_synth, y_train_synth)
y_pred = pipe.predict(df_test)
predictions_df = clean_predictor(y_pred)

# Having it fit the desired format
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False) 