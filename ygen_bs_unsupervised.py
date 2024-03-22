from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SVMSMOTE
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]
predict_best = pd.read_csv("test_predictions_best.csv")["Cover_Type"]

# Un-one-hot-encoding the categorical variables
soil_types = [f"Soil_Type{i}" for i in range(1, 41)]
wilderness_areas = [f"Wilderness_Area{i}" for i in range(1,5)]
df_test["Wilderness_Area_Synth"] = df_test[wilderness_areas] @ range(1,5)
df_train["Wilderness_Area_Synth"] = df_train[wilderness_areas] @ range(1,5)
df_test["Soil_Type_Synth"] = df_test[soil_types] @ range(1,41)
df_train["Soil_Type_Synth"] = df_train[soil_types] @ range(1,41)
df_train = df_train.drop(columns=wilderness_areas + soil_types)
df_test = df_test.drop(columns=wilderness_areas + soil_types)

### 1. OVERSAMPLING CLASS 2 AND 1
ovs_strat = {1: 30_000, 2: 35_000}

# Separating train 
X_train = df_train.drop(columns=['Cover_Type'], axis=1)
y_train = df_train['Cover_Type']

# Oversampling
svmsmote = SVMSMOTE(sampling_strategy=ovs_strat)
X_train_synth, y_train_synth = svmsmote.fit_resample(X_train, y_train)
X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)

### 2. KMEANS 

# With ID
km_test = KMeans(n_clusters=25, n_init=5, init="k-means++")
km_test.fit_predict(df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
df_test["kmean_cluster"] = km_test.labels_
X_train_synth["kmean_cluster"] = km_test.predict(X_train_synth.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])

# Without ID
km = KMeans(n_clusters=12, n_init=5, init="k-means++")
df_test["kmean_2"] = km.fit_predict(df_test.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])
X_train_synth["kmean_2"] = km.predict(X_train_synth.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])

### 3. PCA

# With ID
pca_1 = PCA(n_components=4)
pca_cols_1 = ["PCA_1", "PCA_2", "PCA_3", "PCA_4"]
df_test.loc[:, pca_cols_1] = pca_1.fit_transform(df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
X_train_synth.loc[:, pca_cols_1] = pca_1.transform(X_train_synth.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])

# Without ID
pca_2 = PCA(n_components=2)
pca_cols_2 = ["PCA_5", "PCA_6"]
df_test.loc[:, pca_cols_2] = pca_2.fit_transform(df_test.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])
X_train_synth.loc[:, pca_cols_2] = pca_2.transform(X_train_synth.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])


### 4. GENERATING
clf = RandomForestClassifier(n_estimators=150, n_jobs=-1)

clf.fit(X_train_synth, y_train_synth)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)

# Having it fit the desired format
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False) 