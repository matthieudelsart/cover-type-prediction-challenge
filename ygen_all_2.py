import numpy as np
import pandas as pd

from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

from utils_2 import *


def preprocess(df):
    # Un-one-hot-encoding the categorical variables
    df = concat_Soil_Type(df, drop_value=True)
    df = concat_Wilderness_Area(df, drop_value=True)
    return df


df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")

# Parameters
ovs1 = 60_000
ovs2 = 93_500
smote_random = 84
pca_random = 67
ext_random = 69
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 745

# Preprocessing
df_train = preprocess(df_train)
df_test = preprocess(df_test)

# Separating train
X_train = df_train.drop(columns=['Cover_Type'], axis=1)
y_train = df_train['Cover_Type']
base_cols = list(X_train.columns)

# 1. OVERSAMPLING CLASS 2 AND 1
ovs_strat = {1: ovs1, 2: ovs2}

# Oversampling
svmsmote = SVMSMOTE(sampling_strategy=ovs_strat, random_state=smote_random)
X_train_synth, y_train_synth = svmsmote.fit_resample(X_train, y_train)
X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)


# 2. NON-LINEAR FEATURES
# Cross-features
mult_combinator = {"Horizontal_Distance_To_Fire_Points": ["Id", "Elevation", "Horizontal_Distance_To_Roadways"],
                   "Elevation": ["Id", "Horizontal_Distance_To_Roadways"]}

new_cols = []
for key in mult_combinator:
    for value in mult_combinator[key]:
        new_cols.append(f"{key} * {value}")
        df_test[f"{key} * {value}"] = df_test[key] * df_test[value]
        X_train_synth[f"{key} * {value}"] = X_train_synth[key] * \
            X_train_synth[value]

# Log features
log_combinator = ['Id', 'Horizontal_Distance_To_Roadways',
                  'Horizontal_Distance_To_Fire_Points']

for label in log_combinator:
    temp = np.where(df_test[label] > 0.5, df_test[label], -10)
    df_test[f"log({label})"] = np.log(temp, where=temp > 0.5)
    temp = np.where(X_train_synth[label] > 0.5, X_train_synth[label], -10)
    X_train_synth[f"log({label})"] = np.log(temp, where=temp > 0.5)
    X_train_synth[X_train_synth.isna().any(axis=1)][f"log({label})"] = -10


# 3. UNSUPERVISED LEARNING

# KMEANS - Without ID
km = KMeans(n_clusters=30, n_init=5, init="k-means++")
df_test["kmean"] = km.fit_predict(
    df_test.loc[:, "Elevation":"Horizontal_Distance_To_Fire_Points"])
X_train_synth["kmean"] = km.predict(
    X_train_synth.loc[:, "Elevation":"Horizontal_Distance_To_Fire_Points"])

# PCA - With ID
pca_1 = PCA(n_components=4, random_state=pca_random)
pca_cols_1 = ["PCA_1", "PCA_2", "PCA_3", "PCA_4"]
df_test.loc[:, pca_cols_1] = pca_1.fit_transform(
    df_test.loc[:, "Id":"Horizontal_Distance_To_Fire_Points"])

X_train_synth.loc[:, pca_cols_1] = pca_1.transform(
    X_train_synth.loc[:, "Id":"Horizontal_Distance_To_Fire_Points"])
# Dropping PCA_1 (virtually the same as the Id column)
df_test.drop(columns='PCA_1', inplace=True)
X_train_synth.drop(columns='PCA_1', inplace=True)


# 4.CLASSIFYING

clf = ExtraTreesClassifier(n_jobs=-1, max_features=None,
                           min_samples_leaf=min_samples_leaf,
                           min_samples_split=min_samples_split,
                           n_estimators=n_estimators,
                           random_state=ext_random)


clf.fit(X_train_synth, y_train_synth)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)
predictions_df.to_csv(f'with_all_features.csv', index=False)
print(accuracy_score(pd.read_parquet("ground_truth.parquet")
      ["Cover_Type"], predictions_df["Cover_Type"]))
