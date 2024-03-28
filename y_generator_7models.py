from sklearn.pipeline import make_pipeline
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SVMSMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from utils import *
from feature_engineering import *


def preprocess(df):
    df = concat_Soil_Type(df, new_name='ST_concat', drop_value=True)
    df = concat_Wilderness_Area(df, new_name='WA_concat', drop_value=True)
    df = group_climatic_zone(df)
    df = group_geological_zone(df)
    return df


df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")

# 1. OVERSAMPLING CLASS 2 AND 1
ovs_strat = {1: 30_000, 2: 35_000}

# Separating train
X_train = df_train.drop(columns=['Cover_Type'], axis=1)
y_train = df_train['Cover_Type']

# Oversampling
svmsmote = SVMSMOTE(sampling_strategy=ovs_strat)
X_train_synth, y_train_synth = svmsmote.fit_resample(X_train, y_train)
X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)

# 2. KMEANS - With ID
km_test = KMeans(n_clusters=25, n_init=5, init="k-means++")
km_test.fit_predict(df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
df_test["kmean_cluster"] = km_test.labels_
X_train_synth["kmean_cluster"] = km_test.predict(
    X_train_synth.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])

# 3. KMEANS - Without ID
km = KMeans(n_clusters=12, n_init=5, init="k-means++")
df_test["GMM"] = km.fit_predict(
    df_test.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])
X_train_synth["GMM"] = km.predict(
    X_train_synth.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])


def get_X_y(df: pd.DataFrame, target_col=None):
    X = df.copy()
    if target_col == "under2":
        X.drop(["Cover_Type"], axis=1, inplace=True)
    elif target_col == "Cover_Type":
        X.drop(["under2"], axis=1, inplace=True)
    if target_col in X.columns:
        y = X[target_col]
        X.drop([target_col], axis=1, inplace=True)
        return X, y
    else:
        # X.drop(["under2", "Cover_Type"], axis=1, inplace=True)
        return X


def split_in_7_dfs(df: pd.DataFrame):
    # if "under2" not in df.columns:
    #     # Split between cover type >2 and <=2
    #     df["under2"] = (df["Cover_Type"] < 3).astype(int)
    dfs = [df]

    for i in range(1, 8):
        new_df = df.where(df["Cover_Type"] == i)
        new_df.dropna(inplace=True)
        new_df = new_df.astype(int)  # @TODO c'est bizarre
        dfs.append(new_df)

    return dfs


# Load training data
df_train = pd.concat([X_train_synth, y_train_synth], axis=1)
# X_train = X_train_synth.copy()
# y_train = y_train_syntha.copy()

df_train = preprocess(df_train)

# Split the data
df_train, df1, df2, df3, df4, df5, df6, df7 = split_in_7_dfs(df_train)
dfs = [df1, df2, df3, df4, df5, df6, df7]
Xy = []
for df in dfs:
    X = df.drop(columns='Cover_Type')
    y = df['Cover_Type']
    Xy.append([X, y])

# Train  model
# model = xgb.XGBClassifier(n_estimators=200, max_depth=15)
print('debut train')
models = []
for e in Xy:
    model = xgb.XGBClassifier(n_estimators=200, max_depth=15)
    le = LabelEncoder()
    model.fit(e[0], le.fit_transform(e[1]))
    models.append(model)
print('fin train 7 models')
# Concat/stacking
X_train = df_train.drop(columns='Cover_Type')
y_train = df_train['Cover_Type']

print('debut pred sur train')
pred_train = []
for i, m in enumerate(models):
    pred = m.predict_proba(X_train)
    pred_train.append(pd.DataFrame(pred[:, 1], columns=[f'1_{i}']))
print('fin pred 7 models sur train')

new_train = pd.concat(pred_train, axis=1)
print(new_train)
model_final = LGBMClassifier()  # xgb.XGBClassifier(objective="multi:softmax")
le2 = LabelEncoder()

model_final.fit(new_train, le2.fit_transform(y_train))

print('debut test')
# Load Testing data
df_test = preprocess(df_test)

pred_test = []
for i, m in enumerate(models):
    pred = m.predict_proba(df_test)
    pred_test.append(pd.DataFrame(pred[:, 1], columns=[f'1_{i}']))

new_test = pd.concat(pred_test, axis=1)
pred_final = model_final.predict(new_test)
final_pred = pd.DataFrame({'Id': df_test.Id, 'Cover_Type': pred_final+1})
predictions_df = clean_predictor(
    y_pred=final_pred.Cover_Type, Id=final_pred.Id)


# Having it fit the desired format
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]
predict_best = pd.read_csv("test_predictions_best.csv")["Cover_Type"]
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False)
print("FINI")
