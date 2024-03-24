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


def split_in_two_df(df: pd.DataFrame):
    if "under2" not in df.columns:
        # Split between cover type >2 and <=2
        df["under2"] = (df["Cover_Type"] < 3).astype(int)

    # 1 -2
    df_under = df.where(df["under2"] == 1)
    df_under.dropna(inplace=True)
    df_under = df_under.astype(int)  # @TODO c'est bizarre

    # 3-4-5-6-7
    df_above = df.where(df["under2"] == 0)
    df_above.dropna(inplace=True)
    df_above = df_above.astype(int)  # @TODO c'est bizarre

    return df, df_under, df_above


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


def train_first_model(X, y, model, test_size=0.25, seed=42, startify=True):
    if startify:
        strf = y
    else:
        strf = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strf)

    model.fit(X, y)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    return model


def train_second_model(X, y, model, test_size=0.25, seed=42, startify=True):
    if startify:
        strf = y
    else:
        strf = None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strf)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    model.fit(X, le.fit_transform(y))
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # y_pred = le.inverse_transform(y_pred)

    return model


# Load training data
df_train = pd.concat([X_train_synth, y_train_synth], axis=1)
# X_train = X_train_synth.copy()
# y_train = y_train_syntha.copy()

df_train = preprocess(df_train)

# Split the data
df_train, df_train_under, df_train_above = split_in_two_df(df_train)
X_train_under, y_train_under = get_X_y(
    df_train_under, target_col='Cover_Type')
X_train_above, y_train_above = get_X_y(
    df_train_above, target_col='Cover_Type')
X_train, y_train = get_X_y(df_train, target_col='under2')

# Train first model
model = xgb.XGBClassifier(n_estimators=200, max_depth=15)
model1 = train_first_model(X_train, y_train, model)

# Train second model
model2u = xgb.XGBClassifier(n_estimators=200, max_depth=15)
model2u = train_second_model(
    X_train_under, y_train_under, model2u)

model2a = xgb.XGBClassifier(objective="multi:softmax")
model2a = train_second_model(
    X_train_above, y_train_above, model2a)

# Load Testing data
df_test = preprocess(df_test)

X_test = get_X_y(df_test)
y1_pred = model1.predict(X_test)

X_test["under2"] = y1_pred

df_test1, df_test_under, df_test_above = split_in_two_df(X_test)
X_test_under = get_X_y(df_test_under)
X_test_under.drop(["under2"], axis=1, inplace=True)
X_test_above = get_X_y(df_test_above)
X_test_above.drop(["under2"], axis=1, inplace=True)

y2u_pred = model2u.predict(X_test_under)
y2a_pred = model2a.predict(X_test_above)

# Convert prediction into Dataframe
pred_under = pd.DataFrame({'Id': X_test_under.Id, 'Cover_Type': y2u_pred+1})
pred_above = pd.DataFrame({'Id': X_test_above.Id, 'Cover_Type': y2a_pred+3})


final_pred = pd.concat([pred_under, pred_above])
predictions_df = clean_predictor(
    y_pred=final_pred.Cover_Type, Id=final_pred.Id)


# Having it fit the desired format
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]
predict_best = pd.read_csv("test_predictions_best.csv")["Cover_Type"]
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False)
print("FINI")
