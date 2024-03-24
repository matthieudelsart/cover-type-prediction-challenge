from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SVMSMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

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
# print(X_train_synth["GMM"].value_counts())

# 4. GENERATING
SEED = 42
estimators = [
    # , class_weight=c_w)),
    ('rf', RandomForestClassifier(n_estimators=200)),
    ('svr', make_pipeline(StandardScaler(),
                          LinearSVC(dual="auto"))),
    ('xgb', xgb.XGBClassifier(n_estimators=200)),
    # ('xtree'), ExtraTreesClassifier(random_state=SEED),
    ('ada', AdaBoostClassifier(algorithm="SAMME", n_estimators=200)),
    ('xgbrf', xgb.XGBRFClassifier(n_estimators=200)),
    ('lgbm', LGBMClassifier(
        objective='multiclass',
        num_class=7,
        boosting_type='gbdt',
        verbose=0,
        n_jobs=-1
    ))
]
clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500)
    # LGBMClassifier(
    #     objective='multiclass',
    #     num_class=7,
    #     boosting_type='gbdt',
    #     verbose=0,
    #     n_jobs=-1
    # )
)

clf.fit(X_train_synth, y_train_synth)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)

# Having it fit the desired format
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]
predict_best = pd.read_csv("test_predictions_best.csv")["Cover_Type"]
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False)
print("FINI")
