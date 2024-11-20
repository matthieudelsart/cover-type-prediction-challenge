from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")

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
km_test = KMeans(n_clusters=7, n_init=10, init="k-means++")
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

clf = StackingClassifier(estimators=[
                        ('rf', RandomForestClassifier(random_state=42, n_estimators=150)),
                        ('xtr', ExtraTreesClassifier()),
                        ('lgbm', LGBMClassifier()),
                        ('KNN5', KNeighborsClassifier(3)), # Essayer 1
                    ], 
                   final_estimator=RandomForestClassifier(), 
                   cv=5,
                   n_jobs=-1,
                   verbose=1)

pipe = make_pipeline(preprocessor, clf)

pipe.fit(X_train_synth, y_train_synth)
y_pred = pipe.predict(df_test)
predictions_df = clean_predictor(y_pred)

# Having it fit the desired format
predictions_df.to_csv('test_predictions.csv', index=False) 
print("FINI")