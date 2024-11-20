import numpy as np
import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from utils import clean_predictor

# Reading the data
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

# Parameters
ovs1 = 60_000
ovs2 = 93_500
smote_random = 84
min_samples_leaf = 1
min_samples_split = 2
n_estimators = 745

# Separating target and features
X_train = df_train.drop(columns=['Cover_Type'])
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
	# The np.where is here to avoid negative values

# 3. UNSUPERVISED LEARNING

# KMEANS - Without ID
km = KMeans(n_clusters=30, n_init=5, init="k-means++", random_state=0)
df_test["kmean"] = km.fit_predict(
    df_test.loc[:, "Elevation":"Horizontal_Distance_To_Fire_Points"])
X_train_synth["kmean"] = km.predict(
    X_train_synth.loc[:, "Elevation":"Horizontal_Distance_To_Fire_Points"])

# TruncatedSVD - With ID and deleting the first column
svd = TruncatedSVD(n_components=4, n_iter=7, random_state=42)
svd_cols = ["SVD_1", "SVD_2", "SVD_3", "SVD_4"]
df_test.loc[:, svd_cols] = svd.fit_transform(
    df_test.loc[:, "Id":"Horizontal_Distance_To_Fire_Points"]
)
X_train_synth.loc[:, svd_cols] = svd.transform(
    X_train_synth.loc[:, "Id":"Horizontal_Distance_To_Fire_Points"]
)

# Dropping the 1st column
X_train_synth.drop(columns=['SVD_1'], inplace=True)
df_test.drop(columns=['SVD_1'], inplace=True)

# 4. CLASS WEIGHTS - Weighted loss function, calculated using weights.py 
class_weight = {
    1: 15, 2: 1, 3: 16,
    4: 157, 5: 50, 6: 28,
    7: 23
}

# 5.CLASSIFYING
clf = ExtraTreesClassifier(n_jobs=-1, max_features=None,
                           min_samples_leaf=min_samples_leaf,
                           min_samples_split=min_samples_split,
                           n_estimators=n_estimators,
                           random_state=69,
                           class_weight=class_weight)


clf.fit(X_train_synth, y_train_synth)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)

print(f"Current best: {accuracy_score(predict_best, predict_true)}")
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False) 
