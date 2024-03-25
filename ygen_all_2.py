from sklearn.cluster import KMeans
from imblearn.over_sampling import SVMSMOTE
from utils import clean_predictor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
base_cols = list(X_train.columns)

# Oversampling
svmsmote = SVMSMOTE(sampling_strategy=ovs_strat, random_state=1)
X_train_synth, y_train_synth = svmsmote.fit_resample(X_train, y_train)
X_train_synth = pd.DataFrame(X_train_synth, columns=X_train.columns)


#### 2. NON-LINEAR FEATURES
# Cross-features
mult_combinator = {"Horizontal_Distance_To_Fire_Points": ["Id", "Elevation", "Horizontal_Distance_To_Roadways"],
                   "Elevation" : ["Id", "Horizontal_Distance_To_Roadways"]}

new_cols = []
for key in mult_combinator:
    for value in mult_combinator[key]:
        new_cols.append(f"{key} * {value}")
        df_test[f"{key} * {value}"] = df_test[key] * df_test[value]    
        X_train_synth[f"{key} * {value}"] = X_train_synth[key] * X_train_synth[value]   

# Log features
log_combinator = ['Id', 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']

for label in log_combinator:
    temp = np.where(df_test[label] > 0, df_test[label], -10)
    df_test[f"log({label})"] = np.log(temp, where=temp > 0)
    temp = np.where(X_train_synth[label] > 0, X_train_synth[label], -10)
    X_train_synth[f"log({label})"] = np.log(temp, where=temp > 0)
    
        
### 3. UNSUPERVISED LEARNING 
 
# KMEANS - Without ID
km = KMeans(n_clusters=12, n_init=5, init="k-means++")
df_test["kmean"] = km.fit_predict(df_test.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])
X_train_synth["kmean"] = km.predict(X_train_synth.loc[:, "Elevation": "Horizontal_Distance_To_Fire_Points"])

# PCA - With ID
pca_1 = PCA(n_components=4)
pca_cols_1 = ["PCA_1", "PCA_2", "PCA_3", "PCA_4"]
df_test.loc[:, pca_cols_1] = pca_1.fit_transform(df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
X_train_synth.loc[:, pca_cols_1] = pca_1.transform(X_train_synth.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
# Dropping PCA_1 (virtually the same as the Id column)
df_test.drop(columns='PCA_1', inplace=True) 
X_train_synth.drop(columns='PCA_1', inplace=True)


### 4.CLASSIFYING

# clf = RandomForestClassifier(n_estimators=150, n_jobs=-1)
# clf = ExtraTreesClassifier(n_estimators=300, max_features=None, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
clf = ExtraTreesClassifier(n_estimators=150, n_jobs=-1, max_features=None)

clf.fit(X_train_synth, y_train_synth)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)

# Having it fit the desired format
print(f"Current best: {accuracy_score(predict_best, predict_true)}")
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False) 