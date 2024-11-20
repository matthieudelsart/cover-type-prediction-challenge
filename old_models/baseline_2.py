from utils import clean_predictor
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Reading
df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")
predict_true = pd.read_parquet("ground_truth.parquet")["Cover_Type"]

# Un-one-hot-encoding the categorical variables
soil_types = [f"Soil_Type{i}" for i in range(1, 41)]
wilderness_areas = [f"Wilderness_Area{i}" for i in range(1,5)]
df_test["Wilderness_Area_Synth"] = df_test[wilderness_areas] @ range(1,5)
df_train["Wilderness_Area_Synth"] = df_train[wilderness_areas] @ range(1,5)
df_test["Soil_Type_Synth"] = df_test[soil_types] @ range(1,41)
df_train["Soil_Type_Synth"] = df_train[soil_types] @ range(1,41)
df_train = df_train.drop(columns=wilderness_areas + soil_types)
df_test = df_test.drop(columns=wilderness_areas + soil_types)

# Separating features and target
X_train = df_train.drop(columns=['Cover_Type'], axis=1)
y_train = df_train['Cover_Type']

# Scaling
std = StandardScaler()
df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"] = std.fit_transform(df_test.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])
X_train.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"] = std.transform(X_train.loc[:, "Id": "Horizontal_Distance_To_Fire_Points"])

# Setting class weights
coeffs = np.array([2.63, 3.06, 0.43, 0.05, 0.24, 0.27, 0.32])
sample_weight = pd.Series([coeffs[i - 1] for i in df_train["Cover_Type"]])


# Setting classifier
clf = LGBMClassifier(
    objective='multiclass',
    num_class=7,
    verbose=0,
    n_jobs=-1
)

# Predicting
clf.fit(X_train, y_train, categorical_feature=['Wilderness_Area_Synth', 'Soil_Type_Synth'],
        sample_weight=sample_weight)
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)
print(f"Score: {accuracy_score(predictions_df['Cover_Type'], predict_true)}")
predictions_df.to_csv('test_predictions.csv', index=False) 