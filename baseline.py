from utils import clean_predictor
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

# Reading
df_test = pd.read_csv("test-full.csv")
df_train = pd.read_csv("train.csv")

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

# Setting class weights
class_weights = {
    1: 0.4,
    2: 0.45,
    3: 0.04,
    4: 0.01,
    5: 0.04,
    6: 0.04,
    7: 0.04
} 

# Setting classifier
clf = LGBMClassifier(
    objective='multiclass',
    num_class=7,
    class_weight=class_weights,
    verbose=0,
    n_jobs=-1
)

# Predicting
clf.fit(X_train, y_train, categorical_feature=['Wilderness_Area_Synth', 'Soil_Type_Synth'])
y_pred = clf.predict(df_test)
predictions_df = clean_predictor(y_pred)
predictions_df.to_csv('test_predictions.csv', index=False) 