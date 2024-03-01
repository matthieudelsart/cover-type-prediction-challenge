import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import sys
sys.path.insert(1, '../')

from utils import unonehot

test = pd.read_csv("../test-full.csv")
train = pd.read_csv("../train.csv")

test = unonehot(test)
train = unonehot(train)

truth = pd.read_parquet('../check_score/data/ground_truth.parquet')
y_test = truth['Cover_Type'].to_numpy()

y_pred_previous = pd.read_csv('preds.csv')

pred_full = test.copy()
pred_full['Cover_Type'] = y_pred_previous['Cover_Type']

# Remove train rows from prediction
merged = pred_full.merge(train[['Id']], on='Id', how='left', indicator=True)
pred_full = merged[merged['_merge'] == 'left_only'].drop(columns='_merge').copy()

# Select data for augmentation
class_weights = {2: 0.4604913495762566, 1: 0.3627567072625006, 3: 0.06331366649914287,
                 7: 0.04459116162833126, 6: 0.03524023600201029, 5: 0.02767584834736632,
                 4: 0.005931030684392061}
class_samples = {i:int(weight*168_605) for (i, weight) in class_weights.items()}

# Select with a ratio according to class prevalence
selected_data = pd.concat([pred_full[pred_full['Cover_Type'] == cov_type].sample(n=class_samples[cov_type], replace=False) for cov_type in class_samples])
selected_data.reset_index(drop=True, inplace=True)

train_aug = pd.concat([train, selected_data], axis=0)
X_train = train_aug.iloc[:, :-1].to_numpy()
y_train = train_aug['Cover_Type'].to_numpy()
X_test = test.to_numpy()

clf = LGBMClassifier(
    objective='multiclass',
    num_class=7,
    class_weight=class_weights,
    verbose=1,
    n_jobs=-1
)

clf.fit(X_train, y_train, categorical_feature=['Wilderness_Area_Synth', 'Soil_Type_Synth'], feature_name=list(test.columns))
y_pred = clf.predict(X_test)

accuracy = (y_pred == y_test).mean()

print("Accuracy:", accuracy)

