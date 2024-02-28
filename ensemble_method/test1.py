import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from xgboost import XGBClassifier

# Read the CSV file
train = pd.read_csv("../train.csv")

# Convert columns to categorical
cat_cols = [col for col in train.columns if 'Wilderness_Area' in col or 'Soil_Type' in col or col == 'Cover_Type']
train[cat_cols] = train[cat_cols].astype('category')

# Remove unnecessary columns
train.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'], inplace=True)

# # Replace Cover_Type values
# cover_type_map = {1: 'Seg1', 2: 'Seg2', 3: 'Seg3', 4: 'Seg4', 5: 'Seg5', 6: 'Seg6', 7: 'Seg7'}
# train['Cover_Type'] = train['Cover_Type'].map(cover_type_map)

# Split data into features (X) and target (y)
X = train.drop(columns='Cover_Type')
y = train['Cover_Type']

# GBM model
params = {
    'n_estimators': [1000],
    'max_depth': [22],
    'learning_rate': [0.2]
}
gbm = GradientBoostingClassifier()
gbm_grid = GridSearchCV(gbm, params, scoring='roc_auc_ovr_weighted', cv=10, verbose=1, n_jobs=-1, return_train_score=False)
gbm_grid.fit(X, y)
pred1 = gbm_grid.predict(X)
print(confusion_matrix(y, pred1))

# RF model
rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, {'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}, scoring='roc_auc_ovr_weighted', cv=10, verbose=1, n_jobs=-1)
rf_grid.fit(X, y)
pred2 = rf_grid.predict(X)
print(confusion_matrix(y, pred2))

# NB model
nb = GaussianNB()
nb.fit(X, y)
pred3 = nb.predict(X)
print(confusion_matrix(y, pred3))

# Ensemble model
ensemble = VotingClassifier(estimators=[('gbm', gbm_grid), ('rf', rf_grid), ('nb', nb)], voting='soft')
ensemble.fit(X, y)
ensemble_pred = ensemble.predict(X)
print(confusion_matrix(y, ensemble_pred))