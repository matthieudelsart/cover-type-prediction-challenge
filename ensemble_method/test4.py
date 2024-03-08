import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(1, '../')

from utils import unonehot

test = pd.read_csv("../test-full.csv").drop(['Soil_Type7', 'Soil_Type15'], axis=1)
train = pd.read_csv("../train.csv").drop(['Soil_Type7', 'Soil_Type15'], axis=1)

test = unonehot(test)
train = unonehot(train)

truth = pd.read_parquet('../check_score/data/ground_truth.parquet')
y_test = truth['Cover_Type'].to_numpy()

X_test_pca = test.drop(['Id'], axis=1).to_numpy()
X_train_pca = train.drop(['Id'], axis=1).iloc[:, :-1].to_numpy()
y_train_pca = train['Cover_Type'].to_numpy()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

n = 3
dim_red = PCA(n_components=7)
#dim_red = TSNE(n_components=n)
dim_red.fit(X_test_pca)

train_results = dim_red.transform(X_train_pca)
test_results = dim_red.transform(X_test_pca)

# Customize the plot
font_size = 14
font_weight = 'bold'
font_style = 'sans-serif'
plt.rc('font', size=12, family=font_style)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Assigning random color to each label
labels = list(train['Cover_Type'].unique())
color_dict = {label:list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in enumerate(labels)}
colors = [color_dict[value] for value in list(y_train_pca)]

ax.scatter(train_results[:, 0], train_results[:, 1], train_results[:, 2], c=colors, marker='o')

ax.set_xlabel('Component 1', fontsize=font_size, fontweight=font_weight)
ax.set_ylabel('Component 2', fontsize=font_size, fontweight=font_weight)
ax.set_zlabel('Component 3', fontsize=font_size, fontweight=font_weight)
ax.set_title('3D Latent Space Visualized', fontsize=18, fontweight=font_weight)
ax.view_init(elev=-20, azim=-30)

# Display the plot
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
plt.legend(handles=legend_handles, fontsize=14)
plt.savefig('3DUMAP.png', dpi=200)

# Clusterize using HDBSCAN
# hdb = HDBSCAN(min_cluster_size=20)
# hdb.fit(test_results)



# Make predictions
train_pca_df = pd.DataFrame(train_results[:, :n], columns=[f'PCA{i+1}' for i in range(n)])
test_pca_df = pd.DataFrame(test_results[:, :n], columns=[f'PCA{i+1}' for i in range(n)])

train_pca = pd.concat([train, train_pca_df], axis=1)
test_pca = pd.concat([test, test_pca_df], axis=1)

train_pca = train_pca[list(test_pca.columns) + ['Cover_Type']]

cats = ['Wilderness_Area_Synth', 'Soil_Type_Synth']
train_pca[cats] = train_pca[cats].astype('category')
test_pca[cats] = test_pca[cats].astype('category')

X_test = test_pca.to_numpy()
X_train = train_pca.iloc[:, :-1].to_numpy()
y_train = train_pca['Cover_Type'].to_numpy()

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

id_df = test['Id']
preds_df = pd.DataFrame(y_pred, columns=['Cover_Type'])

preds_df = pd.concat([id_df, preds_df], axis=1)
preds_df.to_csv('preds.csv', index=False)

# # Compare the predicted values with the ground truth
accuracy = (y_pred == y_test).mean()

print("Accuracy:", accuracy)


cover_type_counts = preds_df['Cover_Type'].value_counts()
total_predictions = cover_type_counts.sum()
cover_type_ratios = cover_type_counts / total_predictions
cover_type_ratio_dict = cover_type_ratios.to_dict()
print(cover_type_ratio_dict)
