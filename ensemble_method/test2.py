import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

test = pd.read_csv('../test-full.csv')
train = pd.read_csv('../train.csv')

truth = pd.read_parquet('../covertype/ground_truth.parquet')
y_test = truth['Cover_Type'].to_numpy() - 1

X_test_pca = test.to_numpy()
X_train_pca = train.iloc[:, :-1].to_numpy()
y_train_pca = train['Cover_Type'].to_numpy()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

n = 3
dim_red = PCA(n_components=3)
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

# ax.scatter(train_results[:, 0], train_results[:, 1], train_results[:, 2], c=colors, marker='o')

ax.set_xlabel('Component 1', fontsize=font_size, fontweight=font_weight)
ax.set_ylabel('Component 2', fontsize=font_size, fontweight=font_weight)
ax.set_zlabel('Component 3', fontsize=font_size, fontweight=font_weight)
ax.set_title('3D Latent Space Visualized', fontsize=18, fontweight=font_weight)
ax.view_init(elev=20, azim=30)

# Display the plot
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
plt.legend(handles=legend_handles, fontsize=14)
plt.savefig('3DUMAP.png', dpi=200)

# Make predictions
train_pca_df = pd.DataFrame(train_results[:, :n], columns=[f'PCA{i+1}' for i in range(n)])
test_pca_df = pd.DataFrame(test_results[:, :n], columns=[f'PCA{i+1}' for i in range(n)])

train_pca = pd.concat([train, train_pca_df], axis=1)
test_pca = pd.concat([test, test_pca_df], axis=1)

train_pca = train_pca[list(test_pca.columns) + ['Cover_Type']]

X_test = test_pca.to_numpy()
X_train = train_pca.iloc[:, :-1].to_numpy()
y_train = train_pca['Cover_Type'].to_numpy() - 1

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

#y_pred = clf.predict(dim_red.transform(X_test))

# Compare the predicted values with the ground truth
accuracy = (y_pred == y_test).mean()

print("Accuracy:", accuracy)
