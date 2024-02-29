from vae_model import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

train_df = pd.read_csv('train-cleaned.csv')
X_test = train_df[train_df.columns.difference(['Cover_Type'])].to_numpy()
y_test = train_df['Cover_Type'].to_numpy()
X_train = pd.read_csv('test-cleaned.csv').to_numpy()
train_tensor = torch.tensor(X_test)

latent_dims = 3

vae = VariationalAutoencoder(latent_dims=latent_dims, input_dims=len(X_train[0]), output_dims=len(X_train[0]), verbose=False)
vae.load_state_dict(torch.load('models/vae_model.pth'))

encoder = vae.encoder
encoder.eval()

latent_pred = encoder(train_tensor.float()).detach().numpy()

# tsne = TSNE(n_jobs=4, n_components=3)
# tsne.fit(X_train)
# results = tsne(X_test)

# Customize the plot
font_size = 14
font_weight = 'bold'
font_style = 'sans-serif'
plt.rc('font', size=12, family=font_style)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Assigning random color to each label
labels = list(train_df['Cover_Type'].unique())
color_dict = {label:list(mcolors.TABLEAU_COLORS.keys())[i] for i, label in enumerate(labels)}
colors = [color_dict[value] for value in list(y_test)]

ax.scatter(latent_pred[:, 0],latent_pred[:, 1],latent_pred[:, 2], c=colors, marker='o')

ax.set_xlabel('Component 1', fontsize=font_size, fontweight=font_weight)
ax.set_ylabel('Component 2', fontsize=font_size, fontweight=font_weight)
ax.set_zlabel('Component 3', fontsize=font_size, fontweight=font_weight)
ax.set_title('3D Latent Space Visualized', fontsize=18, fontweight=font_weight)
ax.view_init(elev=-20, azim=10)

# Display the plot
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in color_dict.items()]
plt.legend(handles=legend_handles, fontsize=14)
plt.show()