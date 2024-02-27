import argparse
import os
import random
import pandas as pd

import numpy as np

from iop_vae import GaussianVAEIOP

test_df = pd.read_csv('../train-cleaned.csv')
X_test = test_df[test_df.columns.difference(['Cover_Type'])].to_numpy()
y_test = test_df['Cover_Type'].to_numpy()
X_train = pd.read_csv('../test-full.csv').to_numpy()

prior = "iop"
learning_rate = 1e-5
seed = 42
dataset = "Forest_Cover"

dynamic_binarization = False

print(f"Dataset: {dataset} / Prior: {prior}")
print(f"Dynamic Binarization: {dynamic_binarization}/ Learning Rate: {learning_rate} / Seed: {seed}")

# Set seed
random.seed(seed)
np.random.seed(seed)

# Get dataset
if dynamic_binarization:
    # for binary data
    X_valid = np.random.binomial(1, X_valid)
    X_test = np.random.binomial(1, X_test)

# Train
save_dir = f"save/{dataset}/"
save_path = f"save/{dataset}/model_{learning_rate}_{seed}"
os.makedirs(save_dir, exist_ok=True)

if prior == "normal":
    model = GaussianVAE(n_in=X_train.shape[1], n_latent=40, n_h=500)
else:
    model = GaussianVAEIOP(n_in=X_train.shape[1], n_latent=40, n_h=500)

print(f"Model: {type(model)}")

if prior == "normal":
    model.fit(X_train, k=1, batch_size=100,
                learning_rate=learning_rate, n_epoch=1000,
                dynamic_binarization=dynamic_binarization,
                warm_up=True, warm_up_epoch=100, is_stoppable=True,
                X_valid=X_valid, path=save_path)
else:
    model.fit(X_train, k=1, batch_size=100,
                learning_rate_primal=learning_rate, learning_rate_dual=1e-3,
                n_epoch_primal=1000, n_epoch_dual=10,
                dynamic_binarization=dynamic_binarization,
                warm_up=True, warm_up_epoch=100, is_stoppable=True,
                X_valid=X_test, path=save_path)

# Test
test_score = model.importance_sampling(X_test, k=10)
print("Test Score: ", np.mean(test_score))

# Save numpy files
os.makedirs("npy/", exist_ok=True)
np.save(f"npy/exp_{dataset}_{prior}_train_loss_{learning_rate}_{seed}.npy", np.array(model.train_losses))
np.save(f"npy/exp_{dataset}_{prior}_train_time_{learning_rate}_{seed}.npy", np.array(model.train_times))
np.save(f"npy/exp_{dataset}_{prior}_valid_loss_{learning_rate}_{seed}.npy", np.array(model.valid_losses))
np.save(f"npy/exp_{dataset}_{prior}_RE_{learning_rate}_{seed}.npy", np.array(model.reconstruction_errors))
np.save(f"npy/exp_{dataset}_{prior}_KL_{learning_rate}_{seed}.npy", np.array(model.kl_divergences))
np.save(f"npy/exp_{dataset}_{prior}_test_score_{learning_rate}_{seed}.npy", test_score)