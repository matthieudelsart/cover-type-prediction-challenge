import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

import ot

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, input_dims):  
        super(VariationalEncoder, self).__init__()

        self.latent_dims = min(latent_dims, 50)

        self.fc1 = nn.Linear(input_dims, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, self.latent_dims)
        self.fc7 = nn.Linear(64, self.latent_dims)

        self.N = torch.distributions.Normal(0, 1) # Try a prior which is a mixture of gaussians?
        self.kl = 0

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        x = F.tanh(self.fc5(x))
        mu = self.fc6(x)
        sigma = torch.exp(self.fc7(x))
        N = self.N.sample(mu.shape)
        z = mu + sigma * N
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class VariationalDecoder(nn.Module):
    def __init__(self, latent_dims, output_dims):
        super(VariationalDecoder, self).__init__()

        self.latent_dims = min(latent_dims, 50)
        
        self.fc1 = nn.Linear(latent_dims, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_dims)
        
    def forward(self, z):
        z = F.tanh(self.fc1(z))
        z = F.tanh(self.fc2(z))
        z = F.tanh(self.fc3(z))
        z = F.tanh(self.fc4(z))
        z = F.tanh(self.fc5(z))
        z = self.fc6(z)
        return z
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, input_dims, output_dims, verbose):
        super(VariationalAutoencoder, self).__init__()
        self.verbose = verbose
        self.encoder = VariationalEncoder(latent_dims, input_dims)
        self.decoder = VariationalDecoder(latent_dims, output_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
### Training function
def train_vae(vae, X_train_input, X_train_output, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    batch = 25
    train_loss = 0.0
    verbose = vae.verbose
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for i in range(0, len(X_train_input), batch):
        batch_X_input = X_train_input[i:i+batch].float()
        batch_X_output = X_train_output[i:i+batch].float()
        
        x_hat = vae(batch_X_input)

        # Evaluate loss
        loss = ((batch_X_output - x_hat)**2).sum() + vae.encoder.kl
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10000 == 0 and verbose:
            # Print batch loss
            print('[%i] \t partial train loss (single batch): %f' % (i, loss.item()))

        train_loss += loss.item()

    return train_loss / len(X_train_input)


### Testing function
def test_vae(vae, X_test):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0

    with torch.no_grad(): # No need to track the gradients
        for i in range(len(X_test)):
            x = X_test[i].clone().detach().float()
            
            # Decode data
            x_test = x.unsqueeze(0)
            x_hat = vae(x_test)

            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(X_test)

# # GENERATION
# # Import noise array
# noise = np.load('data/noise.npy')[:, :latent_dims]
# indx_range = np.arange(0, len(noise))
# indx_selected = np.random.choice(indx_range, size=1000, replace=False)
# noise = torch.from_numpy(noise[indx_selected])

# # Load the model
# generator = vae.decoder
# generator.eval()

# # Generate the distribution
# yields_gen_tensor = generator(noise)
# yields_gen_numpy = yields_gen_tensor.detach().numpy()

# yields_gen_df = pd.DataFrame(yields_gen_numpy, columns=["YIELD_1", "YIELD_2", "YIELD_3", "YIELD_4"])

# print(ot.sliced.sliced_wasserstein_distance(yields_df.to_numpy(), yields_gen_df.to_numpy(), seed=0))

# # Save the DataFrame to a CSV file
# yields_gen_df.to_csv('CSVs/vae_yields_subset.csv', index=False)