from vae.vae_model import *

train_df = pd.read_csv('test-cleaned.csv')
train_df = train_df.sample(frac=1).reset_index(drop=True)
X_train = train_df.to_numpy()

train_tensor = torch.tensor(X_train)

verbose = True
epochs = 80
lr = 1e-5
latent_dims = 40

vae = VariationalAutoencoder(latent_dims=latent_dims, 
                             input_dims=len(X_train[0]), output_dims=len(X_train[0]),
                             verbose=verbose)

optimizer = torch.optim.Adam(vae.parameters(), lr=lr) #, weight_decay=1e-3)

# Train
# ----------------------------------------------------------
for epoch in range(epochs):
    train_loss = train_vae(vae,train_tensor, train_tensor, optimizer)
    torch.cuda.empty_cache()
    if epoch % 1 == 0:
        print('EPOCH {}/{} \t train loss {:.3f}'.format(epoch + 1, epochs,train_loss))

# SAVE MODEL
torch.save(vae.state_dict(), 'models/vae_model.pth')