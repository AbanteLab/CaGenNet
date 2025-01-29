#%%
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

####################################################################################################
# functions
####################################################################################################

# Define a function to print messages with timestamps
def superprint(message):
    
    # Get the current date and time
    now = datetime.now()
    
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Print the message with the timestamp
    print(f"{timestamp} {message}")

# Define a dataset class for the calcium imaging data
class CalciumDataset(Dataset):
    def __init__(self, data):
        """
        Initializes the dataset with the given data.
        
        Args:
            data (numpy.ndarray): The data to be used in the dataset.
        """
        
        # check if it's a tensor or numpy array
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.from_numpy(data).float()

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            tuple: The sample and a placeholder value (0).
        """
        return self.data[idx], 0  # The second value is a placeholder

def load_data(file_path):
    """
    Loads data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        numpy.ndarray: The loaded data.
    """
    
    # Check if the file has 'csv' or 'csv.gz' in the suffix and use ',' as separator in that case
    if file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
        superprint("Loading data from CSV file")
        data = pd.read_csv(file_path, header=None, sep=',')
    else:
        superprint("Loading data from TSV file")
        data = pd.read_csv(file_path, header=None, sep='\t')
    
    # convert to float 32
    data = data.astype(np.float32)
    
    # Normalize the data
    data = (data - data.min()) / (data.max() - data.min()) - 0.5

    # return the data values
    return data.values

# Define the VAE model
class VAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers
        self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_l2 = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def encode(self, x):
         
        # get output of lstm
        _,(hd,_) = self.encoder_lstm(x.unsqueeze(1))
        
        # remove unnecessary dimension
        hd = hd.squeeze(0)
        
        # get parameters q(z|x)
        mu = self.encoder_fc_mu(hd)
        logvar = self.encoder_fc_logvar(hd)
        
        return mu, logvar
    
    def sample_z(self, mu, logvar):
        
        # reparametrize
        std = torch.exp(0.5 * logvar)
        
        # sample eps~(0,1)
        eps = torch.randn_like(std)

        # return z
        return mu + eps * std
    
    def decode(self, z):
        
        # get linear layer output
        out = self.decoder_l1(z)
        
        # get lstm output (batch_size,1,)
        out, _ = self.decoder_l2(out.unsqueeze(1))
        
        return out.squeeze(1)
    
    def forward(self, x):
        
        # Encode
        mu, logvar = self.encode(x)
        
        # Reparameterize
        z = self.sample_z(mu, logvar)
        
        # Decode
        reconstructed_x = self.decode(z)
        
        return reconstructed_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Computes the loss function for the VAE.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
    
    Returns:
        torch.Tensor: The computed loss.
    """
    
    # Assuming recon_x and x are your reconstructed and original tensors
    MSE = F.mse_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalize by same number of elements as in reconstruction
    return (MSE + KLD) / x.size(0)

# Define a function to train the VAE model
def train_vae(model, train_loader, optimizer, device):
    """
    Trains the VAE model on the given data.

    Args:
        model (VAE): The VAE model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training.

    Returns:
        float: The average training loss.
    """
    
    # Set the model to training mode
    model.train()
    
    # Initialize the training loss
    train_loss = 0
    
    # Iterate over the training data
    for batch_idx, (data, _) in enumerate(train_loader):
        
        # Move the data to the device
        data = data.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform a forward pass
        recon_batch, mu, logvar = model(data)
        
        # Compute the loss
        loss = loss_function(recon_batch, data, mu, logvar)
        
        # Perform a backward pass
        loss.backward()
        
        # Clip the gradients
        clip_grad_norm_(model.parameters(), 1)
        
        # Step the optimizer
        optimizer.step()
        
        # Add the loss to the total loss
        train_loss += loss.item()
        
    # Return the average training loss
    return train_loss / len(train_loader.dataset)

####################################################################################################
# main
####################################################################################################
#%%

# Define the main function
def main():
    
    # argument parser options
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE) on calcium imaging data.")
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('-l', '--latent', type=int, default=4, help='Dimension of latent space (default: 4)')
    parser.add_argument('--hidden', type=int, default=64, help='Dimension of hidden layer (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='RNG seed (default: 0)')
    parser.add_argument('-r', '--rate', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--beta_kl', type=float, default=1, help='KL divergence beta (default: 1)')
    parser.add_argument('--retrain', type=bool, default=False, help='Whether to retrain the model (default: False)')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the model (default: True)')
    parser.add_argument('--outdir', type=str, default='sndgm', help='Directory to save models, embeddings, and losses (default: sndgm)')
    
    # Parse arguments
    args = parser.parse_args()

    #%%
    
    # Access the arguments
    data_path = args.data_path  # data_path='/tmp/test.txt.gz'
    seed = args.seed            # seed=0
    beta = args.beta_kl         # beta=1
    retrain = args.retrain      # retrain=True
    save = args.save            # save=False
    batch = args.batch          # batch=32
    latent = args.latent        # latent=4
    hidden = args.hidden        # hidden=64
    epochs = args.epochs        # epochs=200
    rate = args.rate            # rate=0.005
    outdir = args.outdir        # outdir='/tmp'
    
    # Load and preprocess data
    data = load_data(data_path)
    
    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=seed)
    
    # Create data loaders
    train_dataset = CalciumDataset(train_data)
    val_dataset = CalciumDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        superprint("Inference done on GPU")
    else:
        superprint("Inference done on CPU")
        
    # Initialize model, optimizer, and loss function
    model = VAE(input_dim=train_data.shape[1], latent_dim=latent, hidden_dim=hidden).to(device)
    
    # Define the optimizer
    superprint("Setting up optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Early stopping parameters
    counter = 0
    patience = 10
    min_delta = 0.001
    best_val_loss = float('inf')

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        
        # Training
        train_loss = train_vae(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon_batch, mu, logvar = model(x)
                val_loss += loss_function(recon_batch, x, mu, logvar).item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                superprint(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Step the scheduler at the end of the epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        superprint(f"Epoch: {epoch+1}/{epochs},  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Lr: {current_lr:.4f}")

    # Save model
    if save:
        superprint("Saving model")
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(outdir, "vae.pth"))

    # Save embeddings
    superprint("Saving embeddings")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for x, _ in DataLoader(CalciumDataset(data), batch_size=batch, shuffle=False):
            x = x.to(device)
            mu, _ = model.encode(x)
            embeddings.append(mu.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    
    # Save embeddings to a txt.gz file
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(os.path.join(outdir, "embeddings.txt.gz"), sep='\t', index=False, header=False)

    # Save training and validation losses
    superprint("Saving training and validation losses")
    losses_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    losses_df.to_csv(os.path.join(outdir, "losses.txt.gz"), sep='\t', index=False)
    
    # save hyperparameters of the model
    superprint("Saving hyperparameters")
    hyperparameters = {'seed': seed, 'beta': beta, 'retrain': retrain, 'save': save, 'batch': batch, 'latent': latent, 'hidden': hidden, 'epochs': epochs, 'rate': rate}
    hyperparameters_df = pd.DataFrame(hyperparameters, index=[0])
    hyperparameters_df.to_csv(os.path.join(outdir, "hyperparameters.txt.gz"), sep='\t', index=False)
    
    # print final message
    superprint("Training completed")
    
if __name__ == "__main__":
    main()
# %%
