# dependencies
import numpy as np
import pandas as pd
from datetime import datetime

# torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_

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

def loss_function(recon_x, x, mu, logvar, loss_type='mse'):
    """
    Computes the loss function for the VAE.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
        loss_type (str): The type of reconstruction loss to use ('MSE' or 'MAE').
    
    Returns:
        torch.Tensor: The computed loss.
    """
    
    if loss_type == 'mse':
        # Mean Squared Error loss
        recon_loss = F.mse_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    elif loss_type == 'mae':
        # Mean Absolute Error loss
        recon_loss = F.l1_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    else:
        raise ValueError("Invalid loss_type. Expected 'mse' or 'mae'.")
    
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Normalize by same number of elements as in reconstruction
    return (recon_loss + KLD) / x.size(0)

# Define a function to train the VAE model
def train_vae(model, train_loader, optimizer, device, loss_type="mse"):
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
        loss = loss_function(recon_batch, data, mu, logvar, loss_type=loss_type)
        
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
