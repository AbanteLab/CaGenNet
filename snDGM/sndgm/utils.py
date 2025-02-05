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

def load_data(file_path):
    """
    Loads data from a CSV or TSV file.
    
    Args:
        file_path (str): The path to the file.
    
    Returns:
        pandas.DataFrame: The loaded data.
    """
    
    # Check if the file has 'csv' or 'csv.gz' in the suffix and use ',' as separator in that case
    if file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
        superprint("Loading data from CSV file")
        data = pd.read_csv(file_path, header=None, sep=',')
    else:
        superprint("Loading data from TSV file")
        data = pd.read_csv(file_path, header=None, sep='\t')
    
    # Convert to float32
    data = data.astype(np.float32)
    
    return data.values

def augment_data(data, N, L, seed=0):
    """
    Augments the input data by randomly choosing N different consecutive segments of length L for each row in the data.
    
    Args:
        data (numpy.ndarray): The original data.
        N (int): The number of segments to choose.
        L (int): The length of each segment.
    
    Returns:
        numpy.ndarray: The augmented data.
        numpy.ndarray: The indices of the chosen segments in the original data.
    """
    
    # Initialize lists to store the indices and augmented data
    indices = []
    augmented_data = []
    
    # Iterate over the rows in the data
    for row_idx, row in enumerate(data):
        
        # Initialize lists to store the indices and augmented data for the current row
        row_indices = []
        row_augmented_data = []
        
        # Iterate N times to choose N segments
        for _ in range(N):
            
            # Randomly choose a starting index
            start_idx = np.random.randint(0, len(row) - L)
            
            # Extract the segment
            segment = row[start_idx:start_idx + L]
            
            # Append the segment and the starting index
            row_augmented_data.append(segment)
            row_indices.append(row_idx)
        
        augmented_data.append(row_augmented_data)
        indices.append(row_indices)
    
    # Convert the lists to numpy arrays
    indices = np.array(indices)
    augmented_data = np.array(augmented_data)
    
    return augmented_data,indices

def normalize_data(data):
    """
    Normalizes the data.
    
    Args:
        data (pandas.DataFrame): The data to normalize.
    
    Returns:
        numpy.ndarray: The normalized data.
    """
    
    # Normalize the data
    normalized_data = (data - data.min().min()) / (data.max().max() - data.min().min()) - 0.5
    
    return normalized_data

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

def reconstruction_loss(recon_x, x, loss_type='mse'):
    """
    Computes the reconstruction loss for the VAE.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        loss_type (str): The type of reconstruction loss to use ('mse' or 'mae').
    
    Returns:
        torch.Tensor: The computed reconstruction loss.
    """
    
    if loss_type == 'mse':
        # Mean Squared Error loss
        return F.mse_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    elif loss_type == 'mae':
        # Mean Absolute Error loss
        return F.l1_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    else:
        raise ValueError("Invalid loss_type. Expected 'mse' or 'mae'.")

def kl_divergence_loss(mu, logvar):
    """
    Computes the KL divergence loss for the VAE.
    
    Args:
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
    
    Returns:
        torch.Tensor: The computed KL divergence loss.
    """
    
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def positive_pairwise_loss(mu, roi, metric='cosine'):
    """
    Encourages different views of the same ROI to have similar embeddings.

    Args:
        mu (torch.Tensor): The mean of the latent space (batch_size, latent_dim).
        roi (torch.Tensor): The region of interest identifiers (batch_size,).
        metric (str): The metric to use for computing the pairwise loss ('cosine' or 'euclidean').
        
    Returns:
        torch.Tensor: The computed penalty term.
    """

    # Initialize total loss
    total_loss = 0.0

    # Get unique ROIs
    unique_rois = roi.unique()
    
    # Normalize embeddings
    mu = F.normalize(mu, dim=-1)

    # Iterate over unique ROIs
    for unique_roi in unique_rois:
        
        # Get indices of same ROI
        indices = (roi == unique_roi).nonzero(as_tuple=True)[0]
        
        # If there are multiple indices
        if len(indices) > 1:
            
            # Extract embeddings for this ROI
            mu_roi = mu[indices]  
            
            if metric == 'cosine':
                
                # Compute pairwise cosine similarity
                similarity_matrix = torch.matmul(mu_roi, mu_roi.T)
                
                # Compute loss: minimize distance from perfect similarity (1.0)
                total_loss += (1 - similarity_matrix).mean()
                
            elif metric == 'euclidean':
                
                # Compute pairwise euclidean distance
                distance_matrix = torch.cdist(mu_roi, mu_roi, p=2)
                
                # Compute loss: minimize the distance
                total_loss += distance_matrix.mean()
                
            else:
                
                raise ValueError("Invalid metric. Expected 'cosine' or 'euclidean'.")

    # Return total loss
    return total_loss

def loss_function(recon_x, x, mu, logvar, roi, loss_type='mse', model_type='clavae', bkl=1.0, bpp=1.0, metric='cosine'):
    """
    Computes the total loss function for the VAE, including a penalty term for the distance in z within fluo signals from the same roi.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
        roi (torch.Tensor): The region of interest identifiers.
        loss_type (str): The type of reconstruction loss to use ('mse' or 'mae').
        bkl (float): The weight for the KL divergence loss.
        bpp (float): The weight for the InfoNCE loss.
    
    Returns:
        torch.Tensor: The computed total loss.
    """
    
    # Compute the reconstruction loss
    recon_loss = reconstruction_loss(recon_x, x, loss_type) / x.size(0)
    
    # Compute the KL divergence loss
    kl_loss = bkl * kl_divergence_loss(mu, logvar) / x.size(0)
    
    # Initialize the InfoNCE loss to 0
    pp_loss = 0
    
    # Compute the ROI loss if using the CLAVAE model
    if model_type == 'clavae':
        pp_loss = bpp * 100 * positive_pairwise_loss(mu, roi, metric=metric)  / x.size(0)
    
    # Compute the total loss
    total_loss = recon_loss + kl_loss + pp_loss
    
    return total_loss, recon_loss, kl_loss, pp_loss

# Define a function to train the VAE model
def train_clavae(model, train_loader, optimizer, device, loss_type="mse", bkl=1.0, bpp=1.0, metric='cosine', model_type='clavae'):
    """
    Trains the CLAVAE model using contrastive learning on the given data.

    Args:
        model (CLAVAE): The contrastive learning autoregressive VAE model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        device (torch.device): The device to use for training.

    Returns:
        float: The average training loss.
    """
    
    # Store optional arguments in a dictionary
    loss_args = {
        'loss_type': loss_type,
        'bkl': bkl,
        'bpp': bpp,
        'metric': metric,
        'loss_type': loss_type,
        'model_type': model_type
    }
        
    # Set the model to training mode
    model.train()
    
    # Initialize the training loss
    train_loss = 0
    
    # Initialize variables to store individual losses
    total_recon_loss = 0
    total_kl_loss = 0
    total_pp_loss = 0

    # Iterate over the training data
    for batch_idx, (fluo, roi) in enumerate(train_loader):
        
        # Move the data to the device
        fluo = fluo.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform a forward pass
        recon_batch, mu, logvar = model(fluo)
        
        # Compute the loss
        tot_loss, recon_loss, kl_loss, pp_loss = loss_function(recon_batch, fluo, mu, logvar, roi, **loss_args)
        
        # Perform a backward pass
        tot_loss.backward()
        
        # Clip the gradients
        clip_grad_norm_(model.parameters(), 1)
        
        # Step the optimizer
        optimizer.step()
        
        # Add the losses to the total losses
        train_loss += tot_loss.item()
        total_kl_loss += kl_loss.item()
        total_pp_loss += pp_loss.item()
        total_recon_loss += recon_loss.item()
    
    # Calculate average losses
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_kl_loss = total_kl_loss / len(train_loader.dataset)
    avg_pp_loss = total_pp_loss / len(train_loader.dataset)
    
    # Print average losses
    superprint(f"L_rec: {avg_recon_loss:.6f} | L_kl: {avg_kl_loss:.6f} | L_pp: {avg_pp_loss:.6f}")
        
    # Return the average training loss
    return train_loss / len(train_loader.dataset)
