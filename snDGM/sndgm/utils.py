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
import time

# Define the VAE model
class CLAVAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(CLAVAE, self).__init__()
        
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

# Define the ISOVAE model with separate encoders and decoders for amplitude and phase
class ISOVAE(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ISOVAE, self).__init__()
        
        # dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder layers for amplitude
        self.encoder_amp_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_amp_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_amp_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_amp_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_amp_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Encoder layers for phase
        self.encoder_phase_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_phase_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_phase_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_phase_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_phase_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Compute the number of frequency components
        self.num_components = rfft_num_components(input_dim)
        
        # Decoder layers for amplitude
        self.decoder_amp_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_amp_fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_amp_fc3 = nn.Linear(2 * hidden_dim, self.num_components)
        
        # Decoder layers for amplitude mask
        self.decoder_mask_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_mask_fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_mask_fc3 = nn.Linear(2 * hidden_dim, self.num_components)
        
        # Decoder layers for phase
        self.decoder_phase_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_phase_fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_phase_fc3 = nn.Linear(2 * hidden_dim, self.num_components)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.8)
        
    def encode_amplitude(self, x):
        
        # get output of fully connected layers with ReLU activation
        h = torch.relu(self.encoder_amp_fc1(x))
        h = torch.relu(self.encoder_amp_fc2(h))
        h = self.encoder_amp_fc3(h)
        
        # get parameters q(z|x)
        mu = self.encoder_amp_fc_mu(h)
        logvar = self.encoder_amp_fc_logvar(h)
        
        return mu, logvar
    
    def encode_phase(self, x):
        
        # get output of fully connected layers with ReLU activation
        h = torch.relu(self.encoder_phase_fc1(x))
        h = torch.relu(self.encoder_phase_fc2(h))
        h = self.encoder_phase_fc3(h)
        
        # get parameters q(z|x)
        mu = self.encoder_phase_fc_mu(h)
        logvar = self.encoder_phase_fc_logvar(h)
        
        return mu, logvar
    
    def sample_z(self, mu, logvar):
        
        # reparametrize
        std = torch.exp(0.5 * logvar)
        
        # sample eps~(0,1)
        eps = torch.randn_like(std)

        # return z
        return mu + eps * std
    
    def decode_amplitude(self, z):
        
        # get output of fully connected layers with ReLU activation and dropout
        h = torch.relu(self.decoder_amp_fc1(z))
        h = self.dropout(h)
        h = torch.relu(self.decoder_amp_fc2(h))
        h = self.dropout(h)
        h = self.decoder_amp_fc3(h)
        
        return torch.sigmoid(h)

    def decode_mask(self, z):
        
        # get output of fully connected layers with ReLU activation
        h = torch.relu(self.decoder_mask_fc1(z))
        h = torch.relu(self.decoder_mask_fc2(h))
        h = self.decoder_mask_fc3(h)
        
        return torch.sigmoid(h)
    
    def decode_phase(self, z):
        
        # get output of fully connected layers with ReLU activation and dropout
        h = torch.relu(self.decoder_phase_fc1(z))
        h = self.dropout(h)
        h = torch.relu(self.decoder_phase_fc2(h))
        h = self.dropout(h)
        h = self.decoder_phase_fc3(h)
        
        return torch.sigmoid(h)
    
    def decode_signal(self, za, zp):
        
        # concatenate Za and Zp
        z = torch.cat((za, zp), dim=-1)
        
        # get output of fully connected layers with ReLU activation
        h1 = torch.relu(self.decoder_signal_fc1(z))
        h2 = torch.relu(self.decoder_signal_fc2(h1))
        
        # get final output
        out = self.decoder_signal_fc3(h2)
        
        return out
    
    def forward(self, x):
        
        # Encode amplitude
        mu_a, logvar_a = self.encode_amplitude(x)
        
        # Encode phase
        mu_p, logvar_p = self.encode_phase(x)
        
        # Reparameterize
        za = self.sample_z(mu_a, logvar_a)
        zp = self.sample_z(mu_p, logvar_p)
        
        # Decode amplitude
        mask = self.decode_mask(za)
        ahat = self.decode_amplitude(za)
        
        # Decode phase
        phat = self.decode_phase(zp)
        
        # Decode signal
        # xhat = self.decode_signal(za, zp)
        
        # return ahat, phat, mu_a, logvar_a, mu_p, logvar_p
        # return xhat, ahat, phat, mu_a, logvar_a, mu_p, logvar_p
        return mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p

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

def rfft_num_components(N):
    """
    Returns the number of frequency components for the real FFT (rFFT).

    Args:
        N (int): Number of timepoints in the signal.

    Returns:
        int: Number of rFFT components.
    """
    return (N // 2) + 1

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

# Define a function to print messages with timestamps
def superprint(message):
    
    # Get the current date and time
    now = datetime.now()
    
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    
    # Print the message with the timestamp
    print(f"{timestamp} {message}")

# Define a function to normalize the data
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

# Define a function to compute the reconstruction loss
def reconstruction_loss(xhat, x, rec_loss='mse'):
    """
    Computes the reconstruction loss between a vector X and a reconstruction of this vector Xhat.
    
    Args:
        xhat (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        rec_loss (str): The type of reconstruction loss to use ('mse' or 'mae').
    
    Returns:
        torch.Tensor: The computed reconstruction loss.
    """
    
    if rec_loss == 'mse':
        # Mean Squared Error loss
        return F.mse_loss(xhat.view(-1, xhat.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    elif rec_loss == 'mae':
        # Mean Absolute Error loss
        return F.l1_loss(xhat.view(-1, xhat.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
    else:
        raise ValueError("Invalid rec_loss. Expected 'mse' or 'mae'.")

# Define a function to compute the KL divergence loss
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

# Define a function to compute the pairwise loss
def positive_pairwise_loss(mu, roi, roi_metric='cosine'):
    """
    Encourages different views of the same ROI to have similar embeddings.

    Args:
        mu (torch.Tensor): The mean of the latent space (batch_size, latent_dim).
        roi (torch.Tensor): The region of interest identifiers (batch_size,).
        roi_metric (str): The metric to use for computing the pairwise loss ('cosine' or 'euclidean').
        
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
            
            if roi_metric == 'cosine':
                
                # Compute pairwise cosine similarity
                similarity_matrix = torch.matmul(mu_roi, mu_roi.T)
                
                # Compute loss: minimize distance from perfect similarity (1.0)
                total_loss += (1 - similarity_matrix).mean()
                
            elif roi_metric == 'euclidean':
                
                # Compute pairwise euclidean distance
                distance_matrix = torch.cdist(mu_roi, mu_roi, p=2)
                
                # Compute loss: minimize the distance
                total_loss += distance_matrix.mean()
                
            else:
                
                raise ValueError("Invalid roi_metric. Expected 'cosine' or 'euclidean'.")

    # Return total loss
    return total_loss

# Define a function to compute the total loss
def loss_clavae(recon_x, x, mu, logvar, roi, **optional_args):
    """
    Computes the total loss function for the VAE, including a penalty term for the distance in z within fluo signals from the same roi.
    
    Args:
        recon_x (torch.Tensor): The reconstructed input.
        x (torch.Tensor): The original input.
        mu (torch.Tensor): The mean of the latent space.
        logvar (torch.Tensor): The log variance of the latent space.
        roi (torch.Tensor): The region of interest identifiers.
    
    Returns:
        torch.Tensor: The computed total loss.
    """
    
    # Extract optional arguments
    bkl = optional_args.get('bkl', 1.0)
    broi = optional_args.get('broi', 1.0)
    rec_loss = optional_args.get('rec_loss', 'mse')
    model_type = optional_args.get('model_type', 'vae')
    roi_metric = optional_args.get('roi_metric', 'cosine')
    
    # Compute the reconstruction loss
    recon_loss = reconstruction_loss(recon_x, x, rec_loss) / x.size(0)
    
    # Compute the KL divergence loss
    kl_loss = bkl * kl_divergence_loss(mu, logvar) / x.size(0)
    
    # Initialize the InfoNCE loss to 0
    pp_loss = torch.tensor(0)
    
    # Compute the ROI loss if using the CLAVAE model
    if model_type == 'clavae':
        pp_loss = broi * 100 * positive_pairwise_loss(mu, roi, roi_metric=roi_metric)  / x.size(0)
    
    # Compute the total loss
    total_loss = recon_loss + kl_loss + pp_loss
    
    return total_loss, recon_loss, kl_loss, pp_loss

# Define a function to compute the total loss for the ISOVAE model
def loss_isovae(x, mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p, epoch, **optional_args):
    """
    Computes the total loss function for the ISOVAE.
    
    Args:
        x (torch.Tensor): The original input.
        ahat (torch.Tensor): The reconstructed amplitude.
        phat (torch.Tensor): The reconstructed phase.
        mu_a (torch.Tensor): The mean of the amplitude latent space.
        logvar_a (torch.Tensor): The log variance of the amplitude latent space.
        mu_p (torch.Tensor): The mean of the phase latent space.
        logvar_p (torch.Tensor): The log variance of the phase latent space.
    
    Returns:
        torch.Tensor: The computed total loss.
    """
    
    # NOTE: consider the following:
    # - Most frequencies have amplitude close to zero (very sparse)
    # - Use LSTM in encoders, not in decoders
    # - We could implicitly apply a low pass filter
    # 
    # DONE:
    # - Reconstruction through iFFT
    # - Use real FFT and iFFT to make code more efficient: no need for mask
    # - Use cosine similarity for phase
    # - Implemented KL annealing to avoid posterior collapse
    # 
    # TODO:
    # - Solve posterior collapse leading to std posterior samples
    # - Consider changing the prior of the angle to a uniform distribution?
    # - Solve issue with mask decoder producing only zeros
    # - Generalize scaling of ahat
    
    # Extract optional arguments
    bkl = optional_args.get('bkl', 1.0)
    rec_loss = optional_args.get('rec_loss', 'mae')
    
    # Compute the FFT of the original real signal
    fft_x = torch.fft.rfft(x)
    
    # get amplitudes and phases
    a = torch.abs(fft_x)
    p = torch.angle(fft_x)

    # get a mask
    a_mask = (a > 10).float()
    
    # set a smaller than 1 to zero
    a = a * a_mask
    
    # rescale to compare with decoders' output
    a = a / 2500
    p = (p + np.pi) / (2 * np.pi)
    
    # # superprint the first 10 entries of a and p
    # superprint(f"A values - Min: {a.min().item()}, Max: {a.max().item()}")
    # superprint(f"P values - Min: {p.min().item()}, Max: {p.max().item()}")
    # superprint(f"Ahat values - Min: {ahat.min().item()}, Max: {ahat.max().item()}")
    # superprint(f"Phat values - Min: {phat.min().item()}, Max: {phat.max().item()}")
    
    # # super print mu_a and logvar_a (useful for debugging posterior collapse)
    # superprint(f"Mu_a values - Min: {mu_a.min().item()}, Max: {mu_a.max().item()}")
    # superprint(f"Logvar_a values - Min: {logvar_a.min().item()}, Max: {logvar_a.max().item()}")
    # superprint(f"Mu_p values - Min: {mu_p.min().item()}, Max: {mu_p.max().item()}")
    # superprint(f"Logvar_p values - Min: {logvar_p.min().item()}, Max: {logvar_p.max().item()}")
    
    # apply mask to ahat
    ahat = ahat * (mask > 0.5).float()
    
    # mask loss (low pass filter)
    # mask_loss = torch.tensor(0)
    mask_loss = F.binary_cross_entropy(mask, a_mask, reduction='mean')
    
    # Compute the reconstruction loss for the amplitude
    recon_amp_loss = torch.tensor(0)
    # recon_amp_loss = F.smooth_l1_loss(ahat, a, reduction='mean')
    # recon_amp_loss = reconstruction_loss(ahat, a, 'mae') / x.size(0)
    
    # Compute the reconstruction loss for the phase (cosine similarity)
    recon_phase_loss = torch.tensor(0)
    # dp = p - phat
    # recon_phase_loss = torch.mean(1 - torch.cos((dp * 2 * np.pi) - np.pi)) / x.size(0)
    
    # rescale parameters for iFFT
    ahat = 2500 * ahat
    phat = (phat * 2 * np.pi) - np.pi
    
    # reconstruct signal with negative and positive frequencies
    xhat = torch.fft.irfft(ahat * torch.exp(1j * phat), n=x.shape[1])
    
    # Compute the reconstruction loss for the signal
    # recon_loss = torch.tensor(0)
    recon_loss = reconstruction_loss(xhat, x, rec_loss) / x.size(0)
    
    # Compute the KL divergence loss for amplitude
    kl_loss_a = kl_divergence_loss(mu_a, logvar_a) / x.size(0)
    
    # Compute the KL divergence loss for phase
    kl_loss_p = kl_divergence_loss(mu_p, logvar_p) / x.size(0)
    
    # adjust bkl to avoid posterior collapse
    bkl = min(bkl, (1+epoch) / 500)
    # superprint(f"bkl: {bkl}")
    
    # Compute the total loss
    total_loss = recon_loss + recon_amp_loss + recon_phase_loss + bkl * kl_loss_a + bkl * kl_loss_p + mask_loss
    
    # Return the total loss and individual losses
    return total_loss, mask_loss, recon_loss, recon_amp_loss, recon_phase_loss, kl_loss_a, kl_loss_p

# Define a function to train the CLAVAE model
def train_clavae(model, train_loader, optimizer, device, **optional_args):
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
        tot_loss, recon_loss, kl_loss, pp_loss = loss_clavae(recon_batch, fluo, mu, logvar, roi, **optional_args)
        
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

# Define a function to train the IsoVAE model
def train_isovae(model, train_loader, optimizer, device, epoch, **optional_args):
    """
    Trains the IsoVAE model using the given data.

    Args:
        model (ISOVAE): The IsoVAE model to train.
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
    
    # Initialize variables to store individual losses
    total_mask_loss = 0
    total_kl_loss_a = 0
    total_kl_loss_p = 0
    total_recon_loss = 0
    total_recon_amp_loss = 0
    total_recon_phase_loss = 0

    # Iterate over the training data
    for batch_idx, (x,) in enumerate(train_loader):
        
        # Move the data to the device
        x = x.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform a forward pass
        mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p = model(x)
        
        # Compute the loss
        tot_loss, mask_loss, recon_loss, recon_amp_loss, recon_phase_loss, kl_loss_a, kl_loss_p = loss_isovae(
            x, mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p, epoch, **optional_args)
        
        # Perform a backward pass
        tot_loss.backward()
        
        # Clip the gradients
        clip_grad_norm_(model.parameters(), 1)
        
        # Step the optimizer
        optimizer.step()
        
        # Add the losses to the total losses
        train_loss += tot_loss.item()
        total_mask_loss += mask_loss.item()
        total_kl_loss_a += kl_loss_a.item()
        total_kl_loss_p += kl_loss_p.item()
        total_recon_loss += recon_loss.item()
        total_recon_amp_loss += recon_amp_loss.item()
        total_recon_phase_loss += recon_phase_loss.item()
    
    # Calculate average losses
    avg_kl_loss_a = total_kl_loss_a / len(train_loader.dataset)
    avg_kl_loss_p = total_kl_loss_p / len(train_loader.dataset)
    avg_mask_loss = total_mask_loss / len(train_loader.dataset)
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_recon_amp_loss = total_recon_amp_loss / len(train_loader.dataset)
    avg_recon_phase_loss = total_recon_phase_loss / len(train_loader.dataset)
    
    # Print average losses
    superprint(f"L_rec: {avg_recon_loss:.6f} | L_mask: {avg_mask_loss:.6f} | L_rec_amp: {avg_recon_amp_loss:.6f} | L_rec_phase: {avg_recon_phase_loss:.6f} | L_kl_a: {avg_kl_loss_a:.6f} | L_kl_p: {avg_kl_loss_p:.6f}")
    
    # Return the average training loss
    return train_loss / len(train_loader.dataset)
