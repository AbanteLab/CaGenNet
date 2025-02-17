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
    
    def __init__(self, input_dim, hidden_dim, latent_dim, fft_comp):
        super(ISOVAE, self).__init__()
        
        # dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fft_comp = fft_comp
        
        # Encoder layers for amplitude
        self.encoder_amp_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_amp_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_amp_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_amp_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Encoder layers for phase
        self.encoder_phase_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_phase_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_phase_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_phase_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_phase_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder layers for amplitude
        self.decoder_amp_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_amp_fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_amp_fc3 = nn.Linear(2 * hidden_dim, fft_comp)
        
        # Decoder layers for phase
        self.decoder_phase_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_phase_fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.decoder_phase_fc3 = nn.Linear(2 * hidden_dim, fft_comp)
        
        # Decoder layers for signal reconstruction
        self.decoder_signal_fc1 = nn.Linear(2 * latent_dim, 2 * hidden_dim)
        self.decoder_signal_fc2 = nn.Linear(2 * hidden_dim, 4 * hidden_dim)
        self.decoder_signal_fc3 = nn.Linear(4 * hidden_dim, input_dim)
        
    def encode_amplitude(self, x):
        
        # get output of fully connected layers with ReLU activation
        h1 = F.relu(self.encoder_amp_fc1(x))
        h2 = F.relu(self.encoder_amp_fc2(h1))
        
        # get parameters q(z|x)
        mu = self.encoder_amp_fc_mu(h2)
        logvar = self.encoder_amp_fc_logvar(h2)
        
        return mu, logvar
    
    def encode_phase(self, x):
        
        # get output of fully connected layers with ReLU activation
        h1 = F.relu(self.encoder_phase_fc1(x))
        h2 = F.relu(self.encoder_phase_fc2(h1))
        h2 = F.relu(self.encoder_phase_fc2(h2))
        
        # get parameters q(z|x)
        mu = self.encoder_phase_fc_mu(h2)
        logvar = self.encoder_phase_fc_logvar(h2)
        
        return mu, logvar
    
    def sample_z(self, mu, logvar):
        
        # reparametrize
        std = torch.exp(0.5 * logvar)
        
        # sample eps~(0,1)
        eps = torch.randn_like(std)

        # return z
        return mu + eps * std
    
    def decode_amplitude(self, z):
        
        # get output of fully connected layers with ReLU activation
        h1 = F.relu(self.decoder_amp_fc1(z))
        h2 = F.relu(self.decoder_amp_fc2(h1))
        
        # get final output
        out = self.decoder_amp_fc3(h2)
        
        return out
    
    def decode_phase(self, z):
        
        # get output of fully connected layers with ReLU activation
        h1 = F.relu(self.decoder_phase_fc1(z))
        h2 = F.relu(self.decoder_phase_fc2(h1))
        
        # get final output
        out = self.decoder_phase_fc3(h2)
        
        return out
    
    def decode_signal(self, za, zp):
        
        # concatenate Za and Zp
        z = torch.cat((za, zp), dim=-1)
        
        # get output of fully connected layers with ReLU activation
        h1 = F.relu(self.decoder_signal_fc1(z))
        h2 = F.relu(self.decoder_signal_fc2(h1))
        
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
        reconstructed_amp = self.decode_amplitude(za)
        
        # Decode phase
        reconstructed_phase = self.decode_phase(zp)
        
        # Decode signal
        reconstructed_x = self.decode_signal(za, zp)
        
        return reconstructed_x, reconstructed_amp, reconstructed_phase, mu_a, logvar_a, mu_p, logvar_p

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

def num_freq_components(fs, N):
    """
    Computes the number of frequency components up to the Nyquist frequency (fs/2)
    for a signal of length N sampled at frequency fs.

    Args:
        fs (float): Sampling frequency in Hz.
        N (int): Number of timepoints in the signal.

    Returns:
        int: Number of frequency components up to fs/2.
        float: Frequency resolution (Δf).
        numpy.ndarray: Array of frequency bins up to fs/2.
    """
    import numpy as np

    num_components = N // 2  # Number of bins up to Nyquist frequency
    delta_f = fs / N  # Frequency resolution
    freq_bins = np.linspace(0, fs / 2, num_components)  # Frequency bins

    return num_components, delta_f, freq_bins

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
def loss_isovae(xhat, x, ahat, phat, mu_a, logvar_a, mu_p, logvar_p, **optional_args):
    """
    Computes the total loss function for the ISOVAE.
    
    Args:
        xhat (torch.Tensor): The reconstructed input.
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
    
    # Extract optional arguments
    fs = optional_args.get('fs', 20.0)
    bkl = optional_args.get('bkl', 1.0)
    rec_loss = optional_args.get('rec_loss', 'mae')
    
    # NOTE: consider the following:
    # - We could do the reconstruction of X doing the ifft of Ahat and Phat
    # - We could implicitly apply a low pass filter - this could be problematic though
    
    # TODO:
    # - Make code more efficient by precomputing the mask
    
    # Compute FFT of the original signal
    fft_x = torch.fft.fft(x)
    
    # get amplitudes and phases
    a = torch.abs(fft_x)
    p = torch.angle(fft_x)
    
    # Compute the frequency bins
    freq_bins = torch.fft.fftfreq(x.size(-1), d=1/fs)
    
    # Mask to keep only components with frequency at most fs/2
    mask = (freq_bins >= 0) & (freq_bins <= fs/2)
    
    # Apply the mask to amplitude and phase
    a = a[:, mask]
    p = p[:, mask]
    
    # Normalize amplitude to be between 0 and 1
    a = (a - a.min()) / (a.max() - a.min())
    
    # Compute the reconstruction loss for the signal
    recon_loss = reconstruction_loss(xhat, x, rec_loss) / x.size(0)
    
    # Compute the reconstruction loss for the amplitude
    recon_amp_loss = reconstruction_loss(ahat, a, 'mae') / x.size(0)
    
    # Compute the reconstruction loss for the phase (cosine similarity)
    recon_phase_loss = torch.mean(1 - torch.cos(p - phat)) / x.size(0)
    
    # Compute the KL divergence loss for amplitude
    kl_loss_a = bkl * kl_divergence_loss(mu_a, logvar_a) / x.size(0)
    
    # Compute the KL divergence loss for phase
    kl_loss_p = bkl * kl_divergence_loss(mu_p, logvar_p) / x.size(0)
    
    # Compute the total loss
    total_loss = recon_loss + recon_amp_loss + recon_phase_loss + kl_loss_a + kl_loss_p
    
    # Return the total loss and individual losses
    return total_loss, recon_loss, recon_amp_loss, recon_phase_loss, kl_loss_a, kl_loss_p

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
def train_isovae(model, train_loader, optimizer, device, **optional_args):
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
        recon_batch, recon_amp, recon_phase, mu_a, logvar_a, mu_p, logvar_p = model(x)
        
        # Compute the loss
        tot_loss, recon_loss, recon_amp_loss, recon_phase_loss, kl_loss_a, kl_loss_p = loss_isovae(
            recon_batch, x, recon_amp, recon_phase, mu_a, logvar_a, mu_p, logvar_p, **optional_args)
        
        # Perform a backward pass
        tot_loss.backward()
        
        # Clip the gradients
        clip_grad_norm_(model.parameters(), 1)
        
        # Step the optimizer
        optimizer.step()
        
        # Add the losses to the total losses
        train_loss += tot_loss.item()
        total_recon_loss += recon_loss.item()
        total_recon_amp_loss += recon_amp_loss.item()
        total_recon_phase_loss += recon_phase_loss.item()
        total_kl_loss_a += kl_loss_a.item()
        total_kl_loss_p += kl_loss_p.item()
    
    # Calculate average losses
    avg_recon_loss = total_recon_loss / len(train_loader.dataset)
    avg_recon_amp_loss = total_recon_amp_loss / len(train_loader.dataset)
    avg_recon_phase_loss = total_recon_phase_loss / len(train_loader.dataset)
    avg_kl_loss_a = total_kl_loss_a / len(train_loader.dataset)
    avg_kl_loss_p = total_kl_loss_p / len(train_loader.dataset)
    
    # Print average losses
    superprint(f"L_rec: {avg_recon_loss:.6f} | L_rec_amp: {avg_recon_amp_loss:.6f} | L_rec_phase: {avg_recon_phase_loss:.6f} | L_kl_a: {avg_kl_loss_a:.6f} | L_kl_p: {avg_kl_loss_p:.6f}")
        
    # Return the average training loss
    return train_loss / len(train_loader.dataset)
