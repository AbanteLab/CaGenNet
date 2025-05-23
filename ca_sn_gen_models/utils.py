# dependencies
import numpy as np
import pandas as pd
from datetime import datetime

# sklearn dependencies
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# torch dependencies
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.utils.data import DataLoader, TensorDataset, Dataset

# pyro
import pyro
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO

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
        data = pd.read_csv(file_path, header=None, sep=',').astype(np.float32).values
    elif file_path.endswith('.npy'):
        superprint("Loading data from NPY file")
        data = np.load(file_path).astype(np.float32)
    else:
        superprint("Loading data from TSV file")
        data = pd.read_csv(file_path, header=None, sep='\t').astype(np.float32).values
    
    # Convert to float32
    # data = data.astype(np.float32)
    
    return data

def augment_data(data, N, seed=0):
    """
    Augments the input data by rolling the signal N times with evenly spaced increments.
    
    Args:
        data (numpy.ndarray): The original data.
        N (int): The number of times to roll the signal.
        seed (int): The random seed for reproducibility.
    
    Returns:
        numpy.ndarray: The augmented data.
        numpy.ndarray: The indices of the rolled positions.
    """
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize lists to store the indices and augmented data
    indices = []
    augmented_data = []
    
    # Calculate the increment for rolling
    increment = data.shape[1] // N
    
    # Iterate over the rows in the data
    for row_idx, row in enumerate(data):
        
        # Initialize lists to store the indices and augmented data for the current row
        row_indices = []
        row_augmented_data = []
        
        # Iterate N times to roll the signal
        for i in range(N):
            
            # Calculate the number of positions to roll
            roll_positions = i * increment
            
            # Roll the signal
            rolled_row = np.roll(row, roll_positions)
            
            # Append the rolled signal and the roll positions
            row_augmented_data.append(rolled_row)
            row_indices.append(roll_positions)
        
        augmented_data.append(row_augmented_data)
        indices.append(row_indices)
    
    # Convert the lists to numpy arrays
    indices = np.array(indices)
    augmented_data = np.array(augmented_data)
    
    return augmented_data, indices

def normalize_data(data):
    """
    Normalizes the data.
    
    Args:
        data (pandas.DataFrame): The data to normalize.
    
    Returns:
        numpy.ndarray: The normalized data.
    """
    
    # Normalize the data
    normalized_data = (data - data.min().min()) / (data.max().max() - data.min().min())
    
    # offset mean to zero
    normalized_data = normalized_data - normalized_data.mean()
    
    return normalized_data

def simulate_lognormal_amplitude(num_samples=8449, num_features=2500, num_archetypes=4, num_components=20, archetype_value=10, noise_mean=1, noise_std=0.1):
    """
    Simulates data by generating archetypes and adding noise.

    Args:
        num_samples (int): The number of samples to generate.
        num_features (int): The number of features for each sample.
        num_archetypes (int): The number of archetypes to generate.
        num_components (int): The number of components to set to a specific value in each archetype.
        archetype_value (float): The value to set for the selected components in each archetype.
        noise_mean (float): The mean of the noise to add to the archetypes.
        noise_std (float): The standard deviation of the noise to add to the archetypes.

    Returns:
        torch.Tensor: The simulated data.
        torch.Tensor: The phase data.
        list: The list of archetype indices used for each sample.
    """
    
    # Generate archetypes
    archetypes = []
    for _ in range(num_archetypes):
        base = np.random.lognormal(mean=1, sigma=1, size=num_features)
        idx = np.random.choice(num_features, num_components, replace=False)
        base[idx] = archetype_value
        archetypes.append(base)
    
    # Sample vectors by picking one of the archetypes and adding noise
    y = []
    a = np.zeros((num_samples, num_features))
    for i in range(num_samples):
        idx = np.random.choice(num_archetypes)
        archetype = archetypes[idx]
        noise = np.random.normal(loc=noise_mean, scale=noise_std, size=(num_features,))
        a[i] = archetype + noise
        y.append(idx)
    
    # Ensure a is positive
    a = np.abs(a)
    
    # Convert to tensor
    a = torch.tensor(a, dtype=torch.float32)
    p = torch.tensor(np.random.uniform(-np.pi, np.pi, size=(num_samples, num_features)), dtype=torch.float32)
    
    return a, p, y

############################################################################################################
# INFERENCE
############################################################################################################

# Define a function to dynamically adjust the KL divergence beta
def kl_beta_schedule(epoch, num_epochs, start_beta=0.1, end_beta=2.0):
    
    beta = start_beta + (end_beta - start_beta) * (epoch / num_epochs)
    return min(beta, end_beta)

def train_svi(vae, svi, train_loader, val_loader, num_epochs=500, batch=512, patience=10, min_delta=1e-3, start_beta=0.25, end_beta=0.25, device='cpu', model_type="AmpVAE"):
    """
    Trains a model using Stochastic Variational Inference (SVI) with early stopping.
    Args:
        vae (torch.nn.Module): The pyro model to be trained.
        svi (pyro.infer.SVI): The SVI object for performing optimization.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Default is 500.
        batch (int, optional): Batch size for training. Default is 512.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 10.
        min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Default is 1e-3.
        start_beta (float, optional): Initial value of the beta parameter for KL divergence annealing. Default is 0.25.
        end_beta (float, optional): Final value of the beta parameter for KL divergence annealing. Default is 0.25.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
        model_type (str, optional): Type of VAE model being trained. Default is "AmpVAE".
    Returns:
        tuple: A tuple containing two lists:
            - avg_train_loss_trail (list): List of average training losses per epoch.
            - avg_val_loss_trail (list): List of average validation losses per epoch.
    The function trains the VAE model for a specified number of epochs, updating the beta parameter for KL divergence
    annealing. It computes the training and validation losses for each epoch and implements early stopping based on
    the validation loss to prevent overfitting.
    """
    
    # Early stopping parameters
    counter = 0
    best_val_loss = float('inf')

    # Training loop
    avg_val_loss_trail = []
    avg_train_loss_trail = []
    for epoch in range(num_epochs):
        
        # Update beta
        if model_type != "EFFA":
            beta = kl_beta_schedule(epoch, num_epochs // 2, start_beta=start_beta, end_beta=end_beta)

        # Iterate over batches
        vae.train()
        epoch_loss = 0
        for a_train, p_train in train_loader:
            a_train = a_train.to(device)
            p_train = p_train.to(device)
            if model_type in ["IcasspVAE", "HierarchicalVAE"]:
                loss = svi.step(a_train, p_train)
            elif model_type == "EFFA":
                loss = svi.step(a_train)
            else:
                loss = svi.step(a_train, beta=beta)
            epoch_loss += loss / batch

        # Compute validation loss
        vae.eval()
        val_loss = 0
        for a_val, p_val in val_loader:
            a_val = a_val.to(device)
            p_val = p_val.to(device)
            if model_type in ["IcasspVAE", "HierarchicalVAE"]:
                val_loss += svi.evaluate_loss(a_val, p_val) / batch
            elif model_type == "EFFA":
                val_loss += svi.evaluate_loss(a_val) / batch
            else:
                val_loss += svi.evaluate_loss(a_val, beta=beta) / batch

        # Print average train and validation loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        if model_type != "EFFA":
            superprint(f"Epoch {epoch}, Train Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}, Beta: {beta}")
        else:
            superprint(f"Epoch {epoch}, Train Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}")

        # Append to trail
        avg_val_loss_trail.append(avg_val_loss)
        avg_train_loss_trail.append(avg_epoch_loss)

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                superprint(f'Early stopping triggered at epoch {epoch + 1}')
                break

    return avg_train_loss_trail, avg_val_loss_trail

# Define a function to apply Gram-Schmidt orthogonalization to a matrix
def gram_schmidt(matrix):
    """
    Applies Gram-Schmidt orthogonalization to the input matrix.

    Args:
        matrix (numpy.ndarray): A k x d matrix where k is the number of vectors 
                                and d is the dimensionality of each vector.

    Returns:
        numpy.ndarray: A k x d matrix with k orthogonal vectors.
    """
    k, d = matrix.shape
    orthogonal_matrix = torch.zeros((k, d), device=matrix.device, dtype=matrix.dtype)
    
    for i in range(k):
        # Start with the current vector
        vec = matrix[i]
        
        # Subtract projections onto previously computed orthogonal vectors
        for j in range(i):
            proj = torch.dot(vec, orthogonal_matrix[j]) / torch.dot(orthogonal_matrix[j], orthogonal_matrix[j])
            vec -= proj * orthogonal_matrix[j]
        
        # Normalize the vector
        orthogonal_matrix[i] = vec / torch.norm(vec)
    
    return orthogonal_matrix

def gen_random_mask(n, m, min_seg=5, max_seg=15, masked_fraction=0.2):
    """
    Generates a random boolean mask matrix with missing segments distributed across each row. 
    
    This function creates a mask of shape (n, m) where a specified fraction of each row is set to False (masked/missing),
    distributed in a number of contiguous segments. The number of segments per row is randomly chosen between `min_seg` and `max_seg`.
    The length of each segment is determined such that the total number of masked elements per row matches the `masked_fraction`.
    Args:
        n (int): Number of rows in the mask (e.g., number of samples or ROIs).
        m (int): Number of columns in the mask (e.g., number of timepoints).
        min_seg (int, optional): Minimum number of missing segments per row. Default is 5.
        max_seg (int, optional): Maximum number of missing segments per row. Default is 15.
        masked_fraction (float, optional): Fraction of each row to be masked (set to False). Must be between 0 and 1. Default is 0.2.
    Returns:
        np.ndarray: A boolean mask of shape (n, m), where True indicates observed (unmasked) entries and False indicates masked (missing) entries.
    Notes:
        - The actual number of masked elements per row is approximately `masked_fraction * m`.
        - The masked segments are distributed randomly but are contiguous within each segment.
        - The function uses numpy for random number generation and array manipulation.
    """
    
    # get number of missing timpoints per row
    num_miss_tps = int(masked_fraction * m)

    # Step 1: Generate number of segments per row from randint(1, max_num_miss_seg)
    num_miss_seg = np.random.randint(min_seg, max_seg + 1)

    # Step 2: initialize mask matrix
    mask = np.ones((n, m), dtype=bool)

    # get equally spaced start of segments
    seg_st = np.linspace(0, m - m//num_miss_seg, num_miss_seg)

    # sample random shift for each row
    mask_shift = np.random.randint(0, (seg_st[1]-seg_st[0])//2, size=n)

    for i in range(n):
        
        # allocate missing timepoints across segments for i-th roi
        ith_seg_len = np.random.multinomial(num_miss_tps, [1/num_miss_seg]*num_miss_seg)
        
        # start of missing data intervals for i-th roi
        ith_st = seg_st + mask_shift[i]

        # define intervals
        ith_int = [np.arange(ith_st[j], ith_st[j] + ith_seg_len[j], dtype=int) for j in range(num_miss_seg)]

        ints = np.concatenate(ith_int)
    
        # register i-th row mask
        mask[i, ints] = 0
    
    return mask