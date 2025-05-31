#%%

import os
import glob
import time
import psutil
import argparse
import numpy as np
import pandas as pd

# Pyro
import pyro

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Our modules
from ca_sn_gen_models.utils import superprint

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################
#%%

# Create the parser
parser = argparse.ArgumentParser(description="Training script for DGM models on Ca imaging data.")

# required arguments
parser.add_argument(
    '--indir',
    type=str,
    required=True,
    help='Input directory containing input CSV files'
)

parser.add_argument(
    '--metadata',
    type=str,
    required=True,
    help='Path to the metadata file'
)

# optional arguments
parser.add_argument(
    "-l",
    "--latent_dim", 
    type=int,
    required=False,
    default=100,
    help="Dimension of latent space (default: 100)"
)

parser.add_argument(
    "-e",
    "--num_epochs", 
    type=int, 
    required=False,
    default=5000,
    help="Dimension of latent space (default: 5000)"
)

parser.add_argument(
    "-s",
    "--seed", 
    type=int, 
    required=False, 
    default=0,
    help="RNG seed (default: 0)"
)

parser.add_argument(
    "-r",
    "--rate", 
    type=float, 
    required=False, 
    default=1e-4,
    help="Learning rate (default: 1e-4)"
)

parser.add_argument(
    "-b",
    "--batch_size", 
    type=int, 
    required=False, 
    default=20000,
    help="Batch size for training (default: 20000)"
)

parser.add_argument(
    '--outdir', 
    type=str, 
    required=False,
    default='./output', 
    help='Folder to save output files'
)

# valid model
valid_models = [
    'FixedVarMlpVAE',
    'LearnedVarMlpVAE',
    'FixedVarSupMlpVAE',
    'LearnedVarSupMlpVAE',
    'FixedVarSupMlpDenVAE',
    'LearnedVarSupMlpDenVAE'
]

parser.add_argument(
    '--vae', 
    type=str, 
    required=False,
    default='FixedVarMlpVAE',
    choices=valid_models,
    help='Model to use (default: FixedVarMlpVAE). Options: ' + ', '.join(valid_models)
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
vae = args.vae
lr = args.rate
seed = args.seed
outdir = args.outdir
data_dir = args.indir
batch_size = args.batch_size
num_epochs = args.num_epochs
fluo_noise = args.fluo_noise
latent_dim = args.latent_dim
metadata_path = args.metadata

###############################################################################################
# import model
###############################################################################################
#%%

if vae == 'FixedVarMlpVAE':
    from ca_sn_gen_models.models import FixedVarMlpVAE as vae_model
    supervised = False
    out_var = 'Fixed'
    mask = False
elif vae == 'LearnedVarMlpVAE':
    from ca_sn_gen_models.models import LearnedVarMlpVAE as vae_model
    supervised = False
    out_var = 'Learned'
    mask = False
elif vae == 'FixedVarSupMlpVAE':
    from ca_sn_gen_models.models import FixedVarSupMlpVAE as vae_model
    supervised = True
    out_var = 'Fixed'
    mask = False
elif vae == 'LearnedVarSupMlpVAE':
    from ca_sn_gen_models.models import LearnedVarSupMlpVAE as vae_model
    supervised = True
    out_var = 'Learned'
    mask = False
elif vae == 'FixedVarSupMlpDenVAE':
    from ca_sn_gen_models.models import FixedVarSupMlpDenVAE as vae_model
    supervised = True
    out_var = 'Fixed'
    mask = True
elif vae == 'LearnedVarSupMlpDenVAE':
    from ca_sn_gen_models.models import LearnedVarSupMlpDenVAE as vae_model
    supervised = True
    out_var = 'Learned'
    mask = True
else:
    raise ValueError(f"Model {vae} not recognized. Choose from {valid_models}.")

#########################################################################################################
# Read in
#########################################################################################################

superprint('Reading in data...')

# Read all CSV files in the provided input directory
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {data_dir}")

# Read and concatenate all CSV files, and create batch labels
x_list = []
batch_labels = []
batch_index_to_file = {}
for i, file in enumerate(csv_files):
    
    # Read the CSV file
    superprint(f'Reading file {i+1}/{len(csv_files)}: {file}')
    data = pd.read_csv(file, header=None).values
    data_tensor = torch.tensor(data, dtype=torch.float32)
    x_list.append(data_tensor)
    
    # Assign the same batch label to all rows from this file
    batch_labels.extend([i] * data_tensor.shape[0])
    
    # Map batch index to file name
    batch_index_to_file[i] = file

# Concatenate data
x = torch.cat(x_list, dim=0)

if supervised:

    # Convert batch_labels to a tensor
    batch_labels = torch.tensor(batch_labels, dtype=torch.long)

    # Encode batch_labels using LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(batch_labels)
    encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

    # Convert to one-hot encoding
    num_classes = len(np.unique(encoded_labels))
    y_oh = torch.nn.functional.one_hot(encoded_labels, num_classes=num_classes)

#########################################################################################################
# SPLIT DATA
#########################################################################################################

superprint('Splitting data...')

if supervised:

    # split and create data loader
    train_indices, val_indices = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=seed)

    # split data
    train_data = x[train_indices]
    val_data = x[val_indices]

    # split labels
    train_labels = y_oh[train_indices]
    val_labels = y_oh[val_indices]

    # create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

else:

    # split and create data loader
    train_indices, val_indices = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=seed)

    # split data
    train_data = x[train_indices]
    val_data = x[val_indices]

    # split labels
    train_labels = y_oh[train_indices]
    val_labels = y_oh[val_indices]

    # create datasets
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################

# Create output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

# model path
suffix = f'latent_{latent_dim}_epochs_{num_epochs}_lr_{lr}_seed_{seed}'
model_path = f'{outdir}/parameters_{suffix}.pt'

# init model
superprint(f'Initializing model {vae} with latent dimension {latent_dim}...')
if supervised:
    
    # Initialize model with num_classes
    model = vae_model(data.shape[1], latent_dim, num_classes, device=device)

else:
    
    # Initialize model
    model = vae_model(data.shape[1], latent_dim, device=device)

if not os.path.exists(model_path):
    
    superprint(f'Model does not exist at {model_path}. Starting training...')

    # Start the timer
    start_time = time.time()

    # clear cuda memory
    torch.cuda.empty_cache()

    # Clear Pyro parameters
    pyro.clear_param_store()

    # Train the model
    loss_tr,loss_val = model.train_model(train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=50, min_delta=1e-2)

    # Monitor peak RAM usage
    process = psutil.Process()

    # Get peak memory usage in MB
    peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
    superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

    # stop timer
    end_time = time.time()
    training_time = end_time - start_time
    superprint(f'Training time: {training_time:.2f} seconds')

    # Save the model
    torch.save(model.state_dict(), model_path)
    superprint(f'Model saved to {model_path}')

else:
    
    superprint(f'Model already exists at {model_path}. Loading model...')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

#########################################################################################################
# INFERENCE
#########################################################################################################

superprint('Getting posterior estimates...')

# create loaders for inference
if supervised:
    data_loader = DataLoader(x, y_oh, batch_size=batch_size, shuffle=False)
else:
    data_loader = DataLoader(x, batch_size=batch_size, shuffle=False)

# Set model to evaluation mode
model.eval()

# do forward pass to get posterior estimates
zloc, xhat = model.forward(data_loader)

# Save the results to the output directory
zloc_path = os.path.join(outdir, "zloc_{suffix}.csv.gz")
xhat_path = os.path.join(outdir, "xhat_{suffix}.csv.gz")

# crea dataframes and save as compressed CSV
pd.DataFrame(zloc.numpy()).to_csv(zloc_path, index=False, header=False, compression="gzip")
pd.DataFrame(xhat.numpy()).to_csv(xhat_path, index=False, header=False, compression="gzip")

superprint(f"Saved zloc and xhat to {outdir} as compressed CSV files.")
