#%%

import os
import umap
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pyro
import pyro
import pyro.distributions as dist

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.model_selection import train_test_split

# clustering dependencies
from sklearn.metrics import silhouette_score

# Our modules
from ca_sn_gen_models.models import FA
from ca_sn_gen_models.utils import superprint, normalize_data, find_optimal_epsilon

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################

# Create the parser
parser = argparse.ArgumentParser(description="""
    This script trains a Bayesian Factor Analysis (FA) model on calcium imaging data. The FA model 
    is implemented using PyTorch and Pyro, and is designed to infer latent variables that explain 
    the observed data. The script includes functions for reading and processing calcium imaging 
    traces, normalizing data, and evaluating the model's performance.
    
    Arguments:
        --fluo_noise (float, required): Noise level in fluorescence data (default: 1.0)
        -l, --latent_dim (int, optional): Dimension of latent space (default: 32)
        -e, --epochs (int, optional): Number of training epochs (default: 1000)
        -s, --seed (int, optional): RNG seed (default: 0)
        -r, --rate (float, optional): Learning rate (default: 0.01)
        --norm (str, optional): Normalization method for the data (choices: 'log1p', 'std', 'scale', 'none'; default: 'none')
        --retrain (bool, optional): Whether to retrain the model (default: False)
        --save (bool, optional): Whether to save the model (default: True)
        --target_dist (str, required): Likelihood distribution for the model (choices: 'Normal', 'Laplace', 'Gamma', 'LogNormal'; default: 'Normal')
        --domain (str, required): Domain of the data (choices: 'time', 'frequency'; default: 'time')
        --outdir (str, optional): Folder to save output files (default: './output')
    
    Architecture:
        - Bayesian Factor Analysis: Learns latent factors and their contributions to observed data.
        - Likelihood: Supports multiple distributions (e.g., Normal, Laplace) for modeling data noise.
        - Domain-specific processing: Handles data in both time and frequency domains.
        
    Training:
        - Loads and processes calcium imaging data.
        - Splits the data into training and validation sets.
        - Initializes the FA model and trains it using a variational inference approach.
        - Evaluates the model's reconstruction performance and clustering quality.
        - Saves the trained model, results, and evaluation metrics if specified.
    
    Evaluation:
        - Reconstruction: Measures reconstruction error in both time and frequency domains.
        - Clustering: Evaluates latent space embeddings using metrics like ARI and silhouette score.
        - Dimensionality reduction: Supports visualization of latent variables using UMAP.
""")

# required arguments
parser.add_argument(
    "--fluo_noise",
    type=float,
    required=True,
    default=1.0,
    help="Noise level (default: 1.0)"
)

# optional arguments
parser.add_argument(
    "-l",
    "--latent_dim", 
    type=int, 
    required=False,
    default=32,
    help="Dimension of latent space (default: 32)"
)

parser.add_argument(
    "-e",
    "--epochs", 
    type=int, 
    required=False, 
    default=1000,
    help="Number of training epochs (default: 1000)"
)

parser.add_argument(
    "--norm", 
    type=str, 
    required=False, 
    default='none',
    choices=['log1p', 'std', 'scale', 'none'], 
    help="norm method"
)
parser.add_argument(
    "-s",
    "--seed", 
    type=int, 
    required=False, 
    default=0,
    help="RNG seed (default: 42)"
)

parser.add_argument(
    "-r",
    "--rate", 
    type=float, 
    required=False, 
    default=0.01,
    help="Learning rate (default: 0.01)"
)

parser.add_argument(
    "--retrain", 
    type=bool, 
    required=False, 
    default=False,
    help="Whether to retrain the model (default: False)"
)

parser.add_argument(
    "--save", 
    type=bool, 
    required=False, 
    default=True,
    help="Whether to save the model (default: True)"
)

parser.add_argument(
    "--target_dist", 
    type=str, 
    required=False, 
    default='Normal',
    choices=['Normal', 'Laplace', 'Gamma', 'LogNormal'],
    help="likelihood distribution (default: Normal)"
)

parser.add_argument(
    "--domain",
    type=str,
    required=False,
    default='time',
    choices=['time', 'frequency'],
    help="Domain of the data (default: time)"
)

parser.add_argument(
    '--outdir', 
    type=str, 
    required=False,
    default='./output', 
    help='Folder to save output files'
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
seed = args.seed              # seed=1   
retrain = args.retrain        # retrain=True
save_model = args.save        # save=False
norm = args.norm              # norm='log1p'
latent_dim = args.latent_dim  # latent_dim=8
num_epochs = args.epochs      # num_epochs=200
lr = args.rate                # learning_rate=0.005
fluo_noise = args.fluo_noise  # fluo_noise=1.0
target_dist = getattr(dist, args.target_dist)           # likelihood distribution (default: Normal)
domain = args.domain          # domain = 'time'
outdir = args.outdir          # outdir = './output'

# Comment out the following lines if you want to run the script without command line arguments
# seed = 1
# lr = 0.01
# norm = 'log1p'
# orthog = True
# retrain = True
# domain = 'frequency'
# latent_dim = 200
# fluo_noise = 1.0
# num_epochs = 1000
# save_model = False
# target_dist = getattr(dist, 'LogNormal')

#########################################################################################################
# Testing
#########################################################################################################
#%%

## Testing datasets

# # split data into train and validation
# data = torch.randn(1000, 100, device=device)
# labels = torch.full_like(torch.arange(1000), 0, dtype=torch.long)

# # Load iris data and normalize
# from sklearn.datasets import load_iris
# iris = load_iris()
# labels = iris.target
# data = torch.tensor(iris.data, dtype=torch.float32)
# data = normalize_data(data)

# # Load wine dataset
# from sklearn.datasets import load_wine
# wine = load_wine()
# labels = wine.target
# features = wine.feature_names
# data = torch.tensor(wine.data, dtype=torch.float32)
# data = normalize_data(data)

# # Breast cancer dataset
# from sklearn.datasets import load_breast_cancer
# data, labels = load_breast_cancer(return_X_y=True)
# data = torch.tensor(data, dtype=torch.float32)
# data = normalize_data(data)

## Simulated calcium traces

# included groups
group_set = [1, 2, 3]
sample_set = [0, 1, 2]

# directory
data_dir = '/pool01/projects/abante_lab/snDGM/sims_brb_spring_2025/data/normalized/'

# metadata
metadata_list = []
for group in group_set:
    for sample in sample_set:
        metadata_path_sims = f'{data_dir}params_group_{group}_sample_{sample}.tsv.gz'
        metadata_sims = pd.read_csv(metadata_path_sims, sep='\t')
        metadata_list.append(metadata_sims)

# add excitatory and inhibitory labels
ei = 200 * ['E']  + 800 * ['I']
ei_meta = ei * 9

# concatenate dataframes
meta_df = pd.concat(metadata_list, axis=0, ignore_index=True)
labels = meta_df['firing_type'].values

# Convert labels to numeric values
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = torch.tensor([label_mapping[label] for label in labels])

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}sigma_{fluo_noise}/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
x = torch.cat(data_list, dim=0)

# compute average signal for all xs
x_mean = x.mean(dim=1, keepdim=True)

# do real FFT of the data
xfft = torch.fft.rfft(x, axis=1)

# get amplitude and phase
a = torch.abs(xfft)
p = torch.angle(xfft)

#########################################################################################################
# NORMALIZE AND SPLIT DATA
#########################################################################################################

superprint('Normalizing data...')

if domain == 'time':
    superprint('(Not) normalizing data in time domain...')
    data = x
    norm = None
    # data = normalize_data(x)
else:
    superprint('Normalizing data in frequency domain...')
    # normalize the data
    data = torch.log1p(a)

    # plot histogram of a and ascaled
    plt.hist(data.flatten(), bins=100, alpha=0.5, label='Normalized', color='orange')
    plt.title('Amplitude')
    plt.legend()
    plt.show()

# split and create data loader
train_indices, val_indices = train_test_split(np.arange(len(data)), test_size=0.2, random_state=seed)
train_data = data[train_indices]
val_data = data[val_indices]

# create datasets
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=train_data.shape[0], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=val_data.shape[0], shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################

# Start the timer
start_time = time.time()

# clear cuda memory
torch.cuda.empty_cache()

# Clear Pyro parameters
pyro.clear_param_store()

# Initialize the FA model
effa = FA(data.shape[1], latent_dim, like=target_dist, domain = domain, device=device)

# Train the model
loss_history = effa.train_model(train_loader, num_epochs=num_epochs, lr=lr, patience=20, min_delta=5e0)

# Plot the training loss
plt.plot(loss_history, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Monitor peak RAM usage
process = psutil.Process()

# Get peak memory usage in MB
peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

# stop timer
end_time = time.time()
training_time = end_time - start_time
superprint(f'Training time: {training_time:.2f} seconds')

#########################################################################################################
# Clean data
#########################################################################################################

superprint('Reading in zero noise data...')

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}/sigma_0.0/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
x0 = torch.cat(data_list, dim=0)

# split into train and validation
x0_train = x0[train_indices]
x0_val = x0[val_indices]

# FFT
x0_train_fft = torch.fft.rfft(x0_train, axis=1)
x0_val_fft = torch.fft.rfft(x0_val, axis=1)

# get amplitude and phase
a0_train = torch.abs(x0_train_fft)
a0_val = torch.abs(x0_val_fft)
p0_train = torch.angle(x0_train_fft)
p0_val = torch.angle(x0_val_fft)

#########################################################################################################
# EMBEDDING
#########################################################################################################

superprint('Getting posterior estimates...')

# get posterior values of Z, W, and sigma
Ztrain, W_loc, sigma_loc = effa.get_posterior()

# TODO: comment out until next section

# # Perform UMAP dimensionality reduction
# reducer = umap.UMAP(n_components=2)
# Z_umap = reducer.fit_transform(Ztrain)

# # get labels for training data
# group_labels = meta_df['group'].values[train_indices]
# sample_labels = meta_df['sample'].values[train_indices]
# firing_type_labels = meta_df['firing_type'].values[train_indices]
# mean_firing_label = x_mean[train_indices].squeeze(1).cpu().numpy()

# # Create a DataFrame for easier plotting with seaborn
# umap_df = pd.DataFrame({
#     'UMAP1': Z_umap[:, 0],
#     'UMAP2': Z_umap[:, 1],
#     'Group': group_labels,
#     'Sample': sample_labels,
#     'FiringType': firing_type_labels,
#     'MeanFiring': mean_firing_label
# })

# # Plot using seaborn for Group and Sample
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Group and Sample)")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.legend(title="Groups and Samples", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Firing Type
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='FiringType',
#     style='Group',
#     palette='Set2',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Firing Type)")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.legend(title="Firing Type", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Firing Type
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='MeanFiring',
#     style='Group',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Firing Type)")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.show()

##############################################################################################################
# Correlation between Z and (group x sample ID)
##############################################################################################################

# TODO: comment out until next section when running benchmark

# from sklearn.feature_selection import mutual_info_classif
# from sklearn.preprocessing import LabelEncoder

# # create label combining group and sample
# group_sample_labels = np.array([f'{g}_{s}' for g, s in zip(group_labels, sample_labels)])

# # Encode group_sample_labels using LabelEncoder
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(group_sample_labels)

# # Compute correlation between each component of Ztrain and encoded labels
# mi_z_label = mutual_info_classif(Ztrain, encoded_labels)

# # Order latent dimensions by mutual information
# ordered_indices = np.argsort(mi_z_label)[::-1]
# ordered_mi = mi_z_label[ordered_indices]

# # Barplot of mutual information with ordered dimensions
# plt.figure(figsize=(15, 5))
# plt.bar(range(len(ordered_mi)), ordered_mi)
# plt.xlabel('Ordered Latent Dimension')
# plt.ylabel('Mutual Information with Group-Sample Labels')
# plt.title('Mutual Information between Latent Dimensions and Group-Sample Labels (Ordered)')
# plt.xticks(range(len(ordered_mi)), ordered_indices)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()

# # split latent dimensions into two groups: high vs low mutual information
# high_mi_indices = ordered_indices[:latent_dim//2]
# low_mi_indices = ordered_indices[latent_dim//2:]
# high_mi_z = Ztrain[:, high_mi_indices]
# low_mi_z = Ztrain[:, low_mi_indices]

# # umap of each group
# reducer = umap.UMAP(n_components=2)
# high_mi_umap = reducer.fit_transform(high_mi_z)
# low_mi_umap = reducer.fit_transform(low_mi_z)

# # Create a DataFrame for easier plotting with seaborn
# umap_df_high = pd.DataFrame({
#     'UMAP1': high_mi_umap[:, 0],
#     'UMAP2': high_mi_umap[:, 1],
#     'Group': group_labels,
#     'Sample': sample_labels,
#     'FiringType': firing_type_labels,
#     'MeanFiring': mean_firing_label
# })
# umap_df_low = pd.DataFrame({
#     'UMAP1': low_mi_umap[:, 0],
#     'UMAP2': low_mi_umap[:, 1],
#     'Group': group_labels,
#     'Sample': sample_labels,
#     'FiringType': firing_type_labels,
#     'MeanFiring': mean_firing_label
# })

# # Plot using seaborn for Group and Sample
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df_high,
#     x='UMAP1',
#     y='UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Group and Sample) - High MI")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.legend(title="Groups and Samples", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Group and Sample
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df_low,
#     x='UMAP1',
#     y='UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Group and Sample) - Low MI")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.legend(title="Groups and Samples", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Firing Type
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df_low,
#     x='UMAP1',
#     y='UMAP2',
#     hue='FiringType',
#     style='Group',
#     palette='Set2',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Firing Type) - Low MI")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.legend(title="Firing Type", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Firing Type
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df_low,
#     x='UMAP1',
#     y='UMAP2',
#     hue='MeanFiring',
#     style='Group',
#     s=50
# )
# scatter.set_title("UMAP of Latent Variables Z (Colored by Firing Type) - Low MI")
# scatter.set_xlabel("UMAP Dimension 1")
# scatter.set_ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.show()

##############################################################################################################
# Reconstruction of X
##############################################################################################################

superprint('Evaluating reconstruction...')

# Set model to evaluation mode
effa.eval()

# get posterior value of Z given X
Zval = val_data @ effa.Hhat.cpu()

if domain == 'time':
            
    # reconstruct data
    xhat = Zval @ W_loc

    # set error metrics of amplitude to None
    mae_a = None
    mse_a = None
    
else:
    
    # Revert log1p scaling
    ahat = torch.expm1(Zval @ W_loc)
    
    # Calculate reconstruction error metrics
    mae_a = torch.nn.functional.l1_loss(a0_val, ahat, reduction='mean').item()
    mse_a = torch.nn.functional.mse_loss(a0_val, ahat, reduction='mean').item()

    # do ifft using the reconstructed amplitude and original phase
    xhat = ahat * torch.exp(1j * p0_val)
    xhat = torch.fft.irfft(xhat, n=x.shape[1])

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(len(val_data), 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_x = torch.nn.functional.l1_loss(x0_val, xhat, reduction='mean').item()
mse_x = torch.nn.functional.mse_loss(x0_val, xhat, reduction='mean').item()

#########################################################################################################
# Clustering
#########################################################################################################

superprint('Evaluating embedding...')

# find optimal epsilon for training data
clust_train, eps, ari_train = find_optimal_epsilon(Ztrain, labels[train_indices])
clust_val, eps, ari_val = find_optimal_epsilon(Zval, labels[val_indices])

# Calculate silhouette score
sil_train = silhouette_score(Ztrain, clust_train)
sil_val = silhouette_score(Zval, clust_val)

#########################################################################################################
# Save results
#########################################################################################################

superprint('Storing summary...')

# get distribution name
dist_name = target_dist.__name__.split('.')[-1]

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': [f'EFFA ({domain})'],
    'norm': [norm],
    'beta': [None],
    'lr': [lr],
    'time': [training_time],
    'ram': [peak_memory],
    'latent_dim': [latent_dim],
    'target_dist': [dist_name],
    'num_epochs': [len(loss_history)],
    'last_train_loss': [loss_history[-1]],
    'last_val_loss': [None],
    'mae_amp': [mae_a],
    'mse_amp': [mse_a],
    'mae_x': [mae_x],
    'mse_x': [mse_x],
    'ari_train': [ari_train],
    'ari_val': [ari_val],
    'sil_train': [sil_train],
    'sil_val': [sil_val]
})

#%%

# Append the dataframe to a text file
results_file = f'{outdir}/results_summary_effa_{domain}.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
