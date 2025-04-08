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
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO

# PyTorch  
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# clustering dependencies
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Our modules
from ca_sn_gen_models.models import AmpVAE
from ca_sn_gen_models.utils import superprint, normalize_data, train_svi, find_optimal_epsilon

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

#########################################################################################################
# ARGUMENTS
#########################################################################################################
# %%

# Parse arguments
parser = argparse.ArgumentParser(description='FFT VAE')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--fluo_noise', type=float, default=0.0, help='Level of noise to add to the data')
parser.add_argument('--beta', type=float, default=0.25, help='KL divergence weight')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--outdir', type=str, default='./output', help='Folder to save output files')
parser.add_argument('--latent_dim', type=int, default=10, help='Dimensionality of the latent space')
parser.add_argument('--norm', type=str, default='log1p', choices=['log1p', 'std', 'scale', 'none'], help='norm method')
parser.add_argument('--target_dist', type=str, default='LogNormal', choices=['LogNormal'], help='Target distribution for the VAE')

args = parser.parse_args()

# Summarize the options chosen
superprint(f"Options chosen: seed={args.seed}, beta={args.beta}, lr={args.lr}, norm={args.norm}, target_dist={args.target_dist}, outdir={args.outdir}")

# set seed and norm
lr = args.lr
seed = args.seed
beta = args.beta
norm = args.norm
outdir = args.outdir
latent_dim = args.latent_dim
fluo_noise = args.fluo_noise
target_dist = getattr(dist, args.target_dist)

# %%

# if interactive (do not comment out)
# seed = 0
# beta = 1
# lr = 1e-4
# norm = 'log1p'
# latent_dim = 200
# fluo_noise = 1.0
# target_dist = dist.LogNormal

#########################################################################################################
# Load and preprocess data
#########################################################################################################
#%%

superprint('Loading and preprocessing data...')

# vars
batch=512

# Set the random seed for reproducibility
np.random.seed(seed)

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

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################

# Start the timer
start_time = time.time()

# clear cuda memory
torch.cuda.empty_cache()

# Clear Pyro parameters
pyro.clear_param_store()

# define architecture
enc_dims = [2056, 512, 128]
dec_dims = [128, 512, 2056]
vae = AmpVAE(train_data.shape[1], latent_dim=latent_dim, enc_dims=enc_dims, dec_dims=dec_dims, device=device, target_dist=target_dist)

# Define optimizer
optimizer = ClippedAdam({"lr": 1e-4, "clip_norm": 1.0, "lrd": 0.999, "weight_decay": 1e-6})

# Use Pyro's Stochastic Variational Inference (SVI) for training
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# Train the model
training_params = {
    "num_epochs": 1000,         # Number of epochs
    "patience": 10,             # Patience for early stopping
    "min_delta": 1e-3,          # Minimum change in loss for early stopping
    "start_beta": beta,         # Initial KL divergence weight
    "end_beta": beta           # Final KL divergence weight
}

# Monitor peak RAM usage
process = psutil.Process()

# Call the function
avg_train_loss_trail, avg_val_loss_trail = vae.train_model(train_loader, val_loader, svi, **training_params)

# plot training and validation loss
plt.plot(range(len(avg_train_loss_trail)), avg_train_loss_trail, label='Train Loss')
plt.plot(range(len(avg_train_loss_trail)), avg_val_loss_trail, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

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
# Embedding of data
#########################################################################################################

# set model to evaluation mode
vae.eval()

# get latent representations for the data    
Ztrain = vae.encoder_mu(train_data.to(device))
Zval = vae.encoder_mu(val_data.to(device))

# TODO: comment out until next section

# # Perform UMAP dimensionality reduction
# reducer = umap.UMAP(n_components=2)
# Z_umap = reducer.fit_transform(Ztrain.cpu().detach().numpy())

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

#########################################################################################################
# Reconstruction of A
#########################################################################################################

superprint('Evaluating reconstruction...')

# get expected value of scaled a
a_mu = target_dist(vae.decoder_loc(Zval),vae.decoder_scale(Zval)).mean

# send a_mu to original scale
ahat = torch.expm1(a_mu).cpu().detach()

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
    axs[i].set_ylim(-1, 1)  # Set y-axis limits
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
clust_train, eps, ari_train = find_optimal_epsilon(Ztrain.cpu().detach(), labels[train_indices])
clust_val, eps, ari_val = find_optimal_epsilon(Zval.cpu().detach(), labels[val_indices])

# Calculate silhouette score
sil_train = silhouette_score(Ztrain.cpu().detach(), clust_train)
sil_val = silhouette_score(Zval.cpu().detach(), clust_val)

#########################################################################################################
# Save results
#########################################################################################################

# get distribution name
dist_name = target_dist.__name__.split('.')[-1]

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': ['FFT VAE'],
    'norm': [norm],
    'beta': [beta],
    'lr': [lr],
    'time': [training_time],
    'ram': [peak_memory],
    'latent_dim': [latent_dim],
    'target_dist': [dist_name],
    'num_epochs': [len(avg_train_loss_trail)],
    'last_train_loss': [avg_train_loss_trail[-1]],
    'last_val_loss': [avg_val_loss_trail[-1]],
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
results_file = f'{outdir}/results_summary_fftvae.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')