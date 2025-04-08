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

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.model_selection import train_test_split

# clustering dependencies
from sklearn.metrics import silhouette_score

# Our modules
from ca_sn_gen_models.models import LstmVAE, train_clavae
from ca_sn_gen_models.utils import superprint, find_optimal_epsilon

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

#########################################################################################################
# ARGUMENTS
#########################################################################################################

# Parse arguments
parser = argparse.ArgumentParser(description='FFT VAE')
parser.add_argument(
    '--seed', 
    type=int, 
    default=0,
    help='Random seed'
)
parser.add_argument(
    '--beta', 
    type=float, 
    default=1.0, 
    help='KL divergence weight'
)
parser.add_argument(
    '--lr', 
    type=float, 
    default=2e-3, 
    help='Learning rate for the optimizer'
)
parser.add_argument(
    '--outdir', 
    type=str, 
    default='./output', 
    help='Folder to save output files'
)
parser.add_argument(
    '--latent_dim', 
    type=int, 
    default=10, 
    help='Dimensionality of the latent space'
)
parser.add_argument(
    '--fluo_noise', 
    type=float, 
    default=0.0, 
    help='Level of noise to add to the data'
)

args = parser.parse_args()

# Summarize the options chosen
superprint(f"Options chosen: seed={args.seed}, beta={args.beta}, lr={args.lr}, outdir={args.outdir}")

# transfer parameters to variables
lr = args.lr
seed = args.seed
beta = args.beta
outdir = args.outdir
latent_dim = args.latent_dim
fluo_noise = args.fluo_noise

# if interactive (do not comment out)
# seed = 0
# lr = 2e-3
# beta = 1.0
# latent_dim = 200
# fluo_noise = 1.0

#########################################################################################################
# Load and preprocess data
#########################################################################################################
#%%

superprint('Loading and preprocessing data...')

# Set the random seed for reproducibility
np.random.seed(seed)

# included groups. TODO: include all groups
group_set = [1, 2, 3] # [1, 2, 3]
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
data = torch.cat(data_list, dim=0)

# TODO: get every other timepoint
data = data[:, ::2]

# compute average signal for all xs
x_mean = data.mean(dim=1, keepdim=True)

#########################################################################################################
# SPLIT DATA
#########################################################################################################

superprint('(Not) normalizing data in time domain...')

batch = 256

# split and create data loader
train_indices, val_indices = train_test_split(np.arange(len(data)), test_size=0.2, random_state=seed)
train_data = data[train_indices]
val_data = data[val_indices]

# create datasets
train_dataset = TensorDataset(train_data, torch.tensor(train_indices))
val_dataset = TensorDataset(val_data, torch.tensor(val_indices))

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    
#########################################################################################################
# TRAIN
#########################################################################################################
superprint('Training model...')

# Start the timer
start_time = time.time()

# clear cuda memory
torch.cuda.empty_cache()

# Clear Pyro parameters
pyro.clear_param_store()

# define architecture
model = LstmVAE(train_data.shape[1], latent_dim*4, latent_dim)

# model model to device
model = model.to(device)

# Train the model
training_params = {
    "lr": lr,               # Learning rate
    "patience": 20,         # Patience for early stopping
    "min_delta": 1e-4,      # Minimum change in loss for early stopping
    "bkl": beta,            # KL divergence weight
    "broi": beta,           # Beta for contrastive loss
    "rec_loss": "mae",      # Reconstruction loss
    "num_epochs": 1000,     # Max number of epochs
    "model_type": "vae"     # Model type
}

# Monitor peak RAM usage
process = psutil.Process()

# Call the function
avg_train_loss_trail, avg_val_val_trail = train_clavae(model, train_loader, val_loader, device, **training_params)

# Get peak memory usage in MB
peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

# stop timer
end_time = time.time()
training_time = end_time - start_time
superprint(f'Training time: {training_time:.2f} seconds')

# plot training and validation loss
plt.plot(range(len(avg_train_loss_trail)), avg_train_loss_trail, label='Train Loss')
plt.plot(range(len(avg_val_val_trail)), avg_val_val_trail, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

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

# TODO: get every other timepoint
x0 = x0[:, ::2]

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

# clear cuda memory
torch.cuda.empty_cache()

# Remove gradient information from VAE parameters
for param in model.parameters():
    param.requires_grad = False
    
# set model to evaluation mode
model.eval()

# get latent representations for the data
xhat_train, Ztrain,_ = model(train_data.to(device))

Ztrain = Ztrain.cpu().detach().numpy()
xhat_train = xhat_train.cpu().detach().numpy()

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

# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_selection import mutual_info_classif

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
# high_mi_indices = ordered_indices[:latent_dim//4]
# low_mi_indices = ordered_indices[latent_dim//4:]
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

# get reconstruction of validation data
xhat_val, Zval, _ = model(val_data.to(device))
Zval = Zval.cpu().detach().numpy()
xhat_val = xhat_val.cpu().detach()

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(len(val_data), 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat_val[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_x = torch.nn.functional.l1_loss(x0_val, xhat_val, reduction='mean').item()
mse_x = torch.nn.functional.mse_loss(x0_val, xhat_val, reduction='mean').item()

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

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': ['AVAE'],
    'norm': [None],
    'beta': [beta],
    'lr': [lr],
    'time': [training_time],
    'ram': [peak_memory],
    'latent_dim': [latent_dim],
    'target_dist': [None],
    'num_epochs': [len(avg_train_loss_trail)],
    'last_train_loss': [avg_train_loss_trail[-1]],
    'last_val_loss': [avg_train_loss_trail[-1]],
    'mae_amp': [None],
    'mse_amp': [None],
    'mae_x': [mae_x],
    'mse_x': [mse_x],
    'ari_train': [ari_train],
    'ari_val': [ari_val],
    'sil_train': [sil_train],
    'sil_val': [sil_val]
})

#%%

# Append the dataframe to a text file
results_file = f'{outdir}/results_summary_avae.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
