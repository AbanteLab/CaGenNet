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
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Our modules
from ca_sn_gen_models.models import isoFA
from ca_sn_gen_models.utils import superprint, find_optimal_epsilon

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################
#%%

# Create the parser
parser = argparse.ArgumentParser(description="IsoFA training script")

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
    default=0.05,
    help="Learning rate (default: 0.05)"
)

parser.add_argument(
    "--lreg", 
    type=float, 
    required=False, 
    default=1e6,
    help="Lambda regularization Zx vs Zy (default: 1e6)"
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
lr = args.rate
seed = args.seed
outdir = args.outdir
lambda_reg = args.lreg
num_epochs = args.num_epochs
fluo_noise = args.fluo_noise
latent_dim = args.latent_dim

# TODO: Comment out the following lines if you want to run the script without command line arguments
seed = 1
lr = 0.05
lambda_reg=1e7
latent_dim = 200
fluo_noise = 1.0
num_epochs = 500

#########################################################################################################
# Testing
#########################################################################################################
#%%

## Simulated calcium traces

superprint('Reading in data...')

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

# create label combining group and sample
firing_labels = meta_df['firing_type'].values
group_labels = meta_df['group'].values
sample_labels = meta_df['sample'].values
group_sample_labels = np.array([f'{g}_{s}' for g, s in zip(group_labels, sample_labels)])

# Encode group_sample_labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(group_sample_labels)
encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)
unique_labels = np.unique(encoded_labels)
num_classes = len(unique_labels)

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

# plot three examples from each firing type firing_labels
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
for i, firing_type in enumerate(np.unique(firing_labels)):
    firing_indices = np.where(firing_labels == firing_type)[0]
    random_indices = np.random.choice(firing_indices, 3, replace=False)
    for j, idx in enumerate(random_indices):
        axs[i].plot(x[idx], alpha=0.5)
        axs[i].set_title(f"Firing Type {firing_type} Example {idx}")
        axs[i].set_xlabel("Frequency")
        axs[i].set_ylabel("Amplitude")
plt.tight_layout()
plt.show()

#########################################################################################################
# TESTING CATEGORICAL DISTRIBUTIONS
#########################################################################################################

# # Sample an n by k matrix and a k by r matrix
# n, k = 100, 50  # Example dimensions
# Zy = torch.randn(n, k)  # n by k matrix
# Wy = torch.randn(k, num_classes)  # k by r matrix

# # create a categorical distribution from the logits
# dist_test = dist.Categorical(logits=Zy @ Wy)
# dist_test.sample()

# # evaluate log likelihood
# pyro.sample("Y", dist.Categorical(logits=Zy @ Wy), obs=encoded_labels)

#########################################################################################################
# NORMALIZE AND SPLIT DATA
#########################################################################################################

superprint('Splitting data...')

# do not normalize data
data = x
norm = None

# split and create data loader
train_indices, val_indices = train_test_split(np.arange(len(data)), test_size=0.2, random_state=seed)

train_data = data[train_indices]
val_data = data[val_indices]

train_labels = encoded_labels[train_indices]
val_labels = encoded_labels[val_indices]

# create datasets
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

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
model = isoFA(data.shape[1], latent_dim, num_classes, device=device)

# Train the model
loss_history = model.train_model(train_loader, num_epochs=num_epochs, lr=lr, lambda_reg=lambda_reg)

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

#########################################################################################################
# EMBEDDING
#########################################################################################################

superprint('Getting posterior estimates...')

# Set model to evaluation mode
model.eval()

# get posterior values of Z, W, and sigma
Zx_loc, Zy_loc, Wy_loc, Wxy_loc, sigma_loc = model.get_posterior()
Zxy = np.concatenate((Zx_loc, Zy_loc), axis=-1)

# TODO: comment out until next section

# # Cross correlation between Zx and Zy
# Zx_centered = torch.tensor(Zx_loc) - torch.tensor(Zx_loc).mean(dim=0, keepdim=True)
# Zy_centered = torch.tensor(Zy_loc) - torch.tensor(Zy_loc).mean(dim=0, keepdim=True)
# std_Zx = Zx_centered.std(dim=0, keepdim=True)
# std_Zy = Zy_centered.std(dim=0, keepdim=True)
# covariance_matrix = (Zx_centered.T @ Zy_centered) / (train_data.shape[0] * std_Zx * std_Zy.T)

# # Plot covariance matrix using a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(covariance_matrix.cpu().numpy(), fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
# plt.title("Covariance Matrix Heatmap")
# plt.xlabel("Zx Latent Variables")
# plt.ylabel("Zy Latent Variables")
# plt.tight_layout()
# plt.show()

# # Perform UMAP dimensionality reduction
# reducer = umap.UMAP(n_components=2)
# Zxy_umap = reducer.fit_transform(Zxy)
# Zx_umap = reducer.fit_transform(Zx_loc)
# Zy_umap = reducer.fit_transform(Zy_loc)

# # get labels for training data
# group_labels = meta_df['group'].values[train_indices]
# sample_labels = meta_df['sample'].values[train_indices]
# firing_type_labels = meta_df['firing_type'].values[train_indices]
# mean_firing_label = x_mean[train_indices].squeeze(1).cpu().numpy()

# # Create a DataFrame for easier plotting with seaborn
# umap_df = pd.DataFrame({
#     'Zx_UMAP1': Zx_umap[:, 0],
#     'Zx_UMAP2': Zx_umap[:, 1],
#     'Zy_UMAP1': Zy_umap[:, 0],
#     'Zy_UMAP2': Zy_umap[:, 1],
#     'Zxy_UMAP1': Zxy_umap[:, 0],
#     'Zxy_UMAP2': Zxy_umap[:, 1],
#     'Group': group_labels,
#     'Sample': sample_labels,
#     'FiringType': firing_type_labels,
#     'MeanFiring': mean_firing_label
# })

# # Create a 3x2 panel for plotting
# fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# # Plot Zx colored by Group and shaped by Sample
# sns.scatterplot(
#     ax=axs[0, 0],
#     data=umap_df,
#     x='Zx_UMAP1',
#     y='Zx_UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# axs[0, 0].set_title("Zx: Colored by Group, Shaped by Sample")
# axs[0, 0].set_xlabel("UMAP Dimension 1")
# axs[0, 0].set_ylabel("UMAP Dimension 2")

# # Plot Zx colored by Firing Type and shaped by Group
# sns.scatterplot(
#     ax=axs[0, 1],
#     data=umap_df,
#     x='Zx_UMAP1',
#     y='Zx_UMAP2',
#     hue='FiringType',
#     style='Group',
#     palette='Set2',
#     s=50
# )
# axs[0, 1].set_title("Zx: Colored by Firing Type, Shaped by Group")
# axs[0, 1].set_xlabel("UMAP Dimension 1")
# axs[0, 1].set_ylabel("UMAP Dimension 2")

# # Plot Zy colored by Group and shaped by Sample
# sns.scatterplot(
#     ax=axs[1, 0],
#     data=umap_df,
#     x='Zy_UMAP1',
#     y='Zy_UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# axs[1, 0].set_title("Zy: Colored by Group, Shaped by Sample")
# axs[1, 0].set_xlabel("UMAP Dimension 1")
# axs[1, 0].set_ylabel("UMAP Dimension 2")

# # Plot Zy colored by Firing Type and shaped by Group
# sns.scatterplot(
#     ax=axs[1, 1],
#     data=umap_df,
#     x='Zy_UMAP1',
#     y='Zy_UMAP2',
#     hue='FiringType',
#     style='Group',
#     palette='Set2',
#     s=50
# )
# axs[1, 1].set_title("Zy: Colored by Firing Type, Shaped by Group")
# axs[1, 1].set_xlabel("UMAP Dimension 1")
# axs[1, 1].set_ylabel("UMAP Dimension 2")

# # Plot Zxy colored by Group and shaped by Sample
# sns.scatterplot(
#     ax=axs[2, 0],
#     data=umap_df,
#     x='Zxy_UMAP1',
#     y='Zxy_UMAP2',
#     hue='Group',
#     style='Sample',
#     palette='Set1',
#     s=50
# )
# axs[2, 0].set_title("Zxy: Colored by Group, Shaped by Sample")
# axs[2, 0].set_xlabel("UMAP Dimension 1")
# axs[2, 0].set_ylabel("UMAP Dimension 2")

# # Plot Zxy colored by Firing Type and shaped by Group
# sns.scatterplot(
#     ax=axs[2, 1],
#     data=umap_df,
#     x='Zxy_UMAP1',
#     y='Zxy_UMAP2',
#     hue='FiringType',
#     style='Group',
#     palette='Set2',
#     s=50
# )
# axs[2, 1].set_title("Zxy: Colored by Firing Type, Shaped by Group")
# axs[2, 1].set_xlabel("UMAP Dimension 1")
# axs[2, 1].set_ylabel("UMAP Dimension 2")

# # Adjust layout and show the plot
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

# # Compute mutual information between Zx and encoded labels
# mi_zx_label = mutual_info_classif(Zx_loc, encoded_labels)

# # Compute mutual information between Zy and encoded labels
# mi_zy_label = mutual_info_classif(Zy_loc, encoded_labels)

# # Create a DataFrame for plotting
# mi_df = pd.DataFrame({
#     'Mutual Information': np.concatenate([mi_zx_label, mi_zy_label]),
#     'Latent Space': ['Zx'] * len(mi_zx_label) + ['Zy'] * len(mi_zy_label)
# })

# # Reshape mutual information data for heatmap
# mi_matrix = np.array([mi_zx_label, mi_zy_label])

# # Create a heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(mi_matrix, cmap='coolwarm', xticklabels=[f'Z{i+1}' for i in range(len(mi_zx_label))], yticklabels=['Zx', 'Zy'])
# plt.title('Heatmap of Mutual Information between Z and Sample ID')
# plt.xlabel('Latent Variables')
# plt.ylabel('Latent Space')
# plt.tight_layout()
# plt.show()

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
model.eval()

# get posterior value of Z given X
Zxy_val = val_data @ model.Hhat.cpu()
Zx_val = Zxy_val[:, :latent_dim]
Zy_val = Zxy_val[:, latent_dim:]

# reconstruct data
xhat = Zxy_val @ Wxy_loc

# set error metrics of amplitude to None
mae_a = None
mse_a = None

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(len(val_data), 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_x = torch.nn.functional.l1_loss(x0_val, xhat, reduction='mean').item()
mse_x = torch.nn.functional.mse_loss(x0_val, xhat, reduction='mean').item()

#########################################################################################################
# Clustering
#########################################################################################################
#%%

superprint('Evaluating embedding...')

# find optimal epsilon for training data
clust_train, eps, ari_train = find_optimal_epsilon(Zx_loc, firing_labels[train_indices])
clust_val, eps, ari_val = find_optimal_epsilon(Zx_val, firing_labels[val_indices])

# Calculate silhouette score
sil_train = silhouette_score(Zx_loc, clust_train)
sil_val = silhouette_score(Zx_val, clust_val)

# TODO: comment out until next section when running benchmark

# # cluster using KMeans
# from sklearn.cluster import KMeans

# # cluster in Zx using KMeans
# nclust = 3
# kmeans = KMeans(n_clusters=nclust, random_state=seed)
# clust_train = kmeans.fit_predict(Zx_loc)

# # plot 3 examples from each cluster
# fig, axs = plt.subplots(nclust, 3, figsize=(15, 18), sharex=True, sharey=True)
# for i in range(nclust):
#     cluster_indices = np.where(clust_train == i)[0]
#     random_indices = np.random.choice(cluster_indices, 3, replace=False)
#     for j, ax in zip(random_indices, axs[i]):
#         ax.plot(train_data[j], alpha=0.7)
#         ax.set_title(f"Cluster {i} Example")
#         ax.set_xlabel("Frequency")
#         ax.set_ylabel("Amplitude")
# plt.tight_layout()
# plt.show()

# # Plot umap of Zx with clusters
# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     x=Zx_umap[:, 0],
#     y=Zx_umap[:, 1],
#     hue=clust_train,
#     palette='Set1',
#     s=10
# )
# plt.title("UMAP of Zx with KMeans Clusters")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # take mean of Zxy for each cluster
# cluster_means = []
# for i in range(len(np.unique(clust_train))):
#     cluster_indices = np.where(clust_train == i)[0]
#     cluster_mean = Zxy[cluster_indices].mean(axis=0)
#     mean_xhat_clust = cluster_mean @ Wxy_loc
#     cluster_means.append(mean_xhat_clust)

# # calculate the mean and variance of the clusters
# cluster_means_trace = np.mean(np.array(cluster_means), axis=1)
# cluster_vars_trace = np.var(np.array(cluster_means), axis=1)

# # do a rolling average to smooth the data
# window_size = 50
# cluster_means = [np.convolve(cluster_mean, np.ones(window_size)/window_size, mode='same') for cluster_mean in cluster_means]
# # plot the mean of each cluster
# plt.figure(figsize=(10, 6))
# for i, cluster_mean in enumerate(cluster_means):
#     plt.plot(cluster_mean, label=f'Cluster {i}')
# plt.title("Cluster Means")
# plt.xlabel("Frequency")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.tight_layout()
# plt.show()

#########################################################################################################
# Save results
#########################################################################################################

superprint('Storing summary...')

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': ['IsoFA'],
    'norm': [norm],
    'beta': [None],
    'lr': [lr],
    'time': [training_time],
    'ram': [peak_memory],
    'latent_dim': [latent_dim],
    'target_dist': [None],
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
results_file = f'{outdir}/results_summary_model_isoeffa.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
