#%%
import os
import umap
import argparse
import numpy as np
import pandas as pd
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

# cluster neurons based on latent variables
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Our modules
from sndgm.utils import superprint, load_data
from sndgm.models import AmpVAE, IcasspVAE, HierarchicalVAE, train_vae

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

# TODO:
# - A(f=0)=0?

#########################################################################################################
# ARGUMENTS
#########################################################################################################
# %%

# Parse arguments
parser = argparse.ArgumentParser(description='FFT VAE')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--beta', type=float, default=0.25, help='KL divergence weight')
parser.add_argument('--latent_dim', type=int, default=10, help='Dimensionality of the latent space')
parser.add_argument('--norm', type=str, default='log1p', choices=['log1p', 'std', 'scale', 'none'], help='norm method')
parser.add_argument('--target_dist', type=str, default='Laplace', choices=['Exponential', 'Gamma', 'LogNormal', 'Weibull'], help='Target distribution for the VAE')
parser.add_argument('--outdir', type=str, default='./output', help='Folder to save output files')

args = parser.parse_args()

# Summarize the options chosen
superprint(f"Options chosen: seed={args.seed}, beta={args.beta}, norm={args.norm}, target_dist={args.target_dist}, outdir={args.outdir}")

# set seed and norm
seed = args.seed
beta = args.beta
norm = args.norm
latent_dim = args.latent_dim
target_dist = getattr(dist, args.target_dist)

# %%

# if interactive
# seed = 0
# beta = 0.1
# norm = 'log1p'
# latent_dim = 32
# target_dist = dist.Weibull

#########################################################################################################
# Load and preprocess data
#########################################################################################################
#%%

# vars
batch=512
simulate=False

# Set the random seed for reproducibility
np.random.seed(seed)

if simulate:
    
    # directories
    data_dir = '/pool01/data/private/abante_lab/ca_img/sims_archetypes_hdrep/'

    # Load archetype labels
    y = pd.read_csv(data_dir + 'y.csv').values.flatten().tolist()
    
    # Load traces
    x_dict = np.load(data_dir + 'x.npy', allow_pickle=True).item()
    
    # load amplitude and phase
    a = torch.tensor(load_data(data_dir + 'a.npy'), dtype=torch.float32)
    p = torch.tensor(load_data(data_dir + 'p.npy'), dtype=torch.float32)

    # do real FFT of the data
    xfft = a * torch.exp(1j * p)
    x = torch.fft.irfft(xfft, n=2*a.shape[1])
    
else:

    # path
    data_path = '/pool01/data/private/canals_lab/processed/calcium_imaging/hdrep/xf.csv.gz'

    # Load and preprocess data
    x = torch.tensor(load_data(data_path), dtype=torch.float32)
    
    # # repeat x to get 10 repetitions
    # xaug = np.repeat(x, 10, axis=0)
    
    # # roll the signal by i samples in the first range
    # for i in range(0, 10):
        
    #     # get i-th interval
    #     irg = range(i*x.shape[0],(i+1)*x.shape[0])
        
    #     # roll the signal by i samples in the first range
    #     xaug[irg] = np.roll(xaug[irg], i*x.shape[0]//10, axis=1)
    
    # # add random noise to the data
    # xaug = xaug + np.random.normal(0, 0.5, xaug.shape)
    
    # do real FFT of the data
    xfft = torch.fft.rfft(x, axis=1)

    # get amplitude and phase
    a = torch.abs(xfft)
    p = torch.angle(xfft)

#########################################################################################################
# NORMALIZE AND SPLIT DATA
#########################################################################################################
#%%

# set scaling factor
scaling = 5000

# normalize the data
if norm == 'log1p':
    ascaled = torch.log1p(a)
elif norm == 'std':
    a_mean = a.mean(dim=0, keepdim=True)
    a_std = a.std(dim=0, keepdim=True)
    ascaled = (a - a_mean) / a
elif norm=='scale':
    ascaled = a / scaling
else:
    ascaled = a

# plot a histogram of a feature at random
plt.hist(ascaled[:, np.random.randint(ascaled.shape[1])], bins=100)
plt.title('Amplitude')
plt.show()

# Split data into training and validation sets
atrain, aval, ptrain, pval = train_test_split(ascaled, p, test_size=0.2, random_state=seed)

# Create datasets
train_dataset = TensorDataset(atrain, ptrain)
val_dataset = TensorDataset(aval, pval)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################
#%%

# clear cuda memory
torch.cuda.empty_cache()

# Clear Pyro parameters
pyro.clear_param_store()

# Choose model family
model_type = "AmpVAE"

# define architecture
enc_dims = [2056, 1028, 512, 256, 128]
dec_dims = [128, 256, 512, 1028, 2056]
if model_type == "IcasspVAE":
    vae = IcasspVAE(ascaled.shape[1], latent_dim=latent_dim, enc_dims=enc_dims, dec_dims=dec_dims, device=device, target_dist=target_dist)
elif model_type == "HierarchicalVAE":
    vae = HierarchicalVAE(ascaled.shape[1], latent_dim=latent_dim, enc_dims=enc_dims, dec_dims=dec_dims, device=device, target_dist=target_dist)
else:
    vae = AmpVAE(ascaled.shape[1], latent_dim=latent_dim, enc_dims=enc_dims, dec_dims=dec_dims, device=device, target_dist=target_dist)

# Move model to device
vae.to(device)

# Define optimizer
optimizer = ClippedAdam({"lr": 1e-4, "clip_norm": 2.0, "lrd": 0.999, "weight_decay": 1e-6})

# Use Pyro's Stochastic Variational Inference (SVI) for training
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

# Train the model
training_params = {
    "num_epochs": 1000,         # Number of epochs
    "batch": batch,             # Batch size
    "patience": 10,             # Patience for early stopping
    "min_delta": 1e-3,          # Minimum change in loss for early stopping
    "start_beta": beta,         # Initial KL divergence weight
    "end_beta": beta,           # Final KL divergence weight
    "device": device,           # Device
    "model_type": model_type    # Model type
}

# Call the function
avg_train_loss_trail, avg_val_loss_trail = train_vae(vae, svi, train_loader, val_loader, **training_params)

# plot training and validation loss
plt.plot(range(len(avg_train_loss_trail)), avg_train_loss_trail, label='Train Loss')
plt.plot(range(len(avg_train_loss_trail)), avg_val_loss_trail, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#########################################################################################################
# Embedding of data
#########################################################################################################
# %%

# set model to evaluation mode
vae.eval()

# get latent representations for the data    
Za = vae.encoder_mu(ascaled.to(device))

# plot tSNE of the latent variable
tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
Za_tsne = tsne.fit_transform(Za.cpu().detach().numpy())

# plot UMAP of the latent variable
umap_model = umap.UMAP(n_components=2)
Za_umap = umap_model.fit_transform(Za.cpu().detach().numpy())

if simulate:
    
    # Convert y to numerical values
    y_numerical = pd.factorize(y)[0]

    # plot tSNE and UMAP of the latent variable side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(Za_tsne[:, 0], Za_tsne[:, 1], c=y_numerical, cmap='viridis', s=5)
    axs[0].set_title('tSNE of Amplitude Latent Variable')
    axs[0].set_xlabel('tSNE Dimension 1')
    axs[0].set_ylabel('tSNE Dimension 2')
    scatter1 = axs[0].scatter(Za_tsne[:, 0], Za_tsne[:, 1], c=y_numerical, cmap='viridis', s=5)
    legend1 = axs[0].legend(*scatter1.legend_elements(), title="y")
    axs[0].add_artist(legend1)

    axs[1].scatter(Za_umap[:, 0], Za_umap[:, 1], c=y_numerical, cmap='viridis', s=5)
    axs[1].set_title('UMAP of Amplitude Latent Variable')
    axs[1].set_xlabel('UMAP Dimension 1')
    axs[1].set_ylabel('UMAP Dimension 2')
    scatter2 = axs[1].scatter(Za_umap[:, 0], Za_umap[:, 1], c=y_numerical, cmap='viridis', s=5)
    legend2 = axs[1].legend(*scatter2.legend_elements(), title="y")
    axs[1].add_artist(legend2)
    
    plt.show()

else:
    
    # plot tSNE and UMAP of the latent variable side by side
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].scatter(Za_tsne[:, 0], Za_tsne[:, 1], s=5)
    axs[0].set_title('tSNE of Amplitude Latent Variable')
    axs[0].set_xlabel('tSNE Dimension 1')
    axs[0].set_ylabel('tSNE Dimension 2')

    axs[1].scatter(Za_umap[:, 0], Za_umap[:, 1], s=5)
    axs[1].set_title('UMAP of Amplitude Latent Variable')
    axs[1].set_xlabel('UMAP Dimension 1')
    axs[1].set_ylabel('UMAP Dimension 2')
    
    plt.show()

#########################################################################################################
# Reconstruction of A
#########################################################################################################
# %%

# get expected value of scaled a
if target_dist == dist.Exponential:
    a_mu = dist.Exponential(vae.decoder_loc(Za)).mean
elif target_dist == dist.Gamma:
    a_mu = dist.Gamma(vae.decoder_loc(Za),vae.decoder_scale(Za)).mean
elif target_dist == dist.LogNormal:
    a_mu = dist.LogNormal(vae.decoder_loc(Za),vae.decoder_scale(Za)).mean
elif target_dist == dist.Weibull:
    a_mu = dist.Weibull(vae.decoder_loc(Za),vae.decoder_scale(Za)).mean
else:
    a_mu = vae.decoder_loc(Za)
    
# send a_mu to original scale
if norm == 'log1p':
    ahat = torch.exp(a_mu.cpu().detach()) - 1.0
    ahat = ahat.cpu().detach().numpy()
elif norm == 'std':
    a_std = a_std.cpu().detach().numpy()
    a_mean = a_mean.cpu().detach().numpy()
    ahat = a_std * a_mu.cpu().detach().numpy() + a_mean
elif norm == 'scale':
    ahat = scaling * a_mu.cpu().detach().numpy()
else:
    ahat = a_mu.cpu().detach().numpy()
    
# plot histogram of a and a_mu
plt.hist(ascaled.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label='Original', color='blue')
plt.hist(a_mu.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label='Reconstructed', color='orange')
plt.title('Amplitude')
plt.legend()
plt.xlim(-2, 2)
plt.show()

# plot cross-correlation matrix between a and a_mu
corr_matrix = np.corrcoef(a.cpu().detach().numpy(), ahat)
corr_matrix = corr_matrix[:a.shape[0], a.shape[0]:]

# get indeces corresponding to the comparison of a and a_mu
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title('Cross-Correlation Matrix between a and a_mu')
plt.xlabel('a')
plt.ylabel('ahat')
plt.show()

# Quantify similarity between original and reconstructed data
corr = np.diag(corr_matrix).mean()
superprint(f'Correlation: {corr}')

# # plot scatter plot of a and a_mu
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# axs[0].scatter(a.flatten(), ahat.flatten(), s=5)
# axs[0].set_title('Scatter Plot of a and a_mu')
# axs[0].set_xlabel('a')
# axs[0].set_ylabel('a_mu')

# # Create a density heatmap of a_mu vs a
# heatmap, xedges, yedges = np.histogram2d(a.flatten(), ahat.flatten(), bins=100)

# # Plot the heatmap
# extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
# axs[1].imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
# axs[1].set_title('Density Heatmap of a_mu vs a')
# axs[1].set_xlabel('a')
# axs[1].set_ylabel('a_mu')
# plt.colorbar(axs[1].imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis'), ax=axs[1])

# plt.show()

#########################################################################################################
# Reconstruction of X
#########################################################################################################
# %%
    
# do ifft using the reconstructed amplitude and original phase
xhat = torch.tensor(ahat) * torch.exp(1j * p)
xhat = torch.fft.irfft(xhat, n=x.shape[1])

# plot a few examples of the original and reconstructed data
fig, axs = plt.subplots(4, 2, figsize=(10, 8))

random_indices = np.random.choice(len(x), 8, replace=False)
for i, idx in enumerate(random_indices):
    axs[i // 2, i % 2].plot(x[idx], label='Original', color='blue')
    axs[i // 2, i % 2].plot(xhat[idx], label='Reconstructed', color='orange')
    axs[i // 2, i % 2].set_title(f'Sample {idx}')
    axs[i // 2, i % 2].set_xlabel('Time')
    axs[i // 2, i % 2].set_ylabel('Amplitude')
    axs[i // 2, i % 2].set_ylim(-0.1, 0.5)
    if i % 2 == 0:
        axs[i // 2, i % 2].legend()

plt.tight_layout()
plt.show()

#########################################################################################################
# Clustering
#########################################################################################################
# %%

# Determine the optimal number of clusters using the silhouette score
best_n_clusters = 2
silhouette_scores = []
best_silhouette_score = -1
range_n_clusters = list(range(2, 20))

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels = kmeans.fit_predict(Za.cpu().detach().numpy())
    silhouette_avg = silhouette_score(Za.cpu().detach().numpy(), cluster_labels)
    silhouette_scores.append(silhouette_avg)
    
    if silhouette_avg > best_silhouette_score:
        best_silhouette_score = silhouette_avg
        best_n_clusters = n_clusters

# plot silhouette scores
plt.plot(range_n_clusters, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range_n_clusters)  # Ensure x-axis has integer ticks
plt.show()

# Fit the final model with the optimal number of clusters
kmeans = KMeans(n_clusters=best_n_clusters, random_state=seed)
clusters = kmeans.fit_predict(Za.cpu().detach().numpy())

superprint(f"Optimal number of clusters: {best_n_clusters}")

# plot clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot tSNE with clusters
scatter1 = axs[0].scatter(Za_tsne[:, 0], Za_tsne[:, 1], c=clusters, s=5, cmap='tab20')
axs[0].set_title('tSNE of Amplitude Latent Variable')
axs[0].set_xlabel('tSNE Dimension 1')
axs[0].set_ylabel('tSNE Dimension 2')
legend1 = axs[0].legend(*scatter1.legend_elements(), title="Clusters")
axs[0].add_artist(legend1)

# Plot UMAP with clusters
scatter2 = axs[1].scatter(Za_umap[:, 0], Za_umap[:, 1], c=clusters, s=5, cmap='tab20')
axs[1].set_title('UMAP of Amplitude Latent Variable')
axs[1].set_xlabel('UMAP Dimension 1')
axs[1].set_ylabel('UMAP Dimension 2')
legend2 = axs[1].legend(*scatter2.legend_elements(), title="Clusters")
axs[1].add_artist(legend2)

plt.show()

# plot three examples of reconstructions from each cluster
fig, axs = plt.subplots(best_n_clusters, 3, figsize=(15, 2.5*best_n_clusters))

for i in range(best_n_clusters):
    cluster = i
    idx = np.where(clusters == cluster)[0]
    samples = np.random.choice(idx, 3, replace=False)
    for j, sample in enumerate(samples):
        axs[i, j].plot(x[sample], label='Original', color='blue')
        axs[i, j].plot(xhat[sample], label='Reconstructed', color='orange')
        axs[i, j].set_title(f'Cluster {cluster}')
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Amplitude')
        axs[i, j].set_ylim(-0.1, 1)
        if j == 0:
            axs[i, j].legend()

plt.tight_layout()
plt.show()

#########################################################################################################
# Save results
#########################################################################################################
# %%

# Calculate reconstruction error metrics
mae_a = torch.nn.functional.l1_loss(a, torch.tensor(ahat), reduction='mean').item()
mse_a = torch.nn.functional.mse_loss(a, torch.tensor(ahat), reduction='mean').item()

# calculate x reconstruction error metrics
mae_x = torch.nn.functional.l1_loss(x, xhat, reduction='mean').item()
mse_x = torch.nn.functional.mse_loss(x, xhat, reduction='mean').item()

# get distribution name
dist_name = target_dist.__name__.split('.')[-1]

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'norm': [norm],
    'beta': [beta],
    'latent_dim': [latent_dim],
    'target_dist': [dist_name],
    'num_epochs': [len(avg_train_loss_trail)],
    'last_train_loss': [avg_train_loss_trail[-1]],
    'last_val_loss': [avg_val_loss_trail[-1]],
    'mae_amp': [mae_a],
    'mse_amp': [mse_a],
    'corr_amp': [corr],
    'mae_x': [mae_x],
    'mse_x': [mse_x],
    'best_n_clusters': [best_n_clusters],
    'best_silhouette_score': [best_silhouette_score]
})

# Append the dataframe to a text file
results_file = f'{args.outdir}/results_summary.txt'
header = not os.path.exists(results_file)  # Check if file exists to determine if header is needed
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')

#########################################################################################################
# Embeddings of real data
#########################################################################################################
#%%

# # Load and preprocess data
# data_path = '/pool01/data/private/canals_lab/processed/calcium_imaging/hdrep/xf.csv.gz'

# # Load and preprocess data
# xreal = load_data(data_path)

# # do real FFT of the data
# xfft = torch.fft.rfft(torch.tensor(xreal, dtype=torch.float32), axis=1)

# # get amplitude and phase
# areal = torch.abs(xfft)
# preal = torch.angle(xfft)

# # scale amplitude vector
# areal_scaled = areal / scaling

# # use trained model to get latent representations
# if model_type == "IcasspVAE":
    
#     # get latent representations for the data        
#     Za_real = vae.encoder_a_mu(areal_scaled.to(device))
#     Zp_real = vae.encoder_p_mu(preal.to(device))

#     # get reconstruction of the data
#     a_mu_real = vae.decoder_a_mu(Za_real)
#     a_sigma_real = torch.exp(vae.decoder_a_logvar(Za_real))
#     p_mu_real = vae.decoder_p_mu(torch.cat((Zp_real, areal_scaled.to(device)), dim=1))
#     p_kappa_real = torch.exp(vae.decoder_p_kappa(torch.cat((Zp_real, areal_scaled.to(device)), dim=1)))
    
# elif model_type == "HierarchicalVAE":
    
#     # get latent representations for the data
#     ap_concat_real = torch.stack((areal_scaled, preal), dim=1)
#     Zs_real = vae.encoder_s_mu(ap_concat_real.to(device))
#     Za_real = vae.encoder_a_mu(Zs_real)
#     Zp_real = vae.encoder_p_mu(Zs_real)

#     # get reconstruction of the data
#     a_mu_real = vae.decoder_a_mu(Za_real)
#     a_sigma_real = torch.exp(vae.decoder_a_logvar(Za_real))
#     p_mu_real = vae.decoder_p_mu(Zp_real)
#     p_kappa_real = torch.exp(vae.decoder_p_kappa(Zp_real))
    
# elif model_type in ["ZinfLogNormVAE","LogNormVAE"]:
    
#     # get latent representations for the data
#     Za_real = vae.encoder_mu(areal_scaled.to(device))
    
#     # get reconstruction of the data
#     a_mu_real = torch.exp(vae.decoder_mu(Za_real))
#     a_sigma_real = torch.exp(vae.decoder_logvar(Za_real))

# elif model_type in ["GaussVAE"]:
    
#     # get latent representations for the data
#     Za_real = vae.encoder_mu(areal_scaled.to(device))
    
#     # get reconstruction of the data
#     a_mu_real = vae.decoder_mu(Za_real)
#     a_sigma_real = torch.exp(vae.decoder_logvar(Za_real))

# elif model_type in ["LapVAE"]:
    
#     # get latent representations for the data
#     Za_real = vae.encoder_mu(areal_scaled.to(device))
    
#     # get reconstruction of the data
#     a_mu_real = vae.decoder_loc(Za_real)
#     a_sigma_real = torch.exp(vae.decoder_scale(Za_real))

# elif model_type in ["ExpVAE"]:
    
#     # get latent representations for the data
#     Za_real = vae.encoder_mu(areal_scaled.to(device))
    
#     # get reconstruction of the data
#     a_mu_real = 1 / vae.decoder_mu(Za_real)
    
# # concatenate Za with Za_real
# Za_all = torch.cat((Za, Za_real), dim=0)

# # Determine the optimal number of clusters using the silhouette score
# range_n_clusters = list(range(2, 50))
# best_n_clusters = 2
# best_silhouette_score = -1

# for n_clusters in range_n_clusters:
    
#     kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
#     cluster_labels = kmeans.fit_predict(Za_all.cpu().detach().numpy())
#     silhouette_avg = silhouette_score(Za_all.cpu().detach().numpy(), cluster_labels)
    
#     superprint(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")
    
#     if silhouette_avg > best_silhouette_score:
#         best_silhouette_score = silhouette_avg
#         best_n_clusters = n_clusters

# # Fit the final model with the optimal number of clusters
# kmeans = KMeans(n_clusters=best_n_clusters, random_state=seed)
# clusters = kmeans.fit_predict(Za_all.cpu().detach().numpy())

# print(f"Optimal number of clusters: {best_n_clusters}")

# # create vector with labels from y and NA
# y_all = torch.cat((y, torch.full((areal.shape[0],), -1, dtype=torch.long)), dim=0)

# # plot umap of the latent variable
# umap_model = umap.UMAP(n_components=2)
# Za_all_umap = umap_model.fit_transform(Za_all.cpu().detach().numpy())

# # plot UMAP of the latent variable side by side with clusters
# fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# # Plot UMAP with y_all
# scatter1 = axs[0].scatter(Za_all_umap[:, 0], Za_all_umap[:, 1], c=y_all, cmap='Set2', s=5)
# axs[0].set_title('UMAP of Amplitude Latent Variable with y_all')
# axs[0].set_xlabel('UMAP Dimension 1')
# axs[0].set_ylabel('UMAP Dimension 2')
# legend1 = axs[0].legend(*scatter1.legend_elements(), title="y_all")
# axs[0].add_artist(legend1)

# # Plot UMAP with clusters
# scatter2 = axs[1].scatter(Za_all_umap[:, 0], Za_all_umap[:, 1], c=clusters, cmap='Set2', s=5)
# axs[1].set_title('UMAP of Amplitude Latent Variable with Clusters')
# axs[1].set_xlabel('UMAP Dimension 1')
# axs[1].set_ylabel('UMAP Dimension 2')
# legend2 = axs[1].legend(*scatter2.legend_elements(), title="Clusters")
# axs[1].add_artist(legend2)

# plt.show()

#########################################################################################################
# Reconstruction of real data
#########################################################################################################
#%%

# # reconstruct original data with reconstructed amplitude
# ahat = 2000 * a_mu_real.cpu().detach().numpy()

# # do ifft using the reconstructed amplitude and original phase
# xhat = ahat * np.exp(1j * preal.cpu().detach().numpy())
# xhat = torch.tensor(xhat, dtype=torch.complex64)
# xhat = torch.fft.irfft(xhat, n=2*areal.shape[1])

# # plot a few examples of the original and reconstructed data
# fig, axs = plt.subplots(4, 2, figsize=(10, 8))

# random_indices = np.random.choice(len(xreal), 8, replace=False)

# for i, idx in enumerate(random_indices):
#     axs[i // 2, i % 2].plot(xreal[idx], label='Original', color='blue')
#     axs[i // 2, i % 2].plot(xhat[idx], label='Reconstructed', color='orange')
#     axs[i // 2, i % 2].set_title(f'Sample {idx}')
#     axs[i // 2, i % 2].set_xlabel('Time')
#     axs[i // 2, i % 2].set_ylabel('Amplitude')
#     axs[i // 2, i % 2].set_ylim(-0.1, 0.5)
#     if i % 2 == 0:
#         axs[i // 2, i % 2].legend()

# plt.tight_layout()
# plt.show()

# # rank rows in xreal according to the MAE wrt xhat
# mae = torch.nn.functional.l1_loss(torch.tensor(areal_scaled), torch.tensor(ahat), reduction='none')
# mae = torch.mean(mae, dim=1)
# sorted_indices = torch.argsort(mae)

# # plot top 10 examples with lowest MAE
# fig, axs = plt.subplots(5, 2, figsize=(10, 10))

# for i in range(10):
#     axs[i // 2, i % 2].plot(xreal[sorted_indices[i]], label='Original', color='blue')
#     axs[i // 2, i % 2].plot(xhat[sorted_indices[i]], label='Reconstructed', color='orange')
#     axs[i // 2, i % 2].set_title(f'Sample {sorted_indices[i]}')
#     axs[i // 2, i % 2].set_xlabel('Time')
#     axs[i // 2, i % 2].set_ylabel('Amplitude')
#     axs[i // 2, i % 2].set_ylim(-0.5, 2)
#     if i % 2 == 0:
#         axs[i // 2, i % 2].legend()

# plt.suptitle('Highest MAE Examples')        
# plt.tight_layout()
# plt.show()

# # plot top 20 examples with highest MAE
# fig, axs = plt.subplots(10, 2, figsize=(10, 20))

# for i in range(20):
#     axs[i // 2, i % 2].plot(xreal[sorted_indices[-i-1]], label='Original', color='blue')
#     axs[i // 2, i % 2].plot(xhat[sorted_indices[-i-1]], label='Reconstructed', color='orange')
#     axs[i // 2, i % 2].set_title(f'Sample {sorted_indices[-i-1]}')
#     axs[i // 2, i % 2].set_xlabel('Time')
#     axs[i // 2, i % 2].set_ylabel('Amplitude')
#     axs[i // 2, i % 2].set_ylim(-1, 2)
#     if i % 2 == 0:
#         axs[i // 2, i % 2].legend()

# plt.suptitle('Lowest MAE Examples')
# plt.tight_layout()
# plt.show()

# %%
