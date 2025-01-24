###############################################################################################
# dependancies
###############################################################################################
#%%
# Load deps
import os
import umap
import glob
import torch
import pickle
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset
import torch.nn.functional as F # type: ignore
from sklearn.cluster import DBSCAN#, SpectralClustering
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

###############################################################################################
# functions
###############################################################################################

def superprint(message):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    # Print the message with the timestamp
    print(f"{timestamp} {message}")

def read_roi(path,sample,group):
    
    # read in 
    x = pd.read_csv(path)
    
    # assign sample and group
    x["sample"] = sample
    x["group"] = group
    
    return(x)

def read_trace(path,thin=False):
    
    # read in 
    x = pd.read_csv(path,header=None,sep="\t")
    x = x.iloc[:, :30000]
    
    # thinning
    if thin:
        x = x.iloc[:, ::2]
    
    return(x)

# Create a custom PyTorch Dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Assuming your data is already in the desired format
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a single data sample at the specified index
        sample = self.data[idx]
        return sample

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
        
        # x.shape = (batch_size, input_dim)
        # x.unsqueeze(1).shape = (batch_size, 1, input_dim)
        # out.shape = (batch_size, 1, hidden_dim)
        # hd.shape = (1, batch_size, hidden_dim)
        # hd.squeeze(0).shape = (batch_size, hidden_dim)
        
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
            # NOTE: randn_like means shape like std but still is a N(0,1)

        # return z
        return mu + eps * std
    
    def decode(self, z):
        
        # z.shape = (batch_size, latent_dim)
        # out_l1.shape = (batch_size, hidden_dim)
        # out_l1.unsqueeze(1).shape = (batch_size, 1, hidden_dim)
        # out_l2.shape = (batch_size, input_dim)
        
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

def scatter_param(x,dim_red,param,discrete=False):
    
    # Plot
    plt.figure(figsize=(12, 8))
    pal = 'Set2' if discrete else 'viridis'
    scatter = sns.scatterplot(data=x,x=f'{dim_red}1', y=f'{dim_red}2', hue=param, palette=pal, s=50)

    if not discrete:
        # Add color bar
        norm = plt.Normalize(x[param].min(), x[param].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        scatter.get_legend().remove()
        cbar = plt.colorbar(sm)
        cbar.set_label(param, rotation=270, labelpad=15)

    # Customize the plot
    plt.xlabel(f'{dim_red}1')
    plt.ylabel(f'{dim_red}2')
    
    return scatter

###############################################################################################
# IO
###############################################################################################
#%%
# dirs
bse_dir = "/media/HDD_4TB_1/jordi/calcium_imaging/simulations/set_2/"
rs_dir = f"{bse_dir}/Flat_RS/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_1.0/"
ch_dir = f"{bse_dir}/Flat_CH/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_1.0/"
ib_dir = f"{bse_dir}/Flat_IB/L_0.1/EI_ratio_0.8/Iz_param_noise_0.05/dt_0.001/fluo_noise_1.0/"
outdir = f"{bse_dir}/bVAE_2_lstm_lin/"

# Set global font size
plt.rcParams.update({'font.size': 14})

###############################################################################################
# ROI
###############################################################################################

superprint("Loading metadata")

# ROI files
roi_chat_1_files = [f"{ch_dir}/Flat_CH_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_chat_1 = [read_roi(roi_chat_1_files[i],f"{i+1}","CH") for i in range(0,5)]

# merge dataframes
roi_chat_1 = pd.concat(roi_chat_1, axis=0)

# ROI files
roi_int_1_files = [f"{ib_dir}/Flat_IB_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_int_1 = [read_roi(roi_int_1_files[i],f"{i+1}","IB") for i in range(0,5)]

# merge dataframes
roi_int_1 = pd.concat(roi_int_1, axis=0)

# ROI files
roi_reg_1_files = [f"{rs_dir}/Flat_RS_L_0.1_rois_{i}.txt" for i in range(1,6)]

# read in
roi_reg_1 = [read_roi(roi_reg_1_files[i],f"{i+1}","RS") for i in range(0,5)]

# merge dataframes
roi_reg_1 = pd.concat(roi_reg_1, axis=0)

# concatenate
roi_df = pd.concat([roi_chat_1,roi_int_1,roi_reg_1])

# reset index of ROI df
roi_df.reset_index(inplace=True)

###############################################################################################
# ROI plots
###############################################################################################
#%%

# Create a 1x2 panel plot
fig, axes = plt.subplots(2, 1, figsize=(4, 6.5))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# palette
groups = roi_df['group'].unique()
custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']
color_mapping = dict(zip(groups, custom_palette))

# Define the parameters to plot
sns.scatterplot(data=roi_df, x='a', y='b', hue='group', palette=custom_palette, ax=axes[0], legend=False)
axes[0].set_xlabel(r'$a$')
axes[0].set_ylabel(r'$b$')
sns.scatterplot(data=roi_df, x='v0', y='delta_u', hue='group', palette=custom_palette, ax=axes[1], legend=False)
axes[1].set_xlabel(r'$v_0$')
axes[1].set_ylabel(r'$\Delta u$')

# Adjust layout
plt.tight_layout()

# Save the figure before showing it
plt.savefig(f'{outdir}/izikhievic_params.png')

###############################################################################################
# Traces
###############################################################################################
#%%
superprint("Loading traces")

# files
trace_chat_1_files = [f"{ch_dir}/Flat_CH_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_chat_1 = [read_trace(trace_chat_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_chat_1 = pd.concat(trace_chat_1, axis=0)

# files
trace_int_1_files = [f"{ib_dir}/Flat_IB_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_int_1 = [read_trace(trace_int_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_int_1 = pd.concat(trace_int_1, axis=0)

# files
trace_reg_1_files = [f"{rs_dir}/Flat_RS_L_0.1_calcium_{i}.txt.gz" for i in range(1,6)]

# read in
trace_reg_1 = [read_trace(trace_reg_1_files[i],thin=True) for i in range(0,5)]

# merge dataframes
trace_reg_1 = pd.concat(trace_reg_1, axis=0)

# concatenate
trace_df = pd.concat([trace_chat_1,trace_int_1,trace_reg_1])

###############################################################################################
# Plots of traces
###############################################################################################
# %%

# Select 4 random rows for each group
samples_per_group = 3

selected_indices = []
for group in groups:
    group_indices = roi_df[roi_df['group'] == group].index
    selected_indices.extend(np.random.choice(group_indices, samples_per_group, replace=False))

selected_trace_df = trace_df.iloc[selected_indices]
selected_roi_df = roi_df.iloc[selected_indices]

# Plot in a 4x3 grid layout
fig, axes = plt.subplots(3, 3, figsize=(8, 5), sharex=True, sharey=True)
axes = axes.flatten()
tpvec = 2*np.linspace(1,len(selected_trace_df.columns),len(selected_trace_df.columns)) / 10**3

for i, (idx, row) in enumerate(selected_roi_df.iterrows()):
    ax = axes[i]
    ax.plot(tpvec, trace_df.iloc[idx], color=color_mapping[row["group"]])
    ax.set_xticks([0,10,20,30])
    
plt.tight_layout()

# Save the figure before showing it
plt.savefig(f'{outdir}/raw_trace_examples.pdf')

###############################################################################################
# Loss tables
###############################################################################################
# %%

# Define the lists
latent = [8, 16, 32, 64]
hidden = [128, 256, 512, 1024]
beta = [0.0, 0.25, 0.5, 0.75, 1.0]

# Create an empty list to store the DataFrames
results_list = []

# Iterate over all combinations
for dz, dh in zip (latent, hidden):
    
    for b in beta:
    
        # Define the suffix and file path
        suff = f'latent_{dz}_hidden_{dh}_epochs_200_rate_0.0005_batch_256_beta_{b}'
        curve_file = f'{outdir}/vae_{suff}.txt'
        
        try:
            # Read the table
            tab = pd.read_csv(curve_file)
            
            # iterations until convergence
            iters = tab.shape[0]
            
            # Get the last line
            last_line = tab.iloc[-1]
            
            # Create a DataFrame for the last line and add dz, dh, and b
            last_line_df = pd.DataFrame(last_line).T
            last_line_df['latent'] = dz
            last_line_df['hidden'] = dh
            last_line_df['beta'] = b
            last_line_df['iters'] = iters
            
            # Append the last line DataFrame to the list
            results_list.append(last_line_df)
        except FileNotFoundError:
            print(f'File {curve_file} not found.')
        except pd.errors.EmptyDataError:
            print(f'File {curve_file} is empty.')

# Concatenate all the DataFrames in the list
results_df = pd.concat(results_list, ignore_index=True)

# Reorder columns: make sure dz, dh, and b are the first three columns
# Extract current columns
cols = results_df.columns.tolist()

# Reorder: 'latent', 'hidden', 'beta' first, then remaining columns
desired_order = ['latent', 'hidden', 'beta', 'iters'] + [col for col in cols if col not in ['latent', 'hidden', 'beta']]
results_df = results_df[desired_order]

# Round numerical values to 2 decimal places
results_df = results_df.round(2)

# Save the results to a CSV file
results_df.to_csv(f'{outdir}/loss_table.csv', index=False)

###############################################################################################
# Reconstruction error plots
###############################################################################################
# %%

# Group data by the combination of latent and hidden dimensions
grouped = results_df.groupby(['latent', 'hidden'])

# Set up a figure with 2 subplots side by side
fig, axes = plt.subplots(2, 1, figsize=(4, 6.5))

# Define a color map
colors = sns.color_palette("Set1", 3)

# Plotting Reconstruction Loss
for (key, group), color in zip(grouped, colors):
    axes[0].plot(group['beta'], group['train_rec_loss'], marker='o', linestyle='-', color=color, label=f'$D_1$={key[1]}, $D_2$={key[0]}')

axes[0].set_xlabel(r'$\beta$', fontsize=15)  # 1.5x larger font size
axes[0].set_ylabel(r'$L_R$', fontsize=15)    # 1.5x larger font size
axes[0].legend(title='Architecture', title_fontsize=15, fontsize=12)  # 1.5x larger font size for title and items
axes[0].grid(True)

# Plotting KL Divergence Loss
for (key, group), color in zip(grouped, colors):
    axes[1].plot(group['beta'], group['train_kl_loss'], marker='o', linestyle='-', color=color, label=f'$D_1$={key[1]}, $D_2$={key[0]}')

axes[1].set_xlabel(r'$\beta$', fontsize=15)  # 1.5x larger font size
axes[1].set_ylabel(r'$L_{KL}$', fontsize=15)  # 1.5x larger font size
axes[1].legend(title='Architecture', title_fontsize=15, fontsize=12)  # 1.5x larger font size for title and items
axes[1].grid(True)

# Adjust layout
fig.tight_layout()

# Save the figure before showing it
plt.savefig(f'{outdir}/loss_vs_beta.pdf')

###############################################################################################
# Learned representation
###############################################################################################
# %%

from sklearn.decomposition import PCA

# params of best configuration
beta=0.25
batch_size=256
latent_dim=8
hidden_dim=128
num_epochs=200
learning_rate=0.0005

# output files
suff = f'latent_{latent_dim}_hidden_{hidden_dim}_epochs_{num_epochs}_rate_{learning_rate}_batch_{batch_size}_beta_{beta}'
mod_file = f'{outdir}/vae_{suff}.pkl'
curve_file = f'{outdir}/vae_{suff}.txt'
data_file = f'{outdir}/data_{suff}.npz'

# load data
data = np.load(data_file,allow_pickle=True)
xtrain = data['xtrain']
xval = data['xval']
ytrain = data['ytrain']
yval = data['yval']

# convert to pandas
metadata_cols = ["ROI","X","Y","E/I","Group","a","b","c","d","Sample","-"]
ytrain = pd.DataFrame(ytrain,columns=metadata_cols)
yval = pd.DataFrame(yval,columns=metadata_cols)

# load model
with open(mod_file, 'rb') as file:
    vae = pickle.load(file)
    
# %%

# Forward pass
ztrain, _ = vae.encode(torch.tensor(xtrain).to(device="cuda"))
ztrain = ztrain.detach().cpu().numpy()
zval, _ = vae.encode(torch.tensor(xval).to(device="cuda"))
zval = zval.detach().cpu().numpy()

# Calculate the autocorrelation matrix
autocorr_matrix = np.corrcoef(ztrain, rowvar=False)

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(autocorr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Autocorrelation Matrix Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")

# Create PCA object and fit it to the data
pca = PCA(n_components=None)  # n_components=None keeps all components
pca.fit(xtrain)

# Transform the data into principal components
xtrain_pca = pca.transform(xtrain)

# To get the explained variance ratio of each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# To get the principal components themselves
components = pca.components_

# merge with metadata
ytrain["PCA1"] = xtrain_pca[:,0]
ytrain["PCA2"] = xtrain_pca[:,1]

scatter_param(ytrain, "PCA", "c", discrete=False)
scatter_param(ytrain, "PCA", "d", discrete=False)
scatter_param(ytrain, "PCA", "E/I", discrete=True)
scatter_param(ytrain, "PCA", "Group", discrete=True)

# %%

# fit tSNE
tsne_train = TSNE(n_components=2, random_state=20)
xtrain_tsne = tsne_train.fit_transform(ztrain)

# merge with metadata
ytrain["tSNE1"] = xtrain_tsne[:,0]
ytrain["tSNE2"] = xtrain_tsne[:,1]

scatter_param(ytrain, "tSNE", "c", discrete=False)
plt.savefig(f'{outdir}/tsne_c.png')
scatter_param(ytrain, "tSNE", "d", discrete=False)
plt.savefig(f'{outdir}/tsne_d.png')
scatter_param(ytrain, "tSNE", "E/I", discrete=True)
plt.savefig(f'{outdir}/tsne_ei.png')
scatter_param(ytrain, "tSNE", "Group", discrete=True)
plt.savefig(f'{outdir}/tsne_group.png')

# %%

# best clustering for PCA space
ari_max = 0
for e in np.linspace(0.5,2.0,20):

    # Create DBSCAN clustering model
    pca_clust = DBSCAN(eps=e, min_samples=20).fit(xtrain_pca)

    # PCA metrics
    if adjusted_rand_score(ytrain['Group'], pca_clust.labels_)>ari_max:
        
        # store results
        ari_pca = adjusted_rand_score(ytrain['Group'], ytrain['pca_clust'])
        nmi_pca = normalized_mutual_info_score(ytrain['Group'], ytrain['pca_clust'])
        ami_pca = adjusted_mutual_info_score(ytrain['Group'], ytrain['pca_clust'])
        homogeneity_pca = homogeneity_score(ytrain['Group'], ytrain['pca_clust'])
        completeness_pca = completeness_score(ytrain['Group'], ytrain['pca_clust'])
        v_measure_pca = v_measure_score(ytrain['Group'], ytrain['pca_clust'])
        
        # update max
        ari_max = ari_pca

        # Get cluster labels
        ytrain['pca_clust'] = pca_clust.labels_
        
# best clustering for Z space
ari_max = 0
for e in np.linspace(0.05,0.25,20):
    
    # Create DBSCAN clustering model
    z_clust = DBSCAN(eps=e, min_samples=20).fit(ztrain)

    # PCA metrics
    if adjusted_rand_score(ytrain['Group'], z_clust.labels_)>ari_max:
        
        # store results
        ari_z = adjusted_rand_score(ytrain['Group'], ytrain['z_clust'])
        nmi_z = normalized_mutual_info_score(ytrain['Group'], ytrain['z_clust'])
        ami_z = adjusted_mutual_info_score(ytrain['Group'], ytrain['z_clust'])
        homogeneity_z = homogeneity_score(ytrain['Group'], ytrain['z_clust'])
        completeness_z = completeness_score(ytrain['Group'], ytrain['z_clust'])
        v_measure_z = v_measure_score(ytrain['Group'], ytrain['z_clust'])
        
        # update max
        ari_max = ari_z

        # Get cluster labels
        ytrain['z_clust'] = z_clust.labels_

scatter_param(ytrain, "PCA", "pca_clust", discrete=True)
plt.savefig(f'{outdir}/pca_clust.pdf')
scatter_param(ytrain, "tSNE", "z_clust", discrete=True)
plt.savefig(f'{outdir}/z_clust.pdf')

# Create a DataFrame to hold these metrics
metrics_df = pd.DataFrame({
    'Metric': ['Adjusted Rand Index', 'Normalized Mutual Information', 'Adjusted Mutual Information',
               'Homogeneity', 'Completeness', 'V-measure'],
    'PCA': [ari_pca, nmi_pca, ami_pca, homogeneity_pca, completeness_pca, v_measure_pca],
    'Z': [ari_z, nmi_z, ami_z, homogeneity_z, completeness_z, v_measure_z]
})

metrics_df

scatter_param(ytrain, "tSNE", "Group", discrete=True)
plt.savefig(f'{outdir}/tsne_z_group.png')
scatter_param(ytrain, "tSNE", "z_clust", discrete=True)
plt.savefig(f'{outdir}/tsne_z_clust.png')
scatter_param(ytrain, "tSNE", "c", discrete=False)
plt.savefig(f'{outdir}/tsne z_c.png')
scatter_param(ytrain, "tSNE", "d", discrete=False)
plt.savefig(f'{outdir}/tsne_z_d.png')
scatter_param(ytrain, "tSNE", "E/I", discrete=True)
plt.savefig(f'{outdir}/tsne_z_ei.png')

# %%

for d in range(ztrain.shape[1]):
    
    ytrain[f'z_{d+1}'] = ztrain[:,d]

set1 = ytrain[["a","b","c","d"]].astype(float)
set1.columns = [r"$a$",r"$b$",r"$v_0$",r"$\Delta_u$"]
set2 = ytrain[["z_1","z_2","z_3","z_4","z_5","z_6","z_7","z_8"]].astype(float)
set2.columns = [r"$z_1$",r"$z_2$",r"$z_3$",r"$z_4$",r"$z_5$",r"$z_6$",r"$z_7$",r"$z_8$"]

# Compute pairwise correlations between features of set1 and set2
correlations = pd.DataFrame(index=set1.columns, columns=set2.columns)

for col1 in set1.columns:
    for col2 in set2.columns:
        print(col2)
        correlations.loc[col1, col2] = set1[col1].corr(set2[col2])

# Plotting the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlations.astype(float), annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel(r"$Z$",fontsize=15)
plt.ylabel("Parameters of dynamical model",fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.savefig(f'{outdir}/corrmat_z_params.pdf')

# %%
