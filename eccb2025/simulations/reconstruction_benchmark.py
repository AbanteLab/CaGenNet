###############################################################################################
# dependancies
###############################################################################################
#%%
# Load deps
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set global font size
plt.rcParams.update({'font.size': 14})

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

# compute the MSE of PCA
def compute_pca_mse(x_train, x_val, xtilde_train, xtilde_val, n_components):
    
    # Create PCA object and fit it to the data
    pca = PCA(n_components=n_components)
    pca.fit(xtilde_train)

    # Transform the data into principal components
    xtilde_train_pca = pca.transform(xtilde_train)
    xtilde_val_pca = pca.transform(xtilde_val)

    # Reconstruct the data from the principal components
    xtrain_pca_reconstructed = pca.inverse_transform(xtilde_train_pca)
    xval_pca_reconstructed = pca.inverse_transform(xtilde_val_pca)

    # Compute MSE
    xtrain_pca_mse = np.mean(np.square(x_train - xtrain_pca_reconstructed))
    xval_pca_mse = np.mean(np.square(x_val - xval_pca_reconstructed))

    return xtrain_pca_mse, xval_pca_mse

def compute_vae_mse(input_dim, hidden_dim, latent_dim, mod_file, x_train, x_val, xtilde_train, xtilde_val):
    
    # load model
    superprint("Loading model")
    vae = VAE(input_dim, hidden_dim, latent_dim)
    vae.load_state_dict(torch.load(mod_file,weights_only=True))
    vae.eval()
    
    # move model to gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = vae.to(device)

    # Get the reconstructed training set
    superprint("Running VAE on training set")
    xtilde_train = torch.tensor(xtilde_train).to(device)
    xtrain_vae, _, _ = vae(xtilde_train)
    xtrain_vae = xtrain_vae.cpu().detach().numpy()

    # Get the reconstructed validation set
    superprint("Running VAE on validation set")
    xtilde_val = torch.tensor(xtilde_val).to(device)
    xval_vae, _, _ = vae(xtilde_val)
    xval_vae = xval_vae.cpu().detach().numpy()

    # get mse for validation set
    superprint("Computing MSE")
    xval_vae_mse = np.mean(np.square(x_val - xval_vae))
    xtrain_vae_mse = np.mean(np.square(x_train - xtrain_vae))
    
    return xtrain_vae_mse, xval_vae_mse

###############################################################################################
# IO
###############################################################################################
#%%

# dirs
bse_dir = "/pool01/projects/abante_lab/snDGM/eccb2025/simulations/set_3/AVAE_2_lstm_lin"

# output dirs
outdir = f"{bse_dir}/reconstruction/"

# mkdir if not exists
os.makedirs(outdir, exist_ok=True)
    
###############################################################################################
# MSE table
###############################################################################################
#%%

# mse table
mse_tab_file = f'{outdir}/mse_results.csv'

# check if table exists
if os.path.exists(mse_tab_file):
    
    # load table
    results_df = pd.read_csv(mse_tab_file)

else: 
    
    # Initialize results dictionary
    results = {
        'seed': [],
        'fluo_noise': [],
        'latent_dim': [],
        'hidden_dim': [],
        'xtrain_vae_mse': [],
        'xtrain_pca_mse': [],
        'xval_vae_mse': [],
        'xval_pca_mse': []
    }

    # Define parameter combinations
    latent_dims = [2, 4, 8, 16]
    hidden_dims = [32, 64, 128, 256]

    # loop over beta
    for beta in [0.0,0.5,1.0]:
        
        superprint(f"beta: {beta}")
        
        # iterate over different levels of noise
        for fluo_noise in ["0.5", "1.0", "1.5", "2.0", "2.5"]:
            
            superprint(f"Fluo noise: {fluo_noise}")
            
            # Loop over parameter combinations and seeds
            for seed in range(0, 10):
                
                superprint(f"Seed: {seed}")
                
                # Load groundtruth data
                gt_data_file = f"{bse_dir}/partitions/fluo_noise_0.0/partition__seed_{seed}.npz"
                gt_data = np.load(gt_data_file, allow_pickle=True)
                x_train = gt_data['xtrain']
                x_val = gt_data['xval']
                
                # Load noisy data
                noisy_data_file = f"{bse_dir}/partitions/fluo_noise_{fluo_noise}/partition__seed_{seed}.npz"
                noisy_data = np.load(noisy_data_file, allow_pickle=True)
                xtilde_train = noisy_data['xtrain']
                xtilde_val = noisy_data['xval']
                
                # dimensions
                input_dim = xtilde_train.shape[1]

                # iterate over different architectures
                for latent_dim, hidden_dim in zip(latent_dims, hidden_dims):
                    
                    superprint(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")
                
                    # Output files
                    mod_file = f"{bse_dir}/models/fluo_noise_{fluo_noise}/hidden_{hidden_dim}/latent_{latent_dim}/beta_{beta}/"
                    mod_file += f"avae_model__seed_{seed}.pth"
                    
                    # Check if model file exists
                    if not os.path.exists(mod_file):
                        superprint(f"Model file {mod_file} does not exist. Skipping...")
                        continue
                                
                    # Compute MSE for VAE
                    xtrain_vae_mse, xval_vae_mse = compute_vae_mse(input_dim, hidden_dim, latent_dim, mod_file, x_train, x_val, xtilde_train, xtilde_val)

                    # Compute MSE for PCA
                    xtrain_pca_mse, xval_pca_mse = compute_pca_mse(x_train, x_val, xtilde_train, xtilde_val, n_components=latent_dim)

                    # Store results
                    results['beta'].append(beta)
                    results['seed'].append(seed)
                    results['latent_dim'].append(latent_dim)
                    results['hidden_dim'].append(hidden_dim)
                    results['fluo_noise'].append(fluo_noise)
                    results['xval_vae_mse'].append(xval_vae_mse)
                    results['xval_pca_mse'].append(xval_pca_mse)
                    results['xtrain_vae_mse'].append(xtrain_vae_mse)
                    results['xtrain_pca_mse'].append(xtrain_pca_mse)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(mse_tab_file, index=False)
    
    superprint("Results saved to CSV")

###############################################################################################
# MSE plots
###############################################################################################
# %%

# plot percentage of winning between VAE and PCA in each case
results_df['vae_wins'] = results_df['xval_vae_mse'] <= results_df['xval_pca_mse']
results_df['pca_wins'] = results_df['xval_pca_mse'] < results_df['xval_vae_mse']

## polish training dataframe

# melt dataframe
value_vars = ['xtrain_vae_mse', 'xtrain_pca_mse']
id_vars = ['beta','fluo_noise','latent_dim', 'hidden_dim','seed']
melt_df = pd.melt(results_df, id_vars=id_vars, value_vars=value_vars, var_name='model', value_name='mse')

# give proper model names
melt_df['model'] = melt_df['model'].map({'xtrain_vae_mse':'VAE', 'xtrain_pca_mse':'PCA'})

# add corresponding beta to model name and create new column
melt_df['model_'] = melt_df['model'] + ' (beta=' + melt_df['beta'].astype(str) + ')'

# copy model_ column to model only when model is VAE
melt_df['model'] = np.where(melt_df['model'] == 'VAE', melt_df['model_'], melt_df['model'])

# drop fluo_noise=0.5
melt_df = melt_df[melt_df['fluo_noise'] != 0.5]

## training data 

# do violin plots
g = sns.FacetGrid(melt_df, col="fluo_noise", col_wrap=2, height=5, aspect=1.5)
g.map_dataframe(sns.violinplot, x='latent_dim', y='mse', hue='model', palette='Set3')
g.add_legend(title='Model', title_fontsize='13', fontsize='12', loc='upper center', ncol=4, bbox_to_anchor=(0.34, 1.05))
g.set_axis_labels('Latent Dimension', 'Mean Squared Error (MSE)')
g.set_titles(col_template='Gaussian Noise: {col_name}')
for ax in g.axes.flat:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, melt_df['mse'].max())
plt.tight_layout()
plt.savefig(f'{outdir}/mse_violin_plots_train.png')
plt.show()

## validation data

# polish validation dataframe

# melt dataframe
value_vars = ['xval_vae_mse', 'xval_pca_mse']
id_vars = ['beta','fluo_noise','latent_dim', 'hidden_dim','seed']
melt_df = pd.melt(results_df, id_vars=id_vars, value_vars=value_vars, var_name='model', value_name='mse')

# give proper model names
melt_df['model'] = melt_df['model'].map({'xval_vae_mse':'VAE', 'xval_pca_mse':'PCA'})

# add corresponding beta to model name and create new column
melt_df['model_'] = melt_df['model'] + ' (beta=' + melt_df['beta'].astype(str) + ')'

# copy model_ column to model only when model is VAE
melt_df['model'] = np.where(melt_df['model'] == 'VAE', melt_df['model_'], melt_df['model'])

# drop fluo_noise=0.5
melt_df = melt_df[melt_df['fluo_noise'] != 0.5]

# do violin plots
 

## compare best models

# compare best VAE model for each noise level and latent dimension with the corresponding PCA model
best_vae = results_df.loc[results_df.groupby(['fluo_noise','latent_dim'])['xval_vae_mse'].idxmin()]
    # NOTE: in all scenarios (noise,latent dim) the best VAE beats PCA

###############################################################################################
# Fit linear model to see if beta influences the reconstruction performance
###############################################################################################

# rescale mse to be more manageable for linear model
results_df['xval_vae_mse'] = results_df['xval_vae_mse'] * 1e6

# fit linear model on xval_vae_mse using beta, fluo_noise, and latent_dim
model = sm.OLS.from_formula('xval_vae_mse ~ beta + fluo_noise + latent_dim', data=results_df)

# extract beta coefficients and p-values
results = model.fit()
beta_coefs = results.params
beta_pvals = results.pvalues

# summarize in a table
beta_summary = pd.DataFrame({'beta': beta_coefs, 'p-value': beta_pvals})
beta_summary.to_csv(f'{outdir}/beta_summary.csv')

###############################################################################################
# Example plots
###############################################################################################
# %%

# # params of vae model
# seed=4
# beta=0.0
# batch_size=256
# latent_dim=8
# hidden_dim=128
# num_epochs=200
# learning_rate=0.0005

# # output files
# suff = f'latent_{latent_dim}_hidden_{hidden_dim}_epochs_{num_epochs}_rate_{learning_rate}_batch_{batch_size}_beta_{beta}_seed_{seed}'
# data_file = f'{outdir}/data_{suff}.npz'
# mod_file = f'{outdir}/vae_{suff}.pth'

# # load data
# data = np.load(data_file,allow_pickle=True)
# xtrain = data['xtrain']
# xval = data['xval']

# # Create PCA object and fit it to the data
# pca = PCA(n_components=4)
# pca.fit(xtrain)

# # Transform the data into principal components
# xtrain_pca = pca.transform(xtrain)
# xval_pca = pca.transform(xval)

# # Reconstruct the data from the principal components
# xtrain_pca_reconstructed = pca.inverse_transform(xtrain_pca)
# xval_pca_reconstructed = pca.inverse_transform(xval_pca)

# # load vae model
# input_dim = xtrain.shape[1]
# vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
# vae.load_state_dict(torch.load(mod_file))
# vae.eval()

# # reconstruct with vae
# xtrain_vae, _, _ = vae(torch.tensor(xtrain))
# xval_vae, _, _ = vae(torch.tensor(xval))
# xtrain_vae = xtrain_vae.detach().numpy()
# xval_vae = xval_vae.detach().numpy()

# # plot pca reconstruction for multiple signals
# plt.figure(figsize=(12, 6))
# plt.plot(xtrain[0], label="Noisy")
# plt.plot(xtrain_pca_reconstructed[0], label="PCA")
# plt.plot(xtrain_vae[0], label="AVAE")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Fluorescence")
# plt.title("Reconstruction of a single trace (Training Set)")
# plt.show()

# # plot pca reconstruction for multiple signals
# plt.figure(figsize=(12, 6))
# plt.plot(xval[0], label="Noisy")
# plt.plot(xval_pca_reconstructed[0], label="PCA")
# plt.plot(xval_vae[0], label="AVAE")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Fluorescence")
# plt.title("Reconstruction of a single trace (Validation Set)")
# plt.show()