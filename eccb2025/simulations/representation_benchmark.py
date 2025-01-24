###############################################################################################
# dependancies
###############################################################################################
#%%
# Load deps
import os
import umap
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN#, SpectralClustering
from sklearn.metrics import adjusted_rand_score

###############################################################################################
# functions
###############################################################################################
#%%
def superprint(message):
    # Get the current date and time
    now = datetime.now()
    # Format the date and time
    timestamp = now.strftime("[%Y-%m-%d %H:%M:%S]")
    # Print the message with the timestamp
    print(f"{timestamp} {message}")


def read_trace(path,thin=False):
    
    # read in 
    x = pd.read_csv(path,header=None,sep="\t")
    x = x.iloc[:, :30000]
    
    # thinning
    if thin:
        x = x.iloc[:, ::2]
    
    return(x)


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
    scatter = sns.scatterplot(data=x,x=f'{dim_red}1', y=f'{dim_red}2', hue=param, palette=pal, s=10)

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


def find_optimal_epsilon(min_eps, max_eps, x_emb, y):  
    epsilons = [round(i, 2) for i in np.arange(min_eps, max_eps, 0.01)]
    ari = []
    for epsilon in epsilons:
        clustering = DBSCAN(eps=epsilon, min_samples=50).fit(x_emb) 

        # Get cluster labels
        y['DBSCAN'] = clustering.labels_
        ari.append(adjusted_rand_score(y['Group'], y['DBSCAN']))
    opt_eps = epsilons[ari.index(max(ari))]
    return opt_eps, max(ari)


# Structure of the code
# 1 - Loop over combinations of parameters
# 2 - Load model
# 3 - Embedding with VAE
# 4 - Clustering with DBSCAN
# 5 - MSE clustering

# For PCA
# 1 - Load partitioned data
# 2 - PCA
# 3 - Clustering with DBSCAN
# 4 - MSE clustering


# FALTA
# 1 - Mirar de treure les labels de les dades de cada particio (Quan tenim seed=seed i es fa el merge de totes les X (al ppi de tot, que fa merge i despres particio per fer el training)
#  treure quines son les labels de cada particio i comparar el clustering amb aquestes labels)
# 2 - Calcular ARI per cada particio
# 2 - Fer violin plots
###############################################################################################
# Read in
###############################################################################################
#%%
superprint("Reading in results")

# dirs
bse_dir = "/pool01/projects/abante_lab/snDGM/eccb2025/simulations/set_3/AVAE_2_lstm_lin"

# output dirs
outdir = f"{bse_dir}/representation/"

# mkdir if not exists
os.makedirs(outdir, exist_ok=True)

###############################################################################################
# MSE table
###############################################################################################
#%%

# mse table
ari_tab_file = f'{outdir}/ari_results.csv'

# check if table exists
if os.path.exists(ari_tab_file):
    
    # load table
    results_df = pd.read_csv(ari_tab_file)

else: 
    
    # Initialize results dictionary
    results = {
        'beta': [],
        'seed': [],
        'fluo_noise': [],
        'latent_dim': [],
        'hidden_dim': [],
        'xval_vae_ari': [],
        'xval_pca_ari': [],
        'xtrain_vae_ari': [],
        'xtrain_pca_ari': [],
        'xval_vae_epsilon': [],
        'xval_pca_epsilon': [],
        'xtrain_vae_epsilon': [],
        'xtrain_pca_epsilon': []
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
                ytrain = gt_data['ytrain']
                yval = gt_data['yval']

                # convert to pandas
                metadata_cols = ["ROI","X","Y","E/I","Group","a","b","c","d","Sample","-"]
                ytrain = pd.DataFrame(ytrain,columns=metadata_cols)
                yval = pd.DataFrame(yval,columns=metadata_cols)
                
                # dimensions
                input_dim = x_train.shape[1]

                # iterate over different architectures
                for latent_dim, hidden_dim in zip(latent_dims, hidden_dims):
                    
                    superprint(f"Latent dim: {latent_dim}, Hidden dim: {hidden_dim}")

                    mod_file = f"{bse_dir}/models/fluo_noise_{fluo_noise}/hidden_{hidden_dim}/latent_{latent_dim}/beta_{beta}/"
                    mod_file += f"avae_model__seed_{seed}.pth"

                    # load model
                    superprint("Loading model")
                    vae = VAE(input_dim, hidden_dim, latent_dim)
                    vae.load_state_dict(torch.load(mod_file,weights_only=True))
                    vae.eval()

                    # Check if model file exists
                    if not os.path.exists(mod_file):
                        superprint(f"Model file {mod_file} does not exist. Skipping...")
                        continue

                    vae = vae.to('cpu')


                    # Embedding
                    superprint("Embedding")

                    # get VAE embedding for training data
                    xtrain_emb = vae.encode(torch.from_numpy(x_train))
                    xtrain_emb = xtrain_emb[0].detach().numpy()

                    # get VAE embedding for validation data
                    xval_emb = vae.encode(torch.from_numpy(x_val))
                    xval_emb = xval_emb[0].detach().numpy()


                    # Find optimal epsilon and maximum ari from DBSCAN clustering
                    xtrain_vae_epsilon, xtrain_vae_ari = find_optimal_epsilon(0.01, 0.5, xtrain_emb, ytrain)
                    xval_vae_epsilon, xval_vae_ari = find_optimal_epsilon(0.01, 0.5, xval_emb, yval)
                    # Print the optimal epsilon and ARI for training and validation sets
                    # superprint(f"Optimal epsilon for training set: {xtrain_vae_epsilon}, ARI: {xtrain_vae_ari}")
                    # superprint(f"Optimal epsilon for validation set: {xval_vae_epsilon}, ARI: {xval_vae_ari}")


                    # Create PCA object and fit it to the data
                    pca = PCA(n_components=latent_dim)
                    pca.fit(x_train)

                    # Transform the data into principal components
                    xtrain_pca = pca.transform(x_train) # embedding in the latent space
                    xval_pca = pca.transform(x_val)


                    xtrain_pca_epsilon, xtrain_pca_ari = find_optimal_epsilon(0.5, 2, xtrain_pca, ytrain)
                    xval_pca_epsilon, xval_pca_ari = find_optimal_epsilon(0.5, 2, xval_pca, yval)
                    # superprint(f"Optimal epsilon for training set: {xtrain_pca_epsilon}, ARI: {xtrain_pca_ari}")
                    # superprint(f"Optimal epsilon for validation set: {xval_pca_epsilon}, ARI: {xval_pca_ari}")



                    # Store results
                    results['beta'].append(beta)
                    results['seed'].append(seed)
                    results['latent_dim'].append(latent_dim)
                    results['hidden_dim'].append(hidden_dim)
                    results['fluo_noise'].append(fluo_noise)
                    results['xval_vae_ari'].append(xval_vae_ari)
                    results['xval_pca_ari'].append(xval_pca_ari)
                    results['xtrain_vae_ari'].append(xtrain_vae_ari)
                    results['xtrain_pca_ari'].append(xtrain_pca_ari)
                    results['xval_vae_epsilon'].append(xval_vae_epsilon)
                    results['xval_pca_epsilon'].append(xval_pca_epsilon)
                    results['xtrain_vae_epsilon'].append(xtrain_vae_epsilon)
                    results['xtrain_pca_epsilon'].append(xtrain_pca_epsilon)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results to CSV
    results_df.to_csv(ari_tab_file, index=False)
    
    superprint("Results saved to CSV")

# %%
###############################################################################################
# UMAP
###############################################################################################
#%%
# superprint("UMAP")
# suff = f'latent_{latent_dim}_hidden_{hidden_dim}_beta_{beta}'

# # get embedding for training data
# xtrain_umap = umap.UMAP().fit_transform(xtrain_pca)

# # merge with metadata
# ytrain["UMAP1"] = xtrain_umap[:,0]
# ytrain["UMAP2"] = xtrain_umap[:,1]

# # get embedding for validation data
# xval_umap = umap.UMAP().fit_transform(xval_pca)

# # merge with metadata
# yval["UMAP1"] = xval_umap[:,0]
# yval["UMAP2"] = xval_umap[:,1]

# # plot
# scatter_param(ytrain, "UMAP", "Group", discrete=True)
# # plt.savefig(f'{outdir}/umap_train_{suff}.png')
# scatter_param(yval, "UMAP", "Group", discrete=True)
# # plt.savefig(f'{outdir}/umap_val_{suff}.png')


# %%
# Load data from ari_results.csv
superprint("Loading ARI results")
ari_results_path = f'{outdir}/ari_results.csv'
ari_results_df = pd.read_csv(ari_results_path)

# # polish training dataframe

# # melt dataframe
# value_vars = ['xtrain_vae_ari', 'xtrain_pca_ari']
# id_vars = ['beta','fluo_noise','latent_dim', 'hidden_dim','seed']
# melt_df = pd.melt(ari_results_df, id_vars=id_vars, value_vars=value_vars, var_name='model', value_name='ari')

# # give proper model names
# melt_df['model'] = melt_df['model'].map({'xtrain_vae_ari':'VAE', 'xtrain_pca_ari':'PCA'})

# # add corresponding beta to model name and create new column
# melt_df['model_'] = melt_df['model'] + r' ($\beta_{KL}$=' + melt_df['beta'].astype(str) + ')'

# # copy model_ column to model only when model is VAE
# melt_df['model'] = np.where(melt_df['model'] == 'VAE', melt_df['model_'], melt_df['model'])

# # drop fluo_noise=0.5
# melt_df = melt_df[melt_df['fluo_noise'] != 0.5]


# # Create violin plots excluding the first value of the noise
# g = sns.FacetGrid(melt_df, col="fluo_noise", col_wrap=2, height=5, aspect=1.5)
# g.map_dataframe(sns.violinplot, x='latent_dim', y='ari', hue='model', palette='Set3')
# g.add_legend(title='Model', title_fontsize='13', fontsize='12', loc='upper center', ncol=4, bbox_to_anchor=(0.34, 1.05))
# g.set_axis_labels('Latent Dimension', 'Ajusted Rand Index (ARI)')
# g.set_titles(col_template='Gaussian Noise: {col_name}')
# for ax in g.axes.flat:
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.set_ylim(0, 1)
# # Add a vertical text label on the left side of the figure
# # fig = g.fig
# # fig.text(0.0001, 0.5, 'Adjusted Rand Index (ARI)', va='center', rotation='vertical', fontsize=24)
# plt.tight_layout()
# plt.savefig(f'{outdir}/ari_violin_plots_train.pdf')
# plt.show()



## validation data

# polish validation dataframe

# melt dataframe
value_vars = ['xval_vae_ari', 'xval_pca_ari']
id_vars = ['beta','fluo_noise','latent_dim', 'hidden_dim','seed']
melt_df = pd.melt(ari_results_df, id_vars=id_vars, value_vars=value_vars, var_name='model', value_name='ari')

# give proper model names
melt_df['model'] = melt_df['model'].map({'xval_vae_ari':'VAE', 'xval_pca_ari':'PCA'})

# add corresponding beta to model name and create new column
melt_df['model_'] = melt_df['model'] + ' (beta=' + melt_df['beta'].astype(str) + ')'

# copy model_ column to model only when model is VAE
melt_df['model'] = np.where(melt_df['model'] == 'VAE', melt_df['model_'], melt_df['model'])

# drop fluo_noise=0.5
melt_df = melt_df[melt_df['fluo_noise'] != 0.5]

# Create violin plots excluding the first value of the noise
g = sns.FacetGrid(melt_df, col="fluo_noise", col_wrap=2, height=5, aspect=1.5)
g.map_dataframe(sns.violinplot, x='latent_dim', y='ari', hue='model', palette='Set3')
g.add_legend(title='Model', title_fontsize='13', fontsize='12', loc='upper center', ncol=4, bbox_to_anchor=(0.34, 1.05))
g.set_axis_labels('Latent Dimension', 'Ajusted Rand Index (ARI)')
g.set_titles(col_template='Gaussian Noise: {col_name}')
for ax in g.axes.flat:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{outdir}/ari_violin_plots_val.pdf')
plt.show()
# %%
