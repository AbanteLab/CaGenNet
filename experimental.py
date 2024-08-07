###############################################################################################
# dependancies
###############################################################################################
#%%
# Load deps
import os
import torch # type: ignore
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch.nn as nn # type: ignore
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore

###############################################################################################
# arguments
###############################################################################################
#%%
# Create the parser
parser = argparse.ArgumentParser(description="Trains a VAE on calcium data.")

# Add arguments
parser.add_argument(
    "-l",
    "--latent", 
    type=int, 
    required=False, 
    default=32,
    help="Dimension of latent space (default: 32)"
)

parser.add_argument(
    "--hidden", 
    type=int, 
    required=False, 
    default=256,
    help="Dimension of hidden layer (default: 256)"
)

parser.add_argument(
    "-e",
    "--epochs", 
    type=int, 
    required=False, 
    default=50,
    help="Number of training epochs (default: 50)"
)

parser.add_argument(
    "-b",
    "--batch", 
    type=int, 
    required=False, 
    default=16,
    help="Batch size (default: 16)"
)

parser.add_argument(
    "-r",
    "--rate", 
    type=float, 
    required=False, 
    default=0.001,
    help="Learning rate (default: 0.001)"
)

parser.add_argument(
    "--beta_kl", 
    type=float, 
    required=False, 
    default=1,
    help="KL divergence beta (default: 1)"
)

parser.add_argument(
    "--retrain", 
    type=bool, 
    required=False, 
    default=False,
    help="Whether to retrain the model (default: False)"
)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
beta = args.beta_kl         # beta=1
retrain = args.retrain      # retrain=True
batch_size = args.batch     # batch_size=256
latent_dim = args.latent    # latent_dim=32
hidden_dim = args.hidden    # hidden_dim=512
num_epochs = args.epochs    # num_epochs=200
learning_rate = args.rate   # learning_rate=0.002

#%%
# beta=1
# batch_size=256
# latent_dim=32
# hidden_dim=512
# num_epochs=200
# learning_rate=0.002

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

#%%
# test vae
# vae = VAE(1000, 50, 10)
# x = torch.randn([32, 1000])
# mu, logvar = vae.encode(x)
# z = vae.sample_z(mu,logvar)
# xhat = vae.decode(z)
    
###############################################################################################
# IO
###############################################################################################
#%%
superprint("Checking for existant model")

# output files
indir = "/media/HDD_4TB_1/jordi/calcium_imaging/tau/"
suff = f'latent_{latent_dim}_hidden_{hidden_dim}_epochs_{num_epochs}_rate_{learning_rate}_batch_{batch_size}_beta_{beta}'
mod_file = f'{indir}/vae_{suff}.pkl'
curve_file = f'{indir}/vae_{suff}.txt'
data_file = f'{indir}/data_{suff}.npz'

# check if model file exists
if os.path.exists(mod_file) and not retrain:
    
    superprint("Model file already exists")
    exit(0)

###############################################################################################
# Read in
###############################################################################################
#%%
superprint("Reading in data")
    
# Read in
ctl_file = indir + "DIV12_P2_Well_12_traces_smoothed_everything.csv"
treat_file = indir + "DIV12_P2_Well_42_traces_smoothed_everything.csv"
ctl_data = pd.read_csv(ctl_file)
treat_data = pd.read_csv(treat_file)


# rename time column
ctl_data.rename(columns={'time (s)': 'time'}, inplace=True)
treat_data.rename(columns={'time (s)': 'time'}, inplace=True)

# subsample data
ctl_data = ctl_data.iloc[::2, :]
treat_data = treat_data.iloc[::2, :]

# get dimensions
ntp = len(ctl_data)

# get timepoint vector
tpvec = ctl_data.iloc[:,0]

# convert to numpy array
ctl_data = np.array(ctl_data.iloc[:,1:],dtype=np.float32)
treat_data = np.array(treat_data.iloc[:ntp,1:],dtype=np.float32)

# transpose
ctl_data = ctl_data.T
treat_data = treat_data.T

# get number of cells
nctl = ctl_data.shape[0]
ntrt = treat_data.shape[0]

# reshape data
ctl_data = np.reshape(ctl_data,(nctl,ntp))
treat_data = np.reshape(treat_data,(ntrt,ntp))

# concatenate
xdata = np.concatenate((ctl_data,treat_data),axis=0)
ydata = np.concatenate((np.array([0] * nctl), np.array([1] * ntrt)))

# normalization
xdata = (xdata-xdata.min())/(xdata.max()-xdata.min()) - 0.5

# work with float32
xdata = xdata.astype(np.float32)

# split the data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(xdata, ydata, test_size=0.25, random_state=42)

superprint(f"Training data: {xtrain.shape}")
superprint(f"Validation data: {xval.shape}")

# adapt shape of validation (xtrain is taken care of by loader)
xval = torch.from_numpy(xval)
    
# store data
np.savez(data_file, xtrain=xtrain, xval=xval, ytrain=ytrain, yval=yval)

# create dataset object
dataset = MyDataset(xtrain)

# Create a DataLoader object
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# cuda config
torch.cuda.set_per_process_memory_fraction(0.95, device=torch.device('cuda:0'))

###############################################################################################
# training
###############################################################################################
#%%
superprint("Setting up model")

# dimensions
input_dim = xtrain.shape[1]

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type=='cuda':
    superprint("Inference done on GPU")
    
    # clear cache
    torch.cuda.empty_cache()
    
    # move input data to the GPU
    xval = xval.to(device)
    
else:
    superprint("Inference done on CPU")

# init model
superprint("Initializing VAE")
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# print amount of used memory
superprint(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")

# Define the optimizer
superprint("Setting up optimizer")
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

# Define the maximum norm for gradient clipping
max_grad_norm = 1.0

# Early stopping parameters
counter = 0
patience = 5
min_delta = 0.005
best_val_loss = float('inf')

# store losses
val_loss = np.array([])
train_loss = np.array([])
train_kl_loss = np.array([])
train_rec_loss = np.array([])

#%%

# Training loop
superprint("Training starts now")
vae.train()  # Set the model in training mode
for epoch in range(num_epochs):
    
    # define aux variables
    running_loss = 0.0
    running_kl_loss = 0.0
    running_rec_loss = 0.0
    running_loss_val = 0.0
    
    # iterate over batches
    for batch_idx, inputs in enumerate(data_loader):
        
        # move input data to the GPU
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward pass
        reconstructed_x, mu, logvar = vae(inputs)
        
        # average reconstruction error across samples in batch (equivalent to Gaussian model)
        rec_err = torch.sum(abs(reconstructed_x-inputs)**2) / batch_size
        running_rec_loss += rec_err.detach().cpu().numpy()
        
        # calculate KL divergence D(q(z|x)|p(z))
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        running_kl_loss += kl_divergence.detach().cpu().numpy()
        
        # compute the training loss
        loss = rec_err + beta * kl_divergence
        
        # backward pass
        loss.backward()
        
        # apply gradient clipping
        clip_grad_norm_(vae.parameters(), max_grad_norm)
        
        # update the weights
        optimizer.step()
        
        # accumulate the loss
        running_loss += loss.item()
        
        # empty cache
        if device.type=='cuda':
            torch.cuda.empty_cache()
        
    ## validation loss after all batches are done
        
    # forward pass
    reconstructed_x, mu, logvar = vae(xval)
    
    # average reconstruction error across samples in batch (equivalent to Gaussian model)
    rec_err = torch.sum(abs(reconstructed_x-xval)**2) / reconstructed_x.shape[0]
    rec_err = rec_err.detach().cpu().numpy()
    
    # calculate KL divergence D(q(z|x)|p(z))
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / reconstructed_x.shape[0]
    kl_divergence = kl_divergence.detach().cpu().numpy()
    
    # compute the loss - reconstruction loss has a much larger scale
    loss = rec_err + beta * kl_divergence
    
    # accumulate validation loss
    running_loss_val += loss.item()
        
    # Compute the average loss for the epoch
    mean_loss = running_loss / len(data_loader)
    train_loss = np.append(train_loss,mean_loss)
    
    # Compute the average reconstruction loss for the epoch
    mean_rec_loss = running_rec_loss / len(data_loader)
    train_rec_loss = np.append(train_rec_loss,mean_rec_loss)
    
    # Compute the average kl loss for the epoch
    mean_kl_loss = running_kl_loss / len(data_loader)
    train_kl_loss = np.append(train_kl_loss,mean_kl_loss)
    
    # Compute the average validation loss for the epoch
    mean_loss_val = running_loss_val / len(data_loader)
    val_loss = np.append(val_loss,mean_loss_val)

    # Step the scheduler at the end of the epoch
    scheduler.step()

    # Print the learning rate (optional)
    current_lr = optimizer.param_groups[0]['lr']
    str_print = f"Epoch: {epoch+1}/{num_epochs}, Lr: {mean_rec_loss:.2f}, Lkl: {mean_kl_loss:.2f}, Lt: {mean_loss:.2f}, lr: {current_lr:.4f}"
    superprint(str_print)
    
    # Check for early stopping
    if mean_loss_val < best_val_loss - min_delta:
        best_val_loss = mean_loss_val
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            superprint(f'Early stopping triggered at epoch {epoch+1}')
            break
    
# Save the trained VAE to a file
with open(mod_file, 'wb') as file:
    pickle.dump(vae, file)

# store training curves
train_curves = pd.DataFrame({"train_loss":train_loss,"val_loss":val_loss,"train_rec_loss":train_rec_loss,"train_kl_loss":train_kl_loss})
train_curves.to_csv(curve_file,index=False)

# %%
