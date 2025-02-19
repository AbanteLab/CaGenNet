####################################################################################################
# dependencies
####################################################################################################
#%%
import os
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# specific functions
from torch.utils.data import DataLoader, TensorDataset

# import all functions from utils.py
from sndgm.utils import ISOVAE
from sndgm.utils import superprint, load_data, normalize_data, train_isovae, loss_isovae

####################################################################################################
# main
####################################################################################################
#%%

# Define the main function
def main():
    
    # argument parser options
    parser = argparse.ArgumentParser(description="Train a Variational Autoencoder (VAE) on calcium imaging data.")
    parser.add_argument('data_path', type=str, help='Path to the CSV file containing the data')
    parser.add_argument('--normalize', type=bool, default=False, help='Whether to normalize the data (default: False)')
    parser.add_argument('--fs', type=int, default=20, help='Sampling frequency (default: 20)')
    parser.add_argument('--retrain', type=bool, default=False, help='Whether to retrain the model (default: False)')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the model (default: True)')
    parser.add_argument('--outdir', type=str, default='sndgm', help='Directory to save models, embeddings, and losses (default: sndgm)')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler (default: 10)')
    parser.add_argument('--latent', type=int, default=4, help='Dimension of latent space (default: 4)')
    parser.add_argument('--hidden', type=int, default=64, help='Dimension of hidden layer (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--batch', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default: 0)')
    parser.add_argument('--rate', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--rec_loss', type=str, choices=['mse', 'mae'], default='mae', help='Type of loss function to use (default: mse)')
    parser.add_argument('--bkl', type=float, default=1, help='KL divergence beta (default: 1)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change in validation loss to qualify as an improvement (default: 0.001)')
    
    # Parse arguments
    args = parser.parse_args()

    #%%
    
    # Access the arguments
    data_path = args.data_path      # data_path="/pool01/data/private/canals_lab/processed/calcium_imaging/hdrep/xf.csv.gz"
    seed = args.seed                # seed=0
    bkl = args.bkl                  # bkl=1
    retrain = args.retrain          # retrain=True
    save = args.save                # save=False
    batch = args.batch              # batch=32
    latent = args.latent            # latent=16
    hidden = args.hidden            # hidden=256
    epochs = args.epochs            # epochs=200
    rate = args.rate                # rate=0.005
    outdir = args.outdir            # outdir='/tmp'
    step_size = args.step_size      # step_size=10
    rec_loss = args.rec_loss        # rec_loss='mae'
    min_delta = args.min_delta      # min_delta=0.001
    fs = args.fs                    # fs=20
    
    # Define the dictionary for optional arguments
    optional_args = {'bkl': bkl, 'rec_loss': rec_loss, 'fs' : fs}
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
            
    # Load and preprocess data
    x = load_data(data_path)
    
    # normalize the data
    if args.normalize:
        x = normalize_data(x)
     
    # Get the indices of the data
    idx = np.arange(x.shape[0])
    
    # Split data into training and validation sets
    xtrain, xval, itrain, ival = train_test_split(x, idx, test_size=0.2, random_state=seed)
    
    # get magnitude and phase
    xtrain = torch.tensor(xtrain, dtype=torch.float32)
    xval = torch.tensor(xval, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(xtrain)
    val_dataset = TensorDataset(xval)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        superprint("Inference done on GPU")
    else:
        superprint("Inference done on CPU")
    
    # Initialize model, optimizer, and loss function
    model = ISOVAE(input_dim=xtrain.shape[1], hidden_dim=hidden, latent_dim=latent).to(device)
    
    # Define the optimizer
    superprint("Setting up optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    # Early stopping parameters
    counter = 0
    patience = 20
    best_val_loss = float('inf')

    # Training loop
    val_losses = []
    train_losses = []
    for epoch in range(epochs):
        
        # Training
        train_loss = train_isovae(model, train_loader, optimizer, device, epoch, **optional_args)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (xbatch,) in val_loader:
                xbatch = xbatch.to(device)
                mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p = model(xbatch)
                val_tot_loss, _, _, _, _, _, _ = loss_isovae(xbatch, mask, ahat, phat, mu_a, logvar_a, mu_p, logvar_p, epoch, **optional_args)
                val_loss += val_tot_loss.item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                superprint(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Step the scheduler at the end of the epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        superprint(f"Epoch: {epoch+1}/{epochs},  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Lr: {current_lr:.4f}")

    ## Save model
    if save:
        superprint("Saving model")
        os.makedirs(outdir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(outdir, "vae.pth"))

    ## Save embeddings of centered interval
    
    # Save embeddings to a txt.gz file
    superprint("Saving embeddings")
    mu_a,logvar_a = model.encode_amplitude(torch.tensor(x).to(device))
    mu_p,logvar_p = model.encode_phase(torch.tensor(x).to(device))
    za_df = pd.DataFrame(mu_a.cpu().detach().numpy())
    zp_df = pd.DataFrame(mu_p.cpu().detach().numpy())
    za_df.to_csv(os.path.join(outdir, "za_embeddings.txt.gz"), sep='\t', index=False, header=False)
    zp_df.to_csv(os.path.join(outdir, "zp_embeddings.txt.gz"), sep='\t', index=False, header=False)
        
    ## Save training and validation losses
    superprint("Saving training and validation losses")
    losses_df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
    losses_df.to_csv(os.path.join(outdir, "losses.txt.gz"), sep='\t', index=False)
    
    ## Save hyperparameters of the model
    superprint("Saving hyperparameters")
    hyperparameters_df = pd.DataFrame(optional_args, index=[0])
    hyperparameters_df.to_csv(os.path.join(outdir, "hyperparameters.txt.gz"), sep='\t', index=False)
    
    # print final message
    superprint("Training completed")
    
if __name__ == "__main__":
    main()
# %%
