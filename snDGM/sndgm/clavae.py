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
from sndgm.utils import superprint, load_data, augment_data, normalize_data, VAE, train_clavae, loss_function

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
    parser.add_argument('--retrain', type=bool, default=False, help='Whether to retrain the model (default: False)')
    parser.add_argument('--save', type=bool, default=True, help='Whether to save the model (default: True)')
    parser.add_argument('--model_type', type=str, choices=['avae', 'clavae'], default='clavae', help='Type of model to train (default: clavae)')
    parser.add_argument('--outdir', type=str, default='sndgm', help='Directory to save models, embeddings, and losses (default: sndgm)')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for the learning rate scheduler (default: 10)')
    parser.add_argument('--latent', type=int, default=4, help='Dimension of latent space (default: 4)')
    parser.add_argument('--hidden', type=int, default=64, help='Dimension of hidden layer (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (default: 200)')
    parser.add_argument('--batch', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default: 0)')
    parser.add_argument('--rate', type=float, default=0.005, help='Learning rate (default: 0.005)')
    parser.add_argument('--rec_loss', type=str, choices=['mse', 'mae'], default='mse', help='Type of loss function to use (default: mse)')
    parser.add_argument('--bkl', type=float, default=1, help='KL divergence beta (default: 1)')
    parser.add_argument('--broi', type=float, default=10, help='ROI penalty beta (default: 10)')
    parser.add_argument('--roi_metric', type=str, choices=['cosine', 'euclidean'], default='cosine', help='roi_metric used in ROI penalty (default: cosine)')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change in validation loss to qualify as an improvement (default: 0.001)')
    
    # Parse arguments
    args = parser.parse_args()

    #%%
    
    # Access the arguments
    data_path = args.data_path      # data_path='/tmp/test.txt.gz'
    seed = args.seed                # seed=0
    bkl = args.bkl                  # bkl=1
    broi = args.broi                # broi=10
    retrain = args.retrain          # retrain=True
    save = args.save                # save=False
    batch = args.batch              # batch=32
    latent = args.latent            # latent=4
    hidden = args.hidden            # hidden=64
    epochs = args.epochs            # epochs=200
    rate = args.rate                # rate=0.005
    outdir = args.outdir            # outdir='/tmp'
    step_size = args.step_size      # step_size=10
    rec_loss = args.rec_loss        # rec_loss='mae'
    roi_metric = args.roi_metric    # roi_metric='cosine'
    model_type = args.model_type    # model_type='clavae'
    
    # Define the dictionary for optional arguments
    optional_args = {
        'bkl': bkl, 
        'broi': broi, 
        'rec_loss': rec_loss, 
        'roi_metric': roi_metric,
        'model_type': model_type
    }
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
            
    # Load and preprocess data
    data = load_data(data_path)
    
    # Augment the data if need be
    if model_type == 'clavae':
        superprint("Performing data augmentation for CLAVAE model")
        reps = 10
        length = 6000
        data,idx = augment_data(data,reps,length, seed=seed)
    else:
        superprint("AVAE model does not require data augmentation")
        idx = np.arange(data.shape[0])
        
    # normalize the data
    if args.normalize:
        data = normalize_data(data)
    
    # Split data into training and validation sets
    train_data, val_data, train_idx, val_idx = train_test_split(data, idx, test_size=0.2, random_state=seed)
    
    # flatten the data
    train_data = train_data.reshape(-1, train_data.shape[-1])
    val_data = val_data.reshape(-1, val_data.shape[-1])
    train_idx = train_idx.reshape(-1)
    val_idx = val_idx.reshape(-1)
    
    # Create datasets
    train_dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_idx))
    val_dataset = TensorDataset(torch.tensor(val_data), torch.tensor(val_idx))
    
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
    model = VAE(input_dim=train_data.shape[1], latent_dim=latent, hidden_dim=hidden).to(device)
    
    # Define the optimizer
    superprint("Setting up optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)

    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    # Early stopping parameters
    counter = 0
    patience = 10
    best_val_loss = float('inf')

    # Training loop
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        
        # Training
        train_loss = train_clavae(model, train_loader, optimizer, device, **optional_args)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for fluo, roi in val_loader:
                fluo = fluo.to(device)
                recon_batch, mu, logvar = model(fluo)
                val_tot_loss, _, _, _ = loss_function(recon_batch, fluo, mu, logvar, roi, **optional_args)
                val_loss += val_tot_loss.item()
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Check for early stopping
        if val_loss < best_val_loss - args.min_delta:
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
    
    # Load original data
    superprint("Saving embeddings")
    
    if model_type=='clavae':
        
        # load and augment the data
        reps = 10
        length = 6000
        fluo = load_data(data_path)
        fluo,idx = augment_data(fluo,reps,length, seed=seed)
        
        # keep only the centered interval of the data with 5,000 points
        num_cols = fluo.shape[1]
        center_start = (num_cols - length) // 2
        center_end = center_start + length
        fluo = torch.tensor(fluo[:, center_start:center_end])
        fluo = fluo.to(device)
    
        # Save embeddings to a txt.gz file
        embeddings = model.encode(fluo)[0]
        embeddings_df = pd.DataFrame(embeddings.cpu().detach().numpy())
        embeddings_df.to_csv(os.path.join(outdir, "embeddings.txt.gz"), sep='\t', index=False, header=False)
        
    else:
        
        # Save embeddings to a txt.gz file
        embeddings = model.encode(data.to(device))[0]
        embeddings_df = pd.DataFrame(embeddings.cpu().detach().numpy())
        embeddings_df.to_csv(os.path.join(outdir, "embeddings.txt.gz"), sep='\t', index=False, header=False)
        
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
