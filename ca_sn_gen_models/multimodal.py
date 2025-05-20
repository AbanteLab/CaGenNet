# torch
import torch
import torch.nn as nn

# pyro
import pyro
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO

from ca_sn_gen_models.utils import superprint

class MultimodalElbo(Trace_ELBO):
    
    def __init__(self, lreg=1e5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lreg = lreg

    def loss_zdist(self, zcal, zrna):
        
        # return L1 distance
        return self.lreg * torch.linalg.vector_norm(zcal - zrna, dim=1).sum()

    def loss(self, model, guide, *args, **kwargs):
        """ Calcula la pèrdua ELBO + regularització """
        
        loss = super().loss(model, guide, *args, **kwargs)  # ELBO normal
        superprint(f"ELBO loss: {loss:.4f}")
        
        # get zcal and zrna from the guide
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        zcal = guide_trace.nodes["zcal"]["value"]
        zrna = guide_trace.nodes["zrna"]["value"]
        
        # compute the extra loss
        extra_loss = self.loss_zdist(zcal, zrna)
        superprint(f"L(d(Zcal,Zrna)): {extra_loss:.4f}")
        
        return loss + extra_loss

class MultiModMlpVAE(nn.Module):
    
    def __init__(self, CalD, RnaD, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.device = device
        self.CalD = CalD            # number of time steps in Ca imaging data
        self.RnaD = RnaD            # number of genes in RNA-seq data
        self.K = K                  # latent dimension
        
        ## Calcium imaging data
        
        # define encoder
        self.encoder_cal_mlp = nn.Sequential(
            nn.Linear(CalD, CalD // 2), nn.ReLU(),
            nn.Linear(CalD // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_cal_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_cal_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(K, CalD // 8), nn.ReLU(), 
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
            )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))
        
        # scRNA data
        
        # define encoder
        self.encoder_rna_mlp = nn.Sequential(
            nn.Linear(RnaD, RnaD // 2), nn.ReLU(),
            nn.Linear(RnaD // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_rna_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_rna_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(K, RnaD // 8), nn.ReLU(), 
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
            )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))
        
        # Move model to device
        self.to(self.device)

    def encode_cal(self, x):
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process the combined representation through the encoder network
        x = self.encoder_cal_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_cal_loc(x)
        scale = torch.clamp(self.encoder_cal_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def encode_rna(self, x):
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process the combined representation through the encoder network
        x = self.encoder_rna_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_rna_loc(x)
        scale = torch.clamp(self.encoder_rna_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode_cal(self, z):
        
        # run through MLP
        h = self.decoder_cal_mlp(z)
        
        # compute location [N, D]
        xhat_cal = self.decoder_cal_loc(h)
        
        # return the location and log scale
        return xhat_cal
    
    def decode_rna(self, z):
        
        # run through MLP
        h = self.decoder_rna_mlp(z)
        
        # compute location [N, D]
        xhat_rna = self.decoder_rna_loc(h)
        
        # return the location and log scale
        return xhat_rna
    
    def model(self, xcal, xrna):
        
        # register modules with pyro
        pyro.module("decoder_cal_mlp", self.decoder_cal_mlp)
        pyro.module("decoder_cal_loc", self.decoder_cal_loc)
        pyro.module("decoder_rna_mlp", self.decoder_rna_mlp)
        pyro.module("decoder_rna_loc", self.decoder_rna_loc)
        
        # Get the batch size from input x
        batch_size = xcal.shape[0]
        assert batch_size == xrna.shape[0], "Batch size of xcal and xrna must be the same"

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                zcal = pyro.sample("zcal", dist.Normal(z_loc_prior, z_scl_prior))
                zrna = pyro.sample("zrna", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            xhat_cal = self.decode_cal(zcal)
            xhat_rna = self.decode_rna(zrna)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna):
        
        # register modules with pyro
        pyro.module("encoder_cal_mlp", self.encoder_cal_mlp)
        pyro.module("encoder_cal_loc", self.encoder_cal_loc)
        pyro.module("encoder_cal_scl", self.encoder_cal_scl)
        pyro.module("encoder_rna_mlp", self.encoder_rna_mlp)
        pyro.module("encoder_rna_loc", self.encoder_rna_loc)
        pyro.module("encoder_rna_scl", self.encoder_rna_scl)
        
        # Get the batch size from input x
        batch_size = xcal.shape[0]
        assert batch_size == xrna.shape[0], "Batch size of xcal and xrna must be the same"
        
        # get posterior parameters
        zloc_cal,zscl_cal = self.encode_cal(xcal)
        zloc_rna,zscl_rna = self.encode_rna(xrna)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample zcal
                pyro.sample("zcal", dist.Normal(zloc_cal, zscl_cal))
                
                # sample zrna
                pyro.sample("zrna", dist.Normal(zloc_rna, zscl_rna))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, lreg=1e5):
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=MultimodalElbo(lreg=lreg))
        
        # Training loop
        train_loss_history = []
        val_loss_history = []
        
        # patience counter
        patience_counter = 0
        best_loss = float("inf")

        # Train the model
        for epoch in range(num_epochs):
            
            # Training phase
            self.train()
            train_loss = 0.0
            for batch in data_loader:
                xcal_batch = batch[0].to(self.device)
                xrna_batch = batch[1].to(self.device)
                train_loss += svi.step(xcal_batch,xrna_batch) / xcal_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    val_loss += svi.evaluate_loss(xcal_batch,xrna_batch) / xcal_batch.size(0)                    
            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)
            
            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # return loss history
        return train_loss_history,val_loss_history
