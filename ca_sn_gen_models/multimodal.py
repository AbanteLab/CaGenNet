# torch
import torch
import torch.nn as nn

# pyro
import pyro
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO

from ca_sn_gen_models.utils import superprint, gen_random_mask

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

class MultiModMlpDenVAE(nn.Module):
    
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
            nn.Linear(2 * CalD, CalD), nn.ReLU(),
            nn.Linear(CalD, 4 * K), nn.ReLU(),
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
            nn.Linear(2 * RnaD, RnaD), nn.ReLU(),
            nn.Linear(RnaD, 4 * K), nn.ReLU(),
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

    def encode_cal(self, x, m):
        
        # mask the input data
        x_masked = x * m

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xm = torch.cat((x_masked, m), dim=-1)

        # Process the combined representation through the encoder network
        xm_emb = self.encoder_cal_mlp(xm)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_cal_loc(xm_emb)
        scale = torch.clamp(self.encoder_cal_scl(xm_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def encode_rna(self, x, m):
        
        # mask the input data
        x_masked = x * m

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xm = torch.cat((x_masked, m), dim=-1)
        
        # Process the combined representation through the encoder network
        xm = self.encoder_rna_mlp(xm)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_rna_loc(xm)
        scale = torch.clamp(self.encoder_rna_scl(xm), min=1e-3)
        
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
    
    def model(self, xcal, xrna, _mcal, _mrna):
        
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

    def guide(self, xcal, xrna, mcal, mrna):
        
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
        zloc_cal,zscl_cal = self.encode_cal(xcal, mcal)
        zloc_rna,zscl_rna = self.encode_rna(xrna, mrna)

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
                
                # produce masks
                mcal = gen_random_mask(xcal_batch.shape[0],xcal_batch.shape[1]).to(self.device)
                mrna = (torch.rand(xrna_batch.shape, device=self.device) > 0.1).float()

                train_loss += svi.step(xcal_batch, xrna_batch, mcal, mrna) / xcal_batch.size(0)

            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    
                    # produce masks
                    mcal = gen_random_mask(xcal_batch.shape[0],xcal_batch.shape[1]).to(self.device)
                    mrna = (torch.rand(xrna_batch.shape, device=self.device) > 0.1).float()

                    val_loss += svi.evaluate_loss(xcal_batch,xrna_batch, mcal, mrna) / xcal_batch.size(0)                    
            
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
    
    def forward(self, batch_loader):
        """Forward pass through the model for all batches."""

        zloc_cal_list = []
        xhat_cal_list = []
        zloc_rna_list = []
        xhat_rna_list = []

        for xcal_batch, xrna_batch in batch_loader:
            
            xcal_batch = xcal_batch.to(self.device)
            xrna_batch = xrna_batch.to(self.device)

            with torch.no_grad():

                # Calcium modality
                m_cal = torch.ones_like(xcal_batch).to(device=self.device)
                zloc_cal, _ = self.encode_cal(xcal_batch, m_cal)
                xhat_cal = self.decode_cal(zloc_cal)
                zloc_cal_list.append(zloc_cal)
                xhat_cal_list.append(xhat_cal)

                # RNA modality
                m_rna = torch.ones_like(xrna_batch).to(device=self.device)
                zloc_rna, _ = self.encode_rna(xrna_batch, m_rna)
                xhat_rna = self.decode_rna(zloc_rna)
                zloc_rna_list.append(zloc_rna)
                xhat_rna_list.append(xhat_rna)

        # Concatenate results along the batch dimension
        zloc_cal = torch.cat(zloc_cal_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        zloc_rna = torch.cat(zloc_rna_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc_cal, xhat_cal, zloc_rna, xhat_rna
    
class MultiModMlpVAE_v0(nn.Module):
    
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

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, lreg=1e5, **kwargs):
        
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
    
    def forward(self, batch_loader):
        """Forward pass through the model for all batches."""

        zloc_cal_list = []
        xhat_cal_list = []
        zloc_rna_list = []
        xhat_rna_list = []

        for xcal_batch, xrna_batch in batch_loader:
            xcal_batch = xcal_batch.to(self.device)
            xrna_batch = xrna_batch.to(self.device)

            with torch.no_grad():
                # Calcium modality
                zloc_cal, _ = self.encode_cal(xcal_batch)
                xhat_cal = self.decode_cal(zloc_cal)
                zloc_cal_list.append(zloc_cal)
                xhat_cal_list.append(xhat_cal)

                # RNA modality
                zloc_rna, _ = self.encode_rna(xrna_batch)
                xhat_rna = self.decode_rna(zloc_rna)
                zloc_rna_list.append(zloc_rna)
                xhat_rna_list.append(xhat_rna)

        # Concatenate results along the batch dimension
        zloc_cal = torch.cat(zloc_cal_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        zloc_rna = torch.cat(zloc_rna_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc_cal, xhat_cal, zloc_rna, xhat_rna
    
class MultiModSupMlpVAE_v0(nn.Module):
    
    def __init__(self, CalD, RnaD, NS, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.CalD = CalD            # number of time steps in Ca imaging data
        self.RnaD = RnaD            # number of genes in RNA-seq data
        self.K = K                  # latent dimension
        self.NS = NS                # number of classes
        self.device = device
        
        ## Calcium imaging data
        
        # define encoder
        self.encoder_cal_mlp = nn.Sequential(
            nn.Linear(CalD + NS, (CalD + NS) // 2), nn.ReLU(),
            nn.Linear((CalD + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_cal_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_cal_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(K + NS, CalD // 8), nn.ReLU(), 
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
            )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))
        
        # scRNA data
        
        # define encoder
        self.encoder_rna_mlp = nn.Sequential(
            nn.Linear(RnaD + NS, (RnaD + NS) // 2), nn.ReLU(),
            nn.Linear((RnaD + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_rna_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_rna_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(K + NS, RnaD // 8), nn.ReLU(), 
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
            )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))
        
        # Move model to device
        self.to(self.device)

    def encode_cal(self, x, y):
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        xy = self.encoder_cal_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_cal_loc(xy)
        scl = torch.clamp(self.encoder_cal_scl(xy), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def encode_rna(self, x, y):
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)

        # Process the combined representation through the encoder network
        xy = self.encoder_rna_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_rna_loc(xy)
        scl = torch.clamp(self.encoder_rna_scl(xy), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode_cal(self, z, y):
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)

        # run through MLP
        zy = self.decoder_cal_mlp(zy)
        
        # compute location [N, D]
        xhat_cal = self.decoder_cal_loc(zy)
        
        # return the location and log scale
        return xhat_cal
    
    def decode_rna(self, z, y):
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)

        # run through MLP
        zy = self.decoder_rna_mlp(zy)
        
        # compute location [N, D]
        xhat_rna = self.decoder_rna_loc(zy)
        
        # return the location and log scale
        return xhat_rna
    
    def model(self, xcal, xrna, y):
        
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
            
            # sample label
            alpha_cal_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            ycal = pyro.sample("ycal", dist.OneHotCategorical(alpha_cal_prior), obs=y)

            # sample label
            alpha_rna_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            yrna = pyro.sample("yrna", dist.OneHotCategorical(alpha_rna_prior), obs=y)

            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                zcal = pyro.sample("zcal", dist.Normal(z_loc_prior, z_scl_prior))
                zrna = pyro.sample("zrna", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            xhat_cal = self.decode_cal(zcal,ycal)
            xhat_rna = self.decode_rna(zrna,yrna)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna, y):
        
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
        zloc_cal,zscl_cal = self.encode_cal(xcal, y)
        zloc_rna,zscl_rna = self.encode_rna(xrna, y)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample zcal
                pyro.sample("zcal", dist.Normal(zloc_cal, zscl_cal))
                
                # sample zrna
                pyro.sample("zrna", dist.Normal(zloc_rna, zscl_rna))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, lreg=1e5, **kwargs):
        
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
                y_batch = batch[2].to(self.device)
                train_loss += svi.step(xcal_batch,xrna_batch,y_batch) / xcal_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    y_batch = batch[2].to(self.device)
                    val_loss += svi.evaluate_loss(xcal_batch,xrna_batch,y_batch) / xcal_batch.size(0)                    
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
    
    def forward(self, batch_loader):
        """Forward pass through the model for all batches."""

        zloc_cal_list = []
        xhat_cal_list = []
        zloc_rna_list = []
        xhat_rna_list = []

        for batch in batch_loader:
            
            xcal_batch = batch[0].to(self.device)
            xrna_batch = batch[1].to(self.device)
            y_batch = batch[2].to(self.device)

            with torch.no_grad():
                
                # Calcium modality
                zloc_cal, _ = self.encode_cal(xcal_batch, y_batch)
                xhat_cal = self.decode_cal(zloc_cal, y_batch)
                zloc_cal_list.append(zloc_cal)
                xhat_cal_list.append(xhat_cal)

                # RNA modality
                zloc_rna, _ = self.encode_rna(xrna_batch, y_batch)
                xhat_rna = self.decode_rna(zloc_rna, y_batch)
                zloc_rna_list.append(zloc_rna)
                xhat_rna_list.append(xhat_rna)

        # Concatenate results along the batch dimension
        zloc_cal = torch.cat(zloc_cal_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        zloc_rna = torch.cat(zloc_rna_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc_cal, xhat_cal, zloc_rna, xhat_rna

class MultiModMlpVAE_v1(nn.Module):
    
    def __init__(self, CalD, RnaD, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.device = device
        self.CalD = CalD            # number of time steps in Ca imaging data
        self.RnaD = RnaD            # number of genes in RNA-seq data
        self.K = K                  # latent dimension
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(CalD+RnaD, (CalD+RnaD) // 2), nn.ReLU(),
            nn.Linear((CalD+RnaD) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        ## Calcium imaging data
        
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(K, CalD // 8), nn.ReLU(), 
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
            )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))
        
        # scRNA data
        
        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(K, RnaD // 8), nn.ReLU(), 
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
            )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))
        
        # Move model to device
        self.to(self.device)

    def encode(self, xcal, xrna):
        
        # Combine the input data xcal and xrna
        x = torch.cat((xcal, xrna), dim=-1)
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process the combined representation through the encoder network
        x = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x)
        scale = torch.clamp(self.encoder_scl(x), min=1e-3)
        
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
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            xhat_cal = self.decode_cal(z)
            xhat_rna = self.decode_rna(z)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna):
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = xcal.shape[0]

        # get posterior parameters
        zloc,zscl = self.encode(xcal,xrna)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample zcal
                pyro.sample("z", dist.Normal(zloc, zscl))
                
    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, **kwargs):
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
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
    
    def forward(self, batch_loader):
        """ Forward pass through the model """
        
        # Set model to evaluation mode
        self.eval()
        
        # initialize lists to store results
        zloc_list = []
        xhat_cal_list = []
        xhat_rna_list = []

        # Iterate through the batch loader
        for xcal_batch, xrna_batch in batch_loader:
            
            xcal_batch = xcal_batch.to(device=self.device)
            xrna_batch = xrna_batch.to(device=self.device)

            with torch.no_grad():
            
                # Get posterior parameters
                zloc, _ = self.encode(xcal_batch, xrna_batch)

                # Decode the latent representation
                xhat_cal = self.decode_cal(zloc)
                xhat_rna = self.decode_rna(zloc)

                # Accumulate results
                zloc_list.append(zloc)
                xhat_cal_list.append(xhat_cal)
                xhat_rna_list.append(xhat_rna)

        # Concatenate results along the batch dimension
        zloc = torch.cat(zloc_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc, xhat_cal, xhat_rna
    
class MultiModSupMlpVAE_v1(nn.Module):
    
    def __init__(self, CalD, RnaD, NS, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.CalD = CalD            # number of time steps in Ca imaging data
        self.RnaD = RnaD            # number of genes in RNA-seq data
        self.K = K                  # latent dimension
        self.NS = NS                # number of classes
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(CalD+RnaD+NS, (CalD+RnaD+NS) // 2), nn.ReLU(),
            nn.Linear((CalD+RnaD+NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        ## Calcium imaging data
        
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(K+NS, CalD // 8), nn.ReLU(), 
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
            )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))
        
        # scRNA data
        
        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(K+NS, RnaD // 8), nn.ReLU(), 
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
            )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))
        
        # Move model to device
        self.to(self.device)

    def encode(self, xcal, xrna, y):
        
        # Combine the input data xcal and xrna
        x = torch.cat((xcal, xrna, y), dim=-1)
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process the combined representation through the encoder network
        x = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x)
        scl = torch.clamp(self.encoder_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode_cal(self, z, y):
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)

        # run through MLP
        zy = self.decoder_cal_mlp(zy)
        
        # compute location [N, D]
        xhat_cal = self.decoder_cal_loc(zy)
        
        # return the location and log scale
        return xhat_cal
    
    def decode_rna(self, z, y):
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)

        # run through MLP
        zy = self.decoder_rna_mlp(zy)
        
        # compute location [N, D]
        xhat_rna = self.decoder_rna_loc(zy)
        
        # return the location and log scale
        return xhat_rna
    
    def model(self, xcal, xrna, y):
        
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
            
            # sample label
            alpha_cal_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            ycal = pyro.sample("ycal", dist.OneHotCategorical(alpha_cal_prior), obs=y)

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            yrna = pyro.sample("yrna", dist.OneHotCategorical(alpha_prior), obs=y)

            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            xhat_cal = self.decode_cal(z, ycal)
            xhat_rna = self.decode_rna(z, yrna)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna, y):
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = xcal.shape[0]

        # get posterior parameters
        zloc,zscl = self.encode(xcal,xrna,y)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample zcal
                pyro.sample("z", dist.Normal(zloc, zscl))
                
    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, **kwargs):
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
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
                y_batch = batch[2].to(self.device)
                train_loss += svi.step(xcal_batch,xrna_batch,y_batch) / xcal_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    y_batch = batch[2].to(self.device)
                    val_loss += svi.evaluate_loss(xcal_batch,xrna_batch,y_batch) / xcal_batch.size(0)                    
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
    
    def forward(self, batch_loader):
        """ Forward pass through the model """
        
        # Set model to evaluation mode
        self.eval()
        
        # initialize lists to store results
        zloc_list = []
        xhat_cal_list = []
        xhat_rna_list = []

        # Iterate through the batch loader
        for batch in batch_loader:
            
            xcal_batch = batch[0].to(self.device)
            xrna_batch = batch[1].to(self.device)
            y_batch = batch[2].to(self.device)

            with torch.no_grad():
            
                # Get posterior parameters
                zloc, _ = self.encode(xcal_batch, xrna_batch, y_batch)

                # Decode the latent representation
                xhat_cal = self.decode_cal(zloc, y_batch)
                xhat_rna = self.decode_rna(zloc, y_batch)

                # Accumulate results
                zloc_list.append(zloc)
                xhat_cal_list.append(xhat_cal)
                xhat_rna_list.append(xhat_rna)

        # Concatenate results along the batch dimension
        zloc = torch.cat(zloc_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc, xhat_cal, xhat_rna

class Elbo_MultiModMlpVAE_v2(Trace_ELBO):
    
    def __init__(self, lreg=1e5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.lreg = lreg

    def loss_corr(self, z, zmod):
        """ Computes frobenius norm of the correlation matrix """
        
        # Center
        z_centered = z - z.mean(dim=0, keepdim=True)
        zmod_centered = zmod - zmod.mean(dim=0, keepdim=True)
        
        # Scale by std
        z_std = z_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        zmod_std = zmod_centered.std(dim=0, unbiased=False, keepdim=True) + 1e-8
        z_norm = z_centered / z_std
        zmod_norm = zmod_centered / zmod_std
        
        # Compute correlation matrix
        corr_matrix = torch.mm(z_norm.t(), zmod_norm) / (z.shape[0] - 1)
        
        return self.lreg * torch.linalg.norm(corr_matrix, ord='fro')
        
    def loss(self, model, guide, *args, **kwargs):
        
        loss = super().loss(model, guide, *args, **kwargs)  # ELBO normal
        superprint(f"ELBO loss: {loss:.4f}")
        
        # get zcal and zrna from the guide
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        z = guide_trace.nodes["z"]["value"]
        zcal = guide_trace.nodes["zcal"]["value"]
        zrna = guide_trace.nodes["zrna"]["value"]
        
        # compute the extra loss
        extra_loss = self.loss_corr(z, zcal) + self.loss_corr(z, zrna)
        superprint(f"L(c(Z,Zmod)): {extra_loss:.4f}")
        
        return loss + extra_loss
    
class MultiModMlpVAE_v2(nn.Module):
    def __init__(self, CalD, RnaD, K, device="cpu"):
        super().__init__()
        self.device = device
        self.CalD = CalD
        self.RnaD = RnaD
        self.K = K

        # Encoders for modality-specific latents
        self.encoder_cal_mlp = nn.Sequential(
            nn.Linear(CalD, CalD // 2), nn.ReLU(),
            nn.Linear(CalD // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_cal_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_cal_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        self.encoder_rna_mlp = nn.Sequential(
            nn.Linear(RnaD, RnaD // 2), nn.ReLU(),
            nn.Linear(RnaD // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_rna_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_rna_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        # Encoder for shared latent
        self.encoder_shared_mlp = nn.Sequential(
            nn.Linear(CalD + RnaD, (CalD + RnaD) // 2), nn.ReLU(),
            nn.Linear((CalD + RnaD) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_shared_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_shared_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        # Decoders for each modality (input: shared Z + modality-specific Z)
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(2 * K, CalD // 8), nn.ReLU(),
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
        )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))

        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(2 * K, RnaD // 8), nn.ReLU(),
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
        )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))

        self.to(self.device)

    def encode_cal(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encoder_cal_mlp(x)
        loc = self.encoder_cal_loc(h)
        scale = torch.clamp(self.encoder_cal_scl(h), min=1e-3)
        return loc, scale

    def encode_rna(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encoder_rna_mlp(x)
        loc = self.encoder_rna_loc(h)
        scale = torch.clamp(self.encoder_rna_scl(h), min=1e-3)
        return loc, scale

    def encode_shared(self, xcal, xrna):
        x = torch.cat((xcal, xrna), dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encoder_shared_mlp(x)
        loc = self.encoder_shared_loc(h)
        scale = torch.clamp(self.encoder_shared_scl(h), min=1e-3)
        return loc, scale

    def decode_cal(self, z_shared, z_cal):
        z = torch.cat((z_shared, z_cal), dim=-1)
        h = self.decoder_cal_mlp(z)
        xhat = self.decoder_cal_loc(h)
        return xhat

    def decode_rna(self, z_shared, z_rna):
        z = torch.cat((z_shared, z_rna), dim=-1)
        h = self.decoder_rna_mlp(z)
        xhat = self.decoder_rna_loc(h)
        return xhat

    def model(self, xcal, xrna):
        pyro.module("decoder_cal_mlp", self.decoder_cal_mlp)
        pyro.module("decoder_cal_loc", self.decoder_cal_loc)
        pyro.module("decoder_rna_mlp", self.decoder_rna_mlp)
        pyro.module("decoder_rna_loc", self.decoder_rna_loc)

        batch_size = xcal.shape[0]
        assert batch_size == xrna.shape[0], "Batch size of xcal and xrna must be the same"

        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        with pyro.plate("data", batch_size, dim=-2):
            with pyro.plate("latent_dim", self.K, dim=-1):
                z_shared = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
                zcal = pyro.sample("zcal", dist.Normal(z_loc_prior, z_scl_prior))
                zrna = pyro.sample("zrna", dist.Normal(z_loc_prior, z_scl_prior))

            xhat_cal = self.decode_cal(z_shared, zcal)
            xhat_rna = self.decode_rna(z_shared, zrna)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna):
        pyro.module("encoder_cal_mlp", self.encoder_cal_mlp)
        pyro.module("encoder_cal_loc", self.encoder_cal_loc)
        pyro.module("encoder_cal_scl", self.encoder_cal_scl)
        pyro.module("encoder_rna_mlp", self.encoder_rna_mlp)
        pyro.module("encoder_rna_loc", self.encoder_rna_loc)
        pyro.module("encoder_rna_scl", self.encoder_rna_scl)
        pyro.module("encoder_shared_mlp", self.encoder_shared_mlp)
        pyro.module("encoder_shared_loc", self.encoder_shared_loc)
        pyro.module("encoder_shared_scl", self.encoder_shared_scl)

        batch_size = xcal.shape[0]

        zloc_cal, zscl_cal = self.encode_cal(xcal)
        zloc_rna, zscl_rna = self.encode_rna(xrna)
        zloc_shared, zscl_shared = self.encode_shared(xcal, xrna)

        with pyro.plate("data", batch_size, dim=-2):
            with pyro.plate("latent_dim", self.K, dim=-1):
                pyro.sample("z", dist.Normal(zloc_shared, zscl_shared))
                pyro.sample("zcal", dist.Normal(zloc_cal, zscl_cal))
                pyro.sample("zrna", dist.Normal(zloc_rna, zscl_rna))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, lreg=1e5, **kwargs):
        
        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Elbo_MultiModMlpVAE_v2(lreg=lreg))

        # initialize training and validation loss history
        train_loss_history = []
        val_loss_history = []
        patience_counter = 0
        best_loss = float("inf")

        # Training loop
        for epoch in range(num_epochs):
            self.train()  # set model to training mode
            train_loss = 0.0
            # Iterate over training batches
            for batch in data_loader:
                xcal_batch = batch[0].to(self.device)
                xrna_batch = batch[1].to(self.device)
                # Perform one SVI step and accumulate loss
                train_loss += svi.step(xcal_batch, xrna_batch) / xcal_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)

            self.eval()  # set model to evaluation mode
            val_loss = 0.0
            # Evaluate on validation set without gradients
            with torch.no_grad():
                for batch in val_loader:
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    # Compute validation loss
                    val_loss += svi.evaluate_loss(xcal_batch, xrna_batch) / xcal_batch.size(0)
            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Return loss history for analysis
        return train_loss_history, val_loss_history

    def forward(self, batch_loader):
        self.eval()
        zloc_shared_list = []
        zloc_cal_list = []
        zloc_rna_list = []
        xhat_cal_list = []
        xhat_rna_list = []

        for xcal_batch, xrna_batch in batch_loader:
            xcal_batch = xcal_batch.to(self.device)
            xrna_batch = xrna_batch.to(self.device)
            with torch.no_grad():
                zloc_shared, _ = self.encode_shared(xcal_batch, xrna_batch)
                zloc_cal, _ = self.encode_cal(xcal_batch)
                zloc_rna, _ = self.encode_rna(xrna_batch)
                xhat_cal = self.decode_cal(zloc_shared, zloc_cal)
                xhat_rna = self.decode_rna(zloc_shared, zloc_rna)
                zloc_shared_list.append(zloc_shared)
                zloc_cal_list.append(zloc_cal)
                zloc_rna_list.append(zloc_rna)
                xhat_cal_list.append(xhat_cal)
                xhat_rna_list.append(xhat_rna)

        zloc_shared = torch.cat(zloc_shared_list, dim=0)
        zloc_cal = torch.cat(zloc_cal_list, dim=0)
        zloc_rna = torch.cat(zloc_rna_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc_shared, zloc_cal, zloc_rna, xhat_cal, xhat_rna
 
class MultiModSupMlpVAE_v2(nn.Module):
    def __init__(self, CalD, RnaD, NS, K, device="cpu"):
        super().__init__()
        self.device = device
        self.CalD = CalD
        self.RnaD = RnaD
        self.K = K
        self.NS = NS  # number of classes

        # Encoders for modality-specific latents
        self.encoder_cal_mlp = nn.Sequential(
            nn.Linear(CalD + NS, (CalD + NS) // 2), nn.ReLU(),
            nn.Linear((CalD + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_cal_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_cal_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        self.encoder_rna_mlp = nn.Sequential(
            nn.Linear(RnaD + NS, (RnaD + NS) // 2), nn.ReLU(),
            nn.Linear((RnaD + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_rna_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_rna_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        # Encoder for shared latent
        self.encoder_shared_mlp = nn.Sequential(
            nn.Linear(CalD + RnaD, (CalD + RnaD) // 2), nn.ReLU(),
            nn.Linear((CalD + RnaD) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_shared_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_shared_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())

        # Decoders for each modality (input: shared Z + modality-specific Z)
        self.decoder_cal_mlp = nn.Sequential(
            nn.Linear(2 * K + NS, CalD // 8), nn.ReLU(),
            nn.Linear(CalD // 8, CalD // 4), nn.ReLU(),
            nn.Linear(CalD // 4, CalD // 2)
        )
        self.decoder_cal_loc = nn.Sequential(nn.Linear(CalD // 2, CalD))

        self.decoder_rna_mlp = nn.Sequential(
            nn.Linear(2 * K + NS, RnaD // 8), nn.ReLU(),
            nn.Linear(RnaD // 8, RnaD // 4), nn.ReLU(),
            nn.Linear(RnaD // 4, RnaD // 2)
        )
        self.decoder_rna_loc = nn.Sequential(nn.Linear(RnaD // 2, RnaD))

        self.to(self.device)

    def encode_cal(self, x, y):
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        h = self.encoder_cal_mlp(xy)
        loc = self.encoder_cal_loc(h)
        scale = torch.clamp(self.encoder_cal_scl(h), min=1e-3)
        return loc, scale

    def encode_rna(self, x, y):
        
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        xy = torch.cat((x, y), dim=-1)
        h = self.encoder_rna_mlp(xy)
        loc = self.encoder_rna_loc(h)
        scale = torch.clamp(self.encoder_rna_scl(h), min=1e-3)
        
        return loc, scale

    def encode_shared(self, xcal, xrna):
        x = torch.cat((xcal, xrna), dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.encoder_shared_mlp(x)
        loc = self.encoder_shared_loc(h)
        scale = torch.clamp(self.encoder_shared_scl(h), min=1e-3)
        
        return loc, scale

    def decode_cal(self, z_shared, z_cal, ycal):
        zy = torch.cat((z_shared, z_cal, ycal), dim=-1)
        h = self.decoder_cal_mlp(zy)
        xhat = self.decoder_cal_loc(h)
        return xhat

    def decode_rna(self, z_shared, z_rna, yrna):
        zy = torch.cat((z_shared, z_rna, yrna), dim=-1)
        h = self.decoder_rna_mlp(zy)
        xhat = self.decoder_rna_loc(h)
        return xhat

    def model(self, xcal, xrna, y):
        pyro.module("decoder_cal_mlp", self.decoder_cal_mlp)
        pyro.module("decoder_cal_loc", self.decoder_cal_loc)
        pyro.module("decoder_rna_mlp", self.decoder_rna_mlp)
        pyro.module("decoder_rna_loc", self.decoder_rna_loc)

        batch_size = xcal.shape[0]
        assert batch_size == xrna.shape[0], "Batch size of xcal and xrna must be the same"

        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        with pyro.plate("data", batch_size, dim=-2):

            # sample calcium label
            alpha_cal_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            ycal = pyro.sample("ycal", dist.OneHotCategorical(alpha_cal_prior), obs=y)

            # sample rna label
            alpha_rna_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            yrna = pyro.sample("yrna", dist.OneHotCategorical(alpha_rna_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):
                z_shared = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
                zcal = pyro.sample("zcal", dist.Normal(z_loc_prior, z_scl_prior))
                zrna = pyro.sample("zrna", dist.Normal(z_loc_prior, z_scl_prior))

            xhat_cal = self.decode_cal(z_shared, zcal, ycal)
            xhat_rna = self.decode_rna(z_shared, zrna, yrna)

            with pyro.plate("obs_dim_cal", self.CalD, dim=-1):
                pyro.sample("xcal", dist.NanMaskedNormal(xhat_cal, 1e-3), obs=xcal)
            with pyro.plate("obs_dim_rna", self.RnaD, dim=-1):
                pyro.sample("xrna", dist.NanMaskedNormal(xhat_rna, 1e-3), obs=xrna)

    def guide(self, xcal, xrna, y):
        pyro.module("encoder_cal_mlp", self.encoder_cal_mlp)
        pyro.module("encoder_cal_loc", self.encoder_cal_loc)
        pyro.module("encoder_cal_scl", self.encoder_cal_scl)
        pyro.module("encoder_rna_mlp", self.encoder_rna_mlp)
        pyro.module("encoder_rna_loc", self.encoder_rna_loc)
        pyro.module("encoder_rna_scl", self.encoder_rna_scl)
        pyro.module("encoder_shared_mlp", self.encoder_shared_mlp)
        pyro.module("encoder_shared_loc", self.encoder_shared_loc)
        pyro.module("encoder_shared_scl", self.encoder_shared_scl)

        batch_size = xcal.shape[0]

        zloc_cal, zscl_cal = self.encode_cal(xcal, y)
        zloc_rna, zscl_rna = self.encode_rna(xrna, y)
        zloc_shared, zscl_shared = self.encode_shared(xcal, xrna)

        with pyro.plate("data", batch_size, dim=-2):
            with pyro.plate("latent_dim", self.K, dim=-1):
                pyro.sample("z", dist.Normal(zloc_shared, zscl_shared))
                pyro.sample("zcal", dist.Normal(zloc_cal, zscl_cal))
                pyro.sample("zrna", dist.Normal(zloc_rna, zscl_rna))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, lreg=1e5, **kwargs):
        
        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Elbo_MultiModMlpVAE_v2(lreg=lreg))

        # initialize training and validation loss history
        train_loss_history = []
        val_loss_history = []
        patience_counter = 0
        best_loss = float("inf")

        # Training loop
        for epoch in range(num_epochs):
            self.train()  # set model to training mode
            train_loss = 0.0
            # Iterate over training batches
            for batch in data_loader:
                xcal_batch = batch[0].to(self.device)
                xrna_batch = batch[1].to(self.device)
                y_batch = batch[2].to(self.device)
                # Perform one SVI step and accumulate loss
                train_loss += svi.step(xcal_batch, xrna_batch, y_batch) / xcal_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)

            self.eval()  # set model to evaluation mode
            val_loss = 0.0
            # Evaluate on validation set without gradients
            with torch.no_grad():
                for batch in val_loader:
                    xcal_batch = batch[0].to(self.device)
                    xrna_batch = batch[1].to(self.device)
                    y_batch = batch[2].to(self.device)
                    # Compute validation loss
                    val_loss += svi.evaluate_loss(xcal_batch, xrna_batch, y_batch) / xcal_batch.size(0)
            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Print progress
            superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Return loss history for analysis
        return train_loss_history, val_loss_history

    def forward(self, batch_loader):
        self.eval()
        zloc_shared_list = []
        zloc_cal_list = []
        zloc_rna_list = []
        xhat_cal_list = []
        xhat_rna_list = []

        for xcal_batch, xrna_batch, y_batch in batch_loader:
            xcal_batch = xcal_batch.to(self.device)
            xrna_batch = xrna_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            with torch.no_grad():
                zloc_shared, _ = self.encode_shared(xcal_batch, xrna_batch)
                zloc_cal, _ = self.encode_cal(xcal_batch, y_batch)
                zloc_rna, _ = self.encode_rna(xrna_batch, y_batch)
                xhat_cal = self.decode_cal(zloc_shared, zloc_cal, y_batch)
                xhat_rna = self.decode_rna(zloc_shared, zloc_rna, y_batch)
                zloc_shared_list.append(zloc_shared)
                zloc_cal_list.append(zloc_cal)
                zloc_rna_list.append(zloc_rna)
                xhat_cal_list.append(xhat_cal)
                xhat_rna_list.append(xhat_rna)

        zloc_shared = torch.cat(zloc_shared_list, dim=0)
        zloc_cal = torch.cat(zloc_cal_list, dim=0)
        zloc_rna = torch.cat(zloc_rna_list, dim=0)
        xhat_cal = torch.cat(xhat_cal_list, dim=0)
        xhat_rna = torch.cat(xhat_rna_list, dim=0)

        return zloc_shared, zloc_cal, zloc_rna, xhat_cal, xhat_rna