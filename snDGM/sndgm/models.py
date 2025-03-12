# Description: Pyro models for the snDGM and VAE models.

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# pyro
import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO

# distributions
from sndgm.dist import ZeroInflatedLogNormal
from sndgm.utils import superprint, kl_beta_schedule

########################################################################################
# Encoder and Decoder classes
########################################################################################

class ConvEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)
            
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, enc_dims):
        
        super().__init__()
        
        # generate layers according to the dimensions
        layers = [nn.Linear(input_dim, enc_dims[0]), nn.ReLU()]
        for i in range(len(enc_dims) - 1):
            layers.extend([nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU()])
        
        layers.append(nn.Linear(enc_dims[-1], latent_dim))

        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, dec_dims, output=None):
        
        super().__init__()
        
        layers = [nn.Linear(latent_dim, dec_dims[0]), nn.ReLU()]
        for i in range(len(dec_dims) - 1):
            layers.extend([nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dec_dims[-1], output_dim))
        
        if output is None:
            self.net = nn.Sequential(*layers)
        elif output=="softplus":
            self.net = nn.Sequential(*layers, nn.Softplus())
        elif output=="sigmoid":
            self.net = nn.Sequential(*layers, nn.Sigmoid())
        elif output=="exponential":
            self.net = nn.Sequential(*layers, nn.Lambda(lambda x: torch.exp(x)))
        else:
            raise ValueError("Output activation not recognized.")
        
    def forward(self, x):
        
        return self.net(x)
    
########################################################################################
# Amplitude generative model
########################################################################################

class AmpVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for FFT amplitude data.
    
    This model consists of a single latent variable:
    - Z: Latent variable
    
    Attributes:
        
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent variable Z.
        hidden_dim (int): Dimension of the hidden layers in the neural networks.
        target_dist (pyro.distributions): Target distribution for the data.
        device (str): Device to use for the model.
        
    Methods:
    
        model(a):
            Defines the generative model.
        guide(a):
            Defines the inference model.
            
    """
    def __init__(self, input_dim, latent_dim=16, enc_dims=None, dec_dims=None, target_dist=dist.Normal, device='cpu'):
        super().__init__()
        
        # store to be accessed elsewhere
        self.zinf = False
        self.device = device
        self.loc_and_scale = False
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.target_dist = target_dist

        # Default dimensions if not provided
        if enc_dims is None:
            enc_dims = [input_dim // 2, input_dim // 4, input_dim // 8]
        if dec_dims is None:
            dec_dims = [latent_dim * 2, latent_dim * 4, latent_dim * 8]
        
        # check if model is loc or loc+scale
        if target_dist != dist.Exponential:
            self.loc_and_scale = True
        
        # check conditions
        if target_dist in [dist.Exponential,dist.Gamma,dist.Weibull]:
            # need positive location
            self.decoder_loc = Decoder(latent_dim, input_dim, dec_dims, output='softplus')
        else:
            self.decoder_loc = Decoder(latent_dim, input_dim, dec_dims)
            
        # add decoder for scale if necessary
        if self.loc_and_scale:
            self.decoder_scale = Decoder(latent_dim, input_dim, dec_dims, output='softplus')
        
        # define encoders
        self.encoder_mu = Encoder(input_dim, latent_dim, enc_dims)
        self.encoder_logvar = Encoder(input_dim, latent_dim, enc_dims)

        # Move model to device
        self.to(device)

    def model(self, a, beta=1.0):
        
        batch_size = a.shape[0]
        
        # Move data to the correct device
        a = a.to(self.device)

        # Latent prior
        mu_prior = torch.zeros(batch_size, self.latent_dim, device=self.device)
        sigma_prior = torch.ones(batch_size, self.latent_dim, device=self.device)
        
        # register modules
        pyro.module("decoder_loc", self.decoder_loc)
        if self.loc_and_scale:
            pyro.module("decoder_scale", self.decoder_scale)

        # plate for mini-batch
        with pyro.poutine.scale(scale=beta):
            with pyro.plate("batch", batch_size):
            
                # Sample latent variable
                Za = pyro.sample("Za", dist.Normal(mu_prior, sigma_prior).to_event(1))
            
                # get location
                loc = self.decoder_loc(Za)
                
                # get scale if not exponential
                if self.loc_and_scale:
                    
                    ## NOTE: tried clamping scale but ELBO doesn't improve
                    # scale = torch.clamp(self.decoder_scale(Za), min=0.1)
                    
                    # Decode scale
                    scale = self.decoder_scale(Za) + 1e-6
                    
                    # Target likelihood
                    pyro.sample("A_obs", self.target_dist(loc, scale).to_event(1), obs=a)        
                
                else:
                    
                    # Target likelihood
                    pyro.sample("A_obs", self.target_dist(loc).to_event(1), obs=a)

    def guide(self, a, beta=1.0):
        
        batch_size = a.shape[0]

        # Move data to the correct device
        a = a.to(self.device)

        # register modules
        pyro.module("encoder_mu", self.encoder_mu)
        pyro.module("encoder_logvar", self.encoder_logvar)
        
        # plate for mini-batch
        with pyro.poutine.scale(scale=beta):
            with pyro.plate("batch", batch_size, dim=-1):
            
                # Compute approximate posterior for Z
                mu_Za = self.encoder_mu(a)
                sigma_Za = torch.exp(self.encoder_logvar(a)) + 1e-6

                # Sample latent variable
                Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))

###########################################################################################################
# Inference
###########################################################################################################                    
def train_vae(vae, svi, train_loader, val_loader, num_epochs=500, batch=512, patience=10, min_delta=1e-3, start_beta=0.25, end_beta=0.25, device='cpu', model_type="AmpVAE"):
    """
    Trains a Variational Autoencoder (VAE) using Stochastic Variational Inference (SVI) with early stopping.
    Args:
        vae (torch.nn.Module): The VAE model to be trained.
        svi (pyro.infer.SVI): The SVI object for performing optimization.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        num_epochs (int, optional): Number of epochs to train the model. Default is 500.
        batch (int, optional): Batch size for training. Default is 512.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Default is 10.
        min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Default is 1e-3.
        start_beta (float, optional): Initial value of the beta parameter for KL divergence annealing. Default is 0.25.
        end_beta (float, optional): Final value of the beta parameter for KL divergence annealing. Default is 0.25.
        device (str, optional): Device to run the training on ('cpu' or 'cuda'). Default is 'cpu'.
        model_type (str, optional): Type of VAE model being trained. Default is "AmpVAE".
    Returns:
        tuple: A tuple containing two lists:
            - avg_train_loss_trail (list): List of average training losses per epoch.
            - avg_val_loss_trail (list): List of average validation losses per epoch.
    The function trains the VAE model for a specified number of epochs, updating the beta parameter for KL divergence
    annealing. It computes the training and validation losses for each epoch and implements early stopping based on
    the validation loss to prevent overfitting.
    """
    
    # Early stopping parameters
    counter = 0
    best_val_loss = float('inf')

    # Training loop
    avg_val_loss_trail = []
    avg_train_loss_trail = []
    for epoch in range(num_epochs):
        # Update beta
        beta = kl_beta_schedule(epoch, num_epochs // 2, start_beta=start_beta, end_beta=end_beta)

        # Iterate over batches
        vae.train()
        epoch_loss = 0
        for a_train, p_train in train_loader:
            a_train = a_train.to(device)
            p_train = p_train.to(device)
            if model_type in ["IcasspVAE", "HierarchicalVAE"]:
                loss = svi.step(a_train, p_train)
            else:
                loss = svi.step(a_train, beta=beta)
            epoch_loss += loss / batch

        # Compute validation loss
        vae.eval()
        val_loss = 0
        for a_val, p_val in val_loader:
            a_val = a_val.to(device)
            p_val = p_val.to(device)
            if model_type in ["IcasspVAE", "HierarchicalVAE"]:
                val_loss += svi.evaluate_loss(a_val, p_val) / batch
            else:
                val_loss += svi.evaluate_loss(a_val, beta=beta) / batch

        # Print average train and validation loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        superprint(f"Epoch {epoch}, Train Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}, Beta: {beta}")

        # Append to trail
        avg_val_loss_trail.append(avg_val_loss)
        avg_train_loss_trail.append(avg_epoch_loss)

        # Check for early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                superprint(f'Early stopping triggered at epoch {epoch + 1}')
                break

    return avg_train_loss_trail, avg_val_loss_trail

###########################################################################################################
# CHECK THESE MODELS
###########################################################################################################

class ZinfLogNormVAE(nn.Module):
    """
    Variational Autoencoder (VAE) for FFT amplitude data.
    
    This model consists of a single latent variable:
    - Z: Latent variable
    
    Attributes:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent variable Z.
        hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
    Methods:
        model(a):
            Defines the generative model.
        guide(a):
            Defines the inference model.
    """
    def __init__(self, input_dim, latent_dim=16, enc_dims=None, dec_dims=None, device='cpu'):
        super().__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default dimensions if not provided
        if enc_dims is None:
            enc_dims = [input_dim // 2, input_dim // 4, input_dim // 8]
        if dec_dims is None:
            dec_dims = [latent_dim * 2, latent_dim * 4, latent_dim * 8]
        
        # define decoders
        self.decoder_mu = Decoder(latent_dim, input_dim, dec_dims)
        self.decoder_logvar = Decoder(latent_dim, input_dim, dec_dims)
        self.decoder_zinf = Decoder(latent_dim, input_dim, dec_dims)

        # define encoders
        self.encoder_mu = Encoder(input_dim, latent_dim, enc_dims)
        self.encoder_logvar = Encoder(input_dim, latent_dim, enc_dims)

        # Move model to device
        self.to(device)

    def model(self, a):
        
        batch_size = a.shape[0]
        
        # Move data to the correct device
        a = a.to(self.device)

        # Latent prior
        mu_prior = torch.zeros(batch_size, self.latent_dim, device=self.device)
        sigma_prior = torch.ones(batch_size, self.latent_dim, device=self.device)
        
        # register modules
        pyro.module("decoder_mu", self.decoder_mu)
        pyro.module("decoder_zinf", self.decoder_zinf)
        pyro.module("decoder_logvar", self.decoder_logvar)
        
        with pyro.plate("batch", batch_size):
            
            # Sample latent variable
            Za = pyro.sample("Za", dist.Normal(mu_prior, sigma_prior).to_event(1))

            # Decode 0 probability
            p_zero = torch.sigmoid(self.decoder_zinf(Za))
            
            # Decode to amplitude parameters
            loc_A = self.decoder_mu(Za)
            scale_A = torch.exp(self.decoder_logvar(Za)) + 1e-6
            
            # Zero-inflated LogNormal likelihood
            pyro.sample("A_obs", ZeroInflatedLogNormal(loc_A, scale_A, gate=p_zero).to_event(1), obs=a)

    def guide(self, a):
        
        batch_size = a.shape[0]

        # Move data to the correct device
        a = a.to(self.device)
        
        # register modules
        pyro.module("encoder_mu", self.encoder_mu)
        pyro.module("encoder_logvar", self.encoder_logvar)

        with pyro.plate("batch", batch_size, dim=-1):
            
            # Compute approximate posterior for Z
            mu_Za = self.encoder_mu(a)
            sigma_Za = torch.exp(self.encoder_logvar(a)) + 1e-6
            pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))
                        
class HierarchicalVAE(nn.Module):
    """
    
    Hierarchical Variational Autoencoder (VAE) with a three-level latent structure.
    
    This model consists of three latent variables:
    - Zs: Global latent variable
    - Za: Amplitude latent variable
    - Zp: Phase latent variable
    
    Graphical Model:
        Zs
       /  \
      Za   Zp
      |    |
      A   Phi
    
    Attributes:
        input_dim (int): Dimension of the input data.
        latent_dim_s (int): Dimension of the global latent variable Zs.
        latent_dim_a (int): Dimension of the amplitude latent variable Za.
        latent_dim_p (int): Dimension of the phase latent variable Zp.
        hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
    Methods:
        model(a, p):
            Defines the generative model.
        guide(a, p):
            Defines the inference model.
    
    """
    def __init__(self, input_dim, latent_dim_s=64, latent_dim_a=32, latent_dim_p=32, hidden_dim=512, device='cpu'):
        
        super().__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim_s = latent_dim_s
        self.latent_dim_a = latent_dim_a
        self.latent_dim_p = latent_dim_p

        # decoder Zs to Za and Zp
        self.mu_a_net = Decoder(latent_dim_s, latent_dim_a, [latent_dim_s])
        self.sigma_a_net = Decoder(latent_dim_s, latent_dim_a, [latent_dim_s])
        self.mu_p_net = Decoder(latent_dim_s, latent_dim_p, [latent_dim_s])
        self.sigma_p_net = Decoder(latent_dim_s, latent_dim_p, [latent_dim_s])

        # decoder amplitude Za to A (needs mu to be positive for lognormal - relu)
        self.decoder_a_mu = Decoder(latent_dim_a, input_dim, [128, 512, 1028])
        self.decoder_a_logvar = Decoder(latent_dim_a, input_dim, [128, 512, 1028])
        
        # decoder phi Zp to Phi (needs mu to be [-pi, pi] and kappa to be positive)
        self.decoder_p_mu = Decoder(latent_dim_p, input_dim, [128, 512, 1028])
        self.decoder_p_kappa = Decoder(latent_dim_p, input_dim, [128, 512, 1028])

        # Use ConvEncoder for encoder_s_mu
        self.encoder_s_mu = ConvEncoder(input_channels=2, output_dim=latent_dim_s, hidden_dim=hidden_dim)
        self.encoder_s_logvar = ConvEncoder(input_channels=2, output_dim=latent_dim_s, hidden_dim=hidden_dim)

        # Encoders for Za
        self.encoder_a_mu = Encoder(latent_dim_s, latent_dim_a, [latent_dim_s])
        self.encoder_a_logvar = Encoder(latent_dim_s, latent_dim_a, [latent_dim_s])
        
        # Encoders for Zp
        self.encoder_p_mu = Encoder(latent_dim_s, latent_dim_p, [latent_dim_s])
        self.encoder_p_logvar = Encoder(latent_dim_s, latent_dim_p, [latent_dim_s])

        # Initialize parameters
        self._initialize_parameters()

        # Move model to device
        self.to(self.device)

    # def _initialize_parameters(self):
    #     # Initialize weights and biases for decoders
    #     for decoder in [self.decoder_a_mu, self.decoder_a_logvar, self.decoder_p_mu, self.decoder_p_kappa]:
    #         for layer in decoder.net:
    #             if isinstance(layer, nn.Linear):
    #                 nn.init.kaiming_normal_(layer.weight)
    #                 nn.init.constant_(layer.bias, 0)

    #     # Initialize weights and biases for encoders
    #     for encoder in [self.encoder_s_mu, self.encoder_s_logvar, self.encoder_a_mu, self.encoder_a_logvar, self.encoder_p_mu, self.encoder_p_logvar]:
    #         for layer in encoder.net:
    #             if isinstance(layer, nn.Linear):
    #                 nn.init.kaiming_normal_(layer.weight)
    #                 nn.init.constant_(layer.bias, 0)

    #     # Initialize ConvEncoder separately
    #     for conv_encoder in [self.encoder_s_mu, self.encoder_s_logvar]:
    #         for layer in conv_encoder.net:
    #             if isinstance(layer, nn.Conv1d):
    #                 nn.init.kaiming_normal_(layer.weight)
    #                 nn.init.constant_(layer.bias, 0)

    def model(self, a, p):
        
        batch_size = a.shape[0]
        
        # Move data to the correct device
        a = a.to(self.device)
        p = p.to(self.device)

        # register modules
        pyro.module("mu_a_net", self.mu_a_net)
        pyro.module("sigma_a_net", self.sigma_a_net)
        pyro.module("mu_p_net", self.mu_a_net)
        pyro.module("sigma_p_net", self.sigma_a_net)
        pyro.module("decoder_a_mu", self.decoder_a_mu)
        pyro.module("decoder_a_logvar", self.decoder_a_logvar)
        pyro.module("decoder_p_mu", self.decoder_p_mu)
        pyro.module("decoder_p_kappa", self.decoder_p_kappa)
        
        # Latent prior
        mu_prior = torch.zeros(batch_size, self.latent_dim_s, device=self.device)
        sigma_prior = torch.ones(batch_size, self.latent_dim_s, device=self.device)
        
        # plate
        with pyro.plate("batch", batch_size, dim=-1):  # Define a batch plate
            
            # Global latent variable Zs
            Zs = pyro.sample("Zs",dist.Normal(mu_prior, sigma_prior).to_event(1))

            # Amplitude latent variable Za
            mu_a = self.mu_a_net(Zs)
            sigma_a = torch.exp(self.sigma_a_net(Zs)) + 1e-6  # Ensure positivity
            Za = pyro.sample("Za", dist.Normal(mu_a, sigma_a).to_event(1))

            # Phase latent variable Zp
            mu_p = self.mu_p_net(Zs)
            sigma_p = torch.exp(self.sigma_p_net(Zs)) + 1e-6  # Ensure positivity
            Zp = pyro.sample("Zp", dist.Normal(mu_p, sigma_p).to_event(1))

            # Decode amplitude and phase
            mu_A = self.decoder_a_mu(Za)
            sigma_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-6
            
            # Amplitude likelihood (Gaussian)
            pyro.sample("A_obs", dist.LogNormal(mu_A, sigma_A).to_event(1), obs=a)
            
            # Decode phase
            mu_P = self.decoder_p_mu(Zp)
            kappa_P = torch.exp(self.decoder_p_kappa(Zp)) + 1e-6  # Ensure positivity

            # Phase likelihood (Von Mises)
            pyro.sample("P_obs", dist.VonMises(mu_P, kappa_P).to_event(1), obs=p)

    def guide(self, a, p):
    
        batch_size = a.shape[0]

        # Move data to the correct device
        a = a.to(self.device)
        p = p.to(self.device)
        
        # register modules
        pyro.module("encoder_s_mu", self.encoder_s_mu)
        pyro.module("encoder_s_logvar", self.encoder_s_logvar)
        pyro.module("encoder_a_mu", self.encoder_a_mu)
        pyro.module("encoder_a_logvar", self.encoder_a_logvar)
        pyro.module("encoder_p_mu", self.encoder_p_mu)
        pyro.module("encoder_p_logvar", self.encoder_p_logvar)
        
        # plate
        with pyro.plate("batch", batch_size, dim=-1):
            
            # Compute approximate posterior for Zs using a neural network
            ap_concat = torch.stack((a, p), dim=1)
            # ap_concat = torch.cat((a, p), dim=1)
            mu_Zs = self.encoder_s_mu(ap_concat)
            sigma_Zs = torch.exp(self.encoder_s_logvar(ap_concat)) + 1e-6  # Ensure positivity
            Zs = pyro.sample("Zs", dist.Normal(mu_Zs, sigma_Zs).to_event(1))

            # Compute approximate posterior for Za (depends on Zs)
            # Zs_a_concat = torch.cat((Zs, a), dim=1)
            mu_Za = self.encoder_a_mu(Zs)
            sigma_Za = torch.exp(self.encoder_a_logvar(Zs)) + 1e-6
            Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))

            # Compute approximate posterior for Zp (depends on Zs and p)
            # Zs_p_concat = torch.cat((Zs, p), dim=1)
            mu_Zp = self.encoder_p_mu(Zs)
            sigma_Zp = torch.exp(self.encoder_p_logvar(Zs)) + 1e-6
            Zp = pyro.sample("Zp", dist.Normal(mu_Zp, sigma_Zp).to_event(1))

class IcasspVAE(nn.Module):
    """
    
    Variational Autoencoder (VAE) with a two-level latent structure.
    
    This model consists of two latent variables:
    - Za: Amplitude latent variable
    - Zp: Phase latent variable
    
    Graphical Model:
        Za   Zp
        |    |
        A - Phi
    
    Attributes:
        input_dim (int): Dimension of the input data.
        latent_dim_a (int): Dimension of the amplitude latent variable Za.
        latent_dim_p (int): Dimension of the phase latent variable Zp.
        hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
    Methods:
        model(a, p):
            Defines the generative model.
        guide(a, p):
            Defines the inference model.
    
    """
    def __init__(self, input_dim, latent_dim_a=64, latent_dim_p=64, hidden_dim=1024, device='cpu'):
        
        super().__init__()
        
        # define fields
        self.device = device
        self.input_dim = input_dim
        self.latent_dim_a = latent_dim_a
        self.latent_dim_p = latent_dim_p

        # decoder amplitude: Za to A (needs mu to be positive for lognormal)
        self.decoder_a_mu = nn.Sequential(
            nn.Linear(latent_dim_a, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2 * hidden_dim), 
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, input_dim),
            nn.ReLU()
        )
        self.decoder_a_logvar = nn.Sequential(
            nn.Linear(latent_dim_a, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2 * hidden_dim), 
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Linear(4 * hidden_dim, input_dim)
        )
        
        # decoder phi: (Zp,A) to Phi (needs mu to be real and kappa to be positive)
        self.decoder_p_mu = nn.Sequential(
            nn.Linear(latent_dim_p + input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(), 
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(), 
            nn.Linear(4 * hidden_dim, input_dim)
        )
        self.decoder_p_kappa = nn.Sequential(
            nn.Linear(latent_dim_p + input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(), 
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.ReLU(), 
            nn.Linear(4 * hidden_dim, input_dim)
        )

        # initialization to help cover the phi range
        self.decoder_p_mu[-1].weight.data.uniform_(-torch.pi, torch.pi)
        self.decoder_p_mu[-1].bias.data.zero_()
        
        # encoder: A to Za
        self.encoder_a_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim_a)
        )
        self.encoder_a_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim_a)
        )

        # encoders: P to Zp
        self.encoder_p_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim_p)
        )
        self.encoder_p_logvar = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim_p)
        )

        # Move model to device
        self.to(self.device)

    def model(self, a, p):
        
        batch_size = a.shape[0]
        
        # Move data to the correct device
        a = a.to(self.device)
        p = p.to(self.device)

        # standard normal parameters
        mu_prior = torch.zeros(batch_size, self.latent_dim_a, device=self.device)
        sigma_prior = torch.ones(batch_size, self.latent_dim_a, device=self.device)
        
        with pyro.plate("batch", batch_size, dim=-1):  # Define a batch plate
            
            # Sample latent variables
            Za = pyro.sample("Za",dist.Normal(mu_prior, sigma_prior).to_event(1))
            Zp = pyro.sample("Zp",dist.Normal(mu_prior, sigma_prior).to_event(1))

            # Decode amplitude and phase
            mu_A = self.decoder_a_mu(Za)
            sigma_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-6
            # a_A = torch.exp(self.decoder_a_mu(Za)) + 1e-3
            # b_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-3
            
            # Amplitude likelihood (Gaussian)
            pyro.sample("A_obs", dist.LogNormal(mu_A, sigma_A).to_event(1), obs=a)
            # pyro.sample("A_obs", dist.Normal(mu_A, sigma_A).to_event(1), obs=a)
            # pyro.sample("A_obs", dist.Beta(a_A, b_A).to_event(1), obs=a)

            # Decode phase
            mu_P = self.decoder_p_mu(torch.cat((Zp, a), dim=1))
            kappa_P = torch.exp(self.decoder_p_kappa(torch.cat((Zp, a), dim=1))) + 1e-6  # Ensure positivity

            # Phase likelihood (Von Mises)
            pyro.sample("P_obs", dist.VonMises(mu_P, kappa_P).to_event(1), obs=p)

    def guide(self, a, p):
    
        batch_size = a.shape[0]

        # Move data to the correct device
        a = a.to(self.device)
        p = p.to(self.device)

        with pyro.plate("batch", batch_size, dim=-1):  # Define the same batch plate
            
            # Compute approximate posterior for Za (depends on Zs)
            mu_Za = self.encoder_a_mu(a)
            sigma_Za = torch.exp(self.encoder_a_logvar(a)) + 1e-6
            Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))

            # Compute approximate posterior for Zp (depends on Zs and p)
            mu_Zp = self.encoder_p_mu(p)
            sigma_Zp = torch.exp(self.encoder_p_logvar(p)) + 1e-6
            Zp = pyro.sample("Zp", dist.Normal(mu_Zp, sigma_Zp).to_event(1))
