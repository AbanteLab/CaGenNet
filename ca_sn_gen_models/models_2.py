# Description: Pyro models for the snDGM and VAE models.

# torch
from statistics import variance
import torch
import torch.nn as nn
import torch.nn.functional as F

# pyro
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO

# sklearn
from sklearn.metrics import accuracy_score

# distributions
from ca_sn_gen_models.utils import superprint, gen_random_mask

########################################################################################
# Encoder and Decoder classes
########################################################################################
# Convolutional encoder for time domain data with circular padding
class ConvEncoder(nn.Module):
    """
    A convolutional encoder module for transforming input data into latent space representations.
    This module consists of convolutional layers followed by fully connected layers to compute
    the mean (`mu`) and log-variance (`logvar`) of the latent space distribution.
    Args:
        input_dim (int): The size of the input dimension (length of the input sequence).
        latent_dim (int): The size of the latent space (dimensionality of the latent representation).
    Attributes:
        conv1 (nn.Conv1d): First 1D convolutional layer with circular padding.
        conv2 (nn.Conv1d): Second 1D convolutional layer with circular padding.
        flatten (nn.Flatten): Layer to flatten the output of the convolutional layers.
        fc_mu (nn.Linear): Fully connected layer to compute the mean of the latent space.
        fc_logvar (nn.Linear): Fully connected layer to compute the log-variance of the latent space.
    Example:
        >>> import torch
        >>> from sndgm.models import ConvEncoder
        >>> input_dim = 100  # Length of the input sequence
        >>> latent_dim = 10  # Dimensionality of the latent space
        >>> encoder = ConvEncoder(input_dim, latent_dim)
        >>> x = torch.randn(16, input_dim)  # Batch of 16 sequences, each of length 100
        >>> loc,scl = encoder(x)
        >>> print(loc.shape)  # Expected output: torch.Size([16, 10])
        >>> print(scl.shape)  # Expected output: torch.Size([16, 10])
    """
    def __init__(self, input_dim, latent_dim, kernel_size = 3, padding = 5):
        super().__init__()
        
        # parameters
        self.padding = padding
        self.kernel_size = kernel_size
        
        # channels of convolutional layers
        self.num_chan_1 = 8
        self.num_chan_2 = 16
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, self.num_chan_1, kernel_size=self.kernel_size, padding=self.padding, padding_mode='circular')
        self.conv2 = nn.Conv1d(self.num_chan_1, self.num_chan_2, kernel_size=self.kernel_size, padding=self.padding, padding_mode='circular')
        
        # Global pooling
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Calculate the correct flattened size after convolutional layers
        # NOTE: conv_output_size = input_dim + 2 * padding - kernel_size + 1
        conv_output_size = input_dim + 2 * self.padding - self.kernel_size + 1
        conv_output_size += 2 * self.padding - self.kernel_size + 1
        flattened_size = self.num_chan_2 * conv_output_size
        
        # # Add additional fully connected layers for location
        # self.fc_loc_1 = nn.Linear(flattened_size, flattened_size // 2)
        # self.fc_loc_2 = nn.Linear(flattened_size // 2, flattened_size // 4)
        
        # # Add additional fully connected layers for logscale layers
        # self.fc_lgs_1 = nn.Linear(flattened_size, flattened_size // 2)
        # self.fc_lgs_2 = nn.Linear(flattened_size // 2, flattened_size // 4)
        
        # # Update fully connected layers with the correct size
        # self.fc_loc_out = nn.Linear(flattened_size // 4, latent_dim)
        # self.fc_lgs_out = nn.Linear(flattened_size // 4, latent_dim)
        
        # Add additional fully connected layers for location
        # self.fc_loc_1 = nn.Linear(self.num_chan_2, self.num_chan_2)
        # self.fc_lgs_1 = nn.Linear(self.num_chan_2, self.num_chan_2)
        
        # Update fully connected layers with the correct size
        self.fc_loc_out = nn.Linear(flattened_size, latent_dim)
        self.fc_lgs_out = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        
        # Reshape input from [batch_size, input_size] to [batch_size, 1, input_size]
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Global pooling layer
        # x = self.global_avg_pool(x).squeeze(-1) 
        
        # convert to 2D tensor
        x = self.flatten(x)
        
        # # compute 1st intermediate fully connected layers for loc and scale
        # loc_int = torch.relu(self.fc_loc_1(x))
        # lgs_int = torch.relu(self.fc_lgs_1(x))
        
        # # compute 2nd intermediate fully connected layers for loc and scale
        # loc_int = torch.relu(self.fc_loc_2(loc_int))
        # lgs_int = torch.relu(self.fc_lgs_2(lgs_int))
        
        # # compute loc and scale of distribution
        # loc = torch.tanh(self.fc_loc_out(loc_int))
        # scale = torch.exp(self.fc_lgs_out(lgs_int)) + 1e-6
        
        # # compute intermediate fully connected layers for loc
        # loc_int = torch.relu(self.fc_loc_1(x))
        # lgs_int = torch.relu(self.fc_lgs_1(x))
        
        # compute loc within [-5, 5]
        loc = 5.0 * torch.tanh(self.fc_loc_out(x))
        
        # compute scale of distribution
        scale = torch.exp(self.fc_lgs_out(x)) + 1e-6
        
        # return output
        return loc, scale

class ConvDecoder(nn.Module):
    """
    A convolutional decoder module for decoding latent representations into 
    reconstructed inputs. This module uses a fully connected layer followed 
    by transposed convolutional layers to upsample the latent representation.
    Attributes:
        input_dim (int): The dimensionality of the input data.
        fc (nn.Linear): Fully connected layer to project latent space to 
                        intermediate feature space.
        conv1 (nn.ConvTranspose1d): First transposed convolutional layer.
        conv2 (nn.ConvTranspose1d): Second transposed convolutional layer.
    Args:
        input_dim (int): The dimensionality of the input data.
        latent_dim (int): The dimensionality of the latent space.
    Methods:
        forward(z):
            Forward pass of the decoder. Takes a latent vector `z` and 
            reconstructs the input.
    Example:
        >>> import torch
        >>> from sndgm.models import ConvDecoder
        >>> latent_dim,input_dim,batch_size = 10,100,16
        >>> decoder = ConvDecoder(latent_dim, input_dim)
        >>> z = torch.randn(batch_size, latent_dim)  # Random latent vectors
        >>> print(z.shape)
        >>> loc,scl = decoder(z)
        >>> print(loc.shape)  # Expected output: torch.Size([16, 100])
        >>> print(scl.shape)  # Expected output: torch.Size([16, 100])
    """
    def __init__(self, latent_dim, input_dim, kernel_size = 3, padding = 5):
        super().__init__()
        
        # store dimensions
        self.padding = padding
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        
        # channels of convolutional layers
        self.num_chan_1 = 16
        self.num_chan_2 = 8
        
        # Fully connected layers
        self.fc1 = nn.Linear(latent_dim, self.num_chan_1 * input_dim)
        
        # Convolutional layers
        self.conv1 = nn.ConvTranspose1d(self.num_chan_1, self.num_chan_2, kernel_size=self.kernel_size, padding=self.padding)
        self.conv2 = nn.ConvTranspose1d(self.num_chan_2, 1, kernel_size=self.kernel_size, padding=self.padding)

        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Calculate the correct flattened size after convolutional layers 
        # 1. (batch_size, self.num_chan_1, input_dim) -> (batch_size, self.num_chan_2, dim_out_conv1)
        # 2. (batch_size, self.num_chan_2, dim_out_conv1) -> (batch_size, 1, dim_out_conv2)
        dim_out_conv1 = (input_dim - 1) - 2 * self.padding + self.kernel_size
        dim_out_conv2 = (dim_out_conv1 - 1) - 2 * self.padding + self.kernel_size
        
        # # Add intermediate fully connected layers
        # self.fc_loc_1 = nn.Linear(dim_out_conv2, input_dim)
        # self.fc_lgs_1 = nn.Linear(dim_out_conv2, input_dim)
        
        # # linear layer from conv2 to input_dim
        # self.fc_loc_out = nn.Linear(input_dim, input_dim)
        # self.fc_lgs_out = nn.Linear(input_dim, input_dim)
        
        # linear layer from conv2 to input_dim
        self.fc_loc_out = nn.Linear(dim_out_conv2, input_dim)
        self.fc_lgs_out = nn.Linear(dim_out_conv2, input_dim)

    def forward(self, z):
        
        # Fully connected layers
        x = self.fc1(z).view(-1, self.num_chan_1, self.input_dim)
        
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # convert to 2D tensor
        x = self.flatten(x)
        
        # # intermediate fully connected layer
        # loc_int = torch.relu(self.fc_loc_1(x))
        # lgs_int = torch.relu(self.fc_lgs_1(x))
        
        # # compute loc and scale
        # loc = torch.tanh(self.fc_loc_out(loc_int))
        # scale = torch.exp(self.fc_lgs_out(lgs_int)) + 1e-6
        
        # compute loc and scale
        loc = torch.tanh(self.fc_loc_out(x))
        scale = torch.exp(self.fc_lgs_out(x)) + 1e-6
        
        # return output
        return loc, scale

# Linear encoder
class Encoder(nn.Module):
    """
    Encoder neural network module for dimensionality reduction and feature extraction.
    This class defines an encoder that maps input data to a latent space representation
    using a series of fully connected layers with ReLU activations. The architecture
    is dynamically generated based on the provided dimensions.
    Attributes:
        net (nn.Sequential): A sequential container of layers forming the encoder network.
    Args:
        input_dim (int): The dimensionality of the input data.
        latent_dim (int): The dimensionality of the latent space representation.
        enc_dims (list of int): A list of integers specifying the number of units in each
            hidden layer of the encoder.
    Methods:
        forward(x):
            Passes the input data through the encoder network to produce the latent
            representation.
    """
    def __init__(self, input_dim, latent_dim, enc_dims, output=None):
        
        super().__init__()
        
        # generate layers according to the dimensions
        layers = [nn.Linear(input_dim, enc_dims[0]), nn.ReLU()]
        for i in range(len(enc_dims) - 1):
            layers.extend([nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU()])
        
        layers.append(nn.Linear(enc_dims[-1], latent_dim))

        if output is None:
            self.net = nn.Sequential(*layers)
        elif output == "softplus":
            self.net = nn.Sequential(*layers, nn.Softplus())
        elif output == "sigmoid":
            self.net = nn.Sequential(*layers, nn.Sigmoid())
        elif output == "tanh":
            self.net = nn.Sequential(*layers, nn.Tanh())
        else:
            raise ValueError("Output activation not recognized.")
        
    def forward(self, x):
        return self.net(x)

# Linear decoder
class Decoder(nn.Module):
    """
    A neural network decoder module for transforming latent representations into output data.
    This class defines a fully connected feedforward neural network with customizable architecture
    and output activation functions. It is typically used in generative models to decode latent
    variables into reconstructed data.
    Attributes:
        net (nn.Sequential): The sequential container holding the layers of the decoder network.
    Args:
        latent_dim (int): The dimensionality of the latent space (input to the decoder).
        output_dim (int): The dimensionality of the output space (output of the decoder).
        dec_dims (list of int): A list specifying the number of units in each hidden layer of the decoder.
        output (str, optional): The activation function to apply to the output layer. Options are:
            - None: No activation function is applied.
            - "softplus": Applies the Softplus activation function.
            - "sigmoid": Applies the Sigmoid activation function.
            Defaults to None.
    Methods:
        forward(x):
            Performs a forward pass through the decoder network.
            Args:
                x (torch.Tensor): The input tensor of shape (batch_size, latent_dim).
            Returns:
                torch.Tensor: The output tensor of shape (batch_size, output_dim).
    """
    def __init__(self, latent_dim, output_dim, dec_dims, output=None):
        
        super().__init__()
        
        layers = [nn.Linear(latent_dim, dec_dims[0]), nn.ReLU()]
        for i in range(len(dec_dims) - 1):
            layers.extend([nn.Linear(dec_dims[i], dec_dims[i + 1]), nn.ReLU()])
        layers.append(nn.Linear(dec_dims[-1], output_dim))
        
        if output is None:
            self.net = nn.Sequential(*layers)
        elif output == "softplus":
            self.net = nn.Sequential(*layers, nn.Softplus())
        elif output == "sigmoid":
            self.net = nn.Sequential(*layers, nn.Sigmoid())
        elif output == "tanh":
            self.net = nn.Sequential(*layers, nn.Tanh())
        else:
            raise ValueError("Output activation not recognized.")
        
    def forward(self, x):
        
        return self.net(x)

########################################################################################
# Generative model based on Exponential Family Factor Analysis (EFFA)
########################################################################################
class FA(nn.Module):
    def __init__(self, D, K, like=dist.Normal, domain = 'time', device="cpu"):
        """
        Factor analysis model with flexible likelihood.

            X = ZW + eps, eps ~ like(0, sigma)
            
        where Z is the latent variables, W are the factor loadings, and sigma is the likelihood scale.

        Args:
            D (int): Number of observed features.
            K (int): Number of latent dimensions.
            like (function): Pyro distribution for likelihood.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.D = D
        self.K = K
        self.like = like
        self.device = device
        self.domain = domain
        self.Hhat = None
        
        # Move model to device
        self.to(self.device)
                         
    def model(self, x):
        """
        Defines the generative model P(X | Z, W, sigma^2).
        """

        # Move data to the correct device
        x = x.to(self.device)
        
        # Get the batch size from input x
        batch_size = x.shape[0]
        
        # prior parameters for sigma
        x_scl_sigma = torch.ones(self.D, device=self.device)
        
        # Prior for factor loadings W
        with pyro.plate("observed_dim", self.D, dim=-1):
            
            # Prior for likelihood scale (non-isotropic model)
            sigma = pyro.sample("sigma", dist.HalfNormal(scale=x_scl_sigma))
            
            with pyro.plate("latent_dim", self.K, dim=-2):
                
                # prior scale of W
                w_scl_prior = torch.ones(self.K, self.D, device=self.device)
                
                if self.domain == 'time':
                    
                    # prior parameters for W
                    w_loc_prior = torch.zeros(self.K, self.D, device=self.device)
                    
                    # sample
                    W = pyro.sample("W", dist.Normal(w_loc_prior, w_scl_prior))
                        
                else:
                    
                    # prior parameters for W
                    w_loc_prior = torch.ones(self.K, self.D, device=self.device)
                    
                    # Gamma has low variance
                    W = pyro.sample("W", dist.Gamma(w_loc_prior, w_scl_prior))

        # Prior for latent variables Z
        with pyro.plate("data", batch_size):
    
            # prior scale of Z
            z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
                
            if self.domain == 'time':
                
                # prior parameters for Z
                z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
                
                # Standard normal prior for latent variables
                Z = pyro.sample("Z", dist.Normal(z_loc_prior, z_scl_prior).to_event(1))
            
            else:
                
                # prior parameters for Z
                z_loc_prior = torch.ones(batch_size, self.K, device=self.device)
                
                # Gamma prior for latent variables
                Z = pyro.sample("Z", dist.Gamma(z_loc_prior, z_scl_prior).to_event(1))
        
            # Likelihood P(X | Z, W, sigma)
            X = pyro.sample("X", self.like(Z @ W, sigma).to_event(1), obs=x)

    def guide(self, x):
        """
        Defines the variational guide q(W, Z, sigma).
        """
        
        # Move data to the correct device
        x = x.to(self.device)

        # Get the batch size from input x
        batch_size = x.shape[0]

        # Variational parameters for W
        W_scale = pyro.param("W_scale", torch.full((self.K, self.D), 1.0, device=self.device), constraint=dist.constraints.positive)

        # generate posterior for W
        with pyro.plate("observed_dim", self.D, dim=-1):
            
            # # Sample posterior of sigma^2 (isotropic covariance - pPCA)
            # sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)
            # sigma = pyro.sample("sigma", dist.Delta(sigma_loc))
            
            # Sample posterior of sigma^2 (non-isotropic covariance)
            sigma_loc = pyro.param("sigma_loc", torch.ones(self.D, device=self.device), constraint=dist.constraints.positive)
            sigma = pyro.sample("sigma", dist.Delta(sigma_loc))
            
            with pyro.plate("latent_dim", self.K, dim=-2):
                if self.domain == 'time':
                    
                    # Variational parameters for W
                    W_loc = pyro.param("W_loc", torch.zeros(self.K, self.D, device=self.device))
                    
                    # Sample posterior for W (isotropic covariance)
                    W = pyro.sample("W", dist.Normal(W_loc, W_scale))
                    
                else:
                    
                    # Variational parameters for W
                    W_loc = pyro.param("W_loc", torch.full((self.K, self.D), 1.0, device=self.device), constraint=dist.constraints.positive)
                    
                    # Sample posterior for W (Gamma distribution)
                    W = pyro.sample("W", dist.Gamma(W_loc, W_scale))
        
        # Variational parameters for Z
        with pyro.plate("data", batch_size):
            
            # Variational parameters for Z
            Z_scale = pyro.param("Z_scale", torch.ones(batch_size, self.K, device=self.device), constraint=dist.constraints.positive)
            
            if self.domain == 'time':
                
                # Variational parameters for Z
                Z_loc = pyro.param("Z_loc", torch.zeros(batch_size, self.K, device=self.device))
                
                # Sample posterior for Z (isotropic covariance)
                Z = pyro.sample("Z", dist.Normal(Z_loc, Z_scale).to_event(1))
            
            else:
                
                # sample posterior for Z (Gamma distribution)
                Z_loc = pyro.param("Z_loc", torch.full((batch_size, self.K), 1.0, device=self.device), constraint=dist.constraints.positive)
                
                # Sample posterior for Z (Gamma distribution)
                Z = pyro.sample("Z", dist.Gamma(Z_loc, Z_scale).to_event(1))
                
    def train_model(self, data_loader, num_epochs=5000, lr=5e-3, patience=500, min_delta=1e-3):
        """
        Trains the FA model using SVI.

        Args:
            data (torch.Tensor): Input data of shape [N, D].
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            patience (int): Early stopping patience.
            min_delta (float): Minimum change in validation loss for early stopping.

        Returns:
            list: Training loss history.
        """
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=TraceMeanField_ELBO())

        # Training loop
        patience_counter = 0
        loss_history = []
        best_loss = float("inf")

        # Train the model
        for epoch in range(num_epochs):
            
            epoch_loss = 0.0
            
            # Training phase
            for batch in data_loader:
                x_batch = batch[0].to(self.device)
                epoch_loss += svi.step(x_batch)
            epoch_loss /= len(data_loader.dataset)
            loss_history.append(epoch_loss)

            # Early stopping
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            superprint(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # compute Hhat
        with torch.no_grad():
            # E[Z|X]= XHhat
            W_loc = pyro.param("W_loc")
            sigma_loc = pyro.param("sigma_loc")
            Sm = torch.diag(1.0 / sigma_loc) # Sm.shape
            A = W_loc @ Sm @ W_loc.T
            Am = torch.linalg.inv(A + torch.eye(A.shape[0], device=A.device))
            self.Hhat = Sm @ W_loc.T @ Am
        
        # return loss history
        return loss_history

    def get_posterior(self):
        """
        Returns the variational parameters of latent variables Z, matrix W, and sigma given the observed data.

        Returns:
            tuple: Posterior of Z (torch.Tensor), W (torch.Tensor), and sigma (torch.Tensor).
        """
        print(self.parameters())
        self.eval()
        with torch.no_grad():
            Z_loc = pyro.param("Z_loc").cpu().detach().numpy()
            W_loc = pyro.param("W_loc").cpu().detach().numpy()
            sigma_loc = pyro.param("sigma_loc").cpu().detach().numpy()

        return Z_loc, W_loc, sigma_loc

########################################################################################
# Generative model based on Exponential Family Factor Analysis (EFFA) with Zx and Zy
########################################################################################
class CustomELBO(pyro.infer.Trace_ELBO):
    def __init__(self, lambda_reg=1e6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_reg = lambda_reg

    def compute_orthogonality_loss(self, Zx, Zy):
        """ Penalitza la correlació entre Zx i Zy per forçar independència. """
        Zx_centered = Zx - Zx.mean(dim=0, keepdim=True)
        Zy_centered = Zy - Zy.mean(dim=0, keepdim=True)
        
        batch_size = Zx.shape[0]
        std_Zx = Zx_centered.std(dim=0, keepdim=True)
        std_Zy = Zy_centered.std(dim=0, keepdim=True)
        covariance_matrix = (Zx_centered.T @ Zy_centered) / (batch_size * std_Zx * std_Zy.T)
        
        return self.lambda_reg * torch.norm(covariance_matrix, p="fro")

    def loss(self, model, guide, *args, **kwargs):
        """ Calcula la pèrdua ELBO + regularització """
        
        loss = super().loss(model, guide, *args, **kwargs)  # ELBO normal
        print(f"ELBO loss: {loss:.4f}")
        
        # Executa el guia per obtenir Zx i Zy
        guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
        Zx = guide_trace.nodes["Zx"]["value"]
        Zy = guide_trace.nodes["Zy"]["value"]
        
        # Calcula la pèrdua d’ortogonalitat
        ortho_loss = self.compute_orthogonality_loss(Zx, Zy)
        print(f"Orthogonality loss: {ortho_loss:.4f}")
        
        return loss + ortho_loss  # Retorna la pèrdua total
    
class isoFA(nn.Module):
    def __init__(self, D, K, NS, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.D = D
        self.K = K
        self.NS = NS
        self.Hhat = None
        self.device = device
        
        # prior parameters
        self.x_scl_sigma = torch.ones(self.D, device=self.device)
        
        # prior parameters for Wxy
        self.Wxy_loc_prior = torch.zeros(self.K, self.D, device=self.device)
        self.Wxy_scl_prior = torch.ones(self.K, self.D, device=self.device)
        
        # prior parameters for Wy
        self.Wy_loc_prior = torch.zeros(self.K // 2, self.NS, device=self.device)
        self.Wy_scl_prior = torch.ones(self.K // 2, self.NS, device=self.device)
        
        # Move model to device
        self.to(self.device)
                         
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, W, sigma^2).
        """

        # Move data to the correct device
        x = x.to(self.device)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K // 2, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K // 2, device=self.device)
        
        # Prior for factor loadings W
        with pyro.plate("observed_dim", self.D, dim=-1):
            
            # Prior for likelihood scale (non-isotropic model)
            sigma = pyro.sample("sigma", dist.HalfNormal(scale=self.x_scl_sigma))
            
            with pyro.plate("latent_dim_wxy", self.K, dim=-2):
                
                # sample matrix to reconstruct X
                Wxy = pyro.sample("Wxy", dist.Normal(self.Wxy_loc_prior, self.Wxy_scl_prior))
        
        # Prior for factor loadings W
        with pyro.plate("num_samps", self.NS, dim=-1):
            
            with pyro.plate("latent_dim_wy", self.K // 2, dim=-2):
                
                # sample vector to reconstruct Y
                Wy = pyro.sample("Wy", dist.Normal(self.Wy_loc_prior, self.Wy_scl_prior))

        # Prior for latent variables Z
        with pyro.plate("data", batch_size):
            
            # Standard normal prior for latent variables
            Zx = pyro.sample("Zx", dist.Normal(z_loc_prior, z_scl_prior).to_event(1))
            Zy = pyro.sample("Zy", dist.Normal(z_loc_prior, z_scl_prior).to_event(1))
            
            # concatenate Zx and Zy
            Zxy = torch.cat((Zx, Zy), dim=-1)
            
            # Likelihood P(X | Zxy, Wxy, sigma)
            pyro.sample("X", dist.Normal(Zxy @ Wxy, sigma).to_event(1), obs=x)
            
            # Likelihood P(Y | Zy, Wy, sigma)
            pyro.sample("Y", dist.Categorical(logits=Zy @ Wy).to_event(1), obs=y)

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma).
        """
        
        # Move data to the correct device
        x = x.to(self.device)

        # Get the batch size from input x
        batch_size = x.shape[0]

        # Variational parameters for Wxy
        Wxy_loc = pyro.param("Wxy_loc", torch.zeros(self.K, self.D, device=self.device))
        Wxy_scale = pyro.param("Wxy_scale", torch.full((self.K, self.D), 1.0, device=self.device), constraint=dist.constraints.positive)

        # Variational parameters for X noise
        sigma_loc = pyro.param("sigma_loc", torch.ones(self.D, device=self.device), constraint=dist.constraints.positive)
        
        # generate posterior for Wxy
        with pyro.plate("observed_dim", self.D, dim=-1):
            
            # Sample posterior of sigma^2 (non-isotropic covariance)
            pyro.sample("sigma", dist.Delta(sigma_loc))
            
            with pyro.plate("latent_dim_wxy", self.K, dim=-2):
                
                # Sample posterior for Wxy
                pyro.sample("Wxy", dist.Normal(Wxy_loc, Wxy_scale))

        # Variational parameters for Wy
        Wy_loc = pyro.param("Wy_loc", torch.zeros(self.K // 2, self.NS, device=self.device))
        Wy_scale = pyro.param("Wy_scale", torch.full((self.K // 2, self.NS), 1.0, device=self.device), constraint=dist.constraints.positive)
        
        # generate posterior for Wy
        with pyro.plate("num_samps", self.NS, dim=-1):
            
            with pyro.plate("latent_dim_wy", self.K // 2, dim=-2):
                
                # Sample posterior for Wy
                pyro.sample("Wy", dist.Normal(Wy_loc, Wy_scale))
        
        # Variational parameters for Z
        with pyro.plate("data", batch_size):
            
            # Variational parameters for Zx
            Zx_loc = pyro.param("Zx_loc", torch.zeros(batch_size, self.K // 2, device=self.device))
            Zx_scale = pyro.param("Zx_scale", torch.ones(batch_size, self.K // 2, device=self.device), constraint=dist.constraints.positive)
            
            # Sample posterior for Z
            pyro.sample("Zx", dist.Normal(Zx_loc, Zx_scale).to_event(1))
            
            # Variational parameters for Zy
            Zy_loc = pyro.param("Zy_loc", torch.zeros(batch_size, self.K // 2, device=self.device))
            Zy_scale = pyro.param("Zy_scale", torch.ones(batch_size, self.K // 2, device=self.device), constraint=dist.constraints.positive)
            
            # Sample posterior for Z
            pyro.sample("Zy", dist.Normal(Zy_loc, Zy_scale).to_event(1))
                
    def train_model(self, data_loader, num_epochs=5000, lr=1e-1, patience=20, min_delta=5, lambda_reg=1e6):
        """
        Trains the FA model using SVI.

        Args:
            data (torch.Tensor): Input data of shape [N, D].
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            patience (int): Early stopping patience.
            min_delta (float): Minimum change in validation loss for early stopping.

        Returns:
            list: Training loss history.
        """
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=CustomELBO(lambda_reg=lambda_reg))
        # svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Training loop
        loss_history = []
        patience_counter = 0
        best_loss = float("inf")

        # Train the model
        for epoch in range(num_epochs):
            
            epoch_loss = 0.0
            
            # Training phase
            for batch in data_loader:
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                epoch_loss += svi.step(x_batch, y_batch)
                
            epoch_loss /= len(data_loader.dataset)
            loss_history.append(epoch_loss)
    
            # Early stopping
            if epoch_loss < best_loss - min_delta:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            superprint(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # compute Hhat
        with torch.no_grad():
            # E[Z|X]= XHhat
            Wxy_loc = pyro.param("Wxy_loc")
            sigma_loc = pyro.param("sigma_loc")
            Sm = torch.diag(1.0 / sigma_loc) # Sm.shape
            A = Wxy_loc @ Sm @ Wxy_loc.T
            Am = torch.linalg.inv(A + torch.eye(A.shape[0], device=A.device))
            self.Hhat = Sm @ Wxy_loc.T @ Am
        
        # return loss history
        return loss_history

    def get_posterior(self):
        """
        Returns the variational parameters of latent variables Z, matrix W, and sigma given the observed data.

        Returns:
            tuple: Posterior of Z (torch.Tensor), W (torch.Tensor), and sigma (torch.Tensor).
        """
        self.eval()
        with torch.no_grad():
            Zx_loc = pyro.param("Zx_loc").cpu().detach().numpy()
            Zy_loc = pyro.param("Zy_loc").cpu().detach().numpy()
            Wy_loc = pyro.param("Wy_loc").cpu().detach().numpy()
            Wxy_loc = pyro.param("Wxy_loc").cpu().detach().numpy()
            sigma_loc = pyro.param("sigma_loc").cpu().detach().numpy()

        return Zx_loc, Zy_loc, Wy_loc, Wxy_loc, sigma_loc

############################################################################################################
# Validated deep generative models for time-series data
############################################################################################################
class FixedVarMlpVAE(nn.Module):
    """
    MlpVAE: A Multi-Layer Perceptron Variational Autoencoder (VAE) for time-series data that assumes a fixed
    scale of the likelihood function p(x|z).

    This class implements a VAE with fully connected layers for both the encoder and decoder.
    The encoder maps input data to a latent space, and the decoder reconstructs the input from
    the latent representation. The model uses Pyro for probabilistic programming and supports
    variational inference.

    Attributes:
        D (int): Input dimension (number of time steps).
        K (int): Latent dimension.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Methods:
        encode(x): Encodes the input data into a latent representation.
        decode(z): Decodes the latent representation into the original data space.
        model(x): Defines the generative model P(X | Z).
        guide(x): Defines the variational guide q(Z | X).
        train_model(data_loader, val_loader, num_epochs, lr, patience, min_delta):
            Trains the VAE using Stochastic Variational Inference (SVI).
    """
    def __init__(self, D, K, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "FixedVarMlpVAE"
        self.fix_scl = True
        self.sup = False

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.scl = scl          # fixed scale of the likelihood
        self.device = device

        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(),
            nn.Linear(D // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x):
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Process the combined representation through the encoder network
        x = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x)
        scale = torch.clamp(self.encoder_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z):
        
        # run through MLP
        z = self.decoder_mlp(z)
        
        # compute location [N, D]
        loc = self.decoder_loc(z)
        
        # return the location and log scale
        return loc
    
    def model(self, x):
        
        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x):
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, roll=False):
        
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

                x_batch = batch[0].to(self.device)

                # apply random roll to x_batch
                if roll:
                    x_batch = torch.roll(x_batch, shifts=torch.randint(0, x_batch.size(1), (1,)).item(), dims=1)
                    
                train_loss += svi.step(x_batch) / x_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device)
                    val_loss += svi.evaluate_loss(x_batch) / x_batch.size(0)                    
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                z_loc, _ = self.encode(x_batch)
                xhat = self.decode(z_loc)
        
        return z_loc, xhat

class LearnedVarMlpVAE(nn.Module):
    """
    MlpVAE: A Multi-Layer Perceptron Variational Autoencoder (VAE) for time-series data that learns the
    scale of the likelihood function p(x|z).

    This class implements a VAE with fully connected layers for both the encoder and decoder.
    The encoder maps input data to a latent space, and the decoder reconstructs the input from
    the latent representation. The model uses Pyro for probabilistic programming and supports
    variational inference.

    Attributes:
        D (int): Input dimension (number of time steps).
        K (int): Latent dimension.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Methods:
        encode(x): Encodes the input data into a latent representation.
        decode(z): Decodes the latent representation into the original data space.
        model(x): Defines the generative model P(X | Z).
        guide(x): Defines the variational guide q(Z | X).
        train_model(data_loader, val_loader, num_epochs, lr, patience, min_delta):
            Trains the VAE using Stochastic Variational Inference (SVI).
    """
    def __init__(self, D, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "LearnedVarMlpVAE"
        self.fix_scl = False
        self.sup = False

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(),
            nn.Linear(D // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        self.decoder_scl = nn.Sequential(nn.Linear(D // 2, D), nn.Softplus())
        
        # Move model to device
        self.to(self.device)

    def encode(self, x):
        
        # Process the combined representation through the encoder network
        x = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x)
        scl = torch.clamp(self.encoder_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z):
        
        # run through MLP
        z = self.decoder_mlp(z)
        
        # compute location [N, D]
        loc = self.decoder_loc(z)
        
        # compute scale [N, D]
        scl = torch.clamp(self.decoder_scl(z), min=1e-3)
        
        # return the location and log scale
        return loc, scl
    
    def model(self, x):
        
        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        pyro.module("decoder_scl", self.decoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc,scl = self.decode(z)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, scl), obs=x)

    def guide(self, x):
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                x_batch = batch[0].to(self.device)
                train_loss += svi.step(x_batch) / x_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device)
                    val_loss += svi.evaluate_loss(x_batch) / x_batch.size(0)                    
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                z_loc, _ = self.encode(x_batch)
                xhat, _ = self.decode(z_loc)
        
        return z_loc, xhat

class FixedVarSupMlpVAE(nn.Module):
    """
    FixedVarSupMlpVAE is a supervised variational autoencoder (VAE) model with fixed variance, implemented using PyTorch and Pyro. 
    It is designed for semi-supervised learning tasks where both input data and class labels are available. The model uses 
    multi-layer perceptrons (MLPs) for both the encoder and decoder networks, and incorporates class label information 
    via concatenation with the input and latent variables. The encoder maps input data and labels to a latent space, 
    while the decoder reconstructs the input from the latent representation and labels. The model supports training 
    with early stopping and tracks both training and validation loss histories.
        D (int): Number of input features or time steps.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    Attributes:
        D (int): Number of input features or time steps.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes.
        device (str): Device used for computation.
        encoder_mlp (nn.Sequential): Encoder MLP network.
        encoder_loc (nn.Sequential): Network to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Network to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): Decoder MLP network.
        decoder_loc (nn.Sequential): Network to compute reconstructed data mean.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into reconstructed data.
        model(x, y): Defines the generative model for Pyro.
        guide(x, y): Defines the variational guide for Pyro.
        train_model(data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3): 
            Trains the model using SVI with early stopping and returns loss histories.
    """
    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "FixedVarSupMlpVAE"
        self.fix_scl = True
        self.sup = True
        self.beta = 1.0         # weight of the KL divergence term

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # scale of the output distribution
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D + NS, (D + NS) // 2), nn.ReLU(),
            nn.Linear((D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scl = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, W, sigma^2).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            # Weigh KL divergence by scaling factor (e.g., beta)
            # with pyro.poutine.scale(scale=self.beta):  # Change 0.1 to your desired KL weight
            with pyro.plate("latent_dim", self.K, dim=-1):
                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
                
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma) with beta weighting for KL divergence.
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):
            # Weigh KL divergence by scaling factor (e.g., beta)
            # with pyro.poutine.scale(scale=self.beta):
            with pyro.plate("latent_dim", self.K, dim=-1):
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, roll=False):
        
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
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                # apply random roll to x_batch
                if roll:
                    x_batch = torch.roll(x_batch, shifts=torch.randint(0, x_batch.size(1), (1,)).item(), dims=1)

                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)

                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    
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
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch)
                xhat = self.decode(z_loc,y_batch)
        
        return z_loc, xhat



class FixedVarSupMlpBVAE(nn.Module):
    """
    FixedVarSupMlpVAE is a supervised variational autoencoder (VAE) model with fixed variance, implemented using PyTorch and Pyro. 
    It is designed for semi-supervised learning tasks where both input data and class labels are available. The model uses 
    multi-layer perceptrons (MLPs) for both the encoder and decoder networks, and incorporates class label information 
    via concatenation with the input and latent variables. The encoder maps input data and labels to a latent space, 
    while the decoder reconstructs the input from the latent representation and labels. The model supports training 
    with early stopping and tracks both training and validation loss histories.
        D (int): Number of input features or time steps.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    Attributes:
        D (int): Number of input features or time steps.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes.
        device (str): Device used for computation.
        encoder_mlp (nn.Sequential): Encoder MLP network.
        encoder_loc (nn.Sequential): Network to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Network to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): Decoder MLP network.
        decoder_loc (nn.Sequential): Network to compute reconstructed data mean.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into reconstructed data.
        model(x, y): Defines the generative model for Pyro.
        guide(x, y): Defines the variational guide for Pyro.
        train_model(data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3): 
            Trains the model using SVI with early stopping and returns loss histories.
    """
    def __init__(self, D, K, NS, scl=1e-3, beta = 1.0, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "FixedVarSupMlpVAE"
        self.fix_scl = True
        self.sup = True
        self.beta = beta        # weight of the KL divergence term

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # scale of the output distribution
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D + NS, (D + NS) // 2), nn.ReLU(),
            nn.Linear((D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scl = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, W, sigma^2).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            # Weigh KL divergence by scaling factor (e.g., beta)
            with pyro.poutine.scale(scale=self.beta):  # Change 0.1 to your desired KL weight
                with pyro.plate("latent_dim", self.K, dim=-1):
                    # Standard normal prior for latent variables
                    z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
                
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma) with beta weighting for KL divergence.
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):
            # Weigh KL divergence by scaling factor (e.g., beta)
            with pyro.poutine.scale(scale=self.beta):
                with pyro.plate("latent_dim", self.K, dim=-1):
                    # sample z
                    pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, roll=False):
        
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
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                # apply random roll to x_batch
                if roll:
                    x_batch = torch.roll(x_batch, shifts=torch.randint(0, x_batch.size(1), (1,)).item(), dims=1)

                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)

                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    
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
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch)
                xhat = self.decode(z_loc,y_batch)
        
        return z_loc, xhat
    
############################################################################################################
# Supervised Deep Generative Model for time domain data (X) based on MLP embeddings that learns the 
# output scale.
############################################################################################################
class LearnedVarSupMlpVAE(nn.Module):
    """
    A supervised variational autoencoder (VAE) model with learned variance, implemented using multilayer 
    perceptrons (MLPs) for both encoder and decoder networks. This model is designed for semi-supervised 
    or supervised learning tasks where both input data and class labels are available. The encoder network
    maps the concatenated input data and one-hot encoded labels into a latent space, parameterizing the mean
    and scale of the latent distribution. The decoder reconstructs the input data from the latent 
    representation and labels, also predicting the mean and scale of the output distribution.
    
    The model leverages Pyro for probabilistic modeling and inference, supporting stochastic variational 
    inference (SVI) for training. It includes methods for encoding, decoding, defining the generative 
    model and variational guide, and a training loop with early stopping.
        D (int): Number of input features (time steps).
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoding).
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Default is "cpu".
    Attributes:
        D (int): Number of input features.
        K (int): Latent dimension.
        NS (int): Number of classes.
        device (str): Device used for computation.
        encoder_mlp (nn.Sequential): Encoder MLP network.
        encoder_loc (nn.Sequential): Encoder network for latent mean.
        encoder_scl (nn.Sequential): Encoder network for latent scale.
        decoder_mlp (nn.Sequential): Decoder MLP network.
        decoder_loc (nn.Sequential): Decoder network for output mean.
        decoder_scl (nn.Sequential): Decoder network for output scale.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into reconstructed data mean and scale.
        model(x, y): Defines the generative model P(X | Z, Y) for Pyro.
        guide(x, y): Defines the variational guide q(Z | X, Y) for Pyro.
        train_model(data_loader, val_loader, num_epochs, lr, patience, min_delta): Trains the model using SVI with early stopping.
    """
    def __init__(self, D, K, NS, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "LearnedVarSupMlpVAE"
        self.fix_scl = False
        self.sup = True

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D + NS, (D + NS) // 2), nn.ReLU(),
            nn.Linear((D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        nn.init.zeros_(self.encoder_loc[0].weight)
        nn.init.zeros_(self.encoder_loc[0].bias)
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        nn.init.ones_(self.encoder_scl[0].weight)
        nn.init.ones_(self.encoder_scl[0].bias)
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        self.decoder_scl = nn.Sequential(nn.Linear(D // 2, D), nn.Softplus())
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # compute scale [N, D]
        scl = torch.clamp(self.decoder_scl(zy), min=1e-3)
        
        # return the location and log scale
        return loc, scl
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z,Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        pyro.module("decoder_scl", self.decoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc,scl = self.decode(z, y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(Theta,Z | X,Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)
                
                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch)
                xhat, _ = self.decode(z_loc,y_batch)
        
        return z_loc, xhat
    
############################################################################################################
# Experimental models
############################################################################################################
class LearnedVarLstmVAE(nn.Module):
    """
    Variational Autoencoder (VAE) with LSTM-based encoder and decoder for sequential data.
    This model is designed for learning latent variable representations of time series data using a 
    bidirectional LSTM encoder and LSTM-based decoder. The encoder maps input sequences to a latent 
    space, parameterized by a mean and scale, while the decoder reconstructs the input from the latent 
    variables. The model is compatible with Pyro for probabilistic modeling and variational inference.
    Args:
        D (int): Number of time steps in the input sequence.
        K (int): Dimensionality of the latent variable space.
        device (str, optional): Device to run the model on ('cpu' or 'cuda'). Default is 'cpu'.
    Attributes:
        D (int): Number of time steps.
        K (int): Latent dimension.
        device (str): Device for computation.
        encoder_x (nn.LSTM): Bidirectional LSTM encoder for input sequences.
        int_layer (nn.Sequential): Intermediate linear layer for encoder output.
        emb_to_loc (nn.Sequential): Linear layer to compute latent mean.
        emb_to_scl (nn.Sequential): Linear + Softplus layer to compute latent scale.
        decoder_mlp (nn.Sequential): MLP to map latent variables to decoder input.
        decoder_loc (nn.LSTM): LSTM decoder for reconstructing sequence mean.
        decoder_scl (nn.LSTM): LSTM decoder for reconstructing sequence scale.
        tanh (nn.Tanh): Tanh activation function.
        softplus (nn.Softplus): Softplus activation function.
    Methods:
        encode(x): Encodes input sequence x into latent mean and scale.
        decode(z): Decodes latent variable z into reconstructed sequence mean and scale.
        model(x): Pyro model definition for variational inference.
        guide(x): Pyro guide definition for variational inference.
    """
    def __init__(self, D, K, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "LearnedVarLstmVAE"
        self.fix_scl = False
        self.sup = False

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.device = device
        
        # Memory LSTM ∝ batch_size × seq_len × hidden_dim × num_layers × (2 if bidirectional)
        
        # define proper LSTM encoder
        self.encoder_x = nn.LSTM(1, K, batch_first=True, bidirectional=True, num_layers=2)
        self.int_layer = nn.Sequential(nn.Linear(2 * K, K))
        self.emb_to_loc = nn.Sequential(nn.Linear(K, K))
        self.emb_to_scl = nn.Sequential(nn.Linear(K, K), nn.Softplus())
        
        # define proper LSTM decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D)
            )
        self.decoder_loc = nn.LSTM(1, 1, batch_first=True, bidirectional=False, num_layers=2)
        self.decoder_scl = nn.LSTM(1, 1, batch_first=True, bidirectional=False, num_layers=2)
        
        # non-linear activation functions
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        # Move model to device
        self.to(self.device)

    def encode(self, x):
        
        # BiLSTM representation of sequence [N, D, 1] -> [N, D, 2 * K]
        _,(h,_) = self.encoder_x(x.unsqueeze(-1))
        h = h[2:4].transpose(0, 1).reshape(x.shape[0], -1)
        
        # Process the LSTM embedding through another encoder
        h = self.int_layer(h)
        
        # Compute the mean and scale of the latent representation
        loc = self.emb_to_loc(h)
        scl = self.emb_to_scl(h)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z):
        
        batch_size = z.shape[0]
        
        # run through MLP [N, D, 1]
        h = self.decoder_mlp(z).unsqueeze(-1)
        
        # run embedding through location biLSTM
        loc,_ = self.decoder_loc(h)
        
        # run embedding through scale biLSTM
        scl,_ = self.decoder_scl(h)
        
        # return the location and log scale
        return loc.squeeze(-1),self.softplus(scl.squeeze(-1))
    
    def model(self, x):

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        pyro.module("decoder_scl", self.decoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc,scl = self.decode(z)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, scl), obs=x)

    def guide(self, x):
        
        # register modules with pyro
        pyro.module("encoder_x", self.encoder_x)
        pyro.module("int_layer", self.int_layer)
        pyro.module("emb_to_loc", self.emb_to_loc)
        pyro.module("emb_to_scl", self.emb_to_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x)

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                x_batch = batch[0].to(self.device)
                train_loss += svi.step(x_batch) / x_batch.size(0)
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device)
                    val_loss += svi.evaluate_loss(x_batch) / x_batch.size(0)
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                z_loc, _ = self.encode(x_batch)
                xhat, _ = self.decode(z_loc)
        
        return z_loc, xhat
    
class LearnedVarSupLstmVAE(nn.Module):
    """
    A supervised variational autoencoder (VAE) model with LSTM-based encoder and decoder
    networks, designed for sequential data with class labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_x (nn.LSTM): Bidirectional LSTM encoder for input sequences.
        xy_to_emb (nn.Sequential): Linear layer to combine LSTM output and label embedding.
        emb_to_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        emb_to_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.LSTM): LSTM decoder for reconstructing sequence mean.
        decoder_scl (nn.LSTM): LSTM decoder for reconstructing sequence scale.
        tanh (nn.Tanh): Tanh activation function.
        softplus (nn.Softplus): Softplus activation function.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence mean of shape [N, D].
        scl (torch.Tensor): Reconstructed sequence scale (stddev) of shape [N, D].

    The model learns a latent representation of input sequences conditioned on both the data and their associated
    labels, enabling class-conditional generation and inference. The encoder utilizes a
    bidirectional LSTM to capture temporal dependencies in the input, while the decoder
    reconstructs the original sequence from the latent variables and class information
    using a combination of MLP and LSTM layers. The model is compatible with Pyro for
    probabilistic programming and supports training with stochastic variational inference
    (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "LearnedVarSupLstmVAE"
        self.fix_scl = False
        self.sup = True

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.device = device
        
        # Memory LSTM ∝ batch_size × seq_len × hidden_dim × num_layers × (2 if bidirectional)
        
        # define proper LSTM encoder
        self.encoder_x = nn.LSTM(1, K, batch_first=True, bidirectional=True, num_layers=2)
        self.xy_to_emb = nn.Sequential(nn.Linear(2 * K + NS, K))
        self.emb_to_loc = nn.Sequential(nn.Linear(K, K))
        self.emb_to_scl = nn.Sequential(nn.Linear(K, K), nn.Softplus())
        
        # define proper LSTM decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D)
            )
        self.decoder_loc = nn.LSTM(1, 1, batch_first=True, bidirectional=False, num_layers=2)
        self.decoder_scl = nn.LSTM(1, 1, batch_first=True, bidirectional=False, num_layers=2)
        
        # non-linear activation functions
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # BiLSTM representation of sequence [N, D, 1] -> [N, D, 2 * K]
        _,(h,_) = self.encoder_x(x.unsqueeze(-1))
        h = h[2:4].transpose(0, 1).reshape(x.shape[0], -1)
        
        # Combine the encoded representations (e.g., via concatenation or addition)
        xy = torch.cat((h, y), dim=-1)
        
        # Process the combined representation through another encoder
        xy_emb = self.xy_to_emb(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.emb_to_loc(xy_emb)
        scale = self.emb_to_scl(xy_emb)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        batch_size = z.shape[0]
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D, 1]
        zy = self.decoder_mlp(zy).unsqueeze(-1)
        
        # run embedding through location biLSTM [N, D, 1] -> [N, D]
        loc,_ = self.decoder_loc(zy)
        
        # run embedding through scale biLSTM [N, D, 1] -> [N, D]
        scl,_ = self.decoder_scl(zy)
        
        # return the location and log scale
        return loc.squeeze(-1),self.softplus(scl.squeeze(-1))
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, W, sigma^2).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        pyro.module("decoder_scl", self.decoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc,scl = self.decode(z, y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma).
        """
        
        # register modules with pyro
        pyro.module("encoder_x", self.encoder_x)
        pyro.module("xy_to_emb", self.xy_to_emb)
        pyro.module("emb_to_loc", self.emb_to_loc)
        pyro.module("emb_to_scl", self.emb_to_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)
                
                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch)
                xhat,_ = self.decode(z_loc,y_batch)
        
        return z_loc, xhat
    
class FixedVarSupMlpDenVAE(nn.Module):
    """
    FixedVarSupMlpDenVAE is a supervised variational autoencoder (VAE) model for denoising
    and imputation with fixed variance, implemented using PyTorch and Pyro. This model is 
    designed for semi-supervised learning tasks where both input data, class labels, and 
    a mask indicating observed/missing values are available. The encoder uses multi-layer 
    perceptrons (MLPs) to map masked input data, labels, and the mask itself to a latent 
    space, parameterizing the mean and scale of the latent distribution. The decoder 
    reconstructs the original input from the latent representation and labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_mlp (nn.Sequential): Encoder MLP network for masked input, labels, and mask.
        encoder_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.Sequential): Linear layer to reconstruct the sequence mean.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence location of shape [N, D].

    The model learns a latent representation of masked input sequences conditioned on both 
    the data, their associated labels, and the mask, enabling class-conditional denoising 
    and imputation. The encoder processes the masked input, labels, and mask, while the 
    decoder reconstructs the original sequence from the latent variables and class information.
    The model is compatible with Pyro for probabilistic programming and supports training with 
    stochastic variational inference (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "FixedVarSupMlpDenVAE"
        self.fix_scl = True
        self.sup = True

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # fixed scale of the likelihood
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2 * D + NS, (2 * D + NS) // 2), nn.ReLU(),
            nn.Linear((2 * D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y, m):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # mask the input data
        x_masked = x * m

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x_masked = torch.nan_to_num(x_masked, nan=0.0, posinf=0.0, neginf=0.0)

        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xym = torch.cat((x_masked, y, m), dim=-1)

        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xym)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y, _):
        """
        Defines the generative model P(X | Z, Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y, m):
        """
        Defines the variational guide q(Theta, Z | X, Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y, m)

        # print if there are nans   
        if torch.isnan(loc).any():
            superprint("Warning: NaNs found in loc during guide!")
        if torch.isnan(scl).any():
            superprint("Warning: NaNs found in scl during guide!")

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                m_batch = gen_random_mask(x_batch.shape[0], x_batch.shape[1]).to(self.device)

                train_loss += svi.step(x_batch, y_batch, m_batch) / x_batch.size(0)
                    
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)
                    m_batch = gen_random_mask(x_batch.shape[0],x_batch.shape[1]).to(self.device)
                        
                    val_loss += svi.evaluate_loss(x_batch, y_batch, m_batch) / x_batch.size(0)
                    
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
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                m = torch.ones_like(x_batch).to(self.device)  # assuming all data is observed
                z_loc, _ = self.encode(x_batch,y_batch,m)
                xhat = self.decode(z_loc,y_batch)
        
        return z_loc, xhat

class FixedVarSupMlpDenCensVAE(nn.Module):
    """
    FixedVarSupMlpDenVAE is a supervised variational autoencoder (VAE) model for denoising
    and imputation with fixed variance, implemented using PyTorch and Pyro. This model is 
    designed for semi-supervised learning tasks where both input data, class labels, and 
    a mask indicating observed/missing values are available. The encoder uses multi-layer 
    perceptrons (MLPs) to map masked input data, labels, and the mask itself to a latent 
    space, parameterizing the mean and scale of the latent distribution. The decoder 
    reconstructs the original input from the latent representation and labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_mlp (nn.Sequential): Encoder MLP network for masked input, labels, and mask.
        encoder_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.Sequential): Linear layer to reconstruct the sequence mean.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence location of shape [N, D].

    The model learns a latent representation of masked input sequences conditioned on both 
    the data, their associated labels, and the mask, enabling class-conditional denoising 
    and imputation. The encoder processes the masked input, labels, and mask, while the 
    decoder reconstructs the original sequence from the latent variables and class information.
    The model is compatible with Pyro for probabilistic programming and supports training with 
    stochastic variational inference (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # fixed scale of the likelihood
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(3 * D + NS, (3 * D + NS) // 2), nn.ReLU(),
            nn.Linear((3 * D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y, m_cens, m):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # mask the input data
        x_masked = x * m 

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x_masked = torch.nan_to_num(x_masked, nan=0.0, posinf=0.0, neginf=0.0)

        m_cens = torch.nan_to_num(m_cens, nan=0.0, posinf=0.0, neginf=0.0)

        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xym = torch.cat((x_masked, y, m_cens, m), dim=-1)

        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xym)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y, m_cens, _):
        """
        Defines the generative model P(X | Z, Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y, m_cens, m):
        """
        Defines the variational guide q(Theta, Z | X, Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y, m_cens, m)

        # print if there are nans   
        if torch.isnan(loc).any():
            superprint("Warning: NaNs found in loc during guide!")
        if torch.isnan(scl).any():
            superprint("Warning: NaNs found in scl during guide!")

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, censor_mask=None):
        
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
                x_batch = batch[0].to(self.device) 
                y_batch = batch[1].to(self.device)
                m_batch = gen_random_mask(x_batch.shape[0], x_batch.shape[1]).to(self.device)

                if censor_mask is not None:
                    m_censor_batch = batch[2].to(self.device)
                else:
                    m_censor_batch = torch.ones_like(x_batch).to(self.device)

                train_loss += svi.step(x_batch, y_batch, m_censor_batch, m_batch) / x_batch.size(0)
                    
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device) 
                    y_batch = batch[1].to(self.device)
                    m_batch = gen_random_mask(x_batch.shape[0], x_batch.shape[1]).to(self.device)

                    if censor_mask is not None:
                        m_censor_batch = batch[2].to(self.device)
                    else:
                        m_censor_batch = torch.ones_like(x_batch).to(self.device)
                    
                    val_loss += svi.evaluate_loss(x_batch, y_batch, m_censor_batch, m_batch) / x_batch.size(0)
                    
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

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                m = torch.ones_like(x_batch).to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch,m)
                xhat = self.decode(z_loc,y_batch)
        
        return z_loc, xhat

class FixedVarSupMlpCensVAE(nn.Module):
    """
    FixedVarSupMlpDenVAE is a supervised variational autoencoder (VAE) model for denoising
    and imputation with fixed variance, implemented using PyTorch and Pyro. This model is 
    designed for semi-supervised learning tasks where both input data, class labels, and 
    a mask indicating observed/missing values are available. The encoder uses multi-layer 
    perceptrons (MLPs) to map masked input data, labels, and the mask itself to a latent 
    space, parameterizing the mean and scale of the latent distribution. The decoder 
    reconstructs the original input from the latent representation and labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_mlp (nn.Sequential): Encoder MLP network for masked input, labels, and mask.
        encoder_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.Sequential): Linear layer to reconstruct the sequence mean.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence location of shape [N, D].

    The model learns a latent representation of masked input sequences conditioned on both 
    the data, their associated labels, and the mask, enabling class-conditional denoising 
    and imputation. The encoder processes the masked input, labels, and mask, while the 
    decoder reconstructs the original sequence from the latent variables and class information.
    The model is compatible with Pyro for probabilistic programming and supports training with 
    stochastic variational inference (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # fixed scale of the likelihood
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2 * D + NS, (2 * D + NS) // 2), nn.ReLU(),
            nn.Linear((2 * D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y, m_cens):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        m_cens = torch.nan_to_num(m_cens, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xym = torch.cat((x, y, m_cens), dim=-1)

        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xym)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y, _):
        """
        Defines the generative model P(X | Z, Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y, m_cens):
        """
        Defines the variational guide q(Theta, Z | X, Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y, m_cens)

        # print if there are nans   
        if torch.isnan(loc).any():
            superprint("Warning: NaNs found in loc during guide!")
        if torch.isnan(scl).any():
            superprint("Warning: NaNs found in scl during guide!")

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, censor_mask=True):
        
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
                x_batch = batch[0].to(self.device) 
                y_batch = batch[1].to(self.device)

                if censor_mask:
                    m_censor_batch = batch[2].to(self.device)
                else:
                    m_censor_batch = torch.ones_like(x_batch).to(self.device)

                train_loss += svi.step(x_batch, y_batch, m_censor_batch) / x_batch.size(0)
                    
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device) 
                    y_batch = batch[1].to(self.device)

                    if censor_mask:
                        m_censor_batch = batch[2].to(self.device)
                    else:
                        m_censor_batch = torch.ones_like(x_batch).to(self.device)
                    
                    val_loss += svi.evaluate_loss(x_batch, y_batch, m_censor_batch) / x_batch.size(0)
                    
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
        return train_loss_history, val_loss_history

    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.
        
        Args:
            loader (DataLoader): DataLoader containing the input data.
        
        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()
        
        with torch.no_grad():
            for batch in loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                m = torch.ones_like(x_batch).to(self.device)
                z_loc, _ = self.encode(x_batch,y_batch,m)
                xhat = self.decode(z_loc,y_batch)
        
        return z_loc, xhat

class FixedVarSupMlpMaskVAE(nn.Module):
    """
    FixedVarSupMlpDenVAE is a supervised variational autoencoder (VAE) model for denoising
    and imputation with fixed variance, implemented using PyTorch and Pyro. This model is 
    designed for semi-supervised learning tasks where both input data, class labels, and 
    a mask indicating observed/missing values are available. The encoder uses multi-layer 
    perceptrons (MLPs) to map masked input data, labels, and the mask itself to a latent 
    space, parameterizing the mean and scale of the latent distribution. The decoder 
    reconstructs the original input from the latent representation and labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_mlp (nn.Sequential): Encoder MLP network for masked input, labels, and mask.
        encoder_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.Sequential): Linear layer to reconstruct the sequence mean.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence location of shape [N, D].

    The model learns a latent representation of masked input sequences conditioned on both 
    the data, their associated labels, and the mask, enabling class-conditional denoising 
    and imputation. The encoder processes the masked input, labels, and mask, while the 
    decoder reconstructs the original sequence from the latent variables and class information.
    The model is compatible with Pyro for probabilistic programming and supports training with 
    stochastic variational inference (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # fixed scale of the likelihood
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D + NS, ( D + NS) // 2), nn.ReLU(),
            nn.Linear((D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)

        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        
        # return the location and log scale
        return loc
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(Theta, Z | X, Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)

        # print if there are nans   
        if torch.isnan(loc).any():
            superprint("Warning: NaNs found in loc during guide!")
        if torch.isnan(scl).any():
            superprint("Warning: NaNs found in scl during guide!")

        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                x_batch = batch[0].to(self.device) 
                y_batch = batch[1].to(self.device)

                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                    
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(self.device) 
                    y_batch = batch[1].to(self.device)
                    
                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    
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
        return train_loss_history, val_loss_history

class LearnedVarSupMlpDenVAE(nn.Module):
    """
    FixedVarSupMlpDenVAE is a supervised variational autoencoder (VAE) model for denoising
    and imputation with fixed variance, implemented using PyTorch and Pyro. This model is 
    designed for semi-supervised learning tasks where both input data, class labels, and 
    a mask indicating observed/missing values are available. The encoder uses multi-layer 
    perceptrons (MLPs) to map masked input data, labels, and the mask itself to a latent 
    space, parameterizing the mean and scale of the latent distribution. The decoder 
    reconstructs the original input from the latent representation and labels.

    Attributes:
        D (int): Number of input features or time steps in each sequence.
        K (int): Dimensionality of the latent space.
        NS (int): Number of classes (for one-hot encoded labels).
        device (str): Device used for computation ('cpu' or 'cuda').
        encoder_mlp (nn.Sequential): Encoder MLP network for masked input, labels, and mask.
        encoder_loc (nn.Sequential): Linear layer to compute mean of latent distribution.
        encoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of latent distribution.
        decoder_mlp (nn.Sequential): MLP to map latent variables and labels to decoder input.
        decoder_loc (nn.Sequential): Linear layer to reconstruct the location of X|Z.
        decoder_scl (nn.Sequential): Linear + Softplus layer to compute scale of X|Z.

    Input:
        x (torch.Tensor): Input data of shape [N, D], where N is batch size and D is sequence length.
        y (torch.Tensor): One-hot encoded labels of shape [N, NS].

    Output:
        loc (torch.Tensor): Reconstructed sequence location of shape [N, D].
        scl (torch.Tensor): Reconstructed sequence scale of shape [N, D].      

    The model learns a latent representation of masked input sequences conditioned on both 
    the data, their associated labels, and the mask, enabling class-conditional denoising 
    and imputation. The encoder processes the masked input, labels, and mask, while the 
    decoder reconstructs the original sequence from the latent variables and class information.
    The model is compatible with Pyro for probabilistic programming and supports training with 
    stochastic variational inference (SVI), including early stopping based on validation loss.
    """
    def __init__(self, D, K, NS, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(2 * D + NS, (2 * D + NS) // 2), nn.ReLU(),
            nn.Linear((2 * D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, D // 8), nn.ReLU(), 
            nn.Linear(D // 8, D // 4), nn.ReLU(),
            nn.Linear(D // 4, D // 2)
            )
        self.decoder_loc = nn.Sequential(nn.Linear(D // 2, D))
        self.decoder_scl = nn.Sequential(nn.Linear(D // 2, D), nn.Softplus())
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y, m):
        """
        Encodes the input data into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Encoded latent representation of shape [N, K].
        """
        
        # mask the input data
        x_masked = x * m

        # replace nans with zeros if any (ensures mask is 0 too when NaN in data)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = torch.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xym = torch.cat((x_masked, y, m), dim=-1)

        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xym)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scale = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
    def decode(self, z, y):
        """
        Decodes the latent representation into the original data space.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: Reconstructed data of shape [N, D].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # run through MLP [N, D // 2]
        zy = self.decoder_mlp(zy)
        
        # compute location [N, D]
        loc = self.decoder_loc(zy)
        scale = torch.clamp(self.decoder_scl(zy), min=1e-3)
        
        # return the location and log scale
        return loc, scale
    
    def model(self, x, y, _):
        """
        Defines the generative model P(X | Z, Y).
        """

        # register modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)
        pyro.module("decoder_scl", self.decoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # prior parameters for Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)
        
        # Prior for latent variables Z
        with pyro.plate("data", batch_size, dim=-2):

            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / (1.0 * self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc, scl = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, scl), obs=x)

    def guide(self, x, y, m):
        """
        Defines the variational guide q(Theta, Z | X, Y).
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y, m)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                pyro.sample("z", dist.Normal(loc, scl))

    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3):
        
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
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                m_batch = gen_random_mask(x_batch.shape[0],x_batch.shape[1]).to(self.device)
                
                train_loss += svi.step(x_batch, y_batch, m_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)
                    m_batch = gen_random_mask(x_batch.shape[0],x_batch.shape[1]).to(self.device)
                
                    val_loss += svi.evaluate_loss(x_batch, y_batch, m_batch) / x_batch.size(0)
                    
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

class SupMlpGpVAE(nn.Module):
    """
    SupMlpGpVAE
    A supervised Gaussian Process Variational Autoencoder (GP-VAE) with a fixed output variance and MLP-based encoder/decoder. 
    This model combines the representational power of deep neural networks with the flexibility of Gaussian Processes, 
    allowing for structured modeling of temporal or sequential data with class supervision. The encoder maps input data and 
    class labels to a latent space, while the decoder reconstructs the data via a GP whose mean is parameterized by the 
    latent variables and class labels. The model supports stochastic variational inference (SVI) for training and includes 
    methods for encoding, decoding, GP prediction, and model training with early stopping.
        D (int): Number of time steps (input dimension).
        K (int): Latent dimension.
        NS (int): Number of classes (for one-hot labels).
        M (int, optional): Number of inducing points for the GP. Default is 50.
        scl (float, optional): Fixed scale (variance) of the output distribution. Default is 1e-3.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Default is "cpu".
    Attributes:
        name (str): Name of the model.
        fix_scl (bool): Whether the output variance is fixed.
        sup (bool): Whether the model is supervised.
        D (int): Number of time steps.
        K (int): Latent dimension.
        M (int): Number of inducing points.
        NS (int): Number of classes.
        scl (float): Output distribution scale.
        device (str): Device for computation.
        encoder_mlp (nn.Sequential): MLP encoder network.
        encoder_loc (nn.Sequential): Network for latent mean.
        encoder_scl (nn.Sequential): Network for latent scale.
        decoder_mlp (nn.Sequential): MLP decoder network.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into GP mean at inducing points.
        gp_predict(lengthscale, variance, gp_loc): Computes GP predictive mean and covariance.
        model(x, y): Defines the generative model for Pyro SVI.
        guide(x, y): Defines the variational guide for Pyro SVI.
        train_model(data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3): Trains the model using SVI with early stopping.
        forward(loader): Runs the model forward pass to obtain latent means and reconstructions.
    """

    def __init__(self, D, K, NS, M=50, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "SupMlpGpVAE"
        self.fix_scl = True
        self.sup = True

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.M = M              # number of inducing points
        self.NS = NS            # number of classes
        self.scl = scl          # scale of the output distribution
        self.device = device
        
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D + NS, (D + NS) // 2), nn.ReLU(),
            nn.Linear((D + NS) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, M // 4), nn.ReLU(), 
            nn.Linear(M // 4, M // 2), nn.ReLU(),
            nn.Linear(M // 2, M)
        )
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data and labels into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            tuple: (loc, scl)
            loc (torch.Tensor): Mean of the latent distribution, shape [N, K].
            scl (torch.Tensor): Scale (stddev) of the latent distribution, shape [N, K].
        """
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine the input data x and labels y
        xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        xy_emb = self.encoder_mlp(xy)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(xy_emb)
        scl = torch.clamp(self.encoder_scl(xy_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z, y):
        """
        Decodes the latent representation and class labels into the GP mean at the inducing points.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: GP mean at the inducing points, shape [N, M].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # return the location and log scale
        return self.decoder_mlp(zy)
    
    def matern32(self, diff, lengthscale, variance):
        r = torch.abs(diff) / lengthscale
        return variance * (1.0 + torch.sqrt(torch.tensor(3.0)) * r) * \
            torch.exp(-torch.sqrt(torch.tensor(3.0)) * r)

    def matern52(self, diff, lengthscale, variance):
        r = torch.abs(diff) / lengthscale
        sqrt5 = torch.sqrt(torch.tensor(5.0))
        return variance * (1.0 + sqrt5 * r + 5.0 * r**2 / 3.0) * \
            torch.exp(-sqrt5 * r)
        
    def gp_predict(self, lengthscale, variance, gp_loc):
        """
        Computes the predictive mean and covariance for the GP decoder.

        Args:
        lengthscale (torch.Tensor): GP lengthscale parameter (scalar tensor).
        variance (torch.Tensor): GP variance parameter (scalar tensor).
        gp_loc (torch.Tensor): Latent mean at inducing points, shape [batch_size, M].

        Returns:
        pred_mean (torch.Tensor): Predictive mean at all points, shape [batch_size, D].
        cov (torch.Tensor): Predictive covariance diagonal, shape [D].
        """

        # Register inducing points Tu (assume evenly spaced)
        T = torch.arange(self.D, device=self.device).float().unsqueeze(1)  # [D, 1]
        Tu = torch.linspace(0, self.D - 1, self.M, device=self.device).unsqueeze(1)  # [M, 1]

        # Compute covariance between inducing points: Kuu € [M, M]
        diff_uu = Tu - Tu.T  # [M, M]
            # --- Matérn kernel ---
        Kuu = self.matern52(diff_uu, lengthscale, variance)
        Kuu = Kuu + 1e-5 * torch.eye(self.M, device=self.device)

        # Kuu = variance * torch.exp(-0.5 * (diff_uu ** 2) / (lengthscale ** 2))
        # Kuu = Kuu + 1e-5 * torch.eye(self.M, device=self.device)  # jitter for stability

        # Compute covariance between inducing points and full points: Kuf € [M, D]
        diff_uf = Tu - T.T  # [M, D]
        # --- Matérn kernel ---
        Kuf = self.matern52(diff_uf, lengthscale, variance)
        # Kuf = variance * torch.exp(-0.5 * (diff_uf ** 2) / (lengthscale ** 2))  # [M, D]

        # Compute covariance at full points: Kff € [D, D]
        Kff_diag = variance * torch.ones(self.D, device=self.device)  # only diagonal needed

        # Cholesky of Kuu
        Luu = torch.linalg.cholesky(Kuu)  # [M, M]

        # Compute gamma = Kuu^{-1} (mean_Tu) for all batch elements: [batch_size, M]
        gamma = torch.cholesky_solve(gp_loc.unsqueeze(-1), Luu).squeeze(-1)  # [batch_size, M]

        # Predictive mean at X for all batch elements: [batch_size, D]
        loc = torch.matmul(gamma, Kuf)  # [batch_size, D]

        # Compute predictive covariance (same for all batches): [D, D]
        A = torch.cholesky_solve(Kuf, Luu)  # [M, D]
        scl = Kff_diag - (Kuf.T @ A).diagonal()
        scl = torch.clamp(scl, min=1e-4)

        return loc, scl
        
    def get_likelihood(self, x, y):
        """
        Computes the average normalized likelihood (log probability) of the observed data x under the current model parameters.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            float: Averaged normalized log likelihood over all samples.
        """
        self.eval()
        with torch.no_grad():
            batch_size = x.shape[0]

            # Get GP kernel parameters from param store
            lengthscale = pyro.param("lengthscale_loc").exp()
            variance = pyro.param("variance_loc").exp()

            # # GP kernel hyperparameters
            # lengthscale = pyro.param("gp_lengthscale", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)
            # variance = pyro.param("gp_variance", torch.tensor(0.1, device=self.device), constraint=dist.constraints.positive)

            # Encode and decode to get GP mean at inducing points
            z_loc, _ = self.encode(x, y)
            gp_loc = self.decode(z_loc, y)
            # GP predictive mean and variance
            loc, scl = self.gp_predict(lengthscale, variance, gp_loc)
            # Compute log likelihood for each sample and dimension
            log_prob = dist.Normal(loc, scl.sqrt()).log_prob(x)
            # Normalize per dimension and per sample, then average over all samples
            return log_prob.mean(dim=-1).mean().item()
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, Y) with a Gaussian Process likelihood.

        This method specifies the probabilistic generative process for the supervised GP-VAE model.
        Given input data x and class labels y, it samples latent variables z from a standard normal prior,
        decodes z and y to obtain the mean at the GP inducing points, and uses a Gaussian Process
        to generate the observed data x with a learned kernel. The GP kernel hyperparameters
        (lengthscale and variance) are learned as Pyro parameters. The likelihood is modeled as a
        Normal distribution with mean and variance given by the GP predictive equations.
        """

        # Register decoder modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # Prior for latent variables Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        # GP kernel hyperparameters (fixed). TODO: put a prior on these
        lengthscale = pyro.param("gp_lengthscale", torch.tensor(4, device=self.device), constraint=dist.constraints.positive)
        variance = pyro.param("gp_variance", torch.tensor(0.3, device=self.device), constraint=dist.constraints.positive)

        # pyro sample statements
        with pyro.plate("data", batch_size, dim=-2):
            
            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / float(self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            # with pyro.poutine.scale(scale=2.0):  # scale to emphasize latent sampling
            with pyro.plate("latent_dim", self.K, dim=-1):
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))

            # # Add custom loss (negative means penalty, positive means reward)
            # # kBET(zloc_tr, group_sample_labels[train_indices])
            # kbet_loss = kBET(z, y)  
            # pyro.factor("custom_term", -kbet_loss)

            # Decoder output as GP mean
            loc, scl = self.gp_predict(lengthscale, variance, self.decode(z, y))

            with pyro.plate("output_dim", self.D, dim=-1):
                
                # sample the output x from a Normal distribution
                pyro.sample("x", dist.NanMaskedNormal(loc, scl.sqrt()), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(z | x, y) for the supervised GP-VAE.
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)

        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                z = pyro.sample("z", dist.Normal(loc, scl))


    def train_model(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3, roll=False):
        """
        Trains the model using stochastic variational inference (SVI). This method performs model training over a specified number of epochs, using the provided training and validation data loaders.
        It supports early stopping based on validation loss improvements. The training and validation loss histories are returned for further analysis.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            num_epochs (int, optional): Maximum number of training epochs. Default is 5000.
            lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Default is 100.
            min_delta (float, optional): Minimum change in validation loss to qualify as an improvement. Default is 1e-3.
        Returns:
            tuple: A tuple containing two lists:
                - train_loss_history (list of float): Training loss values for each epoch.
                - val_loss_history (list of float): Validation loss values for each epoch.
        """
        
        # Set model to training mode
        self.train()

        # Define optimizer and SVI
        optimizer = ClippedAdam({"lr": lr})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        
        # Training loop
        train_loss_history = []
        val_loss_history = []
        val_likelihood_history = []
        
        # patience counter
        patience_counter = 0
        best_loss = float("inf")

        # Train the model
        for epoch in range(num_epochs):
            
            # Training phase
            self.train()
            train_loss = 0.0
            for batch in data_loader:
                
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                
                # apply random roll to x_batch
                if roll:
                    x_batch = torch.roll(x_batch, shifts=torch.randint(0, x_batch.size(1), (1,)).item(), dims=1)

                train_loss += svi.step(x_batch, y_batch) / x_batch.size(0)
                
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_likelihood = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    x_batch = batch[0].to(self.device)
                    y_batch = batch[1].to(self.device)

                    val_loss += svi.evaluate_loss(x_batch, y_batch) / x_batch.size(0)
                    # val_likelihood += self.get_likelihood(x_batch, y_batch) / x_batch.size(0)
                    # print("Size val_ll ", val_likelihood.size())
                    # print('x_batch size ', x_batch.size())
                    # print('y_batch size ', y_batch.size())
                    # print('val_loss size ', val_loss.size())

            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            # val_likelihood /= len(val_loader.dataset)
            # val_likelihood_history.append(val_likelihood)

            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Likelihood: {val_likelihood:.4f}")

            # superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Kernel Lengthscale: {pyro.get_param_store().get_param('gp_lengthscale').item():.4f}, Kernel Variance: {pyro.get_param_store().get_param('gp_variance').item():.4f}")

        # return loss history
        return train_loss_history, val_loss_history
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.

        Args:
            loader (DataLoader): DataLoader containing the input data.

        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()

        # Initialize lists to store outputs
        xhat_lst = []
        z_loc_lst = []

        # # GP hyperparameters 
        # lengthscale = pyro.param("lengthscale_loc").exp()
        # variance = pyro.param("variance_loc").exp()

        # GP kernel hyperparameters
        lengthscale = pyro.param("gp_lengthscale", torch.tensor(4.0, device=self.device), constraint=dist.constraints.positive)
        variance = pyro.param("gp_variance", torch.tensor(0.3, device=self.device), constraint=dist.constraints.positive)

        # Iterate through the data loader
        with torch.no_grad():
            for batch in loader:

                # Move batch to device
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # run forward pass
                z_loc_batch, _ = self.encode(x_batch, y_batch)
                xhat_batch, _ = self.gp_predict(lengthscale, variance, self.decode(z_loc_batch, y_batch))

                # Append outputs to lists
                z_loc_lst.append(z_loc_batch)
                xhat_lst.append(xhat_batch)

        # Concatenate the outputs from all batches
        z_loc = torch.cat(z_loc_lst, dim=0)
        xhat = torch.cat(xhat_lst, dim=0)

        # Return the latent mean and reconstructed output
        return z_loc, xhat

class SupMlpGpClVAE(nn.Module):
    """
    SupMlpGpClVAE
    A supervised Gaussian Process Variational Autoencoder (GP-VAE) with a fixed output variance and MLP-based encoder/decoder. 
    This model combines the representational power of deep neural networks with the flexibility of Gaussian Processes, 
    allowing for structured modeling of temporal or sequential data with class supervision. The encoder maps input data and 
    class labels to a latent space, while the decoder reconstructs the data via a GP whose mean is parameterized by the 
    latent variables and class labels. The model supports stochastic variational inference (SVI) for training and includes 
    methods for encoding, decoding, GP prediction, and model training with early stopping.
        D (int): Number of time steps (input dimension).
        K (int): Latent dimension.
        NS (int): Number of classes (for one-hot labels).
        M (int, optional): Number of inducing points for the GP. Default is 50.
        scl (float, optional): Fixed scale (variance) of the output distribution. Default is 1e-3.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Default is "cpu".
    Attributes:
        name (str): Name of the model.
        fix_scl (bool): Whether the output variance is fixed.
        sup (bool): Whether the model is supervised.
        D (int): Number of time steps.
        K (int): Latent dimension.
        M (int): Number of inducing points.
        NS (int): Number of classes.
        scl (float): Output distribution scale.
        device (str): Device for computation.
        encoder_mlp (nn.Sequential): MLP encoder network.
        encoder_loc (nn.Sequential): Network for latent mean.
        encoder_scl (nn.Sequential): Network for latent scale.
        decoder_mlp (nn.Sequential): MLP decoder network.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into GP mean at inducing points.
        gp_predict(lengthscale, variance, gp_loc): Computes GP predictive mean and covariance.
        model(x, y): Defines the generative model for Pyro SVI.
        guide(x, y): Defines the variational guide for Pyro SVI.
        train_model(data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3): Trains the model using SVI with early stopping.
        forward(loader): Runs the model forward pass to obtain latent means and reconstructions.
    """

    def __init__(self, D, K, NS, M=50, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "SupMlpGpClVAE"
        self.fix_scl = True
        self.sup = True

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.M = M              # number of inducing points
        self.NS = NS            # number of classes
        self.scl = scl          # scale of the output distribution
        self.device = device

        # GP hyperparameters as nn.Parameter (log-space to keep them positive)
        self.log_lengthscale = nn.Parameter(torch.tensor(1.0))  # learnable
        self.log_variance = nn.Parameter(torch.tensor(0.0))     # learnable
         
        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D, (D) // 2), nn.ReLU(),
            nn.Linear((D) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K + NS, M // 4), nn.ReLU(), 
            nn.Linear(M // 4, M // 2), nn.ReLU(),
            nn.Linear(M // 2, M)
        )
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data and labels into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            tuple: (loc, scl)
            loc (torch.Tensor): Mean of the latent distribution, shape [N, K].
            scl (torch.Tensor): Scale (stddev) of the latent distribution, shape [N, K].
        """
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # # Combine the input data x and labels y
        # xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        x_emb = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x_emb)
        scl = torch.clamp(self.encoder_scl(x_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z, y):
        """
        Decodes the latent representation and class labels into the GP mean at the inducing points.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: GP mean at the inducing points, shape [N, M].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        zy = torch.cat((z, y), dim=-1)
        
        # return the location and log scale
        return self.decoder_mlp(zy)
    
    def gp_predict(self, gp_loc):
        """
        Computes the predictive mean and covariance for the GP decoder.

        Args:
        lengthscale (torch.Tensor): GP lengthscale parameter (scalar tensor).
        variance (torch.Tensor): GP variance parameter (scalar tensor).
        gp_loc (torch.Tensor): Latent mean at inducing points, shape [batch_size, M].

        Returns:
        pred_mean (torch.Tensor): Predictive mean at all points, shape [batch_size, D].
        cov (torch.Tensor): Predictive covariance diagonal, shape [D].
        """

        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)

        # Register inducing points Tu (assume evenly spaced)
        T = torch.arange(self.D, device=self.device).float().unsqueeze(1)  # [D, 1]
        Tu = torch.linspace(0, self.D - 1, self.M, device=self.device).unsqueeze(1)  # [M, 1]

        # Compute covariance between inducing points: Kuu € [M, M]
        diff_uu = Tu - Tu.T  # [M, M]
        Kuu = variance * torch.exp(-0.5 * (diff_uu ** 2) / (lengthscale ** 2))
        Kuu = Kuu + 1e-5 * torch.eye(self.M, device=self.device)  # jitter for stability

        # Compute covariance between inducing points and full points: Kuf € [M, D]
        diff_uf = Tu - T.T  # [M, D]
        Kuf = variance * torch.exp(-0.5 * (diff_uf ** 2) / (lengthscale ** 2))  # [M, D]

        # Compute covariance at full points: Kff € [D, D]
        Kff_diag = variance * torch.ones(self.D, device=self.device)  # only diagonal needed

        # Cholesky of Kuu
        Luu = torch.linalg.cholesky(Kuu)  # [M, M]

        # Compute gamma = Kuu^{-1} (mean_Tu) for all batch elements: [batch_size, M]
        gamma = torch.cholesky_solve(gp_loc.unsqueeze(-1), Luu).squeeze(-1)  # [batch_size, M]

        # Predictive mean at X for all batch elements: [batch_size, D]
        loc = torch.matmul(gamma, Kuf)  # [batch_size, D]

        # Compute predictive covariance (same for all batches): [D, D]
        A = torch.cholesky_solve(Kuf, Luu)  # [M, D]
        scl = Kff_diag - (Kuf.T @ A).diagonal()
        scl = torch.clamp(scl, min=1e-4)

        return loc, scl
        
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, Y) with a Gaussian Process likelihood.

        This method specifies the probabilistic generative process for the supervised GP-VAE model.
        Given input data x and class labels y, it samples latent variables z from a standard normal prior,
        decodes z and y to obtain the mean at the GP inducing points, and uses a Gaussian Process
        to generate the observed data x with a learned kernel. The GP kernel hyperparameters
        (lengthscale and variance) are learned as Pyro parameters. The likelihood is modeled as a
        Normal distribution with mean and variance given by the GP predictive equations.
        """

        # Register decoder modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # Prior for latent variables Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        # lengthscale = torch.exp(self.log_lengthscale)
        # variance = torch.exp(self.log_variance)

        # # GP kernel hyperparameters (fixed). TODO: put a prior on these
        # lengthscale = pyro.param("gp_lengthscale", torch.tensor(4, device=self.device), constraint=dist.constraints.positive)
        # variance = pyro.param("gp_variance", torch.tensor(0.3, device=self.device), constraint=dist.constraints.positive)

        # pyro sample statements
        with pyro.plate("data", batch_size, dim=-2):
            
            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / float(self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)

            with pyro.plate("latent_dim", self.K, dim=-1):
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))

            # Decoder output as GP mean
            loc, scl = self.gp_predict(self.decode(z, y))

            with pyro.plate("output_dim", self.D, dim=-1):
                
                # sample the output x from a Normal distribution
                pyro.sample("x", dist.NanMaskedNormal(loc, scl.sqrt()), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(z | x, y) for the supervised GP-VAE.
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)

        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                z = pyro.sample("z", dist.Normal(loc, scl))

    def train_joint(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, 
                                    patience=50, min_delta=1e-3, lambda_cls=1e6):
        """
        Trains the SupMlpGpClVAE with a classifier on latent variables z.
        The loss is ELBO + lambda_cls * classification loss.
        """

        # Move model to device
        self.to(self.device)

        # Classifier to predict labels from latent z
        classifier = nn.Linear(self.K, self.NS).to(self.device)

        # Collect different parameter groups
        gp_params = [self.log_lengthscale, self.log_variance]
        vae_params = [p for name, p in self.named_parameters() 
                    if name not in ["log_lengthscale", "log_variance"]]
        
        # Optimizer: updates encoder/decoder + classifier
        optimizer = torch.optim.Adam([
            {"params": vae_params + list(classifier.parameters()), "lr": 1e-3},       # VAE, classifier, etc.
            {"params": gp_params, "lr": 0.01}         # GP kernel hyperparams
        ])

        # Pyro ELBO
        elbo = TraceMeanField_ELBO()

        # Training loop
        train_loss_history = []
        train_elbo_loss_history = []
        train_cls_loss_history = []
        val_loss_history = []
        val_accuracy_history = []

        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            classifier.train()
            train_loss = 0.0
            all_preds, all_labels = [], []

            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_size = x_batch.size(0)

                # ----------------------------
                # 1) Compute ELBO
                # ----------------------------
                # Compute ELBO and extract KL divergence
                elbo_trace = pyro.poutine.trace(self.model).get_trace(x_batch, y_batch)
                guide_trace = pyro.poutine.trace(self.guide).get_trace(x_batch, y_batch)
                elbo_loss = elbo.differentiable_loss(self.model, self.guide, x_batch, y_batch)
                # Extract KL divergence for z
                z_node = guide_trace.nodes["z"]
                z_dist = z_node["fn"]
                z_value = z_node["value"]
                # Prior for z
                z_prior_node = elbo_trace.nodes["z"]
                z_prior_dist = z_prior_node["fn"]
                kl_div_z = torch.distributions.kl_divergence(z_dist, z_prior_dist).sum()
                # You can now use kl_div_z as needed (e.g., log, track, etc.)

                # ----------------------------
                # 2) Sample z from encoder (reparameterized)
                # ----------------------------
                z_loc, z_scl = self.encode(x_batch, y_batch)
                qz = pyro.distributions.Normal(z_loc, z_scl)
                z_sample = qz.rsample()

                # ----------------------------
                # 3) Classifier loss on latent z
                # ----------------------------
                # print('z_sample.shape: ', z_sample.shape)
                logits = classifier(z_sample)
                # print('logits.shape: ', logits.shape)
                # print('logits: ', logits)
                cls_loss = F.cross_entropy(logits, torch.argmax(y_batch, dim=1))

                # ----------------------------
                # 4) Combined loss
                # ----------------------------
                combined_loss = elbo_loss - lambda_cls * cls_loss
                
                # clear old gradients
                optimizer.zero_grad() 
                # backpropagate the combined loss
                combined_loss.backward()
                # update parameters
                optimizer.step()

                train_loss += combined_loss.item()
                train_elbo_loss_history.append(elbo_loss.item())
                train_cls_loss_history.append(cls_loss.item())
                all_preds.append(logits.detach())
                all_labels.append(y_batch.detach())

            # ----------------------------
            # 5) Epoch metrics
            # ----------------------------
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            train_elbo_loss_history.append(sum(train_elbo_loss_history) / len(train_elbo_loss_history))
            train_cls_loss_history.append(sum(train_cls_loss_history) / len(train_cls_loss_history))

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            predicted_labels = torch.argmax(all_preds, dim=1)
            true_labels = torch.argmax(all_labels, dim=1)
            accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
            
            # ----------------------------
            # 6) Validation
            # ----------------------------
            self.eval()
            classifier.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)

                    # ELBO loss
                    elbo_val = elbo.differentiable_loss(self.model, self.guide, x_val, y_val)

                    # Obtain latent z
                    z_loc_val, z_scl_val = self.encode(x_val, y_val)
                    # qz_val = pyro.distributions.Normal(z_loc_val, z_scl_val)
                    # # Sample latent distribution
                    # z_sample_val = qz_val.rsample()

                    # Obtain predictions 
                    logits_val = classifier(z_loc_val)
                    # Classifier loss
                    cls_loss_val = F.cross_entropy(logits_val, torch.argmax(y_val, dim=1))
                    # Total validation loss
                    combined_val_loss = elbo_val - lambda_cls * cls_loss_val
                    
                    val_loss += combined_val_loss.item() * x_val.size(0)

                    val_preds.append(logits_val)
                    val_labels.append(y_val)

            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            val_preds = torch.cat(val_preds, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            # Compute total validation accuracy (from all batches)
            val_accuracy = accuracy_score(torch.argmax(val_labels, dim=1).cpu(),
                                        torch.argmax(val_preds, dim=1).cpu())
            val_accuracy_history.append(val_accuracy)

            superprint(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.6f}, "
                f"Train Acc={accuracy:.4f}, Val Acc={val_accuracy:.4f}, KL Divergence={kl_div_z:.4f}")
            print(f'GP hyperparameters: Lengthscale={torch.exp(self.log_lengthscale):.4f}, Variance={torch.exp(self.log_variance):.4f}')


            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        return train_loss_history, val_loss_history, train_elbo_loss_history, train_cls_loss_history, self, classifier
    
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.

        Args:
            loader (DataLoader): DataLoader containing the input data.

        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()

        # Initialize lists to store outputs
        xhat_lst = []
        z_loc_lst = []

        # # GP hyperparameters 
        # lengthscale = pyro.param("lengthscale_loc").exp()
        # variance = pyro.param("variance_loc").exp()

        # # GP kernel hyperparameters
        # lengthscale = pyro.param("gp_lengthscale", torch.tensor(4.0, device=self.device), constraint=dist.constraints.positive)
        # variance = pyro.param("gp_variance", torch.tensor(0.3, device=self.device), constraint=dist.constraints.positive)

        # Iterate through the data loader
        with torch.no_grad():
            for batch in loader:

                # Move batch to device
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # run forward pass
                z_loc_batch, _ = self.encode(x_batch, y_batch)
                xhat_batch, _ = self.gp_predict(self.decode(z_loc_batch, y_batch))

                # Append outputs to lists
                z_loc_lst.append(z_loc_batch)
                xhat_lst.append(xhat_batch)

        # Concatenate the outputs from all batches
        z_loc = torch.cat(z_loc_lst, dim=0)
        xhat = torch.cat(xhat_lst, dim=0)

        # Return the latent mean and reconstructed output
        return z_loc, xhat
    
class MlpClVAE(nn.Module):
    """
    MlpClVAE
    A supervised Gaussian Process Variational Autoencoder (GP-VAE) with a fixed output variance and MLP-based encoder/decoder. 
    This model combines the representational power of deep neural networks with the flexibility of Gaussian Processes, 
    allowing for structured modeling of temporal or sequential data with class supervision. The encoder maps input data and 
    class labels to a latent space, while the decoder reconstructs the data via a GP whose mean is parameterized by the 
    latent variables and class labels. The model supports stochastic variational inference (SVI) for training and includes 
    methods for encoding, decoding, GP prediction, and model training with early stopping.
        D (int): Number of time steps (input dimension).
        K (int): Latent dimension.
        NS (int): Number of classes (for one-hot labels).
        M (int, optional): Number of inducing points for the GP. Default is 50.
        scl (float, optional): Fixed scale (variance) of the output distribution. Default is 1e-3.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Default is "cpu".
    Attributes:
        name (str): Name of the model.
        fix_scl (bool): Whether the output variance is fixed.
        sup (bool): Whether the model is supervised.
        D (int): Number of time steps.
        K (int): Latent dimension.
        M (int): Number of inducing points.
        NS (int): Number of classes.
        scl (float): Output distribution scale.
        device (str): Device for computation.
        encoder_mlp (nn.Sequential): MLP encoder network.
        encoder_loc (nn.Sequential): Network for latent mean.
        encoder_scl (nn.Sequential): Network for latent scale.
        decoder_mlp (nn.Sequential): MLP decoder network.
    Methods:
        encode(x, y): Encodes input data and labels into latent mean and scale.
        decode(z, y): Decodes latent variables and labels into GP mean at inducing points.
        gp_predict(lengthscale, variance, gp_loc): Computes GP predictive mean and covariance.
        model(x, y): Defines the generative model for Pyro SVI.
        guide(x, y): Defines the variational guide for Pyro SVI.
        train_model(data_loader, val_loader, num_epochs=5000, lr=1e-3, patience=100, min_delta=1e-3): Trains the model using SVI with early stopping.
        forward(loader): Runs the model forward pass to obtain latent means and reconstructions.
    """

    def __init__(self, D, K, NS, M=50, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "MlpClVAE"
        self.fix_scl = True
        self.sup = False
        self.scl = 1e-3

        # Store variables
        self.D = D              # number of time steps
        self.K = K              # latent dimension
        self.NS = NS            # number of classes
        self.scl = scl          # scale of the output distribution
        self.device = device

        # define encoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(D, (D) // 2), nn.ReLU(),
            nn.Linear((D) // 2, 4 * K), nn.ReLU(),
            nn.Linear(4 * K, 2 * K)
        )
        self.encoder_loc = nn.Sequential(nn.Linear(2 * K, K))
        self.encoder_scl = nn.Sequential(nn.Linear(2 * K, K), nn.Softplus())
        
        # define decoder
        self.decoder_mlp = nn.Sequential(
            nn.Linear(K, D // 4), nn.ReLU(), 
            nn.Linear(D // 4, D// 2), nn.ReLU(),
            nn.Linear(D // 2, D)
        )
        
        # Move model to device
        self.to(self.device)

    def encode(self, x, y):
        """
        Encodes the input data and labels into a latent representation.

        Args:
            x (torch.Tensor): Input data of shape [N, D].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            tuple: (loc, scl)
            loc (torch.Tensor): Mean of the latent distribution, shape [N, K].
            scl (torch.Tensor): Scale (stddev) of the latent distribution, shape [N, K].
        """
        
        # replace nans with zeros if any
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # # Combine the input data x and labels y
        # xy = torch.cat((x, y), dim=-1)
        
        # Process the combined representation through the encoder network
        x_emb = self.encoder_mlp(x)
        
        # Compute the mean and scale of the latent representation
        loc = self.encoder_loc(x_emb)
        scl = torch.clamp(self.encoder_scl(x_emb), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scl
    
    def decode(self, z, y):
        """
        Decodes the latent representation and class labels into the GP mean at the inducing points.

        Args:
            z (torch.Tensor): Latent representation of shape [N, K].
            y (torch.Tensor): One-hot encoded labels of shape [N, NS].

        Returns:
            torch.Tensor: GP mean at the inducing points, shape [N, M].
        """
        
        # concatenate z [batch, K] and y [batch, NS] -> [batch, K + NS] 
        # zy = torch.cat((z, y), dim=-1)
        
        # return the location and log scale
        return self.decoder_mlp(z)
    
    def model(self, x, y):
        """
        Defines the generative model P(X | Z, Y) with a Gaussian Process likelihood.

        This method specifies the probabilistic generative process for the supervised GP-VAE model.
        Given input data x and class labels y, it samples latent variables z from a standard normal prior,
        decodes z and y to obtain the mean at the GP inducing points, and uses a Gaussian Process
        to generate the observed data x with a learned kernel. The GP kernel hyperparameters
        (lengthscale and variance) are learned as Pyro parameters. The likelihood is modeled as a
        Normal distribution with mean and variance given by the GP predictive equations.
        """

        # Register decoder modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        
        # Get the batch size from input x
        batch_size = x.shape[0]

        # Prior for latent variables Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        # pyro sample statements
        with pyro.plate("data", batch_size, dim=-2):
            
            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / float(self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)

            with pyro.plate("latent_dim", self.K, dim=-1):
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))

            loc = self.decode(z, y)

            with pyro.plate("output_dim", self.D, dim=-1):
                
                # sample the output x from a Normal distribution
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(z | x, y) for the supervised GP-VAE.
        """
        
        # register modules with pyro
        pyro.module("encoder_mlp", self.encoder_mlp)
        pyro.module("encoder_loc", self.encoder_loc)
        pyro.module("encoder_scl", self.encoder_scl)

        # Get the batch size from input x
        batch_size = x.shape[0]

        # get posterior parameters
        loc, scl = self.encode(x, y)
        
        # samples posterior
        with pyro.plate("data", batch_size, dim=-2):

            with pyro.plate("latent_dim", self.K, dim=-1):
            
                # sample z
                z = pyro.sample("z", dist.Normal(loc, scl))

    def train_joint(self, data_loader, val_loader, num_epochs=5000, lr=1e-3, 
                                    patience=50, min_delta=1e-3, lambda_cls=1e6):
        """
        Trains the SupMlpGpClVAE with a classifier on latent variables z.
        The loss is ELBO + lambda_cls * classification loss.
        """

        # Move model to device
        self.to(self.device)

        # Classifier to predict labels from latent z
        classifier = nn.Linear(self.K, self.NS).to(self.device)

        # Collect different parameter groups
        vae_params = [p for name, p in self.named_parameters()]
        
        # Optimizer: updates encoder/decoder + classifier
        optimizer = torch.optim.Adam([
            {"params": vae_params + list(classifier.parameters()), "lr": lr}
        ])

        # Pyro ELBO
        elbo = TraceMeanField_ELBO()

        # Training loop
        train_loss_history = []
        train_elbo_loss_history = []
        train_cls_loss_history = []
        val_loss_history = []
        val_accuracy_history = []

        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            self.train()
            classifier.train()
            train_loss = 0.0
            all_preds, all_labels = [], []

            for x_batch, y_batch in data_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                batch_size = x_batch.size(0)

                # ----------------------------
                # 1) Compute ELBO
                # ----------------------------
                # Compute ELBO and extract KL divergence
                elbo_trace = pyro.poutine.trace(self.model).get_trace(x_batch, y_batch)
                guide_trace = pyro.poutine.trace(self.guide).get_trace(x_batch, y_batch)
                elbo_loss = elbo.differentiable_loss(self.model, self.guide, x_batch, y_batch)
                # Extract KL divergence for z
                z_node = guide_trace.nodes["z"]
                z_dist = z_node["fn"]
                z_value = z_node["value"]
                # Prior for z
                z_prior_node = elbo_trace.nodes["z"]
                z_prior_dist = z_prior_node["fn"]
                kl_div_z = torch.distributions.kl_divergence(z_dist, z_prior_dist).sum()
                # You can now use kl_div_z as needed (e.g., log, track, etc.)

                # ----------------------------
                # 2) Sample z from encoder (reparameterized)
                # ----------------------------
                z_loc, z_scl = self.encode(x_batch, y_batch)
                qz = pyro.distributions.Normal(z_loc, z_scl)
                z_sample = qz.rsample()

                # ----------------------------
                # 3) Classifier loss on latent z
                # ----------------------------
                # print('z_sample.shape: ', z_sample.shape)
                logits = classifier(z_sample)
                # print('logits.shape: ', logits.shape)
                # print('logits: ', logits)
                cls_loss = F.cross_entropy(logits, torch.argmax(y_batch, dim=1))

                # ----------------------------
                # 4) Combined loss
                # ----------------------------
                combined_loss = elbo_loss - lambda_cls * cls_loss
                
                # clear old gradients
                optimizer.zero_grad() 
                # backpropagate the combined loss
                combined_loss.backward()
                # update parameters
                optimizer.step()

                train_loss += combined_loss.item()
                train_elbo_loss_history.append(elbo_loss.item())
                train_cls_loss_history.append(cls_loss.item())
                all_preds.append(logits.detach())
                all_labels.append(y_batch.detach())

            # ----------------------------
            # 5) Epoch metrics
            # ----------------------------
            train_loss /= len(data_loader.dataset)
            train_loss_history.append(train_loss)
            train_elbo_loss_history.append(sum(train_elbo_loss_history) / len(train_elbo_loss_history))
            train_cls_loss_history.append(sum(train_cls_loss_history) / len(train_cls_loss_history))

            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            predicted_labels = torch.argmax(all_preds, dim=1)
            true_labels = torch.argmax(all_labels, dim=1)
            accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
            
            # ----------------------------
            # 6) Validation
            # ----------------------------
            self.eval()
            classifier.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)

                    # ELBO loss
                    elbo_val = elbo.differentiable_loss(self.model, self.guide, x_val, y_val)

                    # Obtain latent z
                    z_loc_val, z_scl_val = self.encode(x_val, y_val)
                    # qz_val = pyro.distributions.Normal(z_loc_val, z_scl_val)
                    # # Sample latent distribution
                    # z_sample_val = qz_val.rsample()

                    # Obtain predictions 
                    logits_val = classifier(z_loc_val)
                    # Classifier loss
                    cls_loss_val = F.cross_entropy(logits_val, torch.argmax(y_val, dim=1))
                    # Total validation loss
                    combined_val_loss = elbo_val - lambda_cls * cls_loss_val
                    
                    val_loss += combined_val_loss.item() * x_val.size(0)

                    val_preds.append(logits_val)
                    val_labels.append(y_val)

            val_loss /= len(val_loader.dataset)
            val_loss_history.append(val_loss)

            val_preds = torch.cat(val_preds, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            # Compute total validation accuracy (from all batches)
            val_accuracy = accuracy_score(torch.argmax(val_labels, dim=1).cpu(),
                                        torch.argmax(val_preds, dim=1).cpu())
            val_accuracy_history.append(val_accuracy)

            superprint(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.6f}, "
                f"Train Acc={accuracy:.4f}, Val Acc={val_accuracy:.4f}, KL Divergence={kl_div_z:.4f}")

            # Early stopping
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        return train_loss_history, val_loss_history, train_elbo_loss_history, train_cls_loss_history, self, classifier
    
    
    def forward(self, loader):
        """
        Forward pass through the model for a given data loader.

        Args:
            loader (DataLoader): DataLoader containing the input data.

        Returns:
            tuple: (z_loc, xhat) where z_loc is the latent mean and xhat is the reconstructed output.
        """
        
        self.eval()

        # Initialize lists to store outputs
        xhat_lst = []
        z_loc_lst = []

        # Iterate through the data loader
        with torch.no_grad():
            for batch in loader:

                # Move batch to device
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)

                # run forward pass
                z_loc_batch, _ = self.encode(x_batch, y_batch)
                xhat_batch = self.decode(z_loc_batch, y_batch)

                # Append outputs to lists
                z_loc_lst.append(z_loc_batch)
                xhat_lst.append(xhat_batch)

        # Concatenate the outputs from all batches
        z_loc = torch.cat(z_loc_lst, dim=0)
        xhat = torch.cat(xhat_lst, dim=0)

        # Return the latent mean and reconstructed output
        return z_loc, xhat


def compute_mmd(x, y, kernel='rbf', bandwidths=None):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples x and y.
    
    Args:
        x: Tensor of shape (n, d) - samples from distribution P
        y: Tensor of shape (m, d) - samples from distribution Q
        kernel: 'rbf' (default). Can be extended for linear/poly kernels.
        bandwidths: list of bandwidths (σ) for multi-kernel RBF.
                    If None, defaults to [1, 2, 4, 8, 16].
                    
    Returns:
        scalar tensor: MMD^2(x, y)
    """
    n, m = x.size(0), y.size(0)

    if bandwidths is None:
        bandwidths = [1, 2, 4, 8, 16]

    # Compute pairwise squared distances
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    # Multi-kernel RBF
    def rbf(d2, sigma):
        return torch.exp(-d2 / (2 * sigma**2))

    mmd2 = 0.0
    for sigma in bandwidths:
        k_xx = rbf(xx, sigma)
        k_yy = rbf(yy, sigma)
        k_xy = rbf(xy, sigma)

        # Unbiased estimator of MMD^2
        mmd2 += (
            k_xx.sum() - k_xx.diag().sum()
        ) / (n * (n - 1))  # exclude diagonal
        mmd2 += (
            k_yy.sum() - k_yy.diag().sum()
        ) / (m * (m - 1))
        mmd2 -= 2 * k_xy.mean()

    return mmd2 / len(bandwidths)

class ELBO_MMD(pyro.infer.Trace_ELBO):
    def __init__(self, lambda_mmd=1e6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_mmd = lambda_mmd

    def loss(self, model, guide, *args, **kwargs):
        """ Calcula la pèrdua ELBO + regularització """
        
        loss = super().loss(model, guide, *args, **kwargs)  # ELBO normal
        print(f"ELBO loss: {loss:.4f}")

        x, y = args  # Extract x and y from the batch data
        
        # Sample z from the guide
        z = guide(x, y)

        # Extract unique batch labels from y
        unique_labels = y.unique()

        if len(unique_labels) < 2:
            # skip that batch
            return loss

        # num_labels = torch.argmax(unique_labels) # now we have a 0-8 label for each y
        
        # Compare exp_1 with global pool 
        for exp in unique_labels:
            # select z corresponding to exp_label
            z_exp = z[y == exp]

        # Compute MMD for z by batch
        mmd = compute_mmd(z_exp, z_pool)

        return loss + self.lambda_mmd * mmd


