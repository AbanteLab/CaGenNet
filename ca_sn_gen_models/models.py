# Description: Pyro models for the snDGM and VAE models.

# torch
import torch
import torch.nn as nn

# pyro
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI, TraceMeanField_ELBO, Trace_ELBO

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
            
            with pyro.plate("latent_dim", self.K, dim=-1):

                # Standard normal prior for latent variables
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))
            
            # parameters of posterior
            loc = self.decode(z,y)

            with pyro.plate("obs_dim", self.D, dim=-1):
                
                pyro.sample("x", dist.NanMaskedNormal(loc, self.scl), obs=x)

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma).
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

class FixedVarSupMlpGpVAE(nn.Module):

    def __init__(self, D, K, NS, scl=1e-3, device="cpu"):
        
        # Inherit from nn.Module
        super().__init__()
        
        # define properties of model
        self.name = "FixedVarSupMlpGpVAE"
        self.fix_scl = True
        self.sup = True

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
        Defines the generative model P(X | Z, Y) with a Gaussian Process likelihood.
        """

        # Register decoder modules with pyro
        pyro.module("decoder_mlp", self.decoder_mlp)
        pyro.module("decoder_loc", self.decoder_loc)

        batch_size = x.shape[0]

        # Prior for latent variables Z
        z_loc_prior = torch.zeros(batch_size, self.K, device=self.device)
        z_scl_prior = torch.ones(batch_size, self.K, device=self.device)

        # GP kernel hyperparameters (fixed or learnable)
        lengthscale = pyro.param("gp_lengthscale", torch.tensor(10.0, device=self.device), constraint=dist.constraints.positive)
        variance = pyro.param("gp_variance", torch.tensor(1.0, device=self.device), constraint=dist.constraints.positive)
        noise = self.scl if self.fix_scl else pyro.param("gp_noise", torch.tensor(self.scl, device=self.device), constraint=dist.constraints.positive)

        # Frequency/time points for GP (assume evenly spaced)
        t = torch.arange(self.D, device=self.device).float().unsqueeze(0)  # [1, D]

        with pyro.plate("data", batch_size, dim=-2):
            
            # sample label
            alpha_prior = torch.ones([batch_size, self.NS], device=self.device) / float(self.NS)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)

            with pyro.plate("latent_dim", self.K, dim=-1):
                z = pyro.sample("z", dist.Normal(z_loc_prior, z_scl_prior))

            # Decoder output as GP mean
            mean = self.decode(z, y)  # [batch, D]

            ## TODO: fix pyro GP implementation for better performance
            
            # Use Pyro's SparseGPRegression for efficient GP likelihood
            # Set up inducing points (here, use a subset of t or evenly spaced points)
            num_inducing = min(32, self.D)
            Xu = torch.linspace(0, self.D - 1, num_inducing, device=self.device).unsqueeze(-1)  # [M, 1]
            X = t.T  # [D, 1]

            # Define RBF kernel
            kernel = gp.kernels.RBF(input_dim=1, variance=variance, lengthscale=lengthscale)

            # Register a single GP model for all observations in the batch
            # The GP model expects input shape [N*D, input_dim], so we stack batch and time dims
            # Here, we use a batched GP model for efficiency

            # Expand X to [batch_size * D, 1], y_obs to [batch_size * D, 1], mean to [batch_size * D, 1]
            X_batch = X.repeat(batch_size, 1)  # [batch_size * D, 1]
            y_obs_batch = x.reshape(-1, 1)     # [batch_size * D, 1]
            mean_batch = mean.reshape(-1, 1)   # [batch_size * D, 1]

            # Inducing points for the full batch
            Xu_batch = Xu  # [M, 1]

            # Use a single SparseGPRegression for the whole batch
            # mean_function should return a tensor of shape [N, 1] for input X_ of shape [N, 1]
            # Here, we assume mean_batch is aligned with X_batch, so we can use a lambda that returns the correct slice
            sgp = gp.models.SparseGPRegression(
                X_batch, y_obs_batch, kernel, Xu_batch,
                noise=noise, mean_function=lambda X_: mean_batch
            )
            sgp.model()  # Registers the GP likelihood with Pyro

            ## NOTE: torch implementation of GP kernel matrix not very optimal for large D

            # # Compute GP kernel matrix (RBF)
            # # K[i,j] = variance * exp(-0.5 * (t_i - t_j)^2 / lengthscale^2)
            # diff = t.unsqueeze(-1) - t.unsqueeze(-2)  # [1, D, D]
            # K = variance * torch.exp(-0.5 * (diff ** 2) / (lengthscale ** 2))  # [1, D, D]
            # K = K + noise * torch.eye(self.D, device=self.device, dtype=t.dtype).unsqueeze(0)  # Add noise to diagonal

            # # diff = t.unsqueeze(-1) - t.unsqueeze(-2)  # [1, D, D]
            # # K = variance * torch.exp(-0.5 * (diff ** 2) / (lengthscale ** 2))  # [1, D, D]
            # # # K = K.expand(batch_size, self.D, self.D)  # [batch, D, D]
            # # K = K + noise * torch.eye(self.D, device=self.device).unsqueeze(0)  # Add noise to diagonal

            # # Multivariate Normal likelihood for each batch element
            # pyro.sample("x",dist.MultivariateNormal(mean, covariance_matrix=K),obs=x)

            # for i in range(batch_size):
            #     pyro.sample(
            #         f"x_{i}",
            #         dist.MultivariateNormal(mean[i], covariance_matrix=K[i]),
            #         obs=x[i]
            #     )

    def guide(self, x, y):
        """
        Defines the variational guide q(W, Z, sigma).
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