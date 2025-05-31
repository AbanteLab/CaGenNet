# Description: Pyro models for the snDGM and VAE models.

# torch
import torch
import torch.nn as nn

# pyro
import pyro
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
# Deep Generative Model for time domain data (X) based on MLP embeddings that fixes the output scale.
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

############################################################################################################
# Deep Generative Model for time domain data (X) based on MLP embeddings that learns the output scale.
############################################################################################################
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
        scale = torch.clamp(self.encoder_scl(x), min=1e-3)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
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

############################################################################################################
# Supervised Deep Generative Model for time domain data (X) based on LSTM embeddings that learns the 
# output scale.
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
        scale = self.emb_to_scl(h)
        
        # Return the mean and scale of the latent representation
        return loc, scale
    
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

############################################################################################################
# Supervised Deep Generative Model for time domain data (X) based on MLP embeddings that fixes the 
# output scale.
############################################################################################################
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
    
############################################################################################################
# Supervised Deep Generative Model for time domain data (X) based on LSTM embeddings
############################################################################################################
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

########################################################################################
# Deep generative model time domain for denoising and imputation
########################################################################################

########################################################################################
# Deep generative model time domain for denoising and imputation
########################################################################################
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

########################################################################################
# Deep generative model frequency domain (Amplitude)
########################################################################################
# class AmpVAE(nn.Module):
#     """
#     Variational Autoencoder (VAE) for FFT amplitude data.
    
#     This model consists of a single latent variable:
#     - Z: Latent variable
    
#     Attributes:
        
#         input_dim (int): Dimension of the input data.
#         latent_dim (int): Dimension of the latent variable Z.
#         hidden_dim (int): Dimension of the hidden layers in the neural networks.
#         target_dist (pyro.distributions): Target distribution for the data.
#         device (str): Device to use for the model.
        
#     Methods:
    
#         model(a):
#             Defines the generative model.
#         guide(a):
#             Defines the inference model.
            
#     """
#     def __init__(self, input_dim, latent_dim=16, enc_dims=None, dec_dims=None, target_dist=dist.Normal, device='cpu'):
#         super().__init__()
        
#         # store to be accessed elsewhere
#         self.zinf = False
#         self.device = device
#         self.loc_and_scale = False
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.target_dist = target_dist

#         # Default dimensions if not provided
#         if enc_dims is None:
#             enc_dims = [input_dim // 2, input_dim // 4, input_dim // 8]
#         if dec_dims is None:
#             dec_dims = [latent_dim * 2, latent_dim * 4, latent_dim * 8]
        
#         # check if model is loc or loc+scale
#         if target_dist != dist.Exponential:
#             self.loc_and_scale = True
        
#         # check conditions
#         if target_dist in [dist.Exponential,dist.Gamma,dist.Weibull]:
#             # need positive location
#             self.decoder_loc = Decoder(latent_dim, input_dim, dec_dims, output='softplus')
#         else:
#             self.decoder_loc = Decoder(latent_dim, input_dim, dec_dims)
            
#         # add decoder for scale if necessary
#         if self.loc_and_scale:
#             self.decoder_scale = Decoder(latent_dim, input_dim, dec_dims, output='softplus')
        
#         # define encoders
#         self.encoder_mu = Encoder(input_dim, latent_dim, enc_dims)
#         self.encoder_logvar = Encoder(input_dim, latent_dim, enc_dims)

#         # Move model to device
#         self.to(device)

#     def model(self, a, bkl=1.0):
        
#         batch_size = a.shape[0]
        
#         # Move data to the correct device
#         a = a.to(self.device)

#         # Latent prior
#         mu_prior = torch.zeros(batch_size, self.latent_dim, device=self.device)
#         sigma_prior = torch.ones(batch_size, self.latent_dim, device=self.device)
        
#         # register modules
#         pyro.module("decoder_loc", self.decoder_loc)
#         if self.loc_and_scale:
#             pyro.module("decoder_scale", self.decoder_scale)

#         # plate for mini-batch
#         with pyro.poutine.scale(scale=bkl):
#             with pyro.plate("batch", batch_size):
            
#                 # Sample latent variable
#                 Za = pyro.sample("Za", dist.Normal(mu_prior, sigma_prior).to_event(1))
            
#                 # get location
#                 loc = self.decoder_loc(Za)
                
#                 # get scale if not exponential
#                 if self.loc_and_scale:
                    
#                     ## NOTE: tried clamping scale but ELBO doesn't improve
#                     # scale = torch.clamp(self.decoder_scale(Za), min=0.1)
                    
#                     # Decode scale
#                     scale = self.decoder_scale(Za) + 1e-6
        
#                     # Target likelihood
#                     pyro.sample("A_obs", self.target_dist(loc, scale).to_event(1), obs=a)        
                
#                 else:
                    
#                     # Target likelihood
#                     pyro.sample("A_obs", self.target_dist(loc).to_event(1), obs=a)

#     def guide(self, a, bkl=1.0):
        
#         batch_size = a.shape[0]

#         # Move data to the correct device
#         a = a.to(self.device)

#         # register modules
#         pyro.module("encoder_mu", self.encoder_mu)
#         pyro.module("encoder_logvar", self.encoder_logvar)
        
#         # plate for mini-batch
#         with pyro.poutine.scale(scale=bkl):
#             with pyro.plate("batch", batch_size, dim=-1):
            
#                 # Compute approximate posterior for Z
#                 mu_Za = self.encoder_mu(a)
#                 sigma_Za = torch.exp(self.encoder_logvar(a)) + 1e-6

#                 # Sample latent variable
#                 Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))
    
#     def train_model(self, train_loader, val_loader, svi, num_epochs=1000, patience=10, min_delta=1e-3, start_beta=0.25, end_beta=0.25):
#         """
#         Trains the model using Stochastic Variational Inference (SVI).

#         Args:
#             train_loader (DataLoader): DataLoader for training data.
#             val_loader (DataLoader): DataLoader for validation data.
#             svi (SVI): Pyro SVI object.
#             num_epochs (int): Number of epochs to train.
#             patience (int): Early stopping patience.
#             min_delta (float): Minimum change in validation loss for early stopping.

#         Returns:
#             tuple: Training and validation loss history.
#         """
        
#         # vectors to store history
#         val_loss_history = []
#         train_loss_history = []
        
#         # early stopping parameters
#         patience_counter = 0
#         best_val_loss = float('inf')

#         for epoch in range(num_epochs):
            
#             # get beta value for epoch
#             bkl = kl_beta_schedule(epoch, num_epochs // 2, start_beta=start_beta, end_beta=end_beta)
            
#             # Training phase
#             self.train()
#             train_loss = 0.0
#             for X_batch in train_loader:
#                 X_batch = X_batch[0].to(self.device)
#                 train_loss += svi.step(X_batch, bkl=bkl) / X_batch.size(0)
#             train_loss /= len(train_loader.dataset)
#             train_loss_history.append(train_loss)

#             # Validation phase
#             self.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for X_batch in val_loader:
#                     X_batch = X_batch[0].to(self.device)
#                     val_loss += svi.evaluate_loss(X_batch, bkl=bkl) / X_batch.size(0)
#             val_loss /= len(val_loader.dataset)
#             val_loss_history.append(val_loss)

#             # Early stopping
#             if val_loss < best_val_loss - min_delta:
#                 best_val_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     superprint(f"Early stopping at epoch {epoch + 1}")
#                     break

#             superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#         return train_loss_history, val_loss_history
    
###########################################################################################################
# CHECK THESE MODELS
###########################################################################################################

# class ConvVAE(nn.Module):
#     def __init__(self, input_dim, latent_dim=16, target_dist=dist.Normal, device='cpu'):
#         super().__init__()
        
#         # store to be accessed elsewhere
#         self.device = device
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
#         self.target_dist = target_dist
        
#         # define encoder and decoders for location and scale
#         self.encoder = ConvEncoder(input_dim, latent_dim)
#         self.decoder = ConvDecoder(latent_dim, input_dim)

#         # Move model to device
#         self.to(device)

#     def model(self, x, bkl=1.0):
        
#         batch_size = x.shape[0]
        
#         # Move data to the correct device
#         x = x.to(self.device)

#         # Latent prior
#         mu_prior = torch.zeros(batch_size, self.latent_dim, device=self.device)
#         sigma_prior = torch.ones(batch_size, self.latent_dim, device=self.device)
        
#         # register modules
#         pyro.module("decoder", self.decoder)
    
#         # plate for mini-batch
#         with pyro.poutine.scale(scale=bkl):
#             with pyro.plate("batch", batch_size):
            
#                 # Sample latent variable
#                 Z = pyro.sample("Z", dist.Normal(mu_prior, sigma_prior).to_event(1))
            
#                 # get location and scale
#                 loc_x,scale_x = self.decoder(Z)
    
#                 # Target likelihood
#                 pyro.sample("X", self.target_dist(loc_x, scale_x).to_event(1), obs=x)

#     def guide(self, x, bkl=1.0):
        
#         batch_size = x.shape[0]

#         # Move data to the correct device
#         x = x.to(self.device)

#         # register modules
#         pyro.module("encoder", self.encoder)
        
#         # plate for mini-batch
#         with pyro.poutine.scale(scale=bkl):
#             with pyro.plate("batch", batch_size, dim=-1):
            
#                 # Compute approximate posterior for Z
#                 loc_Z,scale_Z = self.encoder(x)

#                 # Sample latent variable
#                 Z = pyro.sample("Z", dist.Normal(loc_Z, scale_Z).to_event(1))
    
#     def train_model(self, train_loader, val_loader, svi, num_epochs=1000, patience=10, min_delta=1e-3, start_beta=0.25, end_beta=0.25):
#         """
#         Trains the model using Stochastic Variational Inference (SVI).

#         Args:
#             train_loader (DataLoader): DataLoader for training data.
#             val_loader (DataLoader): DataLoader for validation data.
#             svi (SVI): Pyro SVI object.
#             num_epochs (int): Number of epochs to train.
#             patience (int): Early stopping patience.
#             min_delta (float): Minimum change in validation loss for early stopping.

#         Returns:
#             tuple: Training and validation loss history.
#         """
        
#         # vectors to store history
#         val_loss_history = []
#         train_loss_history = []
        
#         # early stopping parameters
#         patience_counter = 0
#         best_val_loss = float('inf')

#         for epoch in range(num_epochs):
            
#             # get beta value for epoch
#             bkl = kl_beta_schedule(epoch, num_epochs // 2, start_beta=start_beta, end_beta=end_beta)
            
#             # Training phase
#             self.train()
#             train_loss = 0.0
#             for X_batch in train_loader:
#                 X_batch = X_batch[0].to(self.device)
#                 train_loss += svi.step(X_batch, bkl=bkl) / X_batch.size(0)
#             train_loss /= len(train_loader.dataset)
#             train_loss_history.append(train_loss)

#             # Validation phase
#             self.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for X_batch in val_loader:
#                     X_batch = X_batch[0].to(self.device)
#                     val_loss += svi.evaluate_loss(X_batch, bkl=bkl) / X_batch.size(0)
#             val_loss /= len(val_loader.dataset)
#             val_loss_history.append(val_loss)

#             # Early stopping
#             if val_loss < best_val_loss - min_delta:
#                 best_val_loss = val_loss
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     superprint(f"Early stopping at epoch {epoch + 1}")
#                     break

#             superprint(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#         return train_loss_history, val_loss_history

# class AVAE(nn.Module):
    
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(AVAE, self).__init__()
        
#         # dimensions
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim = latent_dim
        
#         # Encoder layers
#         self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.encoder_fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.encoder_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
#         # Decoder layers
#         self.decoder_l1 = nn.Linear(latent_dim, hidden_dim)
#         self.decoder_l2 = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
#     def encode(self, x):
         
#         # get output of lstm
#         _,(hd,_) = self.encoder_lstm(x.unsqueeze(1))
        
#         # remove unnecessary dimension
#         hd = hd.squeeze(0)
        
#         # get parameters q(z|x)
#         mu = self.encoder_fc_mu(hd)
#         logvar = self.encoder_fc_logvar(hd)
        
#         return mu, logvar
    
#     def sample_z(self, mu, logvar):
        
#         # reparametrize
#         std = torch.exp(0.5 * logvar)
        
#         # sample eps~(0,1)
#         eps = torch.randn_like(std)

#         # return z
#         return mu + eps * std
    
#     def decode(self, z):
        
#         # get linear layer output
#         out = self.decoder_l1(z)
        
#         # get lstm output (batch_size,1,)
#         out, _ = self.decoder_l2(out.unsqueeze(1))
        
#         return out.squeeze(1)
    
#     def forward(self, x):
        
#         # Encode
#         mu, logvar = self.encode(x)
        
#         # Reparameterize
#         z = self.sample_z(mu, logvar)
        
#         # Decode
#         reconstructed_x = self.decode(z)
        
#         return reconstructed_x, mu, logvar
    
# def reconstruction_loss(recon_x, x, rec_loss='mse'):
#     """
#     Computes the reconstruction loss for the VAE.
    
#     Args:
#         recon_x (torch.Tensor): The reconstructed input.
#         x (torch.Tensor): The original input.
#         rec_loss (str): The type of reconstruction loss to use ('mse' or 'mae').
    
#     Returns:
#         torch.Tensor: The computed reconstruction loss.
#     """
    
#     if rec_loss == 'mse':
#         # Mean Squared Error loss
#         return F.mse_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
#     elif rec_loss == 'mae':
#         # Mean Absolute Error loss
#         return F.l1_loss(recon_x.view(-1, recon_x.size(-1)), x.view(-1, x.size(-1)), reduction='sum')
#     else:
#         raise ValueError("Invalid rec_loss. Expected 'mse' or 'mae'.")

# def kl_divergence_loss(mu, logvar):
#     """
#     Computes the KL divergence loss for the VAE.
    
#     Args:
#         mu (torch.Tensor): The mean of the latent space.
#         logvar (torch.Tensor): The log variance of the latent space.
    
#     Returns:
#         torch.Tensor: The computed KL divergence loss.
#     """
    
#     return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# def positive_pairwise_loss(mu, roi, roi_metric='cosine'):
#     """
#     Encourages different views of the same ROI to have similar embeddings.

#     Args:
#         mu (torch.Tensor): The mean of the latent space (batch_size, latent_dim).
#         roi (torch.Tensor): The region of interest identifiers (batch_size,).
#         roi_metric (str): The metric to use for computing the pairwise loss ('cosine' or 'euclidean').
        
#     Returns:
#         torch.Tensor: The computed penalty term.
#     """

#     # Initialize total loss
#     total_loss = 0.0

#     # Get unique ROIs
#     unique_rois = roi.unique()
    
#     # Normalize embeddings
#     mu = F.normalize(mu, dim=-1)

#     # Iterate over unique ROIs
#     for unique_roi in unique_rois:
        
#         # Get indices of same ROI
#         indices = (roi == unique_roi).nonzero(as_tuple=True)[0]
        
#         # If there are multiple indices
#         if len(indices) > 1:
            
#             # Extract embeddings for this ROI
#             mu_roi = mu[indices]  
            
#             if roi_metric == 'cosine':
                
#                 # Compute pairwise cosine similarity
#                 similarity_matrix = torch.matmul(mu_roi, mu_roi.T)
                
#                 # Compute loss: minimize distance from perfect similarity (1.0)
#                 total_loss += (1 - similarity_matrix).mean()
                
#             elif roi_metric == 'euclidean':
                
#                 # Compute pairwise euclidean distance
#                 distance_matrix = torch.cdist(mu_roi, mu_roi, p=2)
                
#                 # Compute loss: minimize the distance
#                 total_loss += distance_matrix.mean()
                
#             else:
                
#                 raise ValueError("Invalid roi_metric. Expected 'cosine' or 'euclidean'.")

#     # Return total loss
#     return total_loss

# def loss_function(xhat, x, mu, logvar, roi, **optional_args):
#     """
#     Computes the total loss function for the VAE, including a penalty term for the distance in z within fluo signals from the same roi.
    
#     Args:
#         xhat (torch.Tensor): The reconstructed input.
#         x (torch.Tensor): The original input.
#         mu (torch.Tensor): The mean of the latent space.
#         logvar (torch.Tensor): The log variance of the latent space.
#         roi (torch.Tensor): The region of interest identifiers.
    
#     Returns:
#         torch.Tensor: The computed total loss.
#     """
    
#     # Extract optional arguments
#     bkl = optional_args.get('bkl', 1.0)
#     broi = optional_args.get('broi', 1.0)
#     rec_loss = optional_args.get('rec_loss', 'mse')
#     model_type = optional_args.get('model_type', 'vae')
#     roi_metric = optional_args.get('roi_metric', 'cosine')
    
#     # Compute the reconstruction loss
#     recon_loss = reconstruction_loss(xhat, x, rec_loss) / x.size(0)
    
#     # Compute the KL divergence loss
#     kl_loss = bkl * kl_divergence_loss(mu, logvar) / x.size(0)
    
#     # Initialize the InfoNCE loss to 0
#     pp_loss = torch.tensor(0)
    
#     # Compute the ROI loss if using the CLAVAE model
#     if model_type == 'clavae':
#         pp_loss = broi * 100 * positive_pairwise_loss(mu, roi, roi_metric=roi_metric)  / x.size(0)
    
#     # Compute the total loss
#     total_loss = recon_loss + kl_loss + pp_loss
    
#     return total_loss, recon_loss, kl_loss, pp_loss

# # Define a function to train the VAE model
# def train_clavae(model, train_loader, val_loader, device, **optional_args):
#     """
#     Trains the CLAVAE model using contrastive learning on the given data.

#     Args:
#         model (CLAVAE): The contrastive learning autoregressive VAE model to train.
#         train_loader (DataLoader): The DataLoader for the training data.
#         optimizer (torch.optim.Optimizer): The optimizer to use for training.
#         device (torch.device): The device to use for training.

#     Returns:
#         float: The average training loss.
#     """

#     # get parameters
#     lr = optional_args.get('lr', 1e-4)
#     patience = optional_args.get('patience', 20)
#     min_delta = optional_args.get('min_delta', 1e-4)
#     num_epochs = optional_args.get('num_epochs', 1000)
    
#     # Define the optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, amsgrad=False)
    
#     # Define the scheduler
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
#     # Initialize early stopping parameters
#     patience_counter = 0
#     best_val_loss = float('inf')
    
#     # Initialize lists to store losses
#     avg_train_loss_trail, avg_val_loss_trail = [], []

#     # Loop over epochs
#     for epoch in range(num_epochs):
        
#         # Training phase
#         model.train()
        
#         # Initialize epoch losses
#         train_loss = 0
#         total_kl_loss = 0
#         total_pp_loss = 0
#         total_recon_loss = 0

#         # Iterate over the training data
#         for batch_idx, (fluo, roi) in enumerate(train_loader):
            
#             # Move the data to the device
#             fluo = fluo.to(device)
            
#             # Zero the gradients
#             optimizer.zero_grad()
            
#             # Perform a forward pass
#             recon_batch, mu, logvar = model(fluo)
            
#             # Compute the loss
#             tot_loss, recon_loss, kl_loss, pp_loss = loss_function(recon_batch, fluo, mu, logvar, roi, **optional_args)
            
#             # Perform a backward pass
#             tot_loss.backward()
            
#             # Clip the gradients
#             clip_grad_norm_(model.parameters(), 1)
            
#             # Step the optimizer for batch
#             optimizer.step()
            
#             # Add the losses to the total losses
#             train_loss += tot_loss.item()
#             total_kl_loss += kl_loss.item()
#             total_pp_loss += pp_loss.item()
#             total_recon_loss += recon_loss.item()

#         # Validation phase
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for fluo, roi in val_loader:
#                 fluo = fluo.to(device)
#                 recon_batch, mu, logvar = model(fluo)
#                 tot_loss, _, _, _ = loss_function(recon_batch, fluo, mu, logvar, roi, **optional_args)
#                 val_loss += tot_loss.item()

#         # Average the losses over batch size
#         val_loss /= len(val_loader.dataset)
#         train_loss /= len(train_loader.dataset)
        
#         # Store the average losses
#         avg_train_loss_trail.append(train_loss)
#         avg_val_loss_trail.append(val_loss)

#         # Early stopping
#         if val_loss < best_val_loss - min_delta:
#             best_val_loss = val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 superprint(f"Early stopping at epoch {epoch + 1}")
#                 break
            
#         # Step the scheduler at the end of the epoch
#         scheduler.step()
#         current_lr = optimizer.param_groups[0]['lr']
        
#         # Print epoch results
#         superprint(f"Epoch: {epoch+1}/{num_epochs},  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Lr: {current_lr:.4f}")
    
#     # Calculate average losses
#     avg_recon_loss = total_recon_loss / len(train_loader.dataset)
#     avg_kl_loss = total_kl_loss / len(train_loader.dataset)
#     avg_pp_loss = total_pp_loss / len(train_loader.dataset)
    
#     # Print average losses
#     superprint(f"L_rec: {avg_recon_loss:.6f} | L_kl: {avg_kl_loss:.6f} | L_pp: {avg_pp_loss:.6f}")
        
#     # Return the average training loss
#     return avg_train_loss_trail, avg_val_loss_trail

# class ZinfLogNormVAE(nn.Module):
#     """
#     Variational Autoencoder (VAE) for FFT amplitude data.
    
#     This model consists of a single latent variable:
#     - Z: Latent variable
    
#     Attributes:
#         input_dim (int): Dimension of the input data.
#         latent_dim (int): Dimension of the latent variable Z.
#         hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
#     Methods:
#         model(a):
#             Defines the generative model.
#         guide(a):
#             Defines the inference model.
#     """
#     def __init__(self, input_dim, latent_dim=16, enc_dims=None, dec_dims=None, device='cpu'):
#         super().__init__()
        
#         self.device = device
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim

#         # Default dimensions if not provided
#         if enc_dims is None:
#             enc_dims = [input_dim // 2, input_dim // 4, input_dim // 8]
#         if dec_dims is None:
#             dec_dims = [latent_dim * 2, latent_dim * 4, latent_dim * 8]
        
#         # define decoders
#         self.decoder_mu = Decoder(latent_dim, input_dim, dec_dims)
#         self.decoder_logvar = Decoder(latent_dim, input_dim, dec_dims)
#         self.decoder_zinf = Decoder(latent_dim, input_dim, dec_dims)

#         # define encoders
#         self.encoder_mu = Encoder(input_dim, latent_dim, enc_dims)
#         self.encoder_logvar = Encoder(input_dim, latent_dim, enc_dims)

#         # Move model to device
#         self.to(device)

#     def model(self, a):
        
#         batch_size = a.shape[0]
        
#         # Move data to the correct device
#         a = a.to(self.device)

#         # Latent prior
#         mu_prior = torch.zeros(batch_size, self.latent_dim, device=self.device)
#         sigma_prior = torch.ones(batch_size, self.latent_dim, device=self.device)
        
#         # register modules
#         pyro.module("decoder_mu", self.decoder_mu)
#         pyro.module("decoder_zinf", self.decoder_zinf)
#         pyro.module("decoder_logvar", self.decoder_logvar)
        
#         with pyro.plate("batch", batch_size):
            
#             # Sample latent variable
#             Za = pyro.sample("Za", dist.Normal(mu_prior, sigma_prior).to_event(1))

#             # Decode 0 probability
#             p_zero = torch.sigmoid(self.decoder_zinf(Za))
            
#             # Decode to amplitude parameters
#             loc_A = self.decoder_mu(Za)
#             scale_A = torch.exp(self.decoder_logvar(Za)) + 1e-6
            
#             # Zero-inflated LogNormal likelihood
#             pyro.sample("A_obs", ZeroInflatedLogNormal(loc_A, scale_A, gate=p_zero).to_event(1), obs=a)

#     def guide(self, a):
        
#         batch_size = a.shape[0]

#         # Move data to the correct device
#         a = a.to(self.device)
        
#         # register modules
#         pyro.module("encoder_mu", self.encoder_mu)
#         pyro.module("encoder_logvar", self.encoder_logvar)

#         with pyro.plate("batch", batch_size, dim=-1):
            
#             # Compute approximate posterior for Z
#             mu_Za = self.encoder_mu(a)
#             sigma_Za = torch.exp(self.encoder_logvar(a)) + 1e-6
#             pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))
                        
# class HierarchicalVAE(nn.Module):
#     """
    
#     Hierarchical Variational Autoencoder (VAE) with a three-level latent structure.
    
#     This model consists of three latent variables:
#     - Zs: Global latent variable
#     - Za: Amplitude latent variable
#     - Zp: Phase latent variable
    
#     Graphical Model:
#         Zs
#        /  \
#       Za   Zp
#       |    |
#       A   Phi
    
#     Attributes:
#         input_dim (int): Dimension of the input data.
#         latent_dim_s (int): Dimension of the global latent variable Zs.
#         latent_dim_a (int): Dimension of the amplitude latent variable Za.
#         latent_dim_p (int): Dimension of the phase latent variable Zp.
#         hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
#     Methods:
#         model(a, p):
#             Defines the generative model.
#         guide(a, p):
#             Defines the inference model.
    
#     """
#     def __init__(self, input_dim, latent_dim_s=64, latent_dim_a=32, latent_dim_p=32, hidden_dim=512, device='cpu'):
        
#         super().__init__()
        
#         self.device = device
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.latent_dim_s = latent_dim_s
#         self.latent_dim_a = latent_dim_a
#         self.latent_dim_p = latent_dim_p

#         # decoder Zs to Za and Zp
#         self.mu_a_net = Decoder(latent_dim_s, latent_dim_a, [latent_dim_s])
#         self.sigma_a_net = Decoder(latent_dim_s, latent_dim_a, [latent_dim_s])
#         self.mu_p_net = Decoder(latent_dim_s, latent_dim_p, [latent_dim_s])
#         self.sigma_p_net = Decoder(latent_dim_s, latent_dim_p, [latent_dim_s])

#         # decoder amplitude Za to A (needs mu to be positive for lognormal - relu)
#         self.decoder_a_mu = Decoder(latent_dim_a, input_dim, [128, 512, 1028])
#         self.decoder_a_logvar = Decoder(latent_dim_a, input_dim, [128, 512, 1028])
        
#         # decoder phi Zp to Phi (needs mu to be [-pi, pi] and kappa to be positive)
#         self.decoder_p_mu = Decoder(latent_dim_p, input_dim, [128, 512, 1028])
#         self.decoder_p_kappa = Decoder(latent_dim_p, input_dim, [128, 512, 1028])

#         # Use ConvEncoder for encoder_s_mu
#         self.encoder_s_mu = ConvEncoder(input_channels=2, output_dim=latent_dim_s, hidden_dim=hidden_dim)
#         self.encoder_s_logvar = ConvEncoder(input_channels=2, output_dim=latent_dim_s, hidden_dim=hidden_dim)

#         # Encoders for Za
#         self.encoder_a_mu = Encoder(latent_dim_s, latent_dim_a, [latent_dim_s])
#         self.encoder_a_logvar = Encoder(latent_dim_s, latent_dim_a, [latent_dim_s])
        
#         # Encoders for Zp
#         self.encoder_p_mu = Encoder(latent_dim_s, latent_dim_p, [latent_dim_s])
#         self.encoder_p_logvar = Encoder(latent_dim_s, latent_dim_p, [latent_dim_s])

#         # Initialize parameters
#         self._initialize_parameters()

#         # Move model to device
#         self.to(self.device)

#     # def _initialize_parameters(self):
#     #     # Initialize weights and biases for decoders
#     #     for decoder in [self.decoder_a_mu, self.decoder_a_logvar, self.decoder_p_mu, self.decoder_p_kappa]:
#     #         for layer in decoder.net:
#     #             if isinstance(layer, nn.Linear):
#     #                 nn.init.kaiming_normal_(layer.weight)
#     #                 nn.init.constant_(layer.bias, 0)

#     #     # Initialize weights and biases for encoders
#     #     for encoder in [self.encoder_s_mu, self.encoder_s_logvar, self.encoder_a_mu, self.encoder_a_logvar, self.encoder_p_mu, self.encoder_p_logvar]:
#     #         for layer in encoder.net:
#     #             if isinstance(layer, nn.Linear):
#     #                 nn.init.kaiming_normal_(layer.weight)
#     #                 nn.init.constant_(layer.bias, 0)

#     #     # Initialize ConvEncoder separately
#     #     for conv_encoder in [self.encoder_s_mu, self.encoder_s_logvar]:
#     #         for layer in conv_encoder.net:
#     #             if isinstance(layer, nn.Conv1d):
#     #                 nn.init.kaiming_normal_(layer.weight)
#     #                 nn.init.constant_(layer.bias, 0)

#     def model(self, a, p):
        
#         batch_size = a.shape[0]
        
#         # Move data to the correct device
#         a = a.to(self.device)
#         p = p.to(self.device)

#         # register modules
#         pyro.module("mu_a_net", self.mu_a_net)
#         pyro.module("sigma_a_net", self.sigma_a_net)
#         pyro.module("mu_p_net", self.mu_a_net)
#         pyro.module("sigma_p_net", self.sigma_a_net)
#         pyro.module("decoder_a_mu", self.decoder_a_mu)
#         pyro.module("decoder_a_logvar", self.decoder_a_logvar)
#         pyro.module("decoder_p_mu", self.decoder_p_mu)
#         pyro.module("decoder_p_kappa", self.decoder_p_kappa)
        
#         # Latent prior
#         mu_prior = torch.zeros(batch_size, self.latent_dim_s, device=self.device)
#         sigma_prior = torch.ones(batch_size, self.latent_dim_s, device=self.device)
        
#         # plate
#         with pyro.plate("batch", batch_size, dim=-1):  # Define a batch plate
            
#             # Global latent variable Zs
#             Zs = pyro.sample("Zs",dist.Normal(mu_prior, sigma_prior).to_event(1))

#             # Amplitude latent variable Za
#             mu_a = self.mu_a_net(Zs)
#             sigma_a = torch.exp(self.sigma_a_net(Zs)) + 1e-6  # Ensure positivity
#             Za = pyro.sample("Za", dist.Normal(mu_a, sigma_a).to_event(1))

#             # Phase latent variable Zp
#             mu_p = self.mu_p_net(Zs)
#             sigma_p = torch.exp(self.sigma_p_net(Zs)) + 1e-6  # Ensure positivity
#             Zp = pyro.sample("Zp", dist.Normal(mu_p, sigma_p).to_event(1))

#             # Decode amplitude and phase
#             mu_A = self.decoder_a_mu(Za)
#             sigma_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-6
            
#             # Amplitude likelihood (Gaussian)
#             pyro.sample("A_obs", dist.LogNormal(mu_A, sigma_A).to_event(1), obs=a)
            
#             # Decode phase
#             mu_P = self.decoder_p_mu(Zp)
#             kappa_P = torch.exp(self.decoder_p_kappa(Zp)) + 1e-6  # Ensure positivity

#             # Phase likelihood (Von Mises)
#             pyro.sample("P_obs", dist.VonMises(mu_P, kappa_P).to_event(1), obs=p)

#     def guide(self, a, p):
    
#         batch_size = a.shape[0]

#         # Move data to the correct device
#         a = a.to(self.device)
#         p = p.to(self.device)
        
#         # register modules
#         pyro.module("encoder_s_mu", self.encoder_s_mu)
#         pyro.module("encoder_s_logvar", self.encoder_s_logvar)
#         pyro.module("encoder_a_mu", self.encoder_a_mu)
#         pyro.module("encoder_a_logvar", self.encoder_a_logvar)
#         pyro.module("encoder_p_mu", self.encoder_p_mu)
#         pyro.module("encoder_p_logvar", self.encoder_p_logvar)
        
#         # plate
#         with pyro.plate("batch", batch_size, dim=-1):
            
#             # Compute approximate posterior for Zs using a neural network
#             ap_concat = torch.stack((a, p), dim=1)
#             # ap_concat = torch.cat((a, p), dim=1)
#             mu_Zs = self.encoder_s_mu(ap_concat)
#             sigma_Zs = torch.exp(self.encoder_s_logvar(ap_concat)) + 1e-6  # Ensure positivity
#             Zs = pyro.sample("Zs", dist.Normal(mu_Zs, sigma_Zs).to_event(1))

#             # Compute approximate posterior for Za (depends on Zs)
#             # Zs_a_concat = torch.cat((Zs, a), dim=1)
#             mu_Za = self.encoder_a_mu(Zs)
#             sigma_Za = torch.exp(self.encoder_a_logvar(Zs)) + 1e-6
#             Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))

#             # Compute approximate posterior for Zp (depends on Zs and p)
#             # Zs_p_concat = torch.cat((Zs, p), dim=1)
#             mu_Zp = self.encoder_p_mu(Zs)
#             sigma_Zp = torch.exp(self.encoder_p_logvar(Zs)) + 1e-6
#             Zp = pyro.sample("Zp", dist.Normal(mu_Zp, sigma_Zp).to_event(1))

# class IcasspVAE(nn.Module):
#     """
    
#     Variational Autoencoder (VAE) with a two-level latent structure.
    
#     This model consists of two latent variables:
#     - Za: Amplitude latent variable
#     - Zp: Phase latent variable
    
#     Graphical Model:
#         Za   Zp
#         |    |
#         A - Phi
    
#     Attributes:
#         input_dim (int): Dimension of the input data.
#         latent_dim_a (int): Dimension of the amplitude latent variable Za.
#         latent_dim_p (int): Dimension of the phase latent variable Zp.
#         hidden_dim (int): Dimension of the hidden layers in the neural networks.
    
#     Methods:
#         model(a, p):
#             Defines the generative model.
#         guide(a, p):
#             Defines the inference model.
    
#     """
#     def __init__(self, input_dim, latent_dim_a=64, latent_dim_p=64, hidden_dim=1024, device='cpu'):
        
#         super().__init__()
        
#         # define fields
#         self.device = device
#         self.input_dim = input_dim
#         self.latent_dim_a = latent_dim_a
#         self.latent_dim_p = latent_dim_p

#         # decoder amplitude: Za to A (needs mu to be positive for lognormal)
#         self.decoder_a_mu = nn.Sequential(
#             nn.Linear(latent_dim_a, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, 2 * hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(2 * hidden_dim, 4 * hidden_dim),
#             nn.ReLU(),
#             nn.Linear(4 * hidden_dim, input_dim),
#             nn.ReLU()
#         )
#         self.decoder_a_logvar = nn.Sequential(
#             nn.Linear(latent_dim_a, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, 2 * hidden_dim), 
#             nn.ReLU(),
#             nn.Linear(2 * hidden_dim, 4 * hidden_dim),
#             nn.ReLU(),
#             nn.Linear(4 * hidden_dim, input_dim)
#         )
        
#         # decoder phi: (Zp,A) to Phi (needs mu to be real and kappa to be positive)
#         self.decoder_p_mu = nn.Sequential(
#             nn.Linear(latent_dim_p + input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, 2 * hidden_dim),
#             nn.ReLU(), 
#             nn.Linear(2 * hidden_dim, 4 * hidden_dim),
#             nn.ReLU(), 
#             nn.Linear(4 * hidden_dim, input_dim)
#         )
#         self.decoder_p_kappa = nn.Sequential(
#             nn.Linear(latent_dim_p + input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, 2 * hidden_dim),
#             nn.ReLU(), 
#             nn.Linear(2 * hidden_dim, 4 * hidden_dim),
#             nn.ReLU(), 
#             nn.Linear(4 * hidden_dim, input_dim)
#         )

#         # initialization to help cover the phi range
#         self.decoder_p_mu[-1].weight.data.uniform_(-torch.pi, torch.pi)
#         self.decoder_p_mu[-1].bias.data.zero_()
        
#         # encoder: A to Za
#         self.encoder_a_mu = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, latent_dim_a)
#         )
#         self.encoder_a_logvar = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, latent_dim_a)
#         )

#         # encoders: P to Zp
#         self.encoder_p_mu = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, latent_dim_p)
#         )
#         self.encoder_p_logvar = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim), 
#             nn.ReLU(), 
#             nn.Linear(hidden_dim, latent_dim_p)
#         )

#         # Move model to device
#         self.to(self.device)

#     def model(self, a, p):
        
#         batch_size = a.shape[0]
        
#         # Move data to the correct device
#         a = a.to(self.device)
#         p = p.to(self.device)

#         # standard normal parameters
#         mu_prior = torch.zeros(batch_size, self.latent_dim_a, device=self.device)
#         sigma_prior = torch.ones(batch_size, self.latent_dim_a, device=self.device)
        
#         with pyro.plate("batch", batch_size, dim=-1):  # Define a batch plate
            
#             # Sample latent variables
#             Za = pyro.sample("Za",dist.Normal(mu_prior, sigma_prior).to_event(1))
#             Zp = pyro.sample("Zp",dist.Normal(mu_prior, sigma_prior).to_event(1))

#             # Decode amplitude and phase
#             mu_A = self.decoder_a_mu(Za)
#             sigma_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-6
#             # a_A = torch.exp(self.decoder_a_mu(Za)) + 1e-3
#             # b_A = torch.exp(self.decoder_a_logvar(Za)) + 1e-3
            
#             # Amplitude likelihood (Gaussian)
#             pyro.sample("A_obs", dist.LogNormal(mu_A, sigma_A).to_event(1), obs=a)
#             # pyro.sample("A_obs", dist.Normal(mu_A, sigma_A).to_event(1), obs=a)
#             # pyro.sample("A_obs", dist.Beta(a_A, b_A).to_event(1), obs=a)

#             # Decode phase
#             mu_P = self.decoder_p_mu(torch.cat((Zp, a), dim=1))
#             kappa_P = torch.exp(self.decoder_p_kappa(torch.cat((Zp, a), dim=1))) + 1e-6  # Ensure positivity

#             # Phase likelihood (Von Mises)
#             pyro.sample("P_obs", dist.VonMises(mu_P, kappa_P).to_event(1), obs=p)

#     def guide(self, a, p):
    
#         batch_size = a.shape[0]

#         # Move data to the correct device
#         a = a.to(self.device)
#         p = p.to(self.device)

#         with pyro.plate("batch", batch_size, dim=-1):  # Define the same batch plate
            
#             # Compute approximate posterior for Za (depends on Zs)
#             mu_Za = self.encoder_a_mu(a)
#             sigma_Za = torch.exp(self.encoder_a_logvar(a)) + 1e-6
#             Za = pyro.sample("Za", dist.Normal(mu_Za, sigma_Za).to_event(1))

#             # Compute approximate posterior for Zp (depends on Zs and p)
#             mu_Zp = self.encoder_p_mu(p)
#             sigma_Zp = torch.exp(self.encoder_p_logvar(p)) + 1e-6
#             Zp = pyro.sample("Zp", dist.Normal(mu_Zp, sigma_Zp).to_event(1))
