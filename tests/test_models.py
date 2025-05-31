import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from ca_sn_gen_models.models import FixedVarMlpVAE, LearnedVarMlpVAE, FixedVarSupMlpVAE, LearnedVarSupMlpVAE

@pytest.fixture
def unsup_loader():
    
    # Create synthetic unsupervised data for testing
    input_dim = 128
    batch_size = 32

    # Generate random data
    data = torch.randn(batch_size, input_dim)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Return the loader
    return loader

@pytest.fixture
def sup_loader():
    
    # Create synthetic supervised data and one-hot labels for testing
    input_dim = 128
    batch_size = 32
    num_classes = 3

    # Generate random data and labels
    data = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

    # Create a dataset and loader
    dataset = TensorDataset(data, one_hot_labels)
    loader = DataLoader(dataset, batch_size=batch_size)

    # Return the loader
    return loader

def test_fixed_var_mlp_vae_forward(unsup_loader):
    
    # Define input and latent dimensions
    input_dim = 128
    latent_dim = 16
    
    # Initialize the FixedVarMlpVAE model
    model = FixedVarMlpVAE(input_dim=input_dim, latent_dim=latent_dim)
    
    # Forward pass through the model
    z,xhat = model.forward(unsup_loader)

    # Check the output types and shapes
    assert isinstance(z, torch.Tensor)
    assert isinstance(xhat, torch.Tensor)
    assert z.shape == (unsup_loader.batch_size, latent_dim)
    assert xhat.shape == unsup_loader.dataset.tensors[0].shape

def test_learned_var_mlp_vae_forward(unsup_loader):

    # Define input and latent dimensions
    input_dim = 128
    latent_dim = 16

    # Initialize the LearnedVarMlpVAE model
    model = LearnedVarMlpVAE(input_dim=input_dim, latent_dim=latent_dim)

    # Forward pass through the model
    z,xhat = model.forward(unsup_loader)

    # Check the output types and shapes
    assert isinstance(z, torch.Tensor)
    assert isinstance(xhat, torch.Tensor)
    assert z.shape == (unsup_loader.batch_size, latent_dim)
    assert xhat.shape == unsup_loader.dataset.tensors[0].shape
    
def test_fixed_var_sup_mlp_vae_forward(sup_loader):
    
    # Deine input and latent dimensions
    input_dim = 128
    latent_dim = 16
    num_classes = 3
    
    # Initialize the FixedVarSupMlpVAE model
    model = FixedVarSupMlpVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)

    # Forward pass through the model
    z,xhat = model.forward(unsup_loader)

    # Check the output types and shapes
    assert isinstance(z, torch.Tensor)
    assert isinstance(xhat, torch.Tensor)
    assert z.shape == (unsup_loader.batch_size, latent_dim)
    assert xhat.shape == unsup_loader.dataset.tensors[0].shape

def test_learned_var_sup_mlp_vae_forward(sup_loader):
    
    # Deine input and latent dimensions
    input_dim = 128
    latent_dim = 16
    num_classes = 3

    # Initialize the LearnedVarSupMlpVAE model
    model = LearnedVarSupMlpVAE(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)
    
    # Forward pass through the model
    z,xhat = model.forward(unsup_loader)

    # Check the output types and shapes
    assert isinstance(z, torch.Tensor)
    assert isinstance(xhat, torch.Tensor)
    assert z.shape == (unsup_loader.batch_size, latent_dim)
    assert xhat.shape == unsup_loader.dataset.tensors[0].shape
