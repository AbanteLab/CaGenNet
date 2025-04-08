import torch
import pytest
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,TensorDataset

from ca_sn_gen_models.utils import (
    superprint, CalciumDataset, augment_data, normalize_data, VAE, train_clavae, reconstruction_loss, kl_divergence_loss, positive_pairwise_loss, loss_function, train_clavae
)

def test_superprint(capsys):
    message = "Test message"
    superprint(message)
    captured = capsys.readouterr()
    assert message in captured.out

def test_calcium_dataset():
    data = np.random.rand(100, 10)
    dataset = CalciumDataset(data)
    assert len(dataset) == 100
    assert torch.equal(dataset[0][0], torch.from_numpy(data[0]).float())

def test_augment_data():
    data = np.random.rand(10, 100)
    N, L = 5, 20
    augmented_data, indices = augment_data(data, N, L)
    assert augmented_data.shape == (10, N, L)
    assert indices.shape == (10, N)

def test_normalize_data():
    data = pd.DataFrame(np.random.rand(100, 10))
    normalized_data = normalize_data(data)
    print(normalized_data.values.mean())
    assert normalized_data.shape == data.shape
    assert np.allclose(normalized_data.values.mean(), 0, atol=1e-1)

def test_vae():
    input_dim, hidden_dim, latent_dim = 10, 5, 2
    model = VAE(input_dim, hidden_dim, latent_dim)
    x = torch.randn(3, input_dim)
    reconstructed_x, mu, logvar = model(x)
    assert reconstructed_x.shape == x.shape
    assert mu.shape == (3, latent_dim)
    assert logvar.shape == (3, latent_dim)

def test_train_vae():
    input_dim, hidden_dim, latent_dim = 10, 5, 2
    model = VAE(input_dim, hidden_dim, latent_dim)
    data = torch.randn(100, input_dim)
    idx = torch.arange(data.shape[0])
    dataset = TensorDataset(data, idx)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters())
    device = torch.device('cpu')
    avg_loss = train_clavae(model, loader, optimizer, device)
    assert avg_loss > 0

def test_reconstruction_loss():
    x = torch.randn(3, 10)
    recon_x = torch.randn(3, 10)
    loss = reconstruction_loss(recon_x, x, rec_loss='mse')
    assert loss > 0

def test_kl_divergence_loss():
    mu = torch.randn(3, 2)
    logvar = torch.randn(3, 2)
    loss = kl_divergence_loss(mu, logvar)
    assert loss > 0

def test_positive_pairwise_loss():
    mu = torch.randn(10, 2)
    roi = torch.randint(0, 3, (10,))
    loss = positive_pairwise_loss(mu, roi, roi_metric='cosine')
    assert loss >= 0

def test_loss_function():
    x = torch.randn(3, 10)
    recon_x = torch.randn(3, 10)
    mu = torch.randn(3, 2)
    logvar = torch.randn(3, 2)
    roi = torch.randint(0, 3, (3,))
    total_loss, recon_loss, kl_loss, pp_loss = loss_function(recon_x, x, mu, logvar, roi)
    assert total_loss > 0
    assert recon_loss > 0
    assert kl_loss > 0
