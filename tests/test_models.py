import pytest
import torch
from ca_sn_gen_models.models import MlpVAE, LstmVAE, supMlpVAE, supLstmVAE

@pytest.fixture
def unsupervised_synthetic_data():
    input_dim = 128
    batch_size = 32
    data = torch.randn(batch_size, input_dim)
    return data

@pytest.fixture
def supervised_synthetic_data():
    input_dim = 128
    batch_size = 32
    num_classes = 3
    data = torch.randn(batch_size, input_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    return data, one_hot_labels

def test_mlp_vae(unsupervised_synthetic_data):
    
    # define the model
    model = MlpVAE(unsupervised_synthetic_data.shape[1], 16)
    
    # test model
    model.model(unsupervised_synthetic_data)
    assert True
    
    # test guide
    model.guide(unsupervised_synthetic_data)
    assert True
    
def test_lstm_vae(unsupervised_synthetic_data):
    
    # define the model
    model = LstmVAE(unsupervised_synthetic_data.shape[1], 16)
    
    # test model
    model.model(unsupervised_synthetic_data)
    assert True
    
    # test guide
    model.guide(unsupervised_synthetic_data)
    assert True

def test_sup_mlp_vae(supervised_synthetic_data):
    
    # Unpack the data and labels
    x, y = supervised_synthetic_data

    # define the model
    model = supMlpVAE(x.shape[1], 16, y.shape[1])
    
    # test model
    model.model(x,y)
    assert True
    
    # test guide
    model.guide(x,y)
    assert True

def test_sup_lstm_vae(supervised_synthetic_data):
    
    # Unpack the data and labels
    x, y = supervised_synthetic_data

    # define the model
    model = supLstmVAE(x.shape[1], 16, y.shape[1])
    
    # test model
    model.model(x,y)
    assert True
    
    # test guide
    model.guide(x,y)
    assert True