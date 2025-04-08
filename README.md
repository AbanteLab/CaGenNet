# Generative Models for Ca imaging data

## Overview
snDGM is a Python package that implements a Variational Autoencoder (VAE) using Long Short-Term Memory (LSTM) networks for processing calcium imaging data. The package includes functionalities for model training, data processing, and evaluation.

## Cellular models

We include various generative models for individual functional traces that **primarily seeks to produce a useful latent representation for downstream tasks**. In addition, we are also exploring the possibilities of these models for other tasks such as denoising and compression.

### Unsupervised

#### Bayesian Factor Analysis (BFA)

- Factor Analysis model with prior distributions on the loading matrix and latent variables
- Learns a latent representation of the data through a linear transformation
- The model has flexible likelihood functions, which can be passed as an argument

#### Autoregressive VAE (AVAE)

- Deep generative model that uses LSTM layers both in the encoder and decoder
- Currently implemented with torch - doesn't explicitely assume a distribution

### Semi-supervised

#### Semi-supervised Bayesian Factor Analysis (SBFA)

- Extension of the BFA model with two latent variables:
    - $Z_y$: captures variability associated to a given covariate of interest
    - $Z_y$: captures rest of variability necessary to reconstruct the traces

## Installation
To install the package, clone the repository and run the following command in the project directory:

```bash
pip install ca_sn_gen_models
```

Alternatively, you can install the required dependencies directly using:

```bash
pip install -r requirements.txt
```

## Testing
Unit tests for the package can be found in the `tests` directory. To run the tests, use the following command:

```bash
pytest snDGM/tests/test_utils.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the Apache License. See the LICENSE file for details.