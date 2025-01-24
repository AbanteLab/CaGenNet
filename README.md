# snDGM Project

## Overview
snDGM is a Python package that implements a Variational Autoencoder (VAE) using Long Short-Term Memory (LSTM) networks for processing calcium imaging data. The package includes functionalities for model training, data processing, and evaluation.

## Features
- LSTM-based encoder and decoder architecture.
- Reparameterization trick for sampling latent variables.
- Custom PyTorch Dataset for handling ROI and trace data.
- Early stopping based on validation loss during training.
- Utilities for data handling and preprocessing.

## Installation
To install the package, clone the repository and run the following command in the project directory:

```bash
pip install ./snDGM
```

Alternatively, you can install the required dependencies directly using:

```bash
pip install -r requirements.txt
```

## Usage
To train the VAE model, you can run the `avae.py` script with the necessary arguments. For example:

```bash
python avae.py x --latent 32 --hidden 256 --epochs 50 --batch 16 --seed 0 --rate 0.001 --beta_kl 1 --retrain False --save True
```

where x needs to be a tsv file.

## Testing
Unit tests for the package can be found in the `tests` directory. To run the tests, use the following command:

```bash
pytest snDGM/tests/test_avae.py
```
## Reproducing Results from ECCB2025 Paper

This repository contains the code to reproduce the results from the paper submitted to ECCB2025. The scripts and instructions provided will allow you to replicate the experiments and analyses described in the paper. Please refer to the `eccb2025` directory for detailed steps and configurations used in the study.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the Apache License. See the LICENSE file for details.