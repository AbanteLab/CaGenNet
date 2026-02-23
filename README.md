# CaGenNet: Calcium Generative Networks for Ca imaging data

## Overview
**CaGenNet** is a Python package that implements various generative models for Ca imaging data, including various generative models that **primarily seeks to produce a useful latent representation of individual neurons for downstream tasks**. The package includes functionalities for model training, data processing, and evaluation.

## Installation
To install the package, clone the repository and run the following command in the project directory:

```bash
pip install ca_sn_gen_models
```

Alternatively, you can install the required dependencies directly using:

```bash
pip install -r requirements.txt
```

## Usage

To get started with training a generative model on Ca imaging data, refer to the example script:

```bash
python ./examples/train_dgm.py
```

This script demonstrates how to train a Deep Generative Model (DGM) using the package. For more details and customization options, please review the script and the package documentation.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Generative models

- **FA (Factor Analysis)**: Exponential-family factor analysis with configurable likelihoods and SVI training (`train_model`, `get_posterior`).
- **isoFA**: FA variant that models paired latents `Zx`/`Zy` for joint modeling of signals and labels (includes `CustomELBO` regularizer).
- **FixedVarMlpVAE / LearnedVarMlpVAE**: Unsupervised MLP VAEs for time-series; differ in whether the output likelihood scale is fixed or learned. Provide `encode`, `decode`, `model`, `guide`, `train_model`, and `forward` helpers.
- **FixedVarSupMlpVAE / FixedVarSupMlpBVAE**: Supervised/semi-supervised VAE variants that incorporate labels into encoding/decoding and support supervised training flows.

## License

This project is licensed under the Apache License. See the LICENSE file for details.

## Papers

- B Ros; M Olives-Verger; C Fuses; JM Canals; J Soriano; J Abante (2026). Integration of Calcium Imaging Traces via Deep Generative Modeling. *2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE (in print).

## Documentation

Full documentation (Sphinx / ReadTheDocs) is available in the `docs/` directory. To build locally:

```bash
pip install -r requirements.txt
pip install sphinx sphinx_rtd_theme myst-parser
cd docs && make html
```
