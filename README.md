# Generative Models for Ca imaging data

## Overview
**calgenn*** is a Python package that implements various generative models for Ca imaging data, including various generative models that **primarily seeks to produce a useful latent representation of individual neurons for downstream tasks**. The package includes functionalities for model training, data processing, and evaluation.

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

## License
This project is licensed under the Apache License. See the LICENSE file for details.
