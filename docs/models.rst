Generative models
==================

This page documents the main generative-model classes implemented in `ca_sn_gen_models.models`.

.. automodule:: ca_sn_gen_models.models
    :members:
    :undoc-members:
    :show-inheritance:

Brief highlights
----------------

- `ConvEncoder` / `ConvDecoder`: convolutional encoder/decoder for time-series (circular padding).
- `Encoder` / `Decoder`: MLP encoder/decoder building blocks.
- `FA`, `isoFA`: factor-analysis models with SVI training and posterior helpers.
- `FixedVarMlpVAE`, `LearnedVarMlpVAE`: unsupervised VAE families.
- `FixedVarSupMlpVAE`, `FixedVarSupMlpBVAE`: supervised / semi-supervised VAE variants.
