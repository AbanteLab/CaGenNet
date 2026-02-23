Evaluation utilities
====================

Functions for evaluating latent representations are in `ca_sn_gen_models.evaluation`.

.. automodule:: ca_sn_gen_models.evaluation
    :members:
    :undoc-members:

Key evaluation functions
------------------------

- `evaluate_latent_svm`: trains/evaluates a linear SVM on latents.
- `get_hdbscan_ari`: HDBSCAN clustering + ARI computation.
- `compute_local_entropy`: local-label entropy in latent neighborhoods.
- `find_optimal_dbscan_epsilon`: sweep DBSCAN epsilon to maximize ARI.
- `kBET`: simple k-nearest-neighbor batch-effect test.
