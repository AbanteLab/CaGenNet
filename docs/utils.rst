Utilities
=========

Key helpers and utilities are implemented in `ca_sn_gen_models.utils`.

.. automodule:: ca_sn_gen_models.utils
    :members:
    :undoc-members:

Common utilities
----------------

- `superprint(msg)`: timestamped logging helper.
- `CalciumDataset`: minimal `Dataset` wrapper for traces.
- `load_data`, `augment_data`, `normalize_data`: I/O and preprocessing helpers.
- `train_svi`, `kl_beta_schedule`: standard SVI training loop and KL schedule.
- `gen_random_mask`, `gen_tail_mask`: missing-data mask generators.
