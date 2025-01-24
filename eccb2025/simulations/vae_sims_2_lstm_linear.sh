#!/bin/bash

set -x

# Define the list of beta_kl values
beta_kl_values=(0.0 0.5 1.0)

# Loop over each beta_kl value
for b in "${beta_kl_values[@]}"
do
    # Loop over seed values from 0 to 9
    for seed in {0..9}
    do
        # Parallelize over fluo_noise values from 0.5 to 2.5 in increments of 0.5
        for fluo_noise in $(seq 0.5 0.5 2.5)
        do
            python vae_sims_2_lstm_linear.py --hidden 32 --latent 2 --beta_kl "$b" --seed "$seed" --fluo_noise "$fluo_noise"
            python vae_sims_2_lstm_linear.py --hidden 64 --latent 4 --beta_kl "$b" --seed "$seed" --fluo_noise "$fluo_noise"
            python vae_sims_2_lstm_linear.py --hidden 128 --latent 8 --beta_kl "$b" --seed "$seed" --fluo_noise "$fluo_noise"
            python vae_sims_2_lstm_linear.py --hidden 256 --latent 16 --beta_kl "$b" --seed "$seed" --fluo_noise "$fluo_noise"
        done
    done
done

# Run the reconstruction benchmark
python reconstruction_benchmark.py
