#%%
import pywt
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%%

# Generate a synthetic signal: a sine wave with a transient spike
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(10 * 2 * np.pi * t) + np.where((t > 0.4) & (t < 0.6), 2, 0)

# Generate a second signal with a different phase for the transient spike
signal2 = np.sin(10 * 2 * np.pi * t) + np.where((t > 0.1) & (t < 0.3), 2, 0)

# Plot the signals
plt.figure(figsize=(6, 3))
plt.plot(t, signal, label='Signal 1')
plt.plot(t, signal2, label='Signal 2')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Signals")
plt.legend()
plt.show()

#%%

# Perform Fast Fourier Transform (FFT) on the first signal
fft_signal = np.fft.fft(signal)

# get amplitudes and phases
a1 = np.abs(fft_signal)
p1 = np.angle(fft_signal)

# Perform Fast Fourier Transform (FFT) on the second signal
fft_signal2 = np.fft.fft(signal2)

# get amplitudes and phases
a2 = np.abs(fft_signal2)
p2 = np.angle(fft_signal2)

# plot a1 as a fuction of frequency
plt.figure(figsize=(6, 3))
frequencies = np.fft.fftfreq(len(signal), 1/1000)
plt.plot(frequencies, a1)
plt.xlim(-25, 25)
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.title("Amplitude Spectrum of Signal 1")
plt.show()

# reconstruct signal2 from the FFT of signal2 but using the phase from signal1
reconstructed_signal2 = np.fft.ifft(a2 * np.exp(1j * p1))

# Plot the reconstructed signal2 and signal1
plt.figure(figsize=(6, 3))
plt.plot(t, signal, label='Signal 1')
plt.plot(t, reconstructed_signal2, label='Shifted Signal 2', linestyle='--')
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.legend()
plt.show()

####################################################################################################
# Real data
####################################################################################################
# %%

## The real data can be well reconstructed using the FFT as well. This shows that we should
## be able to train a model that predicts the coefficients of the FFT and reconstruct the signal.

fs = 20

# input data
xfile="/pool01/data/private/canals_lab/processed/calcium_imaging/hdrep/xf.csv.gz"

# read in the real data
x = pd.read_csv(xfile, header=None)

# subset randomlly 10 rows
xs = x.sample(10).to_numpy()

# %%

# do the FFT of each row
x_fft = np.fft.fft(xs)

# get the amplitudes and phases
a = np.abs(x_fft)
p = np.angle(x_fft)

# Compute the frequency bins
freq_bins = torch.fft.fftfreq(xs.shape[1], d=1/fs)
    
# Mask to keep only components with frequency at most fs/2
mask = (freq_bins >= 0) & (freq_bins <= fs/2)

# count how many positive frequencies we have
n_pos_freqs = mask.sum()

# plot spectrum of the first row
plt.figure(figsize=(6, 3))
plt.plot(frequencies, a[4])
plt.xlim(-30, 30)
plt.ylabel("Amplitude")
plt.xlabel("Frequency (Hz)")
plt.title("Amplitude Spectrum of the First Row")
plt.show()

# apply low pass filter to all rows
a[a < 10] = 0

# reconstruct the data
x_reconstructed = np.real(np.fft.ifft(a * np.exp(1j * p)))

# plot the original and reconstructed data for each of the 10 cases separately using a facet wrap
fig, axes = plt.subplots(5, 2, figsize=(12, 15))
axes = axes.flatten()

# Determine the common y-limits
y_min = min(xs.min().min(), x_reconstructed.min().min())
y_max = max(xs.max().max(), x_reconstructed.max().max())

for i in range(10):
    axes[i].plot(xs[i], label='Original')
    axes[i].plot(x_reconstructed[i], label='Reconstructed', linestyle='--')
    axes[i].set_ylabel("Amplitude")
    axes[i].set_xlabel("Time")
    axes[i].legend()
    axes[i].set_title(f"Case {i+1}")
    axes[i].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()

#%%
#compute MSE of each reconstructed signal
mse = np.mean((xs - x_reconstructed)**2, axis=1)
mse
# %%

# get an index and shift it
idx = 0

# Define a function to shift the signal in the frequency domain
def shift_signal(signal, shift):
    # Perform FFT
    fft_signal = np.fft.fft(signal)
    # Create an array of frequencies
    freqs = np.fft.fftfreq(len(signal))
    # Apply the shift in the frequency domain
    shifted_fft_signal = fft_signal * np.exp(-2j * np.pi * freqs * shift)
    # Perform inverse FFT to get the shifted signal
    shifted_signal = np.fft.ifft(shifted_fft_signal)
    return np.real(shifted_signal)

# Define the shift amount (positive for right shift, negative for left shift)
shift_amount = 1000  # Example: shift by 100 samples

# Apply the shift to all signals in xs
x_shifted = np.array([shift_signal(sig, shift_amount) for sig in xs])

# plot the original and shifted data for each of the 10 cases separately using a facet wrap
fig, axes = plt.subplots(5, 2, figsize=(12, 15))
axes = axes.flatten()

# Determine the common y-limits
y_min = min(xs.min().min(), x_shifted.min().min())
y_max = max(xs.max().max(), x_shifted.max().max())

for i in range(10):
    axes[i].plot(xs[i], label='Original')
    axes[i].plot(x_shifted[i], label='Shifted', linestyle='--')
    axes[i].set_ylabel("Amplitude")
    axes[i].set_xlabel("Time")
    axes[i].legend()
    axes[i].set_title(f"Case {i+1}")
    axes[i].set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()
# %%
