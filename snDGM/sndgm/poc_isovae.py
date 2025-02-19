#%%
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

# input data
xfile="/pool01/data/private/canals_lab/processed/calcium_imaging/hdrep/xf.csv.gz"

# read in the real data
x = pd.read_csv(xfile, header=None)

# get tensor
x = torch.tensor(x.values, dtype=torch.float32)

# do the FFT of each row
x_fft = torch.fft.rfft(x)

# get the amplitudes and phases
a = np.abs(x_fft)
p = np.angle(x_fft)

# range of amplitudes and phases
print(f"Amplitude range: {a.min()} - {a.max()}")
print(f"Phase range: {p.min()} - {p.max()}")

# plot histogram of amplitude and phase
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.hist(a.flatten(), bins=100)
plt.xlabel("Amplitude")
plt.ylabel("Frequency")
plt.title("Histogram of Amplitudes")
plt.subplot(1, 2, 2)
plt.hist(p.flatten(), bins=100)
plt.xlabel("Phase")
plt.ylabel("Frequency")
plt.title("Histogram of Phases")
plt.show()

# %%

# reconstruct the data
x_reconstructed = torch.fft.irfft(a * np.exp(1j * p), n=x.shape[1])

# plot the original and reconstructed data for each of the 10 cases separately using a facet wrap
fig, axes = plt.subplots(5, 2, figsize=(12, 15))
axes = axes.flatten()

# Determine the common y-limits
y_min = min(x.min().min(), x_reconstructed.min().min())
y_max = max(x.max().max(), x_reconstructed.max().max())

for i in range(10):
    axes[i].plot(x[i], label='Original')
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
