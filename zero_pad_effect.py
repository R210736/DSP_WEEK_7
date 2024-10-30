import numpy as np
import matplotlib.pyplot as plt

# Define the sample signal x[n]
x = np.array([2, 4, 6, 8])

# Function to perform zero padding
def zero_pad(signal, num_zeros):
    return np.concatenate((signal, np.zeros(num_zeros)))

# Zero padded signals
x_padded_2 = zero_pad(x, 2)
x_padded_4 = zero_pad(x, 4)

# Function to compute DFT (Discrete Fourier Transform)
def DFT(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Function to compute magnitude spectrum
def magnitude_spectrum(signal):
    X = DFT(signal)
    magnitude = np.abs(X)
    return magnitude

# Compute magnitude spectra
magnitude_original = magnitude_spectrum(x)
magnitude_padded_2 = magnitude_spectrum(x_padded_2)
magnitude_padded_4 = magnitude_spectrum(x_padded_4)

# Plot all three magnitude spectra in a single figure with subplots
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.stem(range(len(magnitude_original)), magnitude_original )
plt.title('Magnitude Spectrum (Original Signal)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

plt.subplot(3, 1, 2)
plt.stem(range(len(magnitude_padded_2)), magnitude_padded_2 )
plt.title('Magnitude Spectrum (Signal with 2 Zeros Padded)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

plt.subplot(3, 1, 3)
plt.stem(range(len(magnitude_padded_4)), magnitude_padded_4)
plt.title('Magnitude Spectrum (Signal with 4 Zeros Padded)')
plt.xlabel('Frequency Bin')
plt.ylabel('Magnitude')

plt.tight_layout()  # Adjust subplots to fit into the figure area.
plt.show()
