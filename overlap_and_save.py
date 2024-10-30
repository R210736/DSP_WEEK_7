import numpy as np
import matplotlib.pyplot as plt

# Function to compute the DFT manually
def DFT(signal):
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Function to compute the inverse DFT manually
def IDFT(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

# Overlap and Save method implementation without using built-in FFT functions
def overlap_save(x, h, L):
    M = len(h)
    N = L + M - 1  # Length of DFT
    
    # Pad the input signal x with M-1 zeros at the beginning
    x = np.concatenate((np.zeros(M-1), x))
    
    # Initialize the output signal
    y = np.zeros(len(x) - M + 1)
    
    # Precompute the DFT of the impulse response h
    h_padded = np.concatenate((h, np.zeros(N - M)))  # Zero-pad h to length N
    H = DFT(h_padded)
    
    for i in range(0, len(x) - M + 1, L):
        # Extract the current segment of the signal
        x_segment = x[i:i+N]
        
        # Compute the DFT of the segment
        X_segment = DFT(x_segment)
        
        # Perform pointwise multiplication in the frequency domain
        Y_segment = X_segment * H
        
        # Compute the inverse DFT to get the time-domain result
        y_segment = IDFT(Y_segment)
        
        # Save the valid part of the convolution (last L samples)
        y[i:i+L] = y_segment[M-1:M-1+L].real
    
    return y

# Example usage
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Input signal
h = np.array([1, -1, 0.5])  # FIR filter
L = 4  # Segment length

# Apply the Overlap and Save method
y = overlap_save(x, h, L)

# Plot the result
plt.stem(y, use_line_collection=True)
plt.title('Output of Overlap and Save Method')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.show()
