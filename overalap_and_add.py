import numpy as np
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
    return x / N

def overlap_add(x, h):
    M = len(h)
    L = len(x)  # We will take segments of length L
    N = L + M - 1
    
    # Zero-pad the filter to length N
    h_padded = np.zeros(N)
    h_padded[:M] = h
    
    # Compute the DFT of the filter
    H = dft(h_padded)
    
    # Initialize the output array
    y = np.zeros(len(x) + M - 1)
    
    # Process each segment
    for i in range(0, len(x), L):
        # Extract the current segment and zero-pad
        x_segment = np.zeros(N)
        x_segment[:L] = x[i:i+L]
        
        # Compute the DFT of the segment
        X = dft(x_segment)
        
        # Multiply in the frequency domain
        Y = X * H
        
        # Compute the inverse DFT
        y_segment = np.real(idft(Y))
        
        # Add the result to the correct part of the output array
        y[i:i+N] += y_segment
    
    return y

# Example usage
x = np.array([3,9,1,2,3,4,5,6,3,4,5,6,7,8,9,8,7,5])
h = np.array([1,2,1,1])
y = overlap_add(x, h)
print("Output:", y)