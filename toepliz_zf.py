import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

def zf_ffe_toeplitz(channel_response, equalizer_length):
    """
    Calculates the Zero-Forcing Feed-Forward Equalizer coefficients
    using a Toeplitz matrix.

    Args:
        channel_response (np.ndarray): The impulse response of the channel.
        equalizer_length (int): The number of taps for the equalizer.

    Returns:
        np.ndarray: The calculated equalizer coefficients.
    """
    # The first column of the Toeplitz matrix is the channel response
    # padded with zeros.
    col = np.zeros(equalizer_length)
    col[:len(channel_response)] = channel_response

    # The first row of the Toeplitz matrix is the first element of the
    # channel response followed by zeros.
    row = np.zeros(equalizer_length)
    row[0] = channel_response[0]

    # Construct the Toeplitz matrix representing the channel convolution.
    H = toeplitz(col, row)
    ''''''
    H=[
        [1, 0.3, 0],
        [-0.2, 1, 0.3 ],
        [0.1, -0.2, 1]
    ]
    '''
    '''
    print("Generated Toeplitz Matrix (H):\n", H)

    # The desired response is an impulse at the center of the equalizer.
    # This is what we want the equalized signal to look like.
    desired_response = np.zeros(equalizer_length)
    desired_response[equalizer_length // 2] = 1.0

    # Solve the linear system Hw = d to find the equalizer coefficients w.
    # We use the pseudo-inverse (pinv) for numerical stability, especially
    # if the matrix is ill-conditioned.
    w = np.linalg.pinv(H).dot(desired_response)

    return w

# --- Main part of the script ---
if __name__ == "__main__":
    # 1. Define a sample channel impulse response that causes distortion.
    # This simulates a simple multipath channel.
    #channel_h = np.array([0.8, 0.4, -0.2])
    #channel_h = np.array([0.3, 1, -0.2, 0.1])
    channel_h = np.array([0.3, 1, -0.2])
    #N_taps = 11  # Number of taps for our FFE filter
    N_taps = 3 
    # 2. Calculate the ZF-FFE coefficients.
    ffe_coeffs = zf_ffe_toeplitz(channel_h, N_taps)
    print("\nCalculated ZF-FFE Coefficients (w):\n", ffe_coeffs)

    # 3. Verify the result by convolving the channel with the equalizer.
    # The result should be close to a single impulse.
    equalized_response = np.convolve(channel_h, ffe_coeffs)
    print("\n Equalized response \n", equalized_response)
    # --- Plotting the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot the original channel response
    ax1.stem(channel_h) # Corrected line
    ax1.set_title("Original Channel Impulse Response (h)")
    ax1.set_xlabel("Tap")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Plot the equalized response
    ax2.stem(equalized_response) # Corrected line
    ax2.set_title("Combined Response of Channel and ZF-FFE (h * w)")
    ax2.set_xlabel("Tap")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()