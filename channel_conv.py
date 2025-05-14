import numpy as np
import matplotlib.pyplot as plt
import serdespy as sdp

channel_taps = np.array([0.1, 0.2, 1])
tx_symbols= np.array([1, -3, -3, 3, 3, -3, 1, 1, -3])
tx_padded = np.concatenate((tx_symbols, np.zeros(len(channel_taps) - 1)))
channel_output = np.convolve(tx_padded, channel_taps, mode='valid') # Keep only valid part
print(channel_output)
