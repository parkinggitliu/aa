import serdespy as sdp
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import scipy as sp
from scipy.linalg import toeplitz

def get_every_kth_element(data_sequence, k):
  """
  Selects every k-th element from a sequence starting from index 0.

  Args:
    data_sequence: The list, tuple, string, or NumPy array to select from.
    k: The step size (select element 0, k, 2k, ...). Must be > 0.

  Returns:
    A new sequence of the same type containing the selected elements.
    Returns an empty sequence if k is invalid or the input sequence is empty.
  """
  if k <= 0:
    print("Error: Step k must be a positive integer.")
    # Return an empty sequence of the same type if possible
    try:
      return type(data_sequence)()
    except TypeError:
      return None # Or raise ValueError("Step k must be positive")

  # Python's slicing [start:stop:step]
  # ::k means start from the beginning, go to the end, with a step of k
  return data_sequence[::k]

def zf_ffe_toeplitz(channel_response, equalizer_length, pre_length=1):
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
   
    print("Generated Toeplitz Matrix (H):\n", H)

    # The desired response is an impulse at the center of the equalizer.
    # This is what we want the equalized signal to look like.
    desired_response = np.zeros(equalizer_length)
    #desired_response[equalizer_length // 2] = 1.0
    desired_response[pre_length] = 1.0
    print("Desired Response:\n", desired_response)
    # Solve the linear system Hw = d to find the equalizer coefficients w.
    # We use the pseudo-inverse (pinv) for numerical stability, especially
    # if the matrix is ill-conditioned.
    w = np.linalg.pinv(H).dot(desired_response)

    return w

f = np.load("./data/f.npy")
w = 2*f*np.pi

h_pulse = np.load("./data/hpulse.npy")
#t = np.load("./data/t.npy")

signal = np.load("./data/signal_test.npy")
#common section
samples_per_symbol = 32
data_rate = 106.125e9
nyquist_f = data_rate/4
t_symbol = 2/data_rate
t_sample = t_symbol/samples_per_symbol
fmax = 0.5/t_sample
UI = t_symbol

#set poles and zeroes for peaking at nyquist freq
#high peaking because channel is high insertion loss
z = 5e10
p = 1.7e11
k = 0.75*p**2/z

#calculate Frequency response of CTLE at given frequencies
w, H_ctle = sp.signal.freqs([k/p**2, k*z/p**2], [1/p**2, 2/p, 1], w)

#bode plot of CTLE transfer function
plt.figure(dpi=100)
plt.semilogx(f,20*np.log10(abs(H_ctle)), color = "red", label = 'CTLE')
plt.title("CTLE Frequency Response")
plt.grid()
plt.axvline(x=25,color = 'grey', label = "Nyquist Frequency")
plt.axvline(x=z/(2*np.pi)*1e-9,color = 'green', label = "Zero Location")
plt.axvline(x=p/(2*np.pi)*1e-9,color = 'blue', label = "Pole Location")
plt.legend()

#%%
h_ctle, t_ctle = sdp.freq2impulse(H_ctle,f)
h_ctle = h_ctle[0:200]
plt.figure(dpi=100)
plt.title("CTLE impulse respnose")
plt.grid()
plt.plot(h_ctle)
t = t_ctle

ntrace=1000

sdp.simple_eye(signal[1000*samples_per_symbol:], samples_per_symbol*3, ntrace, t_sample, "{}Gbps 4-PAM Signal".format(data_rate/1e9), 100)

signal_ctle = sp.signal.convolve(signal,h_ctle)
sdp.simple_eye(signal_ctle[1000*samples_per_symbol:], samples_per_symbol*3, ntrace, t_sample, "{}Gbps 4-PAM Signal with CTLE".format(data_rate/1e9),100)
h_pulse_ctle = sp.signal.convolve(h_pulse,h_ctle)

FFE_pre = 1
FFE_taps = 7
FFE_post = FFE_taps - FFE_pre - 1
DFE_taps = 1



sdp.channel_coefficients(h_pulse[:t.size],t,samples_per_symbol,FFE_pre,FFE_post, 100, "channel coefficients")

h = sdp.channel_coefficients(h_pulse_ctle[:t.size],t,samples_per_symbol,FFE_pre,FFE_post, 100, "channel + CTLE coefficients")
#%%
#h /= h.max()
#print('h: ',h)
channel_main = h.argmax()

#main_cursor = h[channel_main]
main_cursor = 1

ffe_tap_weights2 = zf_ffe_toeplitz(h, len(h))
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Zero Forcing Topelitz FFE tap weights: {ffe_tap_weights2}")
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")

pam4_levels = np.array([-3, -1, 1, 3])
mapping = {0: -3, 1: -1, 2: 1, 3: 3} # Example mapping (can vary)
inv_mapping = {v: k for k, v in mapping.items()} # For checking errors later
# FFE Parameters
num_ffe_pre_taps = FFE_pre      # Number of precursor taps (should match post-cursor ISI length)
#num_ffe_post_taps = 1      # Number of post taps (should match post-cursor ISI length)
num_ffe_post_taps = FFE_post 
num_ffe_taps= num_ffe_pre_taps + num_ffe_post_taps +1

mu_ffe = 4e-5        # Step size (learning rate) for FFE tap adaptation (LMS)
mu_delev = 8e-4     # Step size (learning rate) for delev adaptation (LMS)
delev = 1           # Initial decision level (DLEV) for FFE
ffe_taps = np.concatenate((np.zeros(num_ffe_pre_taps), [1], np.zeros(num_ffe_post_taps)))
#next_rx_data = np.zeros(num_ffe_pre_taps) # Buffer for past decided levels (DLEV)
N_symbols = 9000  # Number of symbols to process
ffe_output_symbols = np.zeros(N_symbols)
slicer_inputs = np.zeros(N_symbols) # Store slicer inputs for analysis/debug
ffe_errors = np.zeros(N_symbols) # Store errors for analysis/debug
ffe_tap1_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap2_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap3_history = np.zeros(N_symbols) # Store tap history for plotting
delev_history = np.zeros(N_symbols)  

print(f"Starting FFE equalization with {num_ffe_taps} taps.")
print(f"Initial FFE taps: {ffe_taps}")
cdr_pos=16
for cdr_pos in range(0,31,4):
  
    received_signal = get_every_kth_element(signal_ctle[cdr_pos:], samples_per_symbol)
        
    for k in range(num_ffe_post_taps, N_symbols-num_ffe_taps):
      
        ffe_input_vector = received_signal[k-num_ffe_post_taps:k+num_ffe_pre_taps+1]  # Get the input vector for FFE taps
        ffe_input_vector_r = ffe_input_vector[::-1]  # Reverse for convolution-like operation
        slicer_input = np.dot(ffe_taps, ffe_input_vector_r)  # Estimate ISI using FFE taps
        slicer_inputs[k] = slicer_input # Store for plotting/analysis

       
        # PAM4 Slicer (Decision Device)
        # Find the closest PAM4 level (DLEV)
        distances = np.abs(slicer_input - pam4_levels)
        closest_level_index = np.argmin(distances)
        current_decision = pam4_levels[closest_level_index] # This is the decided DLEV

        # Store the decision
        ffe_output_symbols[k] = current_decision
        
        # --- 5. LMS Adaptation (Optional) ---
        # Calculate error: difference between slicer input and the decided level
            
        if current_decision == 3 or current_decision == -3: 
            out_of_zone=abs(slicer_input) -3
        elif current_decision == 1 or current_decision == -1:  
            out_of_zone=abs(slicer_input) -1
        else:
            print("Error: Current decision is not a valid PAM4 level.")
        error_dlev = np.sign(out_of_zone)
        
        error = slicer_input - current_decision   # Common error definition
        ffe_errors[k] = error # Store for analysis/debug
        # Update FFE taps using LMS algorithm
        ffe_adj = mu_ffe * error * ffe_input_vector_r # Adjust FFE taps based on error and input vector
 
        ffe_taps = ffe_taps - ffe_adj # Update FFE taps
        ffe_tap1_history[k] = ffe_taps[0] # Store for plotting/analysis
        ffe_tap2_history[k] = ffe_taps[2] # Store for plotting/analysis 
        delev_history[k]= ffe_taps[1]
     
       
    print(f"Final FFE taps after {N_symbols} symbols: {ffe_taps}")
    RX = sdp.Receiver(signal_ctle, samples_per_symbol, nyquist_f, pam4_levels,main_cursor=main_cursor)
    #sdp.simple_eye(RX.signal, samples_per_symbol*3, 800, TX.UI/TX.samples_per_symbol, "Eye Diagram with CTLE")
    RX.FFE(ffe_taps, FFE_pre)
    sdp.simple_eye(RX.signal[int(3000.5*samples_per_symbol):], samples_per_symbol*3, ntrace, t_sample,"Eye Diagram with CTLE and FFE with cdr position{}".format(cdr_pos),100)


#generate binary data for SDP
data = sdp.prqs10(1)[:10000]
signal_BR = sdp.pam4_input_BR(data)
signal_rx = sp.signal.fftconvolve(h, signal_BR)[:len(signal_BR)]
signal_rx_cropped = signal_rx[channel_main:]
reference_signal = signal_BR[:1000]

w_ffe_init = np.zeros([FFE_taps,])
w_dfe_init = np.zeros([DFE_taps,])

w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = sdp.lms_equalizer(signal_rx_cropped, 0.001, len(signal_rx_cropped), w_ffe_init, FFE_pre, w_dfe_init,  pam4_levels, reference=reference_signal[:1000])
    #w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = sdp.lms_equalizer(signal_rx_cropped, 0.001, len(signal_rx_cropped), w_ffe_init, FFE_pre, w_dfe_init,  voltage_levels)
print(f"SDP adapted FFE weights: {w_ffe}")
plt.show()