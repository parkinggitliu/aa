import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import serdespy as sdp

def zf_ffe_toeplitz(channel_response, agc, equalizer_length, pre_length=1):
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
    #col = np.zeros(equalizer_length)
    col = np.zeros(5)
    col[:len(channel_response)] = channel_response

    # The first row of the Toeplitz matrix is the first element of the
    # channel response followed by zeros.
    row = np.zeros(equalizer_length)
    row[0] = channel_response[0]

    # Construct the Toeplitz matrix representing the channel convolution.
    #H = toeplitz(col, row)
    
    H0=np.array([
        [1,    0.3,  0,   0,   0],
        [-0.3, 1,   0.3,  0,   0],
        [0.2, -0.3,  1,  0.3,  0],
        [0.1, 0.2, -0.3,  1,  0.3],
    ])
    
    H = H0*agc
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

def mean_squared_error(y):
    squared_errors = np.square(y)
    mse = np.mean(squared_errors)
    return mse


def generate_prbs7(length):
    """Generates a PRBS7 sequence of the specified length."""
    if length <= 0:
        return np.array([])

    lfsr = np.array([1, 0, 0, 0, 0, 0, 1], dtype=int)  # Initial state (can be non-zero)
    sequence = np.zeros(length, dtype=int)

    for i in range(length):
        feedback = lfsr[6] ^ lfsr[5]  # XOR taps for PRBS7 (x^7 + x^6 + 1)
        sequence[i] = lfsr[0]
        lfsr = np.roll(lfsr, 1)
        lfsr[0] = feedback

    return sequence

def map_prbs7_to_pam4(prbs_sequence, mapping='gray'):
    """Maps a binary PRBS7 sequence to PAM4 symbols.

    Args:
        prbs_sequence (np.ndarray): The binary PRBS7 sequence.
        mapping (str, optional): The binary-to-PAM4 symbol mapping.
                                  'gray' (default), 'natural', or a custom dict.

    Returns:
        np.ndarray: The PAM4 symbol sequence.
    """
    if len(prbs_sequence) % 2 != 0:
        raise ValueError("PRBS sequence length must be even for PAM4 mapping.")

    num_symbols = len(prbs_sequence) // 2
    pam4_symbols = np.zeros(num_symbols)

    if mapping == 'gray':
        mapping_dict = {'00': -3, '01': -1, '11': 1, '10': 3}
    elif mapping == 'natural':
        mapping_dict = {'00': -3, '01': -1, '10': 1, '11': 3}
    elif isinstance(mapping, dict):
        mapping_dict = mapping
    else:
        raise ValueError("Invalid PAM4 mapping specified.")

    for i in range(num_symbols):
        binary_pair = str(prbs_sequence[2 * i]) + str(prbs_sequence[2 * i + 1])
        if binary_pair in mapping_dict:
            pam4_symbols[i] = mapping_dict[binary_pair]
        else:
            raise ValueError(f"Invalid binary pair '{binary_pair}' for PAM4 mapping.")

    return pam4_symbols

# --- Simulation Parameters ---
#N_symbols = 310 
N_symbols = 2900000     # Number of symbols to simulate
prbs7_length = 2*N_symbols  # Must be an even number for PAM4 mapping
prbs7_sequence = generate_prbs7(prbs7_length)
print("Generated PRBS7 Sequence (first 20 bits):", prbs7_sequence[:20])

pam4_symbols = map_prbs7_to_pam4(prbs7_sequence, mapping='gray')
print("PAM4 Symbols (first 10 symbols):", pam4_symbols[:10])

pam4_static_levels = np.array([-3, -1, 1, 3])
mapping = {0: -3, 1: -1, 2: 1, 3: 3} # Example mapping (can vary)
inv_mapping = {v: k for k, v in mapping.items()} # For checking errors later

# Channel Model (Simple FIR Filter: [main_cursor, post_cursor1, post_cursor2, ...])
# This channel introduces ISI from previous symbols
#channel_taps = np.array([0.7, 0.2, 0.1]) # Main tap, 1st post-cursor, 2nd post-cursor
#channel_taps = np.array([0.3, 1, -0.2])
agc_adj=0.25
#agc_adj=1
channel_taps = np.array([0.42, 1, 0.4 , 0.2, 0.1])*agc_adj
#channel_taps = np.array([0.08, 0.2 , 0.10, 0.02, 0.01])*3.2
# Noise Level
noise_stddev = 6e-3  # Standard deviation of Additive White Gaussian Noise (AWGN)

# FFE Parameters
num_ffe_pre_taps = 1      # Number of precursor taps (should match post-cursor ISI length)
num_ffe_post_taps = 3      # Number of post taps (should match post-cursor ISI length)
num_ffe_taps= num_ffe_pre_taps + num_ffe_post_taps +1
num_dfe_taps = 1      # Number of DFE taps (for post-cursor ISI cancellation)

mu_ffe = 4e-6        # Step size (learning rate) for FFE tap adaptation (LMS)
mu_dfe = 8e-6        # Step size (learning rate) for DFE tap adaptation (LMS)
mu_delev = 4e-6     # Step size (learning rate) for delev adaptation (LMS)

# --- 1. Generate PAM4 Signal ---
# Generate random integers (0, 1)
random_indices = np.random.randint(0, 4, N_symbols)
# Map to PAM4 levels (DLEV)
#tx_symbols = np.array([mapping[i] for i in random_indices])
tx_symbols = pam4_symbols # Use the generated PAM4 symbols

# --- 2. Apply Channel ---
# Convolve the transmitted symbols with the channel impulse response
# Add padding for convolution transient
tx_padded = np.concatenate((tx_symbols, np.zeros(len(channel_taps) - 1)))
channel_output = np.convolve(tx_padded, channel_taps, mode='valid') # Keep only valid part

# --- 3. Add Noise ---
noise = np.random.normal(0, noise_stddev, channel_output.shape)

#agc_adj=1
received_signal = (channel_output + noise)
#received_signal = channel_output 
'''
ffe_tap_weights = sdp.forcing_ffe(num_ffe_pre_taps, channel_taps)
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Zero Forcing FFE tap weights: {ffe_tap_weights}")
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")
'''
ffe_tap_weights2 = zf_ffe_toeplitz(channel_taps, agc_adj, 4, 1)
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Zero Forcing Topelitz FFE tap weights: {ffe_tap_weights2}")
print(f"++++++++++++++++++++++++++++++++++++++++++++++++++")

# --- 4. FFE Implementation ---
delev = 1           # Initial decision level (DLEV) for FFE
ffe_taps = np.concatenate((np.zeros(num_ffe_pre_taps), [1], np.zeros(num_ffe_post_taps)))
dfe_taps = 0 
#next_rx_data = np.zeros(num_ffe_pre_taps) # Buffer for past decided levels (DLEV)
ffe_output_symbols = np.zeros(N_symbols)
slicer_inputs = np.zeros(N_symbols) # Store slicer inputs for analysis/debug
ffe_errors    = np.zeros(N_symbols) # Store errors for analysis/debug
dfe_tap1_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap1_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap2_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap3_history = np.zeros(N_symbols) # Store tap history for plotting
ffe_tap2_error_history = np.zeros(N_symbols) # Store tap history for plotting
delev_history = np.zeros(N_symbols)  
delev_error_history = np.zeros(N_symbols)  
print(f"Starting FFE equalization with {num_ffe_taps} taps.")
print(f"Initial FFE taps: {ffe_taps}")
#next_rx_data[0] = received_signal[1]
vga_adj = 1/agc_adj
for k in range(num_ffe_post_taps, N_symbols-num_ffe_taps):
    # Get the current received sample corresponding to symbol k
    # Note: Channel convolution might shift timing, we align simply here
    # In a real system, timing recovery is crucial.
    # For this simple FIR, output 'k' corresponds roughly to input 'k' after delay.
    

    # Calculate ISI estimate from past decisions (DLEV)
    # isi_estimate = sum(dfe_taps[i] * past_decisions[i] for i in range(num_dfe_taps))
    #isi_estimate_x = np.dot(ffe_taps, next_rx_data) # More efficient
 
    ffe_input_vector = received_signal[k-num_ffe_post_taps:k+num_ffe_pre_taps+1]*vga_adj  # Get the input vector for FFE taps
    ffe_input_vector_r = ffe_input_vector[::-1]  # Reverse for convolution-like operation
    previous_decision = ffe_output_symbols[k-1] if k>1 else 0
    slicer_input_ffe = np.dot(ffe_taps, ffe_input_vector_r)  # Estimate ISI using FFE taps
    slicer_input = slicer_input_ffe - previous_decision*dfe_taps # Remove the last decision to avoid feedback loop
    slicer_inputs[k] = slicer_input # Store for plotting/analysis

    # PAM4 Slicer (Decision Device)
    # Find the closest PAM4 level (DLEV)
    pam4_levels = pam4_static_levels 
    distances = np.abs(slicer_input - pam4_levels)
    closest_level_index = np.argmin(distances)
    current_decision = pam4_static_levels[closest_level_index] # This is the decided DLEV

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
    
    #out_of_zone= abs(slicer_input) - abs(current_decision)
    error_dlev = np.sign(out_of_zone)
    error = slicer_input - current_decision   # Common error definition
    ffe_errors[k] = error # Store for analysis/debug
    # Update FFE taps using LMS algorithm
    # new_taps = old_taps + learning_rate * error * input_vector
    # Input vector for FFE taps is the vector of past decisions (DLEV)
    # dfe_taps = dfe_taps *(1-mu_dfe*gamma) + mu_dfe * error * past_decisions
    #ffe_adj_0=  mu_ffe * error * next_rx_data[0] # Precursor tap adjustment 
    #ffe_adj_1=  mu_ffe * error * received_signal[k-1] if k>=1 else 0 # Post 1st tap adjustment
    # Post 2nd tap adjustment
    #ffe_adj = np.array([ffe_adj_0, ffe_adj_1]) # Combine adjustments

    #ffe_adj = mu_ffe * np.sign(error) * np.sign(ffe_input_vector_r) # Adjust FFE taps based on error and input vector
    ffe_adj = mu_ffe * error * (ffe_input_vector_r)
    dfe_adj = mu_dfe * error * previous_decision
    #ffe_adj =  mu_ffe * error * past_decisions
    #ffe_adj =  mu_ffe * error * np.flip(next_rx_data)
    #ffe_adj =  mu_ffe * error * next_rx_data
    
    ffe_taps = ffe_taps - ffe_adj # Update FFE taps
    dfe_taps = dfe_taps + dfe_adj # Update DFE taps
    ffe_taps[2] =0 
    ffe_tap2_error_history[k] = np.sign(error)*np.sign(ffe_input_vector_r[1]) # Store for plotting/analysis
    ffe_tap1_history[k] = ffe_taps[0] # Store for plotting/analysis
    ffe_tap2_history[k] = ffe_taps[1] # Store for plotting/analysis 
    ffe_tap3_history[k] = ffe_taps[2] # Store for plotting/analysis
    dfe_tap1_history[k] = dfe_taps
    # --- Update Past Decisions Buffer ---
    # Shift buffer: oldest decision drops out, newest decision comes in
    #next_rx_data[0] = received_signal[k+2] # Store the latest decided level (DLEV)

    #delev_history[k]= ffe_taps[1]
    
    #delev = delev - mu_delev* error_dlev
    #delev_history[k]= delev # Store for plotting/analysis
    #delev_error_history[k] = error_dlev # Store for plotting/analysis
    #delev = delev + mu_delev*np.sign(error)

print(f"Final FFE taps after {N_symbols} symbols: {ffe_taps}")
print(f"Final DFE taps after {N_symbols} symbols: {dfe_taps}")

w_ffe_init = np.zeros([num_ffe_taps,])
w_dfe_init = np.zeros([num_dfe_taps,])

w_ffe, w_dfe, v_combined_ffe, v_combined_dfe, z_combined, e_combined = sdp.lms_equalizer(received_signal, 0.001, len(received_signal), w_ffe_init, num_ffe_pre_taps, w_dfe_init,  pam4_levels)

print(f"SDP adapted FFE weights: {w_ffe}")
print(f"SDP adapted DFE weights: {w_dfe}")

# --- 6. Performance Evaluation (Symbol Error Rate) ---
# Compare DFE output with original transmitted symbols
# Need to account for any delay introduced by the channel/equalizer setup.
# Here, assuming simple alignment for demonstration.
delay = 3 # Example delay (depends on channel and FFE setup)
ffe_output_delayed = np.concatenate((np.zeros(delay), ffe_output_symbols[:-delay]))
tx_symbols_delayed = np.concatenate((np.zeros(delay), tx_symbols[:-delay])) # Align tx symbols with FFE output
#errors = np.sum(tx_symbols[:N_symbols] != dfe_output_delayed[:N_symbols])
errors =np.sum(abs(ffe_errors[20:N_symbols-20]) > 1) # Count errors based on FFE output
ser_ffe = errors / N_symbols

# For comparison: SER without DFE (slicing the raw received signal)
#no_dfe_decisions = np.zeros(N_symbols)
#for k in range(N_symbols):
#     distances = np.abs(received_signal[k] - pam4_levels)
#     closest_level_index = np.argmin(distances)
#     no_dfe_decisions[k] = pam4_levels[closest_level_index]

#errors_no_dfe = np.sum(tx_symbols[:N_symbols] != no_dfe_decisions[:N_symbols])
#ser_no_dfe = errors_no_dfe / N_symbols

print(f"\n--- Results ---")
#print(f"Symbol Error Rate (SER) WITHOUT DFE: {ser_no_dfe:.4f} ({errors_no_dfe} errors)")
print(f"Symbol Error Rate (SER) WITH FFE:    {ser_ffe:.4f} ({errors} errors)")
print(f"Average Error is : {np.mean(ffe_errors[100:-100]):.4f}")
print(f"Average tx symbol  is : {np.mean(tx_symbols[100:-100]):.4f}")
print(f"Average received rx  is : {np.mean(received_signal[100:-100]):.4f}")

# --- Plotting (Optional) ---
plt.figure(figsize=(12, 8))

# Plot 1: Signal constellations (Scatter plot comparing input/output of slicer)
plt.subplot(2, 2, 1)
# Select a subset of points to avoid overplotting
#plot_subset = min(N_symbols, 500)
#indices_to_plot = np.random.choice(N_symbols, plot_subset, replace=False)
#plt.scatter(slicer_inputs[indices_to_plot], dfe_output_symbols[indices_to_plot], alpha=0.5, s=10)
#plt.vlines([-2, 0, 2], -4, 4, color='r', linestyle='--', label='Slicer Thresh') # Ideal thresholds
#plt.hlines(pam4_levels, np.min(slicer_inputs)-1, np.max(slicer_inputs)+1, color='g', linestyle=':', label='Ideal Levels')
#plt.title('DFE Slicer Input vs. Output Decision (DLEV)')
#plt.xlabel('Slicer Input Voltage (Arbitrary Units)')
#plt.ylabel('Decided PAM4 Level (DLEV)')
#time_axis2 = np.arange(30) # Plot first 100 symbols
time_axis2 = np.arange(N_symbols) 
#plt.plot(time_axis2, dfe_errors[:1000], '.-', label='sgn(error)', color='orange')
#plt.plot(time_axis2, dfe_tap2_history, 'o-', label='DFE Tap2', color='blue')
#plt.plot(time_axis2, delev_error_history[20000:1000]-ffe_tap2_error_history[20000:1000], '.-', label='diff', color='blue', alpha=0.5)
plt.plot(time_axis2, dfe_tap1_history, '--', label='dfe_tap', color='green', alpha=0.5)
#plt.plot(time_axis2, ffe_tap2_error_history[20000:20030], 'o-', label='ffe_error', color='red', alpha=0.5)
txt=  " for channel loss of " + str(agc_adj)
plt.title(txt)
plt.xlabel('Symbol Index')
plt.ylabel('Error')
plt.grid(True)
plt.legend()

received_signal_delayed = np.concatenate((np.zeros(delay), received_signal[:-delay]))
# Plot 2: Received Signal vs Time (subset)
plt.subplot(2, 2, 2)
time_axis = np.arange(100) # Plot first 100 symbols
plt.plot(time_axis, received_signal_delayed[:100], '.-', label='Received Signal Delayed(Noisy + ISI)')
plt.plot(time_axis, tx_symbols[:100], 'o-', label='Original Symbols', alpha=0.7)
plt.title('Signals (First 100 Symbols)')
plt.xlabel('Symbol Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 3: DFE Output vs Time
plt.subplot(2, 2, 3)
#plt.plot(time_axis, ffe_output_symbols[-100:], '.-', label='FFE Output Symbols', color='orange')
plt.plot(time_axis, ffe_output_delayed[-100:], '.-', label='FFE Output Symbols', color='orange')
#plt.plot(time_axis, tx_symbols_delayed[-100:], 'o-', label='Original Symbols Delayed', alpha=0.7)
plt.plot(time_axis, tx_symbols[-100:], '*-', label='Original Symbols', color='green', alpha=0.5)
plt.title('FFE Output (Last 100 Symbols)')
plt.xlabel('Symbol Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot 4: DFE taps convergence (if LMS was used)
# Need to store taps history during the loop for this - omitted for simplicity now
# Placeholder subplot
plt.subplot(2, 2, 4)
time_axis3 = np.arange(N_symbols) # Plot first 100 symbols
plt.plot(time_axis3, ffe_tap1_history, '.-', label='FFE Tap1', color='orange')
plt.plot(time_axis3, ffe_tap2_history, 'o-', label='FFE Tap2', color='blue',alpha=0.5)
plt.plot(time_axis3, ffe_tap3_history, '--', label='FFE Tap3', color='red',alpha=0.35)
#plt.plot(time_axis3, delev_history, '+-', label='delev', color='green',alpha=0.35)
annotation_text = "Channel Taps: " + str(channel_taps) + "FFE Taps: " + str(ffe_taps) 
#plt.title(annotation_text)
plt.title('FFE Tap Convergence')
plt.xlabel('Symbol Index')
plt.ylabel('Tap Value')
plt.grid(True)
'''
bins=100
hist, bin_edges = np.histogram(received_signal, bins=bins)
plt.figure(figsize=(12, 8))
plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), align='edge')

plt.title("Received Signal histogram")
plt.xticks(bin_edges, rotation=45)
plt.xlim(bin_edges[0], bin_edges[-1])
plt.ylim(0, np.max(hist) * 1.1)
plt.axvline(x=0, color='r', linestyle='--', label='Zero Level')            
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)


bins=100
hist, bin_edges = np.histogram(ffe_output_symbols, bins=bins)
plt.figure(figsize=(12, 8))
plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), align='edge')

plt.title("DFE Output histogram")
plt.xticks(bin_edges, rotation=45)
plt.xlim(bin_edges[0], bin_edges[-1])
plt.ylim(0, np.max(hist) * 1.1)
plt.axvline(x=0, color='r', linestyle='--', label='Zero Level')            
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.75)


plt.figure(figsize=(12, 8))
plt.plot(time_axis3, ffe_errors, '.-', label='FFE Tap1', color='orange')
plt.title('Slicer Error')
plt.xlabel('Symbol Index')
plt.ylabel('Tap Value')
plt.tight_layout()
'''
plt.show()

#print(f"max error is : {np.max(ffe_errors):.4f}")
#print(f"meas square error is : {mean_squared_error(ffe_errors):.4f}")



  

