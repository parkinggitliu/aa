import numpy as np
import matplotlib.pyplot as plt

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
N_symbols = 40000       # Number of symbols to simulate
prbs7_length = 2*N_symbols  # Must be an even number for PAM4 mapping
prbs7_sequence = generate_prbs7(prbs7_length)
print("Generated PRBS7 Sequence (first 20 bits):", prbs7_sequence[:20])

pam4_gray_symbols = map_prbs7_to_pam4(prbs7_sequence, mapping='gray')
print("PAM4 Symbols (Gray mapping, first 10 symbols):", pam4_gray_symbols[:10])

pam4_natural_symbols = map_prbs7_to_pam4(prbs7_sequence, mapping='natural')
print("PAM4 Symbols (Natural mapping, first 10 symbols):", pam4_natural_symbols[:10])

pam4_levels = np.array([-3, -1, 1, 3])
mapping = {0: -3, 1: -1, 2: 1, 3: 3} # Example mapping (can vary)
inv_mapping = {v: k for k, v in mapping.items()} # For checking errors later

# Channel Model (Simple FIR Filter: [main_cursor, post_cursor1, post_cursor2, ...])
# This channel introduces ISI from previous symbols
#channel_taps = np.array([0.7, 0.2, 0.1]) # Main tap, 1st post-cursor, 2nd post-cursor
channel_taps = np.array([1, 0.6, 0.2])
# Noise Level
noise_stddev = 0.06  # Standard deviation of Additive White Gaussian Noise (AWGN)

# DFE Parameters
num_dfe_taps = 2      # Number of feedback taps (should match post-cursor ISI length)
mu_dfe = 8e-4        # Step size (learning rate) for DFE tap adaptation (LMS)

# --- 1. Generate PAM4 Signal ---
# Generate random integers (0, 1, 2, 3)
random_indices = np.random.randint(0, 4, N_symbols)
# Map to PAM4 levels (DLEV)
#tx_symbols = np.array([mapping[i] for i in random_indices])
tx_symbols = pam4_natural_symbols # Use the generated PAM4 symbols

# --- 2. Apply Channel ---
# Convolve the transmitted symbols with the channel impulse response
# Add padding for convolution transient
tx_padded = np.concatenate((tx_symbols, np.zeros(len(channel_taps) - 1)))
channel_output = np.convolve(tx_padded, channel_taps, mode='valid') # Keep only valid part

# --- 3. Add Noise ---
noise = np.random.normal(0, noise_stddev, channel_output.shape)
received_signal = channel_output + noise
#received_signal = channel_output 

# --- 4. DFE Implementation ---
dfe_taps = np.array([0.35, 0.15]) # Initialize DFE tap weights
past_decisions = np.zeros(num_dfe_taps) # Buffer for past decided levels (DLEV)
dfe_output_symbols = np.zeros(N_symbols)
slicer_inputs = np.zeros(N_symbols) # Store slicer inputs for analysis/debug
dfe_errors = np.zeros(N_symbols) # Store errors for analysis/debug
dfe_tap1_history = np.zeros(N_symbols) # Store tap history for plotting
dfe_tap2_history = np.zeros(N_symbols) # Store tap history for plotting

print(f"Starting DFE equalization with {num_dfe_taps} taps.")
print(f"Initial DFE taps: {dfe_taps}")

for k in range(N_symbols):
    # Get the current received sample corresponding to symbol k
    # Note: Channel convolution might shift timing, we align simply here
    # In a real system, timing recovery is crucial.
    # For this simple FIR, output 'k' corresponds roughly to input 'k' after delay.
    current_rx_sample = received_signal[k]

    # Calculate ISI estimate from past decisions (DLEV)
    # isi_estimate = sum(dfe_taps[i] * past_decisions[i] for i in range(num_dfe_taps))
    isi_estimate_x = np.dot(dfe_taps, past_decisions) # More efficient
    isi_estimate2 = dfe_taps[1]*past_decisions[1] if k >= 2 else 0 # ISI from 2nd tap
    isi_estimate1 = dfe_taps[0]*past_decisions[0] if k >= 1 else 0 # ISI from 1st tap
    isi_estimate = isi_estimate1 + isi_estimate2 # ISI from main tap and 2 post-cursors
    # Note: In a real DFE, the taps would be adapted based on the error signal.

    # Subtract estimated ISI
    slicer_input = current_rx_sample - isi_estimate
    slicer_inputs[k] = slicer_input # Store for plotting/analysis

    # PAM4 Slicer (Decision Device)
    # Find the closest PAM4 level (DLEV)
    distances = np.abs(slicer_input - pam4_levels)
    closest_level_index = np.argmin(distances)
    current_decision = pam4_levels[closest_level_index] # This is the decided DLEV

    # Store the decision
    dfe_output_symbols[k] = current_decision

    # --- 5. LMS Adaptation (Optional) ---
    # Calculate error: difference between slicer input and the decided level
    error = current_decision - slicer_input # Common DFE error definition
    dfe_errors[k] = error # Store for analysis/debug
    gamma = 0.9
    # Update DFE taps using LMS algorithm
    # new_taps = old_taps + learning_rate * error * input_vector
    # Input vector for DFE taps is the vector of past decisions (DLEV)
  # dfe_taps = dfe_taps *(1-mu_dfe*gamma) + mu_dfe * error * past_decisions
    dfe_adj =  mu_dfe * error * current_decision
    dfe_lkg = (1 - mu_dfe * gamma)*dfe_taps # Leakage term for DFE taps
    dfe_taps = dfe_taps + dfe_adj # Update DFE taps
    dfe_tap1_history[k] = dfe_taps[0] # Store for plotting/analysis
    dfe_tap2_history[k] = dfe_taps[1] # Store for plotting/analysis 
    # --- Update Past Decisions Buffer ---
    # Shift buffer: oldest decision drops out, newest decision comes in
    past_decisions = np.roll(past_decisions, 1)
    past_decisions[0] = current_decision # Store the latest decided level (DLEV)

print(f"Final DFE taps after {N_symbols} symbols: {dfe_taps}")

# --- 6. Performance Evaluation (Symbol Error Rate) ---
# Compare DFE output with original transmitted symbols
# Need to account for any delay introduced by the channel/equalizer setup.
# Here, assuming simple alignment for demonstration.
delay = 2 # Example delay (depends on channel and DFE setup)
dfe_output_delayed = np.concatenate((np.zeros(delay), dfe_output_symbols[:-delay]))
tx_symbols_delayed = np.concatenate((np.zeros(delay), tx_symbols[:-delay])) # Align tx symbols with DFE output
#errors = np.sum(tx_symbols[:N_symbols] != dfe_output_delayed[:N_symbols])
errors =np.sum(abs(dfe_errors[20:N_symbols-20]) > 1) # Count errors based on DFE output
ser_dfe = errors / N_symbols

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
print(f"Symbol Error Rate (SER) WITH DFE:    {ser_dfe:.4f} ({errors} errors)")
print(f"Average Error is : {np.mean(dfe_errors[100:-100]):.4f}")
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
time_axis2 = np.arange(1000) # Plot first 100 symbols
plt.plot(time_axis2, dfe_errors[:1000], '.-', label='DFE Slicer Errors', color='orange')
#plt.plot(time_axis2, dfe_tap2_history, 'o-', label='DFE Tap2', color='blue')
plt.title('DFE Error Convergence')
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
plt.plot(time_axis, dfe_output_delayed[-100:], '.-', label='DFE Output Symbols', color='orange')
#plt.plot(time_axis, tx_symbols_delayed[-100:], 'o-', label='Original Symbols Delayed', alpha=0.7)
plt.plot(time_axis, tx_symbols[-100:], '*-', label='Original Symbols', color='green', alpha=0.5)
plt.title('DFE Output (Last 100 Symbols)')
plt.xlabel('Symbol Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


# Plot 4: DFE taps convergence (if LMS was used)
# Need to store taps history during the loop for this - omitted for simplicity now
# Placeholder subplot
plt.subplot(2, 2, 4)
time_axis3 = np.arange(N_symbols) # Plot first 100 symbols
plt.plot(time_axis3, dfe_tap1_history, '.-', label='DFE Tap1', color='orange')
plt.plot(time_axis3, dfe_tap2_history, 'o-', label='DFE Tap2', color='blue',alpha=0.5)
plt.title('DFE Tap Convergence')
plt.xlabel('Symbol Index')
plt.ylabel('Tap Value')
plt.grid(True)


plt.tight_layout()
plt.show()