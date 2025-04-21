import numpy as np
import matplotlib.pyplot as plt
import collections

# --- Simulation Parameters ---
N_symbols = 10000       # Number of symbols to simulate
sps = 8                # Samples per symbol (oversampling ratio) - for channel sim
dfe_taps_count = 5     # Number of DFE taps
noise_stddev = 0.15    # Standard deviation of Additive White Gaussian Noise (AWGN)
initial_timing_offset = 0.1 * sps # Initial timing offset (in samples)

# PAM4 levels (ideal)
pam4_levels = np.array([-3.0, -1.0, 1.0, 3.0])
pam4_dict = {0: -3.0, 1: -1.0, 2: 1.0, 3: 3.0}
pam4_map_rev = {-3.0: 0, -1.0: 1, 1.0: 2, 3.0: 3}
initial_data_levels = pam4_levels.copy() # Initial guess for data levels

# Adaptation Step Sizes (Mu)
mu_dfe = 0.005          # DFE tap adaptation rate
mu_dlev = 0.001         # Data level adaptation rate
mu_cdr_kp = 0.01        # CDR Proportional gain
mu_cdr_ki = 0.001       # CDR Integral gain

# Channel Impulse Response (CIR) - Example: introduces ISI
# channel_taps = np.array([0.8, -0.3, 0.15, -0.05]) # Main tap + trailing ISI
channel_taps = np.convolve([1], [1, 0.5, -0.2]) # Example with precursor and postcursor
channel_taps /= np.linalg.norm(channel_taps) # Normalize

# Pattern Filtering: Pause adaptation if this many consecutive outer levels occur
pattern_filter_length = 4
pattern_filter_levels = [-3.0, 3.0] # Levels considered for filtering


# --- Helper Functions ---
def pam4_slicer(sample, current_levels):
    """Slices a sample to the nearest PAM4 level based on current_levels."""
    thresholds = (current_levels[:-1] + current_levels[1:]) / 2.0
    if sample < thresholds[0]:
        return current_levels[0]
    elif sample < thresholds[1]:
        return current_levels[1]
    elif sample < thresholds[2]:
        return current_levels[2]
    else:
        return current_levels[3]

def generate_pam4_symbols(n):
    """Generates n random PAM4 symbols."""
    indices = np.random.randint(0, 4, n)
    return np.array([pam4_dict[i] for i in indices])

# --- Simulation Setup ---
# 1. Generate transmit symbols
tx_symbols = generate_pam4_symbols(N_symbols)

# 2. Upsample symbols (simple zero-order hold for waveform)
tx_waveform = np.zeros(N_symbols * sps)
tx_waveform[::sps] = tx_symbols

# 3. Simulate the channel
# Convolve with channel impulse response
rx_waveform_noiseless = np.convolve(tx_waveform, channel_taps, mode='full')
# Add AWGN
noise = np.random.normal(0, noise_stddev, rx_waveform_noiseless.shape)
rx_waveform = rx_waveform_noiseless + noise
# Add some delay for channel
channel_delay = np.argmax(channel_taps) # Approximate group delay center

# --- Receiver Initialization ---
dfe_taps = np.zeros(dfe_taps_count)
data_levels = initial_data_levels.copy()
past_decisions = collections.deque(np.zeros(dfe_taps_count), maxlen=dfe_taps_count)
# CDR state
current_sample_idx_float = float(channel_delay + initial_timing_offset) # Start with offset
cdr_integrator = 0.0

# Storage for analysis
received_samples = []
equalized_samples = []
decided_symbols = []
errors = []
timing_indices = []
dfe_taps_history = [dfe_taps.copy()]
data_levels_history = [data_levels.copy()]
timing_error_history = []
phase_history = [initial_timing_offset]

# Pattern filtering state
recent_decisions_for_pattern = collections.deque(maxlen=pattern_filter_length)
#adapt_enabled = True # Start with adaptation enabled
adapt_enabled = False

# --- Main Simulation Loop ---
for k in range(N_symbols): # Iterate through symbols

    # 1. CDR: Determine exact sample index
    current_sample_idx = int(round(current_sample_idx_float))
    if current_sample_idx < 0 or current_sample_idx >= len(rx_waveform):
        print(f"Warning: Sample index {current_sample_idx} out of bounds at symbol {k}. Stopping.")
        break
    timing_indices.append(current_sample_idx)

    # 2. Get sample from received waveform
    rx_sample = rx_waveform[current_sample_idx]
    received_samples.append(rx_sample)

    # 3. DFE: Calculate feedback based on *past* decisions
    dfe_correction = np.dot(dfe_taps, list(past_decisions))

    # 4. Equalize: Subtract DFE correction
    eq_sample = rx_sample - dfe_correction
    equalized_samples.append(eq_sample)

    # 5. Slice: Make decision based on *current* data levels
    decision = pam4_slicer(eq_sample, data_levels)
    decided_symbols.append(decision)

    # 6. Calculate Error: Difference between equalized sample and the ideal level for the decision
    error = eq_sample - decision
    errors.append(error)

    # 7. Pattern Filtering Check
    recent_decisions_for_pattern.append(decision)
    if len(recent_decisions_for_pattern) == pattern_filter_length:
        is_filtered_pattern = True
        first_level = recent_decisions_for_pattern[0]
        if first_level not in pattern_filter_levels:
             is_filtered_pattern = False
        else:
            for d in range(1, pattern_filter_length):
                if recent_decisions_for_pattern[d] != first_level:
                    is_filtered_pattern = False
                    break
        adapt_enabled = not is_filtered_pattern
        # if not adapt_enabled:
        #      print(f"Symbol {k}: Adaptation PAUSED due to pattern {list(recent_decisions_for_pattern)}")
    else:
        adapt_enabled = True # Not enough history yet

    # --- Adaptation (if enabled by pattern filter) ---
    if adapt_enabled and k > dfe_taps_count : # Allow some buffer buildup

        # 8. DFE Tap Adaptation (LMS)
        # Gradient estimate: error * past_decisions vector
        dfe_taps_update = mu_dfe * error * np.array(list(past_decisions))
        dfe_taps = dfe_taps + dfe_taps_update

        # 9. Data Level (dLev) Adaptation
        #decision_idx = pam4_map_rev[decision]
        ideal_levels = list(pam4_map_rev.keys()) # Get the expected levels
        closest_level = min(ideal_levels, key=lambda level: abs(decision - level))
        decision_idx = pam4_map_rev[closest_level]
        # Simple EMA update for the level corresponding to the decision
        data_levels[decision_idx] = (1 - mu_dlev) * data_levels[decision_idx] + mu_dlev * eq_sample

        # 10. CDR Phase Update
        # Simple timing error detector: error * (current_decision - previous_decision)
        # This is sensitive to noise and assumes transitions carry timing info.
        # A zero-crossing detector on the error signal derivative is often better.
        prev_decision = past_decisions[0] if len(past_decisions)>0 else 0.0
        # Scale transition magnitude to avoid huge updates for outer levels transitions
        transition_magnitude = (decision - prev_decision) / (pam4_levels[-1] - pam4_levels[0])
        timing_error = error * transition_magnitude # Simplified TED

        # PI Loop Filter
        cdr_integrator += mu_cdr_ki * timing_error
        phase_update = mu_cdr_kp * timing_error + cdr_integrator

        # Update sampling phase for *next* symbol (adjust float index)
        # Subtract because a positive timing_error might mean we sampled late
        current_sample_idx_float -= phase_update

        # Store for analysis
        timing_error_history.append(timing_error)
        phase_history.append(current_sample_idx_float - channel_delay) # Store relative phase offset


    # --- Update State for Next Iteration ---
    # Add current decision to the front of the DFE history buffer
    past_decisions.appendleft(decision)

    # Store history (optional, for plotting convergence)
    if k % 50 == 0: # Store less frequently to save memory
         dfe_taps_history.append(dfe_taps.copy())
         data_levels_history.append(data_levels.copy())


# --- Analysis and Plotting ---
decided_symbols = np.array(decided_symbols)
errors = np.array(errors)

# Calculate Symbol Error Rate (SER) - skip initial transient
transient_skip = 500
if N_symbols > transient_skip:
    symbol_errors = np.sum(tx_symbols[transient_skip:len(decided_symbols)] != decided_symbols[transient_skip:])
    ser = symbol_errors / (len(decided_symbols) - transient_skip)
    print(f"Symbol Error Rate (SER): {ser:.2e} (excluding first {transient_skip} symbols)")
else:
    print("Not enough symbols to calculate SER after transient.")

print(f"Final DFE Taps: {dfe_taps}")
print(f"Final Data Levels: {data_levels}")
print(f"Final Relative Phase Offset: {phase_history[-1]:.2f} samples")


# Plotting
plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style if available
fig, axs = plt.subplots(4, 2, figsize=(12, 18))
fig.suptitle('PAM4 DFE Simulation Results', fontsize=16)

# 1. Received vs Equalized Samples (Eye Diagram Concept)
num_eyes = min(N_symbols - transient_skip, 1000) # Limit points for clarity
time_axis_eye = np.arange(len(rx_waveform)) / sps

# Plot received samples around optimal sampling points
axs[0, 0].set_title("Received Samples (around symbol time)")
axs[0, 0].set_ylabel("Amplitude")
axs[0, 0].grid(True)
for i in range(transient_skip, transient_skip + num_eyes):
    start_idx = max(0, timing_indices[i] - sps // 2)
    end_idx = min(len(rx_waveform), timing_indices[i] + sps // 2 + 1)
    time_rel = (np.arange(start_idx, end_idx) - timing_indices[i]) / sps
    axs[0, 0].plot(time_rel, rx_waveform[start_idx:end_idx], 'b-', alpha=0.1)
axs[0, 0].set_xlabel("Time relative to decision (Symbols)")
for level in pam4_levels:
    axs[0,0].axhline(level, color='grey', linestyle=':')


# Plot equalized samples (should form clearer levels)
axs[0, 1].set_title("Equalized Samples (DFE Output)")
axs[0, 1].plot(np.arange(transient_skip, len(equalized_samples)),
               equalized_samples[transient_skip:], 'g.', markersize=1, alpha=0.5)
for level in data_levels: # Plot final adapted levels
    axs[0, 1].axhline(level, color='r', linestyle='--')
for level in pam4_levels: # Plot ideal levels
    axs[0, 1].axhline(level, color='grey', linestyle=':')
axs[0, 1].set_xlabel("Symbol Index")
axs[0, 1].set_ylabel("Amplitude")
axs[0, 1].grid(True)


# 2. DFE Taps Convergence
axs[1, 0].set_title("DFE Taps Convergence")
dfe_taps_history_arr = np.array(dfe_taps_history)
for i in range(dfe_taps_count):
    axs[1, 0].plot(dfe_taps_history_arr[:, i], label=f'Tap {i+1}')
axs[1, 0].set_xlabel("Iterations (x50)")
axs[1, 0].set_ylabel("Tap Value")
axs[1, 0].legend()
axs[1, 0].grid(True)

# 3. Data Levels (dLev) Convergence
axs[1, 1].set_title("Data Level (dLev) Adaptation")
data_levels_history_arr = np.array(data_levels_history)
labels_dlev = ['Level -3', 'Level -1', 'Level +1', 'Level +3']
for i in range(4):
    axs[1, 1].plot(data_levels_history_arr[:, i], label=labels_dlev[i])
    axs[1, 1].axhline(pam4_levels[i], color='grey', linestyle=':', alpha=0.7) # Ideal level
axs[1, 1].set_xlabel("Iterations (x50)")
axs[1, 1].set_ylabel("Level Value")
axs[1, 1].legend()
axs[1, 1].grid(True)

# 4. CDR Phase Convergence
axs[2, 0].set_title("CDR Sampling Phase Offset")
axs[2, 0].plot(phase_history)
axs[2, 0].set_xlabel("Symbol Index")
axs[2, 0].set_ylabel("Phase Offset (samples)")
axs[2, 0].grid(True)

# 5. CDR Timing Error
axs[2, 1].set_title("CDR Timing Error Signal (Filtered)")
axs[2, 1].plot(timing_error_history)
axs[2, 1].set_xlabel("Symbol Index (Adapted)")
axs[2, 1].set_ylabel("Timing Error")
axs[2, 1].grid(True)

# 6. Constellation Diagram (Equalized Samples vs Decisions)
axs[3, 0].set_title("Constellation (Equalized Sample vs Decision)")
axs[3, 0].scatter(decided_symbols[transient_skip:], equalized_samples[transient_skip:],
                  marker='.', alpha=0.1)
axs[3, 0].set_xlabel("Decided Symbol Level")
axs[3, 0].set_ylabel("Equalized Sample Amplitude")
axs[3, 0].grid(True)

# 7. Error Signal
axs[3, 1].set_title("Error Signal (Eq Sample - Decision)")
axs[3, 1].plot(errors[transient_skip:], 'r.', markersize=1, alpha=0.3)
axs[3, 1].set_xlabel("Symbol Index")
axs[3, 1].set_ylabel("Error")
axs[3, 1].grid(True)


plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap
plt.show()
