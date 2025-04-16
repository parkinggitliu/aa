import numpy as np
import matplotlib.pyplot as plt

def generate_random_data(num_symbols):
    """Generates random binary data."""
    return np.random.randint(0, 2, num_symbols) * 2 - 1  # Generates -1 and 1

def apply_channel(data, taps=[0.5, 1.0, 0.3]):
    """Simulates a simple linear channel with ISI."""
    return np.convolve(data, taps, 'full')[:len(data)]

def add_noise(signal, snr_db):
    """Adds Additive White Gaussian Noise (AWGN) to the signal."""
    snr_linear = 10**(snr_db / 10)
    signal_power = np.mean(signal**2)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

def cdr_loop(received_signal, samples_per_symbol=2, initial_phase=0.0, loop_bandwidth=0.01, damping_factor=0.707):
    """Simple digital CDR loop (conceptual)."""
    N = len(received_signal)
    recovered_clock_phase = np.zeros(N, dtype=float)
    sampled_signal = np.zeros(N // samples_per_symbol, dtype=float)
    clock_phase = initial_phase
    phase_error_history = []
    clock_index = 0
    signal_index = 0

    for i in range(N):
        recovered_clock_phase[i] = clock_phase

        # Simple early-late gate phase detector (conceptual)
        if signal_index % samples_per_symbol == samples_per_symbol // 2:
            mid_sample = received_signal[i]
            early_index = max(0, i - 1)
            late_index = min(N - 1, i + 1)
            early_sample = received_signal[early_index]
            late_sample = received_signal[late_index]
            phase_error = late_sample**2 - early_sample**2 # A very simplified error
            phase_error_history.append(phase_error)

            # Loop filter (simplified proportional-integral)
            proportional_gain = 2 * damping_factor * loop_bandwidth
            integral_gain = loop_bandwidth**2
            clock_phase += proportional_gain * phase_error
            clock_phase += integral_gain * np.sum(phase_error_history) # Simplified integration

        if signal_index % samples_per_symbol == 0:
            sampled_signal[clock_index] = received_signal[i]
            clock_index += 1

        signal_index += 1

    return sampled_signal[:clock_index], recovered_clock_phase[:N]

def lms_dfe(received_samples, num_forward_taps, num_feedback_taps, step_size, training_data=None, num_training_symbols=0):
    """LMS Decision Feedback Equalizer."""
    N = len(received_samples)
    forward_weights = np.zeros(num_forward_taps)
    feedback_weights = np.zeros(num_feedback_taps)
    estimated_symbols = np.zeros(N - num_forward_taps, dtype=float)
    delayed_input = np.zeros(num_forward_taps)
    past_decisions = np.zeros(num_feedback_taps)

    if training_data is not None:
        desired_signal = np.concatenate((training_data, np.zeros(N - num_forward_taps - num_training_symbols)))
    else:
        desired_signal = np.zeros(N - num_forward_taps) # Decision-directed

    for i in range(N - num_forward_taps):
        # Form the input vector for the forward filter
        delayed_input = np.concatenate(([received_samples[i + num_forward_taps - 1]], delayed_input[:-1]))

        # Form the input vector for the feedback filter
        feedback_input = past_decisions[::-1]

        # Equalizer output
        equalized_output = np.dot(forward_weights, delayed_input) - np.dot(feedback_weights, feedback_input)

        # Make a decision (slicer)
        estimated_symbol = 1 if equalized_output > 0 else -1
        estimated_symbols[i] = estimated_symbol

        # Error calculation
        if training_data is not None and i < num_training_symbols:
            error = training_data[i] - equalized_output
        else:
            error = estimated_symbol - equalized_output # Decision-directed

        # Update filter weights (LMS algorithm)
        forward_weights += step_size * error * delayed_input
        feedback_weights += step_size * error * feedback_input

        # Update past decisions
        past_decisions = np.concatenate(([estimated_symbol], past_decisions[:-1]))

    return estimated_symbols, forward_weights, feedback_weights

if __name__ == "__main__":
    # System parameters
    num_symbols = 200
    samples_per_symbol = 4
    snr_db = 20
    channel_taps = [0.1, 0.8, 0.5, 0.2]
    num_forward_taps = 7
    num_feedback_taps = 4
    lms_step_size = 0.01
    num_training_symbols = 50

    # Generate data
    transmitted_data = generate_random_data(num_symbols)

    # Upsample the data (for CDR simulation)
    upsampled_data = np.repeat(transmitted_data, samples_per_symbol)

    # Apply channel
    channel_output = apply_channel(upsampled_data, channel_taps)

    # Add noise
    noisy_signal = add_noise(channel_output, snr_db)

    # Clock Data Recovery
    sampled_signal, recovered_clock_phase = cdr_loop(noisy_signal, samples_per_symbol=samples_per_symbol)

    # Prepare training data (first few symbols)
    training_data = transmitted_data[:num_training_symbols]

    # Apply LMS DFE
    equalized_data, forward_weights, feedback_weights = lms_dfe(
        sampled_signal,
        num_forward_taps,
        num_feedback_taps,
        lms_step_size,
        training_data=training_data,
        num_training_symbols=num_training_symbols
    )

    # Evaluate performance (crude BER estimation)
    if len(equalized_data) >= len(transmitted_data[num_training_symbols:]):
        detected_symbols = np.sign(equalized_data[num_training_symbols:])
        transmitted_test = transmitted_data[num_training_symbols:]
        errors = np.sum(detected_symbols != transmitted_test[:len(detected_symbols)])
        ber = errors / len(transmitted_test)
        print(f"Bit Error Rate (BER): {ber:.4f}")
    else:
        print("Not enough equalized data to calculate BER accurately.")

    # Plotting (optional)
    time_upsampled = np.arange(len(noisy_signal))
    time_sampled = np.arange(0, len(noisy_signal), samples_per_symbol)

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_upsampled, noisy_signal)
    plt.title("Received Signal with Noise and ISI")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_sampled[:len(sampled_signal)], sampled_signal)
    plt.title("Sampled Signal (after conceptual CDR)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(transmitted_data[num_training_symbols:], 'b-', label='Transmitted Data')
    plt.plot(np.sign(equalized_data[num_training_symbols:]), 'r.', label='Equalized Data')
    plt.title("Transmitted vs. Equalized Data (after training)")
    plt.xlabel("Symbol Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
