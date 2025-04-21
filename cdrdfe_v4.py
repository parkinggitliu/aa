import numpy as np
import matplotlib.pyplot as plt

def pam4_dfe_cdr(received_signal, num_taps, tap_weights_initial=None, step_size=0.01,
                 decision_levels=[-3, -1, 1, 3], symbol_period=1, samples_per_symbol=4,
                 learning_rate_cdr=0.001, filter_length=5, filter_coefficients=None,
                 pattern_filter_threshold=0.8):
    """
    PAM4 Decision Feedback Equalizer (DFE) with Decision-Directed Level Adjustment (dLev)
    and Clock Data Recovery (CDR) with signal pattern filtering.

    Args:
        received_signal (np.ndarray): The received PAM4 signal (oversampled).
        num_taps (int): Number of feedback taps in the DFE.
        tap_weights_initial (np.ndarray, optional): Initial tap weights for the DFE.
            Defaults to zeros.
        step_size (float): Step size for the DFE adaptation algorithm (LMS).
        decision_levels (list or np.ndarray): The ideal decision levels for PAM4.
        symbol_period (int): Number of samples per symbol interval.
        samples_per_symbol (int): Oversampling factor.
        learning_rate_cdr (float): Step size for the CDR adaptation.
        filter_length (int): Length of the moving average filter for CDR.
        filter_coefficients (np.ndarray, optional): Coefficients for a custom filter.
            If None, a moving average filter is used.
        pattern_filter_threshold (float): Threshold for the pattern filter in CDR.
            A value closer to 1 requires a stronger pattern match.

    Returns:
        tuple: A tuple containing:
            - equalized_signal (np.ndarray): The output of the DFE.
            - detected_symbols (np.ndarray): The detected PAM4 symbols.
            - tap_weights_history (list): History of the DFE tap weights.
            - decision_levels_history (list): History of the decision levels.
            - timing_offset_history (list): History of the estimated timing offset.
    """

    num_samples = len(received_signal)
    detected_symbols = np.zeros(num_samples // samples_per_symbol)
    equalized_signal = np.zeros(num_samples)
    tap_weights = np.zeros(num_taps) if tap_weights_initial is None else tap_weights_initial.copy()
    tap_weights_history = [tap_weights.copy()]
    decision_levels = np.array(decision_levels)
    decision_levels_history = [decision_levels.copy()]
    estimated_timing_offset = 0  # Initial timing offset (in samples)
    timing_offset_history = [estimated_timing_offset]
    phase_accumulator = 0.0

    # CDR Filter Initialization
    if filter_coefficients is None:
        filter_coefficients = np.ones(filter_length) / filter_length  # Moving average filter
    filter_state = np.zeros(filter_length)

    # dLev parameters
    alpha_dlev = 0.01  # Step size for decision level adjustment

    # Symbol buffer for DFE feedback
    symbol_buffer = np.zeros(num_taps)

    for i in range(num_samples):
        # CDR - Clock Data Recovery
        clock_sample_index = int(i + round(estimated_timing_offset))

        if 0 <= clock_sample_index < num_samples:
            current_sample_cdr = received_signal[clock_sample_index]

            # Update filter state
            filter_state = np.concatenate(([current_sample_cdr], filter_state[:-1]))

            # Filter the signal
            filtered_signal = np.dot(filter_coefficients, filter_state)

            # Zero-crossing detection (simplified for PAM4)
            # Look for changes in the filtered signal around the midpoints of levels
            mid_levels = np.array([-2, 0, 2])
            zc_indicator = 0
            for mid in mid_levels:
                if (filtered_signal > mid and filter_state[1] <= mid) or \
                   (filtered_signal <= mid and filter_state[1] > mid):
                    zc_indicator = 1
                    break

            # Pattern Filtering (Simple example: looking for alternating high/low transitions)
            pattern_match = 0
            if i >= 2 * samples_per_symbol:
                prev_symbol = detected_symbols[(i - samples_per_symbol) // samples_per_symbol -1]
                current_symbol_est = np.argmin(np.abs(received_signal[clock_sample_index] - decision_levels))
                current_symbol_val = decision_levels[current_symbol_est]

                prev_symbol_est = np.argmin(np.abs(prev_symbol - decision_levels))
                prev_symbol_val = decision_levels[prev_symbol_est]

                if (current_symbol_val > 0 and prev_symbol_val < 0) or \
                   (current_symbol_val < 0 and prev_symbol_val > 0):
                    pattern_match = 1

            # Update timing offset based on filtered signal and pattern
            if zc_indicator and pattern_match >= pattern_filter_threshold:
                phase_accumulator += np.sign(filtered_signal - filter_state[1]) * learning_rate_cdr
                estimated_timing_offset += phase_accumulator
                phase_accumulator -= np.floor(phase_accumulator) # Keep phase within [0, 1)

        # DFE
        feedback = np.dot(symbol_buffer, tap_weights)
        equalized_sample = received_signal[i] - feedback
        equalized_signal[i] = equalized_sample

        # Symbol detection at the (approximate) symbol boundaries
        if (i % samples_per_symbol) == 0:
            detected_symbol_index = np.argmin(np.abs(equalized_sample - decision_levels))
            detected_symbol = decision_levels[detected_symbol_index]
            symbol_index = i // samples_per_symbol
            if symbol_index < len(detected_symbols):
                detected_symbols[symbol_index] = detected_symbol

            # dLev - Decision-Directed Level Adjustment
            if symbol_index > 0:
                error = detected_symbol - decision_levels[detected_symbol_index] # Error is zero by definition here
                decision_levels[detected_symbol_index] += alpha_dlev * error # No change ideally
                # More robust dLev would consider errors based on slicer output before dLev update
                # For simplicity, we'll adjust based on the detected symbol and current levels.
                # A better approach would involve averaging errors over multiple symbols.
                y_k = equalized_signal[i]
                d_k_hat = detected_symbol
                for j in range(len(decision_levels)):
                    if j != detected_symbol_index:
                        decision_levels[j] -= alpha_dlev * (y_k - d_k_hat) * np.sign(decision_levels[j] - d_k_hat)


            # Update symbol buffer for DFE feedback
            symbol_buffer = np.concatenate(([detected_symbol], symbol_buffer[:-1]))

            # Update DFE tap weights (LMS algorithm)
            error = equalized_sample - detected_symbol
            tap_weights += step_size * error * symbol_buffer

            tap_weights_history.append(tap_weights.copy())
            decision_levels_history.append(decision_levels.copy())
            timing_offset_history.append(estimated_timing_offset)

    return equalized_signal, detected_symbols, tap_weights_history, decision_levels_history, timing_offset_history

if __name__ == '__main__':
    # Simulation parameters
    num_symbols = 200
    samples_per_symbol = 8
    symbol_rate = 1  # Symbols per unit time
    sampling_rate = symbol_rate * samples_per_symbol
    time = np.arange(0, num_symbols / symbol_rate, 1 / sampling_rate)

    # Generate a random PAM4 signal
    import random
    pam4_symbols = random.choices([-3, -1, 1, 3], k=num_symbols)
    upsampled_signal = np.repeat(pam4_symbols, samples_per_symbol)

    # Simulate a channel (e.g., introduce some ISI and noise)
    def channel(signal):
        # Simple ISI: Add a delayed and attenuated version of the signal
        delayed_signal = np.concatenate(([0.0] * (samples_per_symbol // 2), signal[:- (samples_per_symbol // 2)])) * 0.5
        noisy_signal = signal + delayed_signal + 0.1 * np.random.randn(len(signal))
        return noisy_signal

    received_signal = channel(upsampled_signal)

    # DFE and CDR parameters
    num_taps = 5
    step_size_dfe = 0.005
    initial_tap_weights = np.zeros(num_taps)
    decision_levels_initial = np.array([-3, -1, 1, 3])
    learning_rate_cdr = 0.01
    filter_length_cdr = 7
    pattern_filter_threshold = 0.7 # Require a relatively strong alternating pattern

    # Run the PAM4 DFE with dLev and CDR
    equalized_signal, detected_symbols, tap_weights_history, decision_levels_history, timing_offset_history = \
        pam4_dfe_cdr(received_signal, num_taps, tap_weights_initial=initial_tap_weights,
                     step_size=step_size_dfe, decision_levels=decision_levels_initial,
                     samples_per_symbol=samples_per_symbol, learning_rate_cdr=learning_rate_cdr,
                     filter_length=filter_length_cdr, pattern_filter_threshold=pattern_filter_threshold)

    # Plotting (optional)
    time_equalized = np.linspace(0, num_symbols / symbol_rate, len(equalized_signal))
    time_symbols = np.linspace(0, num_symbols / symbol_rate, len(detected_symbols))

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_equalized, received_signal, label='Received Signal')
    plt.title('Received PAM4 Signal with ISI and Noise')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_equalized, equalized_signal, label='Equalized Signal')
    plt.plot(time_symbols, detected_symbols, 'ro', label='Detected Symbols')
    plt.title('Equalized PAM4 Signal and Detected Symbols')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(timing_offset_history)
    plt.title('Estimated Timing Offset (in samples)')
    plt.xlabel('Iteration')
    plt.ylabel('Offset')

    plt.tight_layout()
    plt.show()

    # Plot decision levels over time
    plt.figure(figsize=(10, 6))
    for i in range(len(decision_levels_initial)):
        levels = [levels_at_iter[i] for levels_at_iter in decision_levels_history]
        plt.plot(levels, label=f'Level {decision_levels_initial[i]}')
    plt.title('Decision Level Adjustment over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Decision Level')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
