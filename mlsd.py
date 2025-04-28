import numpy as np

def mlse(received_signal, possible_symbols, channel_response, noise_variance):
    """
    Maximum Likelihood Sequence Estimation (MLSE) using the Viterbi Algorithm.

    Args:
        received_signal (np.ndarray): The received signal samples.
        possible_symbols (np.ndarray): An array of the possible transmitted symbols (alphabet).
        channel_response (np.ndarray): The impulse response of the channel.
        noise_variance (float): The variance of the additive white Gaussian noise.

    Returns:
        np.ndarray: The most likely transmitted sequence of symbols.
    """
    num_received_samples = len(received_signal)
    num_possible_symbols = len(possible_symbols)
    channel_length = len(channel_response)

    # Trellis states are defined by the last channel_length - 1 transmitted symbols.
    num_states = num_possible_symbols**(channel_length - 1)
    state_map = {}  # Map integer state index to the sequence of symbols
    reverse_state_map = {} # Map sequence of symbols to integer state index
    state_counter = 0

    def generate_states(k, current_state):
        nonlocal state_counter
        if k == channel_length - 1:
            state_tuple = tuple(current_state)
            state_map[state_counter] = state_tuple
            reverse_state_map[state_tuple] = state_counter
            state_counter += 1
            return

        for symbol in possible_symbols:
            generate_states(k + 1, current_state + [symbol])

    if channel_length > 1:
        generate_states(0, [])
    else:
        num_states = 1
        state_map[0] = tuple()
        reverse_state_map[tuple()] = 0

    # Initialize metrics and predecessors
    metrics = np.full((num_received_samples + 1, num_states), np.inf)
    predecessors = np.zeros((num_received_samples + 1, num_states), dtype=int)

    # Initial state (assuming we start from a known state, e.g., all zeros if in possible_symbols)
    initial_state = tuple([possible_symbols[0]] * (channel_length - 1)) if channel_length > 1 else tuple()
    start_state_index = reverse_state_map.get(initial_state, 0) # Default to state 0 if initial not found
    metrics[0, start_state_index] = 0

    # Viterbi algorithm
    for t in range(1, num_received_samples + 1):
        for current_state_index in range(num_states):
            current_state_history = list(state_map[current_state_index])
            for next_symbol_index, next_symbol in enumerate(possible_symbols):
                # Hypothetical transmitted sequence leading to the current state and next symbol
                hypothetical_sequence = current_state_history + [next_symbol]

                # Calculate the expected received sample
                if len(hypothetical_sequence) >= channel_length:
                    relevant_symbols = hypothetical_sequence[-(channel_length):]
                    expected_received = np.convolve(relevant_symbols, channel_response, 'valid')[0]
                else:
                    # Handle the initial transient where we don't have enough history
                    padding_length = channel_length - len(hypothetical_sequence)
                    padded_sequence = [possible_symbols[0]] * padding_length + hypothetical_sequence # Assume initial zeros
                    expected_received = np.convolve(padded_sequence, channel_response, 'valid')[0]


                # Calculate the branch metric (Euclidean distance squared)
                branch_metric = np.abs(received_signal[t - 1] - expected_received)**2

                # Calculate the path metric
                previous_state_history = hypothetical_sequence[:-1]
                if channel_length > 1:
                    previous_state_tuple = tuple(previous_state_history[-(channel_length - 1):])
                else:
                    previous_state_tuple = tuple()
                previous_state_index = reverse_state_map.get(previous_state_tuple, 0)

                path_metric = metrics[t - 1, previous_state_index] + branch_metric

                # Update the minimum metric and predecessor
                if path_metric < metrics[t, current_state_index]:
                    metrics[t, current_state_index] = path_metric
                    predecessors[t, current_state_index] = previous_state_index

    # Backtracking to find the most likely sequence
    most_likely_sequence = np.zeros(num_received_samples, dtype=possible_symbols.dtype)
    final_state = np.argmin(metrics[num_received_samples, :])

    for t in range(num_received_samples - 1, -1, -1):
        state_history = list(state_map[final_state])
        most_likely_sequence[t] = state_history[-1] if state_history else possible_symbols[0] # Last symbol of the state
        final_state = predecessors[t + 1, final_state]

    return most_likely_sequence

if __name__ == '__main__':
    # Example usage:
    possible_symbols = np.array([-1, 1])  # BPSK modulation
    channel_response = np.array([0.5, 1, 0.3])  # Example channel
    transmitted_sequence = np.array([1, -1, 1, 1, -1, -1, 1])
    noise_variance = 0.1
    np.random.seed(42)
    noise = np.sqrt(noise_variance) * np.random.randn(len(transmitted_sequence) + len(channel_response) - 1)
    received_signal_full = np.convolve(transmitted_sequence, channel_response, 'full') + noise
    received_signal = received_signal_full[:len(transmitted_sequence)] # Consider only the part corresponding to transmitted symbols

    detected_sequence = mlse(received_signal, possible_symbols, channel_response, noise_variance)

    print("Transmitted Sequence:", transmitted_sequence)
    print("Received Signal:", received_signal)
    print("Detected Sequence (MLSE):", detected_sequence)

    # Example with PAM4:
    possible_symbols_pam4 = np.array([-3, -1, 1, 3])
    channel_response_pam4 = np.array([0.8, 0.4])
    transmitted_sequence_pam4 = np.array([1, -3, 3, -1, 1])
    noise_variance_pam4 = 0.2
    noise_pam4 = np.sqrt(noise_variance_pam4) * np.random.randn(len(transmitted_sequence_pam4) + len(channel_response_pam4) - 1)
    received_signal_pam4_full = np.convolve(transmitted_sequence_pam4, channel_response_pam4, 'full') + noise_pam4
    received_signal_pam4 = received_signal_pam4_full[:len(transmitted_sequence_pam4)]

    detected_sequence_pam4 = mlse(received_signal_pam4, possible_symbols_pam4, channel_response_pam4, noise_variance_pam4)

    print("\nTransmitted PAM4 Sequence:", transmitted_sequence_pam4)
    print("Received PAM4 Signal:", received_signal_pam4)
    print("Detected PAM4 Sequence (MLSE):", detected_sequence_pam4)