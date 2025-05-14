import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def generate_channel_response(main_tap_position, precursor_taps, postcursor_taps, tap_spacing=1):
    """
    Generates a channel impulse response with precursor and post-cursor components.

    Args:
        main_tap_position (int): The time index of the main channel tap.
        precursor_taps (list of tuples): (amplitude, offset) for precursor taps (negative offset).
        postcursor_taps (list of tuples): (amplitude, offset) for post-cursor taps (positive offset).
        tap_spacing (int): The spacing between consecutive taps.

    Returns:
        tuple: (time_axis, impulse_response)
    """
    min_offset = min([offset for _, offset in precursor_taps] + [0])
    max_offset = max([offset for _, offset in postcursor_taps] + [0])
    start_time = main_tap_position + min_offset * tap_spacing
    end_time = main_tap_position + max_offset * tap_spacing
    num_samples = int((end_time - start_time) / tap_spacing) + 1
    time_axis = np.linspace(start_time, end_time, num_samples)
    impulse_response = np.zeros_like(time_axis, dtype=float)

    main_tap_index = np.where(np.isclose(time_axis, main_tap_position))[0]
    if main_tap_index.size > 0:
        impulse_response[main_tap_index[0]] = 1.0

    for amplitude, offset in precursor_taps:
        tap_time = main_tap_position + offset * tap_spacing
        tap_index = np.where(np.isclose(time_axis, tap_time))[0]
        if tap_index.size > 0:
            impulse_response[tap_index[0]] += amplitude

    for amplitude, offset in postcursor_taps:
        tap_time = main_tap_position + offset * tap_spacing
        tap_index = np.where(np.isclose(time_axis, tap_time))[0]
        if tap_index.size > 0:
            impulse_response[tap_index[0]] += amplitude

    return time_axis, impulse_response

def convolve_signal_with_channel(input_signal, channel_response):
    """
    Convolves an input signal with a channel impulse response.

    Args:
        input_signal (np.ndarray): The input signal.
        channel_response (np.ndarray): The channel impulse response.

    Returns:
        np.ndarray: The output of the convolution.
    """
    output_signal = convolve(input_signal, channel_response, mode='full')
    return output_signal

# --- Example Usage ---
if __name__ == "__main__":
    # Channel parameters
    main_position = 5
    precursors = [(0.3, -1), (0.1, -2)]
    postcursors = [(0.004, 1), (0.002, 3)]
    spacing = 1

    # Generate the channel impulse response
    channel_time, channel_impulse_response = generate_channel_response(
        main_position, precursors, postcursors, spacing
    )

    # Create an example input signal (e.g., a pulse)
    input_signal_length = 20
    input_signal = np.zeros(input_signal_length)
    input_signal[5] = 1.0  # A pulse from index 5 to 9

    # Perform the convolution
    output_signal = convolve_signal_with_channel(input_signal, channel_impulse_response)

    # Determine the time axis for the output signal
    output_time = np.arange(len(output_signal))

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.stem(channel_time, channel_impulse_response, basefmt="k-")
    plt.title("Channel Impulse Response")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.stem(np.arange(len(input_signal)), input_signal, basefmt="k-")
    plt.title("Input Signal")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(output_time, output_signal)
    plt.title("Output of Convolution")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()