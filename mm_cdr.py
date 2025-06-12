import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

class MullerMullerCDR_PAM4:
    """
    A class for an ADC-based Muller-Muller Clock and Data Recovery (CDR)
    for PAM4 signals, featuring a PI loop filter and a digital NCO with
    linear interpolation.
    """
    def __init__(self, sps, Kp=0.005, Ki=0.0001):
        """
        Initializes the CDR object.

        Args:
            sps (int): Samples per symbol.
            Kp (float): Proportional gain of the PI loop filter.
            Ki (float): Integral gain of the PI loop filter.
        """
        self.sps = sps
        # PI Loop Filter parameters and state
        self.Kp = Kp
        self.Ki = Ki
        self.integrator = 0.0
        self.loop_filter_output = 0.0
        
        # NCO parameters and state
        self.nco_phase = 0.0
        self.nco_freq = 1.0 / self.sps # Nominal frequency
        
        # Slicer for PAM4 levels
        self.pam4_levels = np.array([-3, -1, 1, 3])

    def _slicer(self, x):
        """Slices the input to the nearest PAM4 level."""
        # This is a simple slicer, more advanced ones might adapt thresholds
        return self.pam4_levels[np.argmin(np.abs(x - self.pam4_levels))]

    def process(self, input_samples):
        """
        Processes a block of input samples to recover symbols and timing.

        Args:
            input_samples (np.array): The received signal samples.

        Returns:
            tuple: A tuple containing:
                - np.array: The recovered symbols.
                - np.array: The timing error signal.
                - np.array: The sample points chosen by the NCO.
        """
        num_input_samples = len(input_samples)
        recovered_symbols = []
        timing_error_log = []
        sample_points = []
        
        # Initialize with dummy previous values for the first iteration
        prev_sample = 0
        prev_decision = 0
        
        # Main processing loop
        i = 0
        while i < num_input_samples - 1:
            # Check for a strobe: when NCO phase wraps around
            if self.nco_phase < 1.0:
                i += 1
                self.nco_phase += self.nco_freq + self.loop_filter_output
                continue

            # NCO phase has wrapped, time to produce a sample
            self.nco_phase -= 1.0
            
            # Linear Interpolation to get the sample at the precise time
            # We need the sample right before (i-1) and after (i) the strobe
            interp_sample = input_samples[i-1] + self.nco_phase * (input_samples[i] - input_samples[i-1])
            
            # Make a decision on the interpolated sample
            current_decision = self._slicer(interp_sample)
            
            # Muller-Muller timing error detector for PAM4
            # This is a common formulation that works well.
            # It correlates the decision at time k-1 with the sample at time k,
            # and vice-versa.
            error = (prev_decision * interp_sample) - (current_decision * prev_sample)
            
            # PI Loop Filter
            self.integrator += self.Ki * error
            self.loop_filter_output = (self.Kp * error) + self.integrator
            
            # Update state for the next iteration
            prev_sample = interp_sample
            prev_decision = current_decision

            # Log data for analysis
            recovered_symbols.append(current_decision)
            timing_error_log.append(error)
            # Store the exact (float) index for plotting
            sample_points.append(i + self.nco_phase) 

        return np.array(recovered_symbols), np.array(timing_error_log), np.array(sample_points)


def generate_pam4_signal(num_symbols, sps, beta=0.3):
    """Generates a pulse-shaped PAM4 signal."""
    # 1. Generate random PAM4 symbols
    pam4_levels = [-3, -1, 1, 3]
    symbols = np.random.choice(pam4_levels, num_symbols)
    
    # 2. Create a Root-Raised Cosine (RRC) filter
    num_taps = sps * 8
    t = np.arange(-num_taps/2, num_taps/2)
    h_rrc = (np.sin(np.pi * t/sps * (1 - beta)) + 4 * beta * t/sps * np.cos(np.pi * t/sps * (1 + beta))) / \
            (np.pi * t/sps * (1 - (4 * beta * t/sps)**2))
    h_rrc[t == 0] = (1 - beta + 4 * beta / np.pi)
    h_rrc[t == sps / (4 * beta)] = beta / np.sqrt(2) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2/np.pi) * np.cos(np.pi / (4*beta)))
    h_rrc[t == -sps / (4 * beta)] = h_rrc[t == sps / (4*beta)]
    h_rrc /= np.sqrt(np.sum(h_rrc**2))
    
    # 3. Upsample and filter
    upsampled_symbols = np.zeros(num_symbols * sps)
    upsampled_symbols[::sps] = symbols
    tx_signal = lfilter(h_rrc, 1, upsampled_symbols)
    
    return tx_signal, symbols, h_rrc

def channel(signal, noise_std=0.1, timing_offset_samples=0):
    """Simulates a simple channel with noise and timing offset."""
    # Add Additive White Gaussian Noise (AWGN)
    noise = np.random.normal(0, noise_std, len(signal))
    # Introduce a timing offset
    noisy_signal = signal + noise
    return np.roll(noisy_signal, timing_offset_samples)


if __name__ == '__main__':
    # --- Simulation Parameters ---
    SPS = 8               # Samples per symbol
    NUM_SYMBOLS = 1000    # Number of symbols to simulate
    RRC_BETA = 0.5        # Roll-off factor for the RRC filter
    NOISE_STD_DEV = 0.15  # Noise level
    TIMING_OFFSET = 3     # Timing offset in samples

    # --- Generate Signal ---
    tx_signal, original_symbols, rrc_filter_taps = generate_pam4_signal(NUM_SYMBOLS, SPS, RRC_BETA)

    # --- Channel ---
    # Apply channel impairments
    rx_signal_no_timing_filter = channel(tx_signal, NOISE_STD_DEV, TIMING_OFFSET)
    # Apply matched filter at the receiver
    rx_signal = lfilter(rrc_filter_taps, 1, rx_signal_no_timing_filter)

    # --- Clock and Data Recovery ---
    cdr = MullerMullerCDR_PAM4(sps=SPS, Kp=0.01, Ki=0.0005)
    recovered_syms, timing_error, sample_idx = cdr.process(rx_signal)
    
    # --- Visualization ---
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("ADC-Based PAM4 Muller-Muller CDR Simulation", fontsize=16)

    # 1. Received Signal with Recovered Sample Points
    ax1 = plt.subplot(3, 2, (1, 2))
    plot_limit = 200 # samples
    ax1.plot(rx_signal[:plot_limit], 'b-', label='Received Signal (post-RRC)')
    # Plot only the sample points within the plot limit
    valid_samples = sample_idx[sample_idx < plot_limit]
    ax1.plot(valid_samples, rx_signal[valid_samples.astype(int)], 'ro', markersize=8, label='Recovered Sample Instants')
    ax1.set_title("Received Signal and CDR Sample Points")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    # 2. Timing Error
    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(timing_error)
    ax2.set_title("Timing Error vs. Symbol")
    ax2.set_xlabel("Recovered Symbol Index")
    ax2.set_ylabel("Error Value")
    ax2.grid(True)
    
    # 3. Loop Filter Output (NCO Control)
    ax3 = plt.subplot(3, 2, 4)
    # We need to re-run to get the loop filter output, or modify class to store it
    # For simplicity, we can approximate it from the phase change
    nco_control = np.diff(sample_idx) - SPS
    ax3.plot(nco_control)
    ax3.set_title("NCO Frequency Control Signal")
    ax3.set_xlabel("Recovered Symbol Index")
    ax3.set_ylabel("Freq. Adjustment")
    ax3.grid(True)

    # 4. Eye Diagram
    ax4 = plt.subplot(3, 2, 5)
    eye_span = 2 * SPS
    for i in range(SPS, len(rx_signal) - eye_span, eye_span):
         ax4.plot(rx_signal[i:i+eye_span], 'b-', alpha=0.1)
    ax4.set_title("Eye Diagram of Received Signal")
    ax4.set_xlabel("Time (samples)")
    ax4.set_ylabel("Amplitude")
    ax4.grid(True)
    
    # 5. Recovered Symbol Constellation
    ax5 = plt.subplot(3, 2, 6)
    ax5.plot(recovered_syms, np.zeros_like(recovered_syms), 'o')
    ax5.set_title("Recovered Symbol Constellation")
    ax5.set_xlabel("In-Phase")
    ax5.set_yticks([])
    ax5.set_xticks([-3, -1, 1, 3])
    ax5.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()