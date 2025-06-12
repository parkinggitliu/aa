import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PAM4Receiver:
    """
    A class to simulate a PAM4 receiver with adaptive equalization and clock recovery.

    This class includes:
    - A Feed-Forward Equalizer (FFE) to handle pre-cursor ISI.
    - A Decision-Feedback Equalizer (DFE) to handle post-cursor ISI.
    - FFE and DFE tap weights adapted using the Least Mean Squares (LMS) algorithm.
    - A Müller-Müller Clock and Data Recovery (CDR) circuit for baud-rate timing recovery.
    - dLev error calculation for PAM4 signals.
    """

    def __init__(self,
                 num_symbols=20000,
                 sps=8,
                 baud_rate=1e9,
                 beta_rrc=0.3,
                 n_taps_ffe=5,
                 n_taps_dfe=3,
                 mu_ffe=0.005,
                 mu_dfe=0.001,
                 mu_cdr=0.01,
                 noise_variance=0.015,
                 channel_impulse_response=np.array([0.9, -0.5, 0.2])):
        """
        Initializes the simulation parameters and receiver state.
        """
        # --- System Parameters ---
        self.num_symbols = num_symbols
        self.sps = sps  # Samples per symbol
        self.baud_rate = baud_rate
        self.beta_rrc = beta_rrc  # Roll-off factor for RRC filter

        # --- Equalizer Parameters ---
        self.n_taps_ffe = n_taps_ffe
        self.n_taps_dfe = n_taps_dfe
        self.mu_ffe = mu_ffe  # FFE learning rate (LMS step size)
        self.mu_dfe = mu_dfe  # DFE learning rate
        
        # --- CDR Parameters ---
        self.mu_cdr = mu_cdr  # CDR learning rate (phase adjustment)

        # --- Channel Parameters ---
        self.noise_variance = noise_variance
        self.channel_ir = channel_impulse_response

        # --- Internal State & History ---
        self.tx_signal = None
        self.original_symbols = None
        self.rx_signal = None
        self.ffe_taps = np.zeros(self.n_taps_ffe)
        self.ffe_taps[self.n_taps_ffe // 2] = 1.0  # Center tap initialization
        self.dfe_taps = np.zeros(self.n_taps_dfe)
        self.history = {
            'strobe_indices': [],
            'recovered_symbols': [],
            'equalized_outputs': [],
            'errors': [],
            'ffe_taps': [],
            'dfe_taps': [],
            'mu_cdr_vals': []
        }

    @staticmethod
    def _pam4_map(bits):
        """Maps binary data to PAM4 levels [-3, -1, 1, 3]."""
        return bits * 2 - 3

    @staticmethod
    def _pam4_slicer(x):
        """Slices a value to the nearest PAM4 level."""
        levels = np.array([-3, -1, 1, 3])
        idx = (np.abs(x - levels)).argmin()
        return levels[idx]

    @staticmethod
    def _dlev_pam4(equalized_output):
        """
        Calculates dLev (decision-level) error for PAM4.
        This is the distance from the equalized sample to the ideal level it was sliced to.
        """
        # Ideal levels are [-3, -1, 1, 3]
        # Slicer thresholds are [-2, 0, 2]
        if equalized_output > 2:
            return equalized_output - 3
        elif equalized_output > 0:
            return equalized_output - 1
        elif equalized_output > -2:
            return equalized_output - (-1)
        else:
            return equalized_output - (-3)

    def _generate_signal(self):
        """Generates a random PAM4 signal with RRC pulse shaping."""
        symbols = np.random.choice([-3, -1, 1, 3], size=self.num_symbols)
        x = np.zeros(self.num_symbols * self.sps)
        x[::self.sps] = symbols

        # Create the RRC filter
        num_taps_rrc = 10 * self.sps + 1
        t = (np.arange(num_taps_rrc) - num_taps_rrc // 2) / self.sps
        # Handle division by zero and the (2*beta*t)**2 == 1 case
        t[t == 0] = 1e-8 
        denom = 1 - (2 * self.beta_rrc * t)**2
        denom[denom == 0] = 1e-8
        rrc_filter = (np.sinc(t) * np.cos(np.pi * self.beta_rrc * t)) / denom
        
        self.tx_signal = signal.convolve(x, rrc_filter, mode='same')
        self.original_symbols = symbols

    def _apply_channel(self):
        """Applies channel impulse response and adds Gaussian noise."""
        if self.tx_signal is None:
            raise ValueError("Signal not generated yet.")
        
        # Convolve with channel and trim to original length
        rx_conv = signal.convolve(self.tx_signal, self.channel_ir, mode='full')
        self.rx_signal = rx_conv[:len(self.tx_signal)]
        
        # Add Additive White Gaussian Noise (AWGN)
        noise = np.sqrt(self.noise_variance) * np.random.randn(len(self.rx_signal))
        self.rx_signal += noise

    def process_signal(self):
        """
        Main processing loop for the receiver.
        This method runs the CDR and adaptive equalizer.
        """
        # 1. Generate signal and apply channel effects
        self._generate_signal()
        self._apply_channel()

        # 2. Initialize buffers and CDR state
        ffe_buffer = np.zeros(self.n_taps_ffe)
        dfe_buffer = np.zeros(self.n_taps_dfe)
        
        # CDR variables
        mu = 0.5  # Fractional interval, starts at mid-point
        strobe_idx = 0 # Integer part of the strobe position
        
        # Process the signal, leaving buffer for filter taps and look-ahead
        for i in range(len(self.rx_signal) - self.sps * (self.n_taps_ffe + 5)):
            # 3. CDR - Interpolate sample at the current strobe point
            # This is the main sampling point for the current symbol.
            main_idx = int(np.round(strobe_idx + mu))
            if main_idx >= len(self.rx_signal): break
            current_sample = self.rx_signal[main_idx]

            # 4. FFE Filtering
            ffe_buffer = np.roll(ffe_buffer, 1)
            ffe_buffer[0] = current_sample
            ffe_output = np.dot(self.ffe_taps, ffe_buffer)

            # 5. DFE Filtering
            dfe_output = np.dot(self.dfe_taps, dfe_buffer)
            
            # 6. Combiner Output
            eq_out = ffe_output - dfe_output
            
            # 7. Slicing and Decision
            decision = self._pam4_slicer(eq_out)
            
            # 8. Error Calculation for LMS
            error = self._dlev_pam4(eq_out)

            # 9. Tap Adaptation (LMS Algorithm)
            self.ffe_taps -= self.mu_ffe * error * ffe_buffer
            self.dfe_taps += self.mu_dfe * error * dfe_buffer # Note: DFE uses '+'
            
            # Update DFE buffer with the current decision for the next symbol's feedback
            dfe_buffer = np.roll(dfe_buffer, 1)
            dfe_buffer[0] = decision
            
            # 10. Müller-Müller CDR Update
            # Get the sample halfway to the *next* strobe point for phase detection.
            halfway_idx = int(np.round(strobe_idx + mu + self.sps / 2))
            if halfway_idx >= len(self.rx_signal): break
            
            # To calculate the phase error, we need the equalized value at the halfway point.
            # This requires a temporary FFE run on the halfway sample. DFE buffer is unchanged.
            halfway_ffe_buffer = np.roll(ffe_buffer, 1) # Use a temp buffer
            halfway_ffe_buffer[0] = self.rx_signal[halfway_idx]
            halfway_eq_out = np.dot(self.ffe_taps, halfway_ffe_buffer) - np.dot(self.dfe_taps, dfe_buffer)
            
            # Müller-Müller Phase Detector logic
            # error = (current decision * halfway_eq_out) - (halfway decision * current_eq_out)
            phase_error = decision * halfway_eq_out - self._pam4_slicer(halfway_eq_out) * eq_out
            mu -= self.mu_cdr * phase_error # Update fractional interval
            
            # Store history for plotting
            self.history['strobe_indices'].append(main_idx)
            self.history['recovered_symbols'].append(decision)
            self.history['equalized_outputs'].append(eq_out)
            self.history['errors'].append(error)
            self.history['ffe_taps'].append(self.ffe_taps.copy())
            self.history['dfe_taps'].append(self.dfe_taps.copy())
            self.history['mu_cdr_vals'].append(mu)

            # 11. Advance to the next symbol's approximate strobe position
            strobe_idx += self.sps

    def plot_results(self):
        """
        Plots various performance metrics of the receiver.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(15, 18))
        
        skip = 5000 # Skip initial transient phase for clearer plots

        # 1. Unequalized Eye Diagram
        ax1 = fig.add_subplot(3, 2, 1)
        for i in range(skip, skip + 2000):
            start = i * self.sps - self.sps
            end = i * self.sps + self.sps
            if end < len(self.rx_signal):
                ax1.plot(np.arange(-self.sps, self.sps), self.rx_signal[start:end], 'b-', alpha=0.05)
        ax1.set_title("1. Unequalized Eye Diagram (Channel Output)")
        ax1.set_xlabel("Sample Index relative to Symbol Center")
        ax1.set_ylabel("Amplitude")

        # 2. Equalized Eye Diagram
        ax2 = fig.add_subplot(3, 2, 2)
        eq_outputs = self.history['equalized_outputs']
        strobes = self.history['strobe_indices']
        reconstructed_signal = np.interp(np.arange(len(self.rx_signal)), strobes, eq_outputs)
        for i in range(skip, len(strobes) - 2000):
            strobe = strobes[i]
            start = strobe - self.sps
            end = strobe + self.sps
            if start > 0 and end < len(self.rx_signal):
                # We need to reconstruct the signal shape around the strobe point
                # For simplicity, we just plot the raw signal, but centered by the recovered clock
                 ax2.plot(np.arange(-self.sps, self.sps), self.rx_signal[start:end], 'g-', alpha=0.05)
        ax2.set_title("2. Equalized Eye Diagram (at Receiver)")
        ax2.set_xlabel("Sample Index relative to Recovered Clock")

        # 3. FFE Tap Weights Convergence
        ax3 = fig.add_subplot(3, 2, 3)
        ffe_taps_history = np.array(self.history['ffe_taps'])
        for i in range(self.n_taps_ffe):
            ax3.plot(ffe_taps_history[skip:, i], label=f'Tap {i}')
        ax3.set_title("3. FFE Tap Weights Convergence")
        ax3.set_xlabel("Symbol Index")
        ax3.set_ylabel("Weight")
        ax3.legend()

        # 4. Error Signal
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(self.history['errors'][skip:], 'r-', alpha=0.7, label='dLev Error')
        # Plot a moving average to see the trend
        error_ma = np.convolve(self.history['errors'][skip:], np.ones(100)/100, mode='valid')
        ax4.plot(error_ma, 'k-', linewidth=2, label='Moving Average')
        ax4.set_title("4. dLev Error Signal (for LMS)")
        ax4.set_xlabel("Symbol Index")
        ax4.set_ylabel("Error")
        ax4.legend()
        ax4.grid(True, which='both')

        # 5. Constellation Diagram
        ax5 = fig.add_subplot(3, 2, 5)
        ax5.scatter(self.history['recovered_symbols'][skip:], self.history['equalized_outputs'][skip:], 
                    alpha=0.1, s=5)
        ax5.set_title("5. Equalized Constellation Diagram")
        ax5.set_xlabel("Decided Symbol (Ideal)")
        ax5.set_ylabel("Equalizer Output (Actual)")
        ax5.set_xticks([-3, -1, 1, 3])
        ax5.grid(True)

        # 6. CDR Fractional Interval `mu`
        ax6 = fig.add_subplot(3, 2, 6)
        ax6.plot(self.history['mu_cdr_vals'][skip:])
        ax6.set_title("6. CDR Fractional Interval ($\mu$) Convergence")
        ax6.set_xlabel("Symbol Index")
        ax6.set_ylabel("Fractional Interval Value")
        ax6.grid(True)

        plt.tight_layout()
        plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Instantiate the receiver
    # You can experiment with these parameters
    receiver = PAM4Receiver(
        num_symbols=30000,
        sps=16, # Higher SPS makes visualization clearer
        n_taps_ffe=7,
        n_taps_dfe=4,
        mu_ffe=0.003,
        mu_dfe=0.0005,
        mu_cdr=0.005,
        noise_variance=0.01,
        channel_impulse_response=np.array([0.85, -0.4, 0.25, -0.1]) # More challenging channel
    )
    
    # Run the simulation
    print("Processing signal...")
    receiver.process_signal()
    print("Processing complete.")
    
    # Print final tap values
    print("\n--- Final Tap Weights ---")
    print(f"Final FFE Taps: {[f'{x:.4f}' for x in receiver.ffe_taps]}")
    print(f"Final DFE Taps: {[f'{x:.4f}' for x in receiver.dfe_taps]}")

    # Plot the results
    print("\nPlotting results...")
    receiver.plot_results()