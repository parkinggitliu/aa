import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

class MullerMullerCDR_PAM4_FalseLockAware:
    """
    Implements an ADC-based Muller-Muller CDR for PAM4 signals with a 
    two-stage, false-lock-aware corrective measure.
    
    Stage 1: Coarse lock using an NRZ-style slicer to avoid false lock points.
    Stage 2: Switch to a PAM4 slicer for fine tracking and data recovery.
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
        self.Kp = Kp
        self.Ki = Ki
        self.pam4_levels = np.array([-3, -1, 1, 3])
        # Reset method to re-initialize state for multiple runs
        self.reset()

    def reset(self):
        """Resets the internal state of the CDR."""
        self.integrator = 0.0
        self.loop_filter_output = 0.0
        self.nco_phase = 0.0
        self.nco_freq = 1.0 / self.sps
        self.current_symbol = 0

    def _pam4_slicer(self, x):
        """Slices the input to the nearest PAM4 level."""
        return self.pam4_levels[np.argmin(np.abs(x - self.pam4_levels))]

    def _nrz_slicer(self, x):
        """
        Slices the input to the outer PAM4 levels (-3, 3) as if it were NRZ.
        This is the key to avoiding false lock during initial acquisition.
        """
        return 3.0 if x > 0 else -3.0

    def process(self, input_samples, use_false_lock_correction=True, mode_switch_symbol=200):
        """
        Processes input samples to recover symbols and timing.

        Args:
            input_samples (np.array): The received signal samples.
            use_false_lock_correction (bool): If True, uses the two-stage locking.
            mode_switch_symbol (int): The symbol count at which to switch from NRZ to PAM4 mode.

        Returns:
            A tuple containing recovered symbols, timing error, and sample points.
        """
        self.reset()
        num_input_samples = len(input_samples)
        recovered_symbols = []
        timing_error_log = []
        sample_points = []
        
        prev_sample = 0
        prev_decision = 0
        
        i = 0
        while i < num_input_samples - 1:
            if self.nco_phase < 1.0:
                i += 1
                self.nco_phase += self.nco_freq + self.loop_filter_output
                continue

            self.nco_phase -= 1.0
            interp_sample = input_samples[i-1] + self.nco_phase * (input_samples[i] - input_samples[i-1])
            
            # --- False Lock Corrective Measure ---
            if use_false_lock_correction and self.current_symbol < mode_switch_symbol:
                # Stage 1: Use NRZ slicer for coarse lock
                current_decision = self._nrz_slicer(interp_sample)
            else:
                # Stage 2: Switch to PAM4 slicer for fine lock
                current_decision = self._pam4_slicer(interp_sample)
            # --- End of Corrective Measure ---

            error = (prev_decision * interp_sample) - (current_decision * prev_sample)
            
            self.integrator += self.Ki * error
            self.loop_filter_output = (self.Kp * error) + self.integrator
            
            prev_sample = interp_sample
            prev_decision = current_decision

            recovered_symbols.append(self._pam4_slicer(interp_sample)) # Always store PAM4 decision
            timing_error_log.append(error)
            sample_points.append(i + self.nco_phase)
            self.current_symbol += 1


            if 100 <= i <= 109 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

            if 200 <= i <= 209 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

            if 1100 <= i <= 1109 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

            if 1500 <= i <= 1709 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

            if 2100 <= i <= 2109 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

            if 3100 <= i <= 3109 :
                print(f"Recovered symbol at index {i}: {recovered_symbols[-1]}, Timing Error: {error}, Sample Point: {sample_points[-1]}")

        return np.array(recovered_symbols), np.array(timing_error_log), np.array(sample_points)

# --- Helper functions from previous simulation (generation, channel, FFE) ---
def generate_pam4_signal(num_symbols, sps, beta=0.3):
    pam4_levels = [-3, -1, 1, 3]
    symbols = np.random.choice(pam4_levels, num_symbols)
    num_taps = sps * 8
    t = np.arange(num_taps) - num_taps/2
    h_rrc = np.zeros(num_taps)
    t_nz = t[t != 0]
    h_rrc[t != 0] = (np.sin(np.pi * t_nz/sps * (1 - beta)) + 4 * beta * t_nz/sps * np.cos(np.pi * t_nz/sps * (1 + beta))) / \
            (np.pi * t_nz/sps * (1 - (4 * beta * t_nz/sps)**2))
    h_rrc[t == 0] = (1 - beta + 4 * beta / np.pi)
    if beta != 0:
        t_sing = sps / (4 * beta)
        h_rrc[np.isclose(np.abs(t), t_sing)] = beta / np.sqrt(2) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2/np.pi) * np.cos(np.pi / (4*beta)))
    h_rrc /= np.sqrt(np.sum(h_rrc**2))
    upsampled_symbols = np.zeros(num_symbols * sps)
    upsampled_symbols[::sps] = symbols
    tx_signal = lfilter(h_rrc, 1, upsampled_symbols)
    return tx_signal, symbols, h_rrc

def channel_with_isi(signal, noise_std=0.1, timing_offset_samples=0):
    noise = np.random.normal(0, noise_std, len(signal))
    return np.roll(signal + noise, timing_offset_samples)

def get_s_curve(signal, sps, use_nrz_slicer):
    """Calculates the Phase Detector characteristic (S-curve)."""
    offsets = np.linspace(-0.5, 0.5, 41) * sps
    errors = []
    pam4_slicer = lambda x: [-3,-1,1,3][np.argmin(np.abs(x-np.array([-3,-1,1,3])))]
    nrz_slicer = lambda x: 3.0 if x > 0 else -3.0
    slicer = nrz_slicer if use_nrz_slicer else pam4_slicer

    for offset in offsets:
        # Resample signal at fixed offsets
        sample_indices = np.arange(sps, len(signal) - sps, sps) + int(round(offset))
        
        # Ensure indices are within bounds
        valid_indices = (sample_indices >= 1) & (sample_indices < len(signal))
        sample_indices = sample_indices[valid_indices]
        
        samples_at_offset = signal[sample_indices]
        prev_samples = signal[sample_indices - sps]
        
        decisions_at_offset = np.array([slicer(s) for s in samples_at_offset])
        prev_decisions = np.array([slicer(s) for s in prev_samples])
        
        error_terms = (prev_decisions * samples_at_offset) - (decisions_at_offset * prev_samples)
        errors.append(np.mean(error_terms))
        
    return offsets / sps, np.array(errors)

if __name__ == '__main__':
    # --- Simulation Parameters ---
    SPS = 8
    NUM_SYMBOLS = 1000
    RRC_BETA = 0.5
    NOISE_STD_DEV = 0.1
    # Start with a significant timing error to induce false lock
    TIMING_OFFSET = int(SPS / 2) -1 # Places initial samples near transitions

    # --- Generate Signal and Pass Through Channel ---
    tx_signal, original_symbols, rrc_filter_taps = generate_pam4_signal(NUM_SYMBOLS, SPS, RRC_BETA)
    rx_signal_post_channel = channel_with_isi(tx_signal, NOISE_STD_DEV, TIMING_OFFSET)
    rx_signal = lfilter(rrc_filter_taps, 1, rx_signal_post_channel)

    # --- Run Both CDRs ---
    cdr_model = MullerMullerCDR_PAM4_FalseLockAware(sps=SPS, Kp=0.015, Ki=0.001)
    # 1. Standard CDR (fails by getting stuck)
    recovered_std, error_std, samples_std = cdr_model.process(rx_signal, use_false_lock_correction=False)
    # 2. Corrected CDR (succeeds)
    recovered_corr, error_corr, samples_corr = cdr_model.process(rx_signal, use_false_lock_correction=True, mode_switch_symbol=250)
    
    # --- Calculate S-Curves to show why false lock happens ---
    offsets, s_curve_pam4 = get_s_curve(rx_signal, SPS, use_nrz_slicer=False)
    _, s_curve_nrz = get_s_curve(rx_signal, SPS, use_nrz_slicer=True)

    # --- Visualization ---
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("CDR False Lock: Problem and Corrective Measure", fontsize=18)

    # 1. S-Curves (Phase Detector Characteristic)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(offsets, s_curve_pam4, 'r-o', label='Standard PAM4 Slicer')
    ax1.plot(offsets, s_curve_nrz, 'g-o', label='NRZ-mode Slicer')
    ax1.axhline(0, color='k', linestyle='--')
    ax1.axvline(0, color='k', linestyle=':', label='Ideal Lock Point')
    ax1.set_title("1. The Cause: Phase Detector S-Curves", fontsize=14)
    ax1.set_xlabel("Timing Offset (Symbols)")
    ax1.set_ylabel("Average Error")
    ax1.legend()
    ax1.grid(True)
    ax1.text(-0.45, 1.5, "False Lock Point!", color='r', bbox=dict(facecolor='white', alpha=0.7, edgecolor='r'))

    # 2. Eye Diagram (to show the problem space)
    ax2 = plt.subplot(2, 2, 2)
    eye_span = 2 * SPS
    for i in range(SPS, 200 * SPS, SPS): # Plot first 200 symbols
        if i + eye_span < len(rx_signal):
             ax2.plot(np.arange(eye_span), rx_signal[i:i+eye_span], 'b-', alpha=0.1)
    ax2.set_title("2. Received Signal Eye Diagram", fontsize=14)
    ax2.set_xlabel("Time (samples)")
    ax2.grid(True)
    
    # 3. NCO Control Signal (shows locking behavior)
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(np.diff(samples_std) - SPS, 'r-', label='Corrected CDR (Achieves True Lock)')
    ax3.plot(np.diff(samples_corr) - SPS, 'g-', label='Standard CDR (Stuck in False Lock)',alpha=0.7)
    ax3.axvline(250, color='k', linestyle='--', label='Mode Switch')
    ax3.set_title("3. NCO Control Signal (Locking Dynamics)", fontsize=14)
    ax3.set_xlabel("Recovered Symbol Index")
    ax3.set_ylabel("Frequency Correction")
    ax3.legend()
    ax3.grid(True)
  
   # 4. Recovered Symbol Constellations
    ax4 = plt.subplot(2, 2, 4)
    y_std = np.ones_like(recovered_std) * 0.5
    y_corr = np.ones_like(recovered_corr) * -0.5
    ax4.plot(recovered_std, y_std, 'ro', alpha=0.5, label='Standard CDR Output')
    ax4.plot(recovered_corr, y_corr, 'go', alpha=0.5, label='Corrected CDR Output')
    ax4.set_title("4. Result: Recovered Constellations", fontsize=14)
    ax4.set_xlabel("Amplitude")
    ax4.set_yticks([-0.5, 0.5], ['Corrected', 'Standard'])
    ax4.set_xticks([-3, -1, 1, 3])
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("Simulation complete. Check the plots for results.")
    