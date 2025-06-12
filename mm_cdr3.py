import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class PAM4_MM_CDR:
    """
    PAM4 Mueller-Muller Clock and Data Recovery with False-Lock Correction
    
    This implementation includes:
    - PAM4 symbol generation and transmission
    - Matched filtering
    - Mueller-Muller timing error detector
    - Loop filter for timing recovery
    - False-lock detection and correction
    - Symbol decision and error tracking
    """
    
    def __init__(self, 
                 symbol_rate=1e9,      # 1 GBaud
                 samples_per_symbol=8,  # Oversampling factor
                 loop_bandwidth=0.01,   # CDR loop bandwidth
                 damping_factor=0.707,  # Loop damping
                 false_lock_threshold=0.1,  # False lock detection threshold
                 correction_gain=0.5):     # False lock correction gain
        
        self.Rs = symbol_rate
        self.sps = samples_per_symbol
        self.Fs = symbol_rate * samples_per_symbol
        self.Ts = 1/symbol_rate
        self.dt = 1/self.Fs
        
        # CDR Loop parameters
        self.Bn = loop_bandwidth
        self.zeta = damping_factor
        self.K0 = 1.0  # VCO gain
        self.Kd = 1.0  # Detector gain
        
        # Calculate loop filter coefficients
        wn = 2 * np.pi * self.Bn
        self.alpha = 2 * self.zeta * wn
        self.beta = wn**2
        
        # False lock detection parameters
        self.false_lock_thresh = false_lock_threshold
        self.correction_gain = correction_gain
        
        # Initialize states
        self.reset_states()
        
        # PAM4 constellation points
        self.constellation = np.array([-3, -1, 1, 3])
        
    def reset_states(self):
        """Reset all internal states"""
        self.timing_error = 0
        self.loop_filter_state = 0
        self.vco_phase = 0
        self.last_sample = 0
        self.last_decision = 0
        self.error_history = []
        self.phase_history = []
        self.timing_history = []
        self.false_lock_counter = 0
        self.lock_state = 'searching'
        
    def generate_pam4_data(self, num_symbols, snr_db=20):
        """Generate PAM4 test data with AWGN"""
        # Generate random PAM4 symbols
        symbols = np.random.choice([0, 1, 2, 3], num_symbols)
        pam4_levels = self.constellation[symbols]
        
        # Create pulse shaping filter (root raised cosine)
        pulse_length = 8 * self.sps
        t = np.arange(-pulse_length//2, pulse_length//2) * self.dt
        alpha = 0.35  # Roll-off factor
        
        # Root raised cosine pulse
        rrc_pulse = np.zeros_like(t)
        for i, time in enumerate(t):
            if time == 0:
                rrc_pulse[i] = 1 - alpha + 4*alpha/np.pi
            elif abs(time) == self.Ts/(4*alpha):
                rrc_pulse[i] = (alpha/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*alpha)) + 
                                                    (1-2/np.pi)*np.cos(np.pi/(4*alpha)))
            else:
                num = np.sin(np.pi*time/self.Ts*(1-alpha)) + 4*alpha*time/self.Ts*np.cos(np.pi*time/self.Ts*(1+alpha))
                den = np.pi*time/self.Ts*(1-(4*alpha*time/self.Ts)**2)
                rrc_pulse[i] = num/den
        
        # Normalize pulse
        rrc_pulse = rrc_pulse / np.sqrt(np.sum(rrc_pulse**2))
        
        # Upsample symbols and apply pulse shaping
        upsampled = np.zeros(num_symbols * self.sps)
        upsampled[::self.sps] = pam4_levels
        
        # Convolve with pulse shaping filter
        tx_signal = np.convolve(upsampled, rrc_pulse, mode='same')
        
        # Add timing offset (simulates clock offset)
        timing_offset = 0.3  # Fraction of symbol period
        offset_samples = int(timing_offset * self.sps)
        tx_signal = np.roll(tx_signal, offset_samples)
        
        # Add AWGN
        signal_power = np.mean(tx_signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(tx_signal))
        rx_signal = tx_signal + noise
        
        return rx_signal, symbols, rrc_pulse
    
    def matched_filter(self, signal, pulse):
        """Apply matched filtering"""
        # Matched filter is time-reversed pulse
        matched_pulse = pulse[::-1]
        filtered = np.convolve(signal, matched_pulse, mode='same')
        return filtered
    
    def mueller_muller_ted(self, current_sample, previous_sample, 
                          current_decision, previous_decision):
        """
        Mueller-Muller Timing Error Detector for PAM4
        
        Error = (current_sample * previous_decision - 
                previous_sample * current_decision)
        """
        error = current_sample * previous_decision - previous_sample * current_decision
        return error
    
    def make_decision(self, sample):
        """Make PAM4 symbol decision"""
        # Find closest constellation point
        distances = np.abs(self.constellation - sample)
        decision_idx = np.argmin(distances)
        return self.constellation[decision_idx], decision_idx
    
    def detect_false_lock(self, timing_errors, window_size=100):
        """
        Detect false lock condition based on timing error statistics
        
        False lock typically occurs at:
        - Half symbol rate (timing error oscillates)
        - Quarter symbol rate 
        - Other fractional rates
        """
        if len(timing_errors) < window_size:
            return False
            
        recent_errors = timing_errors[-window_size:]
        
        # Check for oscillatory behavior (sign changes)
        sign_changes = np.sum(np.diff(np.sign(recent_errors)) != 0)
        oscillation_ratio = sign_changes / len(recent_errors)
        
        # Check for high variance (unstable lock)
        error_variance = np.var(recent_errors)
        
        # Detect false lock if high oscillation or variance
        false_lock_detected = (oscillation_ratio > 0.8 or 
                              error_variance > self.false_lock_thresh)
        
        return false_lock_detected
    
    def correct_false_lock(self):
        """Apply false lock correction by phase jump"""
        # Apply phase correction (typically π/2 or π phase shift)
        phase_correction = np.pi / 2
        self.vco_phase += phase_correction
        self.false_lock_counter += 1
        print(f"False lock detected and corrected (#{self.false_lock_counter})")
    
    def process_cdr(self, rx_signal, pulse_shape):
        """
        Main CDR processing function
        """
        # Apply matched filtering
        matched_output = self.matched_filter(rx_signal, pulse_shape)
        
        # Initialize arrays for results
        num_samples = len(matched_output)
        recovered_symbols = []
        symbol_indices = []
        timing_errors = []
        phases = []
        
        # CDR processing loop
        sample_index = 0
        symbol_count = 0
        
        while sample_index < num_samples - self.sps:
            # Calculate current sampling point based on VCO phase
            sampling_point = sample_index + self.vco_phase / (2 * np.pi) * self.sps
            sampling_point = int(np.round(sampling_point)) % num_samples
            
            # Get current sample
            current_sample = matched_output[sampling_point]
            
            # Make symbol decision
            decision_value, decision_idx = self.make_decision(current_sample)
            
            # Calculate Mueller-Muller timing error
            if symbol_count > 0:  # Need previous sample for comparison
                timing_error = self.mueller_muller_ted(
                    current_sample, self.last_sample,
                    decision_value, self.last_decision
                )
                
                # Store timing error
                timing_errors.append(timing_error)
                self.timing_history.append(timing_error)
                
                # Loop filter (PI controller)
                self.loop_filter_state += self.beta * timing_error
                filter_output = self.alpha * timing_error + self.loop_filter_state
                
                # VCO update
                self.vco_phase += self.K0 * filter_output
                
                # Keep phase in reasonable range
                self.vco_phase = np.mod(self.vco_phase, 2 * np.pi)
                
                # False lock detection and correction
                if len(self.timing_history) > 50 and symbol_count % 50 == 0:
                    if self.detect_false_lock(self.timing_history):
                        self.correct_false_lock()
                        self.lock_state = 'correcting'
                    elif abs(np.mean(self.timing_history[-50:])) < 0.01:
                        self.lock_state = 'locked'
                    else:
                        self.lock_state = 'searching'
            
            # Store results
            recovered_symbols.append(decision_idx)
            symbol_indices.append(sampling_point)
            phases.append(self.vco_phase)
            
            # Update states for next iteration
            self.last_sample = current_sample
            self.last_decision = decision_value
            
            # Advance to next symbol
            sample_index += self.sps
            symbol_count += 1
        
        return {
            'recovered_symbols': np.array(recovered_symbols),
            'symbol_indices': np.array(symbol_indices),
            'timing_errors': np.array(timing_errors),
            'phases': np.array(phases),
            'matched_output': matched_output,
            'lock_state': self.lock_state,
            'false_lock_corrections': self.false_lock_counter
        }
    
    def calculate_ber(self, original_symbols, recovered_symbols):
        """Calculate Bit Error Rate"""
        # Align sequences (account for delays)
        min_len = min(len(original_symbols), len(recovered_symbols))
        
        # Find best alignment by correlation
        corr = np.correlate(original_symbols[:min_len//2], 
                           recovered_symbols[:min_len], mode='valid')
        best_offset = np.argmax(np.abs(corr))
        
        # Compare aligned sequences
        orig_aligned = original_symbols[:min_len-best_offset]
        recv_aligned = recovered_symbols[best_offset:best_offset+len(orig_aligned)]
        
        errors = np.sum(orig_aligned != recv_aligned)
        ber = errors / len(orig_aligned)
        
        return ber, best_offset
    
    def plot_results(self, results, original_symbols=None):
        """Plot CDR performance results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Matched filter output and sampling points
        axes[0,0].plot(results['matched_output'], 'b-', alpha=0.7, label='Matched Filter Output')
        axes[0,0].plot(results['symbol_indices'], 
                      results['matched_output'][results['symbol_indices']], 
                      'ro', markersize=4, label='Sampling Points')
        axes[0,0].set_title('Matched Filter Output with Sampling Points')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Plot 2: Timing errors
        axes[0,1].plot(results['timing_errors'], 'g-', linewidth=1)
        axes[0,1].set_title(f'Timing Errors (False Lock Corrections: {results["false_lock_corrections"]})')
        axes[0,1].set_ylabel('Timing Error')
        axes[0,1].grid(True)
        
        # Plot 3: VCO Phase
        axes[1,0].plot(results['phases'], 'm-', linewidth=1)
        axes[1,0].set_title('VCO Phase Evolution')
        axes[1,0].set_ylabel('Phase (radians)')
        axes[1,0].grid(True)
        
        # Plot 4: Symbol constellation
        recovered_levels = self.constellation[results['recovered_symbols']]
        axes[1,1].scatter(range(len(recovered_levels)), recovered_levels, 
                         alpha=0.6, s=20, c='blue')
        axes[1,1].set_title('Recovered Symbol Constellation')
        axes[1,1].set_ylabel('PAM4 Level')
        axes[1,1].set_ylim([-4, 4])
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Performance summary
        print(f"\n=== CDR Performance Summary ===")
        print(f"Lock State: {results['lock_state']}")
        print(f"False Lock Corrections: {results['false_lock_corrections']}")
        print(f"Final Timing Error RMS: {np.sqrt(np.mean(results['timing_errors'][-100:]**2)):.4f}")
        
        if original_symbols is not None:
            ber, offset = self.calculate_ber(original_symbols, results['recovered_symbols'])
            print(f"Bit Error Rate: {ber:.2e}")
            print(f"Alignment Offset: {offset} symbols")

# Example usage and demonstration
def main():
    """Demonstrate PAM4 Mueller-Muller CDR with false-lock correction"""
    
    print("PAM4 Mueller-Muller CDR with False-Lock Correction")
    print("=" * 50)
    
    # Create CDR instance
    cdr = PAM4_MM_CDR(
        symbol_rate=1e9,           # 1 GBaud
        samples_per_symbol=8,      # 8x oversampling
        loop_bandwidth=0.005,      # 0.5% bandwidth
        damping_factor=0.707,      # Critical damping
        false_lock_threshold=0.05,  # False lock sensitivity
        correction_gain=0.5
    )
    
    # Generate test data
    num_symbols = 2000
    snr_db = 15
    
    print(f"Generating {num_symbols} PAM4 symbols at {snr_db} dB SNR...")
    rx_signal, original_symbols, pulse_shape = cdr.generate_pam4_data(num_symbols, snr_db)
    
    print("Processing CDR...")
    results = cdr.process_cdr(rx_signal, pulse_shape)
    
    # Display and plot results
    cdr.plot_results(results, original_symbols)
    
    return cdr, results, original_symbols

if __name__ == "__main__":
    cdr, results, original_symbols = main()