import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class PAM4_Receiver_Equalizer:
    """
    Complete PAM4 Receiver with:
    - Feed-Forward Equalizer (FFE) with LMS adaptation
    - Decision Feedback Equalizer (DFE) with LMS adaptation
    - Mueller-Muller Clock and Data Recovery (CDR)
    - Integrated timing recovery and equalization
    """
    
    def __init__(self,
                 # System parameters
                 symbol_rate=1e9,
                 samples_per_symbol=4,
                 
                 # FFE parameters
                 ffe_taps=15,
                 ffe_step_size=1e-4,
                 
                 # DFE parameters  
                 dfe_taps=10,
                 dfe_step_size=1e-4,
                 
                 # CDR parameters
                 cdr_bandwidth=0.01,
                 cdr_damping=0.707,
                 
                 # Training parameters
                 training_length=500,
                 decision_directed=True):
        
        # System setup
        self.Rs = symbol_rate
        self.sps = samples_per_symbol
        self.Fs = symbol_rate * samples_per_symbol
        self.Ts = 1/symbol_rate
        self.dt = 1/self.Fs
        
        # FFE parameters
        self.N_ffe = ffe_taps
        self.mu_ffe = ffe_step_size
        self.ffe_weights = np.zeros(self.N_ffe)
        self.ffe_weights[self.N_ffe//2] = 1.0  # Initialize center tap to 1
        self.ffe_buffer = np.zeros(self.N_ffe)
        
        # DFE parameters
        self.N_dfe = dfe_taps
        self.mu_dfe = dfe_step_size
        self.dfe_weights = np.zeros(self.N_dfe)
        self.dfe_buffer = np.zeros(self.N_dfe)
        
        # CDR parameters
        self.cdr_bn = cdr_bandwidth
        self.cdr_zeta = cdr_damping
        wn = 2 * np.pi * self.cdr_bn
        self.cdr_alpha = 2 * self.cdr_zeta * wn
        self.cdr_beta = wn**2
        
        # Training parameters
        self.training_len = training_length
        self.decision_directed = decision_directed
        
        # PAM4 constellation
        self.constellation = np.array([-3, -1, 1, 3])
        self.constellation_normalized = self.constellation / np.sqrt(np.mean(self.constellation**2))
        
        # Initialize states
        self.reset_receiver()
        
    def reset_receiver(self):
        """Reset all receiver states"""
        # CDR states
        self.cdr_phase = 0
        self.cdr_filter_state = 0
        self.last_sample = 0
        self.last_decision = 0
        
        # Equalizer states
        self.ffe_buffer.fill(0)
        self.dfe_buffer.fill(0)
        
        # Performance tracking
        self.mse_history = []
        self.ffe_weight_history = []
        self.dfe_weight_history = []
        self.timing_error_history = []
        self.cdr_phase_history = []
        
        # Symbol tracking
        self.symbol_count = 0
        self.training_mode = True
        
    def generate_channel_with_isi(self, channel_taps=None):
        """Generate a channel with ISI for testing"""
        if channel_taps is None:
            # Example channel with pre-cursor, main, and post-cursor ISI
            channel_taps = np.array([0.1, 0.2, 1.0, 0.3, 0.15, 0.08])
        
        # Normalize channel
        channel_taps = channel_taps / np.sqrt(np.sum(channel_taps**2))
        return channel_taps
    
    def generate_pam4_signal(self, num_symbols, snr_db=20, channel_taps=None):
        """Generate PAM4 signal with ISI channel and noise"""
        # Generate random PAM4 symbols
        symbols = np.random.choice([0, 1, 2, 3], num_symbols)
        pam4_levels = self.constellation[symbols]
        
        # Create pulse shaping filter (root raised cosine)
        pulse_length = 8 * self.sps
        t = np.arange(-pulse_length//2, pulse_length//2) * self.dt
        alpha = 0.35  # Roll-off factor
        
        # Root raised cosine pulse
        rrc_pulse = self.create_rrc_pulse(t, alpha)
        
        # Upsample symbols
        upsampled = np.zeros(num_symbols * self.sps)
        upsampled[::self.sps] = pam4_levels
        
        # Apply pulse shaping
        tx_signal = np.convolve(upsampled, rrc_pulse, mode='same')
        
        # Apply channel with ISI
        if channel_taps is None:
            channel_taps = self.generate_channel_with_isi()
        
        # Upsample channel to match signal sampling rate
        channel_upsampled = np.zeros(len(channel_taps) * self.sps)
        channel_upsampled[::self.sps] = channel_taps
        
        # Apply channel
        rx_signal = np.convolve(tx_signal, channel_upsampled, mode='same')
        
        # Add matched filter (receiver filter)
        rx_signal = np.convolve(rx_signal, rrc_pulse, mode='same')
        
        # Add timing offset
        timing_offset = 0.25  # Fraction of symbol period
        offset_samples = int(timing_offset * self.sps)
        rx_signal = np.roll(rx_signal, offset_samples)
        
        # Add AWGN
        signal_power = np.mean(rx_signal**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(rx_signal))
        rx_signal += noise
        
        return rx_signal, symbols, channel_taps
    
    def create_rrc_pulse(self, t, alpha):
        """Create root raised cosine pulse"""
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
                if den != 0:
                    rrc_pulse[i] = num/den
        
        # Normalize
        rrc_pulse = rrc_pulse / np.sqrt(np.sum(rrc_pulse**2))
        return rrc_pulse
    
    def ffe_filter(self, input_sample):
        """Apply Feed-Forward Equalizer"""
        # Shift buffer and add new sample
        self.ffe_buffer[1:] = self.ffe_buffer[:-1]
        self.ffe_buffer[0] = input_sample
        
        # Compute FFE output
        ffe_output = np.dot(self.ffe_weights, self.ffe_buffer)
        return ffe_output
    
    def dfe_filter(self, decision_feedback=None):
        """Apply Decision Feedback Equalizer"""
        if decision_feedback is not None:
            # Shift buffer and add new decision
            self.dfe_buffer[1:] = self.dfe_buffer[:-1]
            self.dfe_buffer[0] = decision_feedback
        
        # Compute DFE output (subtract ISI from past decisions)
        dfe_output = np.dot(self.dfe_weights, self.dfe_buffer)
        return dfe_output
    
    def make_decision(self, sample):
        """Make PAM4 symbol decision"""
        distances = np.abs(self.constellation - sample)
        decision_idx = np.argmin(distances)
        return self.constellation[decision_idx], decision_idx
    
    def lms_update_ffe(self, error, input_sample):
        """Update FFE weights using LMS algorithm"""
        # Update weights: w(n+1) = w(n) - μ * error * x(n)
        self.ffe_weights -= self.mu_ffe * error * self.ffe_buffer
        
    def lms_update_dfe(self, error):
        """Update DFE weights using LMS algorithm"""
        # Update weights: w(n+1) = w(n) + μ * error * d(n)
        # Note: positive sign because DFE subtracts ISI
        self.dfe_weights += self.mu_dfe * error * self.dfe_buffer
    
    def mueller_muller_error(self, current_sample, previous_sample,
                           current_decision, previous_decision):
        """Compute Mueller-Muller timing error"""
        error = current_sample * previous_decision - previous_sample * current_decision
        return error
    
    def update_cdr(self, timing_error):
        """Update CDR loop filter and VCO"""
        # PI loop filter
        self.cdr_filter_state += self.cdr_beta * timing_error
        filter_output = self.cdr_alpha * timing_error + self.cdr_filter_state
        
        # Update VCO phase
        self.cdr_phase += filter_output
        self.cdr_phase = np.mod(self.cdr_phase, 2 * np.pi)
    
    def interpolate_sample(self, signal, base_index, fractional_delay):
        """Interpolate sample at fractional delay"""
        if base_index + 1 >= len(signal):
            return signal[base_index]
        
        # Linear interpolation
        frac = fractional_delay - int(fractional_delay)
        sample = (1 - frac) * signal[base_index] + frac * signal[base_index + 1]
        return sample
    
    def process_receiver(self, rx_signal, training_symbols=None):
        """
        Main receiver processing function combining equalization and CDR
        """
        num_samples = len(rx_signal)
        
        # Results storage
        recovered_symbols = []
        equalized_samples = []
        timing_errors = []
        mse_values = []
        
        # Processing loop
        sample_index = self.sps  # Start after first symbol period
        
        while sample_index < num_samples - self.sps:
            # CDR: Calculate sampling point based on VCO phase
            fractional_delay = self.cdr_phase / (2 * np.pi)
            sampling_point = sample_index + fractional_delay * self.sps
            base_idx = int(sampling_point)
            frac_delay = sampling_point - base_idx
            
            if base_idx >= num_samples - 1:
                break
                
            # Interpolate sample at optimal timing point
            current_sample = self.interpolate_sample(rx_signal, base_idx, frac_delay)
            
            # Apply FFE
            ffe_output = self.ffe_filter(current_sample)
            
            # Apply DFE (subtract post-cursor ISI)
            dfe_output = self.dfe_filter()
            
            # Combined equalizer output
            equalized_sample = ffe_output - dfe_output
            equalized_samples.append(equalized_sample)
            
            # Make decision
            decision_value, decision_idx = self.make_decision(equalized_sample)
            recovered_symbols.append(decision_idx)
            
            # Compute error for adaptation
            if self.training_mode and training_symbols is not None and self.symbol_count < len(training_symbols):
                # Training mode: use known symbols
                target_symbol = self.constellation[training_symbols[self.symbol_count]]
                error = target_symbol - equalized_sample
            else:
                # Decision-directed mode
                error = decision_value - equalized_sample
                if self.symbol_count >= self.training_len:
                    self.training_mode = False
            
            # LMS adaptation
            self.lms_update_ffe(error, current_sample)
            self.lms_update_dfe(error)
            
            # Update DFE buffer with current decision
            self.dfe_filter(decision_value)
            
            # CDR: Compute timing error using Mueller-Muller
            if self.symbol_count > 0:
                timing_error = self.mueller_muller_error(
                    current_sample, self.last_sample,
                    decision_value, self.last_decision
                )
                timing_errors.append(timing_error)
                self.timing_error_history.append(timing_error)
                
                # Update CDR
                self.update_cdr(timing_error)
            
            # Store performance metrics
            mse = error**2
            mse_values.append(mse)
            self.mse_history.append(mse)
            
            # Store weight histories (decimated for memory)
            if self.symbol_count % 10 == 0:
                self.ffe_weight_history.append(self.ffe_weights.copy())
                self.dfe_weight_history.append(self.dfe_weights.copy())
                self.cdr_phase_history.append(self.cdr_phase)
            
            # Update states for next iteration
            self.last_sample = current_sample
            self.last_decision = decision_value
            self.symbol_count += 1
            
            # Advance to next symbol
            sample_index += self.sps
        
        return {
            'recovered_symbols': np.array(recovered_symbols),
            'equalized_samples': np.array(equalized_samples),
            'timing_errors': np.array(timing_errors),
            'mse_history': np.array(mse_values),
            'ffe_weights_final': self.ffe_weights.copy(),
            'dfe_weights_final': self.dfe_weights.copy(),
            'ffe_weight_evolution': np.array(self.ffe_weight_history),
            'dfe_weight_evolution': np.array(self.dfe_weight_history),
            'cdr_phase_evolution': np.array(self.cdr_phase_history),
            'training_mode_final': self.training_mode,
            'num_symbols_processed': self.symbol_count
        }
    
    def calculate_performance_metrics(self, original_symbols, recovered_symbols):
        """Calculate various performance metrics"""
        # Align sequences
        min_len = min(len(original_symbols), len(recovered_symbols))
        
        # Find best alignment
        if min_len > 100:
            corr = np.correlate(original_symbols[:min_len//2], 
                               recovered_symbols[:min_len], mode='valid')
            best_offset = np.argmax(np.abs(corr))
        else:
            best_offset = 0
        
        # Calculate BER
        end_idx = min(len(original_symbols), len(recovered_symbols) - best_offset)
        orig_aligned = original_symbols[:end_idx]
        recv_aligned = recovered_symbols[best_offset:best_offset + end_idx]
        
        errors = np.sum(orig_aligned != recv_aligned)
        ber = errors / len(orig_aligned) if len(orig_aligned) > 0 else 1.0
        
        # Calculate MSE
        final_mse = np.mean(self.mse_history[-100:]) if len(self.mse_history) > 100 else np.mean(self.mse_history)
        
        # Calculate timing error RMS
        timing_rms = np.sqrt(np.mean(np.array(self.timing_error_history[-100:])**2)) if len(self.timing_error_history) > 100 else 0
        
        return {
            'ber': ber,
            'mse': final_mse,
            'timing_rms': timing_rms,
            'alignment_offset': best_offset,
            'convergence_symbols': len(self.mse_history)
        }
    
    def plot_results(self, results, original_symbols=None, channel_taps=None):
        """Plot comprehensive receiver performance results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Plot 1: Equalized constellation
        eq_samples = results['equalized_samples']
        axes[0,0].scatter(range(len(eq_samples)), eq_samples, alpha=0.6, s=10)
        axes[0,0].axhline(y=-3, color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=-1, color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[0,0].axhline(y=3, color='r', linestyle='--', alpha=0.5)
        axes[0,0].set_title('Equalized Constellation')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].grid(True)
        
        # Plot 2: MSE convergence
        axes[0,1].semilogy(results['mse_history'])
        axes[0,1].set_title('MSE Convergence')
        axes[0,1].set_ylabel('Mean Square Error')
        axes[0,1].grid(True)
        
        # Plot 3: Timing errors
        if len(results['timing_errors']) > 0:
            axes[0,2].plot(results['timing_errors'])
            axes[0,2].set_title('Mueller-Muller Timing Errors')
            axes[0,2].set_ylabel('Timing Error')
            axes[0,2].grid(True)
        
        # Plot 4: FFE weight evolution
        if len(results['ffe_weight_evolution']) > 0:
            for i in range(min(5, len(results['ffe_weights_final']))):
                axes[1,0].plot(results['ffe_weight_evolution'][:, i], 
                              label=f'Tap {i}', alpha=0.7)
            axes[1,0].set_title('FFE Weight Evolution')
            axes[1,0].set_ylabel('Weight Value')
            axes[1,0].legend()
            axes[1,0].grid(True)
        
        # Plot 5: Final FFE weights
        axes[1,1].stem(range(len(results['ffe_weights_final'])), 
                      results['ffe_weights_final'])
        axes[1,1].set_title('Final FFE Weights')
        axes[1,1].set_ylabel('Weight Value')
        axes[1,1].grid(True)
        
        # Plot 6: Final DFE weights
        if np.any(results['dfe_weights_final']):
            axes[1,2].stem(range(len(results['dfe_weights_final'])), 
                          results['dfe_weights_final'])
            axes[1,2].set_title('Final DFE Weights')
            axes[1,2].set_ylabel('Weight Value')
            axes[1,2].grid(True)
        
        # Plot 7: CDR Phase evolution
        if len(results['cdr_phase_evolution']) > 0:
            axes[2,0].plot(results['cdr_phase_evolution'])
            axes[2,0].set_title('CDR Phase Evolution')
            axes[2,0].set_ylabel('Phase (radians)')
            axes[2,0].grid(True)
        
        # Plot 8: Channel response (if available)
        if channel_taps is not None:
            axes[2,1].stem(range(len(channel_taps)), channel_taps)
            axes[2,1].set_title('Channel Impulse Response')
            axes[2,1].set_ylabel('Amplitude')
            axes[2,1].grid(True)
        
        # Plot 9: Symbol comparison (if original available)
        if original_symbols is not None:
            comparison_len = min(100, len(original_symbols), len(results['recovered_symbols']))
            axes[2,2].plot(original_symbols[:comparison_len], 'b-', label='Original', alpha=0.7)
            axes[2,2].plot(results['recovered_symbols'][:comparison_len], 'r--', label='Recovered', alpha=0.7)
            axes[2,2].set_title('Symbol Comparison (First 100)')
            axes[2,2].legend()
            axes[2,2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print performance summary
        if original_symbols is not None:
            metrics = self.calculate_performance_metrics(original_symbols, results['recovered_symbols'])
            print(f"\n=== Receiver Performance Summary ===")
            print(f"Bit Error Rate: {metrics['ber']:.2e}")
            print(f"Final MSE: {metrics['mse']:.4f}")
            print(f"Timing Error RMS: {metrics['timing_rms']:.4f}")
            print(f"Training Mode: {'Active' if results['training_mode_final'] else 'Decision-Directed'}")
            print(f"Symbols Processed: {results['num_symbols_processed']}")
            print(f"FFE Taps: {len(results['ffe_weights_final'])}")
            print(f"DFE Taps: {len(results['dfe_weights_final'])}")

# Demonstration function
def main():
    """Demonstrate PAM4 receiver with FFE/DFE equalization and MM CDR"""
    
    print("PAM4 Receiver with FFE/DFE Equalization and Mueller-Muller CDR")
    print("=" * 65)
    
    # Create receiver
    receiver = PAM4_Receiver_Equalizer(
        symbol_rate=1e9,          # 1 GBaud
        samples_per_symbol=4,     # 4x oversampling
        ffe_taps=15,              # 15-tap FFE
        ffe_step_size=1e-3,       # FFE adaptation rate
        dfe_taps=8,               # 8-tap DFE
        dfe_step_size=1e-3,       # DFE adaptation rate
        cdr_bandwidth=0.01,       # 1% CDR bandwidth
        cdr_damping=0.707,        # Critical damping
        training_length=300,      # Training symbols
        decision_directed=True    # Enable decision-directed mode
    )
    
    # Generate test signal with ISI channel
    num_symbols = 2000
    snr_db = 18
    
    print(f"Generating {num_symbols} PAM4 symbols at {snr_db} dB SNR...")
    print("Channel includes ISI (pre-cursor, main tap, post-cursors)")
    
    rx_signal, original_symbols, channel_taps = receiver.generate_pam4_signal(
        num_symbols, snr_db)
    
    print(f"Channel taps: {channel_taps}")
    
    # Process through receiver
    print("\nProcessing through adaptive receiver...")
    print("- FFE: Feed-forward equalization with LMS adaptation")
    print("- DFE: Decision feedback equalization with LMS adaptation") 
    print("- CDR: Mueller-Muller timing recovery")
    
    # Use first part of symbols for training
    training_symbols = original_symbols[:receiver.training_len]
    
    results = receiver.process_receiver(rx_signal, training_symbols)
    
    # Display results
    receiver.plot_results(results, original_symbols, channel_taps)
    
    return receiver, results, original_symbols, channel_taps

if __name__ == "__main__":
    receiver, results, original_symbols, channel_taps = main()