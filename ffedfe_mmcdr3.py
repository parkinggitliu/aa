import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PAM4Receiver:
    def __init__(self, ffe_taps=5, dfe_taps=3, mu_lms=0.01, mu_timing=0.001, mu_dlev=0.001):
        """
        PAM4 Receiver with FFE/DFE equalization and Mueller-Müller timing recovery
        
        Parameters:
        -----------
        ffe_taps : int, number of FFE taps
        dfe_taps : int, number of DFE taps
        mu_lms : float, LMS adaptation step size for equalizers
        mu_timing : float, timing recovery loop gain
        mu_dlev : float, data level adaptation step size
        """
        # FFE parameters
        self.ffe_taps = ffe_taps
        self.ffe_weights = np.zeros(ffe_taps)
        self.ffe_weights[ffe_taps//2] = 1.0  # Initialize with cursor at center
        self.ffe_buffer = np.zeros(ffe_taps)
        
        # DFE parameters
        self.dfe_taps = dfe_taps
        self.dfe_weights = np.zeros(dfe_taps)
        self.dfe_buffer = np.zeros(dfe_taps)
        
        # LMS parameters
        self.mu_lms = mu_lms
        
        # Timing recovery parameters
        self.mu_timing = mu_timing
        self.timing_error = 0
        self.interpolator_phase = 0
        self.samples_per_symbol = 2  # Assume 2x oversampling
        
        # PAM4 levels (normalized)
        self.pam4_levels = np.array([-3, -1, 1, 3]) / 3.0
        self.dlev = np.copy(self.pam4_levels)  # Adaptive data levels
        self.mu_dlev = mu_dlev
        
        # History buffers for Mueller-Müller
        self.prev_symbol = 0
        self.prev_decision = 0
        
    def interpolate(self, samples, mu):
        """
        Cubic interpolation for timing recovery
        
        Parameters:
        -----------
        samples : array, input samples
        mu : float, interpolation phase (0 to 1)
        """
        if len(samples) < 4:
            return samples[1] if len(samples) > 1 else 0
        
        # Farrow structure cubic interpolator
        v = samples[:4]
        c = np.array([
            -v[0] + 3*v[1] - 3*v[2] + v[3],
            2*v[0] - 5*v[1] + 4*v[2] - v[3],
            -v[0] + v[2],
            v[1]
        ]) / 6.0
        
        return c[0]*mu**3 + c[1]*mu**2 + c[2]*mu + c[3]
    
    def ffe_filter(self, sample):
        """Apply FFE filtering"""
        # Shift buffer and add new sample
        self.ffe_buffer = np.roll(self.ffe_buffer, 1)
        self.ffe_buffer[0] = sample
        
        # Apply FFE
        return np.dot(self.ffe_weights, self.ffe_buffer)
    
    def dfe_feedback(self):
        """Calculate DFE feedback"""
        return np.dot(self.dfe_weights, self.dfe_buffer)
    
    def slicer(self, sample):
        """PAM4 slicer with adaptive levels"""
        # Find closest PAM4 level
        distances = np.abs(self.dlev - sample)
        decision_idx = np.argmin(distances)
        return self.dlev[decision_idx], decision_idx
    
    def pam4_to_bits(self, symbol_idx):
        """Convert PAM4 symbol index to 2 bits"""
        return [(symbol_idx >> 1) & 1, symbol_idx & 1]
    
    def update_dlev(self, sample, decision, decision_idx):
        """Update adaptive data levels"""
        error = sample - decision
        self.dlev[decision_idx] += self.mu_dlev * error
    
    def lms_update(self, error):
        """LMS adaptation for FFE and DFE"""
        # Update FFE weights
        self.ffe_weights += self.mu_lms * error * self.ffe_buffer
        
        # Update DFE weights
        self.dfe_weights += self.mu_lms * error * self.dfe_buffer
    
    def mueller_muller_ted(self, current_sample, current_decision):
        """
        Mueller-Müller timing error detector for PAM4
        
        Returns timing error estimate
        """
        # PAM4 Mueller-Müller: e[n] = a[n-1]*(y[n]-a[n]) - a[n]*(y[n-1]-a[n-1])
        timing_error = (self.prev_decision * (current_sample - current_decision) - 
                       current_decision * (self.prev_symbol - self.prev_decision))
        
        # Update history
        self.prev_symbol = current_sample
        self.prev_decision = current_decision
        
        return timing_error
    
    def process_sample(self, samples, sample_idx):
        """
        Process one baud-rate sample through the receiver
        
        Parameters:
        -----------
        samples : array, input samples
        sample_idx : int, current sample index
        
        Returns:
        --------
        bits : decoded bits (2 bits for PAM4)
        metrics : dict with various metrics
        """
        # Interpolate at current timing phase
        if sample_idx + 4 < len(samples):
            interpolated = self.interpolate(
                samples[sample_idx:sample_idx+4], 
                self.interpolator_phase
            )
        else:
            interpolated = samples[sample_idx]
        
        # FFE filtering
        ffe_out = self.ffe_filter(interpolated)
        
        # DFE feedback subtraction
        dfe_fb = self.dfe_feedback()
        equalizer_out = ffe_out - dfe_fb
        
        # Slicer decision
        decision, decision_idx = self.slicer(equalizer_out)
        
        # Update adaptive data levels
        self.update_dlev(equalizer_out, decision, decision_idx)
        
        # Calculate error for LMS
        error = equalizer_out - decision
        
        # LMS adaptation
        self.lms_update(error)
        
        # Update DFE buffer with decision
        self.dfe_buffer = np.roll(self.dfe_buffer, 1)
        self.dfe_buffer[0] = decision
        
        # Timing error detection
        ted_error = self.mueller_muller_ted(equalizer_out, decision)
        
        # Update timing
        self.timing_error = ted_error
        self.interpolator_phase += self.mu_timing * ted_error
        
        # Wrap phase
        if self.interpolator_phase >= 1.0:
            self.interpolator_phase -= 1.0
        elif self.interpolator_phase < 0:
            self.interpolator_phase += 1.0
        
        # Convert to bits
        bits = self.pam4_to_bits(decision_idx)
        
        # Collect metrics
        metrics = {
            'ffe_out': ffe_out,
            'equalizer_out': equalizer_out,
            'decision': decision,
            'error': error,
            'timing_error': ted_error,
            'phase': self.interpolator_phase,
            'dlev': self.dlev.copy(),
            'ffe_weights': self.ffe_weights.copy(),
            'dfe_weights': self.dfe_weights.copy()
        }
        
        return bits, metrics
    
    def process_signal(self, samples, symbols_to_process=None):
        """
        Process entire signal through receiver
        
        Parameters:
        -----------
        samples : array, input samples
        symbols_to_process : int, number of symbols to process (None for all)
        
        Returns:
        --------
        bits : recovered bits
        metrics_history : dict of metric arrays
        """
        if symbols_to_process is None:
            symbols_to_process = len(samples) // self.samples_per_symbol
        
        # Initialize output arrays
        bits = []
        metrics_history = {
            'ffe_out': [],
            'equalizer_out': [],
            'decisions': [],
            'errors': [],
            'timing_errors': [],
            'phases': [],
            'dlev': [],
            'ffe_weights': [],
            'dfe_weights': []
        }
        
        # Process samples
        sample_idx = 0
        for symbol in range(symbols_to_process):
            # Calculate actual sample index based on timing
            actual_idx = int(sample_idx + self.interpolator_phase)
            
            if actual_idx + 4 >= len(samples):
                break
            
            # Process one symbol
            symbol_bits, metrics = self.process_sample(samples, actual_idx)
            
            # Store results
            bits.extend(symbol_bits)
            for key in metrics_history:
                if key in metrics:
                    metrics_history[key].append(metrics[key])
            
            # Advance by one symbol period
            sample_idx += self.samples_per_symbol
        
        # Convert lists to arrays
        for key in metrics_history:
            metrics_history[key] = np.array(metrics_history[key])
        
        return np.array(bits), metrics_history


def generate_pam4_signal(num_symbols, channel_taps=None, snr_db=30):
    """
    Generate PAM4 test signal with optional channel and noise
    
    Parameters:
    -----------
    num_symbols : int, number of PAM4 symbols
    channel_taps : array, channel impulse response (None for no channel)
    snr_db : float, signal-to-noise ratio in dB
    
    Returns:
    --------
    samples : array, generated samples (2x oversampled)
    symbols : array, transmitted PAM4 symbols
    bits : array, transmitted bits
    """
    # Generate random bits
    bits = np.random.randint(0, 2, size=num_symbols*2)
    
    # Convert to PAM4 symbols
    symbols = np.zeros(num_symbols)
    pam4_map = {(0,0): -3/3, (0,1): -1/3, (1,0): 1/3, (1,1): 3/3}
    
    for i in range(num_symbols):
        bit_pair = (bits[2*i], bits[2*i+1])
        symbols[i] = pam4_map[bit_pair]
    
    # Upsample to 2x
    upsampled = np.zeros(num_symbols * 2)
    upsampled[::2] = symbols
    
    # Apply pulse shaping (raised cosine)
    span = 8
    sps = 2
    alpha = 0.3
    t = np.arange(-span*sps/2, span*sps/2 + 1) / sps
    h = np.sinc(t) * np.cos(np.pi*alpha*t) / (1 - (2*alpha*t)**2 + 1e-10)
    h = h / np.sum(h)
    
    samples = signal.convolve(upsampled, h, mode='same')
    
    # Apply channel if provided
    if channel_taps is not None:
        samples = signal.convolve(samples, channel_taps, mode='same')
    
    # Add noise
    signal_power = np.mean(samples**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(samples))
    samples += noise
    
    return samples, symbols, bits


def plot_results(metrics_history, num_symbols_to_plot=1000):
    """Plot receiver performance metrics"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle('PAM4 Receiver Performance', fontsize=14)
    
    # Eye diagram
    ax = axes[0, 0]
    samples_per_symbol = 2
    eye_length = 3 * samples_per_symbol
    num_traces = min(500, len(metrics_history['equalizer_out']) // eye_length)
    
    for i in range(num_traces):
        start = i * eye_length
        trace = metrics_history['equalizer_out'][start:start+eye_length]
        if len(trace) == eye_length:
            ax.plot(np.arange(eye_length) % samples_per_symbol, trace, 'b', alpha=0.1)
    
    ax.set_title('Eye Diagram (Equalizer Output)')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Equalizer output over time
    ax = axes[0, 1]
    ax.plot(metrics_history['equalizer_out'][:num_symbols_to_plot], alpha=0.7)
    ax.axhline(y=-1, color='r', linestyle='--', alpha=0.5, label='PAM4 levels')
    ax.axhline(y=-1/3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1/3, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
    ax.set_title('Equalizer Output')
    ax.set_xlabel('Symbol Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    # Timing error
    ax = axes[1, 0]
    ax.plot(metrics_history['timing_errors'][:num_symbols_to_plot])
    ax.set_title('Timing Error')
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Error')
    ax.grid(True, alpha=0.3)
    
    # Timing phase
    ax = axes[1, 1]
    ax.plot(metrics_history['phases'][:num_symbols_to_plot])
    ax.set_title('Interpolator Phase')
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Phase')
    ax.grid(True, alpha=0.3)
    
    # FFE weights
    ax = axes[2, 0]
    if len(metrics_history['ffe_weights']) > 0:
        final_ffe = metrics_history['ffe_weights'][-1]
        ax.stem(range(len(final_ffe)), final_ffe)
        ax.set_title('FFE Weights (Final)')
        ax.set_xlabel('Tap')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
    
    # Adaptive levels
    ax = axes[2, 1]
    if len(metrics_history['dlev']) > 0:
        dlev_history = np.array(metrics_history['dlev'])
        for i in range(4):
            ax.plot(dlev_history[:num_symbols_to_plot, i], label=f'Level {i}')
        ax.set_title('Adaptive Data Levels')
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Simulation parameters
    num_symbols = 10000
    
    # Channel model (example: mild ISI)
    channel_taps = np.array([0.1, 0.7, 0.9, 0.3, 0.1])
    channel_taps = channel_taps / np.sum(channel_taps)
    
    # Generate test signal
    print("Generating PAM4 signal...")
    samples, symbols, tx_bits = generate_pam4_signal(
        num_symbols, 
        channel_taps=channel_taps, 
        snr_db=25
    )
    
    # Create receiver
    print("Initializing receiver...")
    receiver = PAM4Receiver(
        ffe_taps=9,
        dfe_taps=4,
        mu_lms=0.005,
        mu_timing=0.0005,
        mu_dlev=0.001
    )
    
    # Process signal
    print("Processing signal...")
    rx_bits, metrics = receiver.process_signal(samples)
    
    # Calculate BER
    min_length = min(len(tx_bits), len(rx_bits))
    ber = np.mean(tx_bits[:min_length] != rx_bits[:min_length])
    print(f"Bit Error Rate: {ber:.6f}")
    
    # Plot results
    print("Plotting results...")
    plot_results(metrics)
    
    # Print final equalizer state
    print("\nFinal Equalizer State:")
    print(f"FFE Weights: {metrics['ffe_weights'][-1]}")
    print(f"DFE Weights: {metrics['dfe_weights'][-1]}")
    print(f"Data Levels: {metrics['dlev'][-1]}")