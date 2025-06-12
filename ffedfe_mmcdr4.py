import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from collections import deque

class PAM4Receiver:
    def __init__(self, 
                 ffe_taps=7, 
                 dfe_taps=5, 
                 mu_ffe=1e-4, 
                 mu_dfe=1e-4,
                 mu_cdr=1e-3,
                 sps=4,  # samples per symbol
                 alpha=0.1):  # loop filter coefficient for CDR
        
        # Equalizer parameters
        self.ffe_taps = ffe_taps
        self.dfe_taps = dfe_taps
        self.mu_ffe = mu_ffe  # LMS step size for FFE
        self.mu_dfe = mu_dfe  # LMS step size for DFE
        
        # CDR parameters
        self.mu_cdr = mu_cdr  # CDR loop gain
        self.sps = sps
        self.alpha = alpha
        
        # Initialize equalizer coefficients
        self.ffe_coeffs = np.zeros(ffe_taps)
        self.ffe_coeffs[ffe_taps//2] = 1.0  # Center tap initialized to 1
        self.dfe_coeffs = np.zeros(dfe_taps)
        
        # Delay lines
        self.ffe_delay_line = deque(maxlen=ffe_taps)
        self.dfe_delay_line = deque(maxlen=dfe_taps)
        
        # Initialize delay lines with zeros
        for _ in range(ffe_taps):
            self.ffe_delay_line.append(0.0)
        for _ in range(dfe_taps):
            self.dfe_delay_line.append(0.0)
        
        # CDR variables
        self.timing_error = 0.0
        self.loop_filter_state = 0.0
        self.sample_phase = 0.0
        self.prev_sample = 0.0
        
        # PAM4 levels (normalized)
        self.pam4_levels = np.array([-3, -1, 1, 3])
        
        # Data level detection
        self.level_history = deque(maxlen=1000)  # For level adaptation
        
    def pam4_slicer(self, sample):
        """PAM4 decision slicer with data level detection (dLev)"""
        # Find closest PAM4 level
        distances = np.abs(sample - self.pam4_levels)
        decision_idx = np.argmin(distances)
        decision = self.pam4_levels[decision_idx]
        
        # Store for level adaptation
        self.level_history.append(sample)
        
        return decision, decision_idx
    
    def adapt_levels(self):
        """Adaptive data level detection (dLev) using histogram analysis"""
        if len(self.level_history) < 100:
            return
        
        # Convert to numpy array for processing
        samples = np.array(list(self.level_history))
        
        # Create histogram
        hist, bin_edges = np.histogram(samples, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find peaks in histogram (should correspond to PAM4 levels)
        peaks, _ = signal.find_peaks(hist, height=np.max(hist) * 0.1)
        
        if len(peaks) >= 3:  # Need at least 3 peaks for PAM4
            peak_positions = bin_centers[peaks]
            peak_positions = np.sort(peak_positions)
            
            # Update PAM4 levels based on detected peaks
            if len(peak_positions) >= 4:
                self.pam4_levels = peak_positions[:4]
            elif len(peak_positions) == 3:
                # Interpolate missing level
                spacing = np.mean(np.diff(peak_positions))
                if peak_positions[0] - spacing > np.min(samples):
                    self.pam4_levels = np.array([peak_positions[0] - spacing, 
                                               peak_positions[0], 
                                               peak_positions[1], 
                                               peak_positions[2]])
                else:
                    self.pam4_levels = np.array([peak_positions[0], 
                                               peak_positions[1], 
                                               peak_positions[2], 
                                               peak_positions[2] + spacing])
    
    def mueller_muller_cdr(self, current_sample, previous_sample, 
                          current_decision, previous_decision):
        """Mueller-Muller Clock and Data Recovery"""
        # Mueller-Muller timing error detector
        timing_error = (current_sample - previous_sample) * previous_decision - \
                      (previous_sample - current_sample) * current_decision
        
        # Loop filter (first-order)
        self.loop_filter_state = (1 - self.alpha) * self.loop_filter_state + \
                                self.alpha * timing_error
        
        # Update sample phase
        self.sample_phase += self.mu_cdr * self.loop_filter_state
        
        return timing_error
    
    def interpolate_sample(self, samples, fractional_delay):
        """Linear interpolation for fractional sample timing"""
        if len(samples) < 2:
            return samples[-1] if samples else 0.0
        
        # Simple linear interpolation
        int_delay = int(fractional_delay)
        frac_delay = fractional_delay - int_delay
        
        if int_delay >= len(samples) - 1:
            return samples[-1]
        
        return samples[-(int_delay+1)] * (1 - frac_delay) + \
               samples[-(int_delay+2)] * frac_delay
    
    def process_sample(self, input_sample):
        """Process a single input sample through the receiver"""
        # Add new sample to FFE delay line
        self.ffe_delay_line.appendleft(input_sample)
        
        # FFE filtering
        ffe_output = np.dot(list(self.ffe_delay_line), self.ffe_coeffs)
        
        # DFE filtering (subtract ISI from previous decisions)
        dfe_output = np.dot(list(self.dfe_delay_line), self.dfe_coeffs)
        
        # Equalized sample
        equalized_sample = ffe_output - dfe_output
        
        # PAM4 decision
        decision, decision_idx = self.pam4_slicer(equalized_sample)
        
        # Add decision to DFE delay line
        self.dfe_delay_line.appendleft(decision)
        
        # Calculate error for LMS adaptation
        error = equalized_sample - decision
        
        # LMS adaptation for FFE
        ffe_input = np.array(list(self.ffe_delay_line))
        ffe_gradient = error * ffe_input
        self.ffe_coeffs -= self.mu_ffe * ffe_gradient
        
        # LMS adaptation for DFE
        dfe_input = np.array(list(self.dfe_delay_line))
        dfe_gradient = error * dfe_input
        self.dfe_coeffs -= self.mu_dfe * dfe_gradient
        
        # Mueller-Muller CDR
        if hasattr(self, 'prev_decision'):
            timing_error = self.mueller_muller_cdr(equalized_sample, 
                                                  self.prev_sample,
                                                  decision, 
                                                  self.prev_decision)
        else:
            timing_error = 0.0
        
        # Store for next iteration
        self.prev_sample = equalized_sample
        self.prev_decision = decision
        
        # Periodically adapt PAM4 levels
        if len(self.level_history) % 100 == 0:
            self.adapt_levels()
        
        return {
            'equalized': equalized_sample,
            'decision': decision,
            'decision_idx': decision_idx,
            'error': error,
            'timing_error': timing_error,
            'ffe_coeffs': self.ffe_coeffs.copy(),
            'dfe_coeffs': self.dfe_coeffs.copy(),
            'pam4_levels': self.pam4_levels.copy()
        }
    
    def process_block(self, input_samples):
        """Process a block of samples"""
        results = []
        for sample in input_samples:
            result = self.process_sample(sample)
            results.append(result)
        return results

# Example usage and simulation
def generate_pam4_signal(symbols, sps=4, pulse_shape='rrc', alpha=0.35):
    """Generate PAM4 signal with pulse shaping"""
    # PAM4 symbol mapping
    pam4_levels = np.array([-3, -1, 1, 3])
    signal_samples = pam4_levels[symbols]
    
    # Upsample
    upsampled = np.zeros(len(signal_samples) * sps)
    upsampled[::sps] = signal_samples
    
    # Pulse shaping (simple raised cosine)
    if pulse_shape == 'rrc':
        # Simple approximation of RRC filter
        t = np.arange(-20, 21) / sps
        h = np.sinc(t) * np.cos(np.pi * alpha * t) / (1 - (2 * alpha * t)**2)
        h[np.abs(2 * alpha * t) == 1] = (np.pi/4) * np.sinc(1/(2*alpha))
        h = h / np.sqrt(np.sum(h**2))
        
        shaped_signal = np.convolve(upsampled, h, mode='same')
    else:
        shaped_signal = upsampled
    
    return shaped_signal

def add_channel_impairments(signal, snr_db=25, isi_taps=None):
    """Add noise and ISI to the signal"""
    # Add AWGN
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    # Add ISI if specified
    if isi_taps is not None:
        signal = np.convolve(signal, isi_taps, mode='same')
    
    return signal + noise

# Simulation example
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    num_symbols = 1000
    sps = 4
    
    # Random PAM4 symbols (0, 1, 2, 3)
    tx_symbols = np.random.randint(0, 4, num_symbols)
    
    # Generate PAM4 signal
    tx_signal = generate_pam4_signal(tx_symbols, sps=sps)
    
    # Add channel impairments
    isi_channel = np.array([0.1, 0.3, 1.0, 0.4, 0.1])  # Simple ISI channel
    rx_signal = add_channel_impairments(tx_signal, snr_db=20, isi_taps=isi_channel)
    
    # Create receiver
    receiver = PAM4Receiver(ffe_taps=7, dfe_taps=5, sps=sps)
    
    # Process received signal
    results = receiver.process_block(rx_signal)
    
    # Extract results
    equalized_samples = [r['equalized'] for r in results]
    decisions = [r['decision'] for r in results]
    errors = [r['error'] for r in results]
    timing_errors = [r['timing_error'] for r in results]
    
    # Plot results
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Original vs received signal
    plt.subplot(3, 2, 1)
    plt.plot(tx_signal[:200], 'b-', label='TX Signal')
    plt.plot(rx_signal[:200], 'r--', alpha=0.7, label='RX Signal')
    plt.title('Transmitted vs Received Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Equalized signal
    plt.subplot(3, 2, 2)
    plt.plot(equalized_samples[:200], 'g-')
    plt.title('Equalized Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plot 3: Error convergence
    plt.subplot(3, 2, 3)
    error_magnitude = np.abs(errors)
    plt.semilogy(error_magnitude)
    plt.title('LMS Error Convergence')
    plt.xlabel('Sample Index')
    plt.ylabel('|Error|')
    plt.grid(True)
    
    # Plot 4: Timing error
    plt.subplot(3, 2, 4)
    plt.plot(timing_errors)
    plt.title('Mueller-Muller Timing Error')
    plt.xlabel('Sample Index')
    plt.ylabel('Timing Error')
    plt.grid(True)
    
    # Plot 5: FFE coefficients evolution
    plt.subplot(3, 2, 5)
    ffe_evolution = np.array([r['ffe_coeffs'] for r in results[-100:]])
    for i in range(receiver.ffe_taps):
        plt.plot(ffe_evolution[:, i], label=f'Tap {i}')
    plt.title('FFE Coefficients (Last 100 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Eye diagram
    plt.subplot(3, 2, 6)
    eye_samples = equalized_samples[100:]  # Skip initial adaptation
    eye_length = sps * 2  # Two symbol periods
    num_traces = min(50, len(eye_samples) // eye_length)
    
    for i in range(num_traces):
        start_idx = i * eye_length
        if start_idx + eye_length < len(eye_samples):
            eye_trace = eye_samples[start_idx:start_idx + eye_length]
            plt.plot(eye_trace, 'b-', alpha=0.3)
    
    # Overlay PAM4 levels
    for level in receiver.pam4_levels:
        plt.axhline(y=level, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Eye Diagram')
    plt.xlabel('Sample Index (within 2 symbols)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    print(f"Final FFE coefficients: {receiver.ffe_coeffs}")
    print(f"Final DFE coefficients: {receiver.dfe_coeffs}")
    print(f"Adapted PAM4 levels: {receiver.pam4_levels}")
    print(f"Final RMS error: {np.sqrt(np.mean(np.array(errors[-100:])**2)):.4f}")