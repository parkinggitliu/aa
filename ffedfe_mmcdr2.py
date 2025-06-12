import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PAM4Receiver:
    def __init__(self, samples_per_symbol=8, rolloff=0.35, filter_span=10):
        """
        PAM4 Receiver with Mueller-Muller CDR
        
        Parameters:
        - samples_per_symbol: Oversampling rate
        - rolloff: Root raised cosine filter rolloff factor
        - filter_span: Filter length in symbols
        """
        self.sps = samples_per_symbol
        self.rolloff = rolloff
        self.filter_span = filter_span
        
        # PAM4 levels: -3, -1, +1, +3
        self.pam4_levels = np.array([-3, -1, 1, 3])
        
        # Generate matched filter (RRC)
        self.matched_filter = self._create_rrc_filter()
        
        # Mueller-Muller CDR parameters
        self.mm_gain = 0.01  # Loop gain
        self.ted_buffer = np.zeros(3)  # For timing error detector
        self.loop_filter_alpha = 0.01
        self.loop_filter_beta = 0.001
        self.integrator = 0
        self.nco_phase = 0
        
    def _create_rrc_filter(self):
        """Create root raised cosine filter"""
        t = np.arange(-self.filter_span*self.sps/2, 
                       self.filter_span*self.sps/2 + 1) / self.sps
        
        h = np.zeros_like(t)
        
        for i, t_val in enumerate(t):
            if t_val == 0:
                h[i] = (1 + self.rolloff * (4/np.pi - 1))
            elif abs(t_val) == 1/(4*self.rolloff):
                h[i] = (self.rolloff/np.sqrt(2)) * \
                       ((1 + 2/np.pi) * np.sin(np.pi/(4*self.rolloff)) + 
                        (1 - 2/np.pi) * np.cos(np.pi/(4*self.rolloff)))
            else:
                h[i] = (np.sin(np.pi*t_val*(1-self.rolloff)) + 
                        4*self.rolloff*t_val*np.cos(np.pi*t_val*(1+self.rolloff))) / \
                       (np.pi*t_val*(1-(4*self.rolloff*t_val)**2))
        
        return h / np.sqrt(np.sum(h**2))
    
    def matched_filter_rx(self, rx_signal):
        """Apply matched filtering to received signal"""
        return signal.convolve(rx_signal, self.matched_filter, mode='same')
    
    def mueller_muller_ted(self, current_sample, previous_sample, decision):
        """
        Mueller-Muller timing error detector
        
        Returns timing error for current symbol
        """
        # Classic Mueller-Muller algorithm
        # e(k) = a(k-1)*x(k) - a(k)*x(k-1)
        # where a(k) is the decision and x(k) is the sample
        
        timing_error = previous_sample * decision - current_sample * self.ted_buffer[1]
        
        # Update buffer
        self.ted_buffer[0] = self.ted_buffer[1]
        self.ted_buffer[1] = decision
        
        return timing_error
    
    def loop_filter(self, timing_error):
        """Second-order loop filter for CDR"""
        # Proportional path
        proportional = self.loop_filter_alpha * timing_error
        
        # Integral path
        self.integrator += self.loop_filter_beta * timing_error
        
        return proportional + self.integrator
    
    def interpolate(self, signal, mu):
        """
        Parabolic interpolation for fractional delay
        
        mu: fractional delay [0, 1)
        """
        if mu < 0 or mu >= 1:
            mu = mu % 1
            
        # Use parabolic (quadratic) interpolation
        # y(mu) = c0 + c1*mu + c2*mu^2
        # where coefficients are computed from neighboring samples
        
        idx = int(self.nco_phase)
        if idx >= len(signal) - 2:
            return 0
        
        y0 = signal[idx-1] if idx > 0 else signal[0]
        y1 = signal[idx]
        y2 = signal[idx+1]
        
        c0 = y1
        c1 = (y2 - y0) / 2
        c2 = (y2 - 2*y1 + y0) / 2
        
        return c0 + c1*mu + c2*mu**2
    
    def pam4_slicer(self, sample):
        """PAM4 decision device (slicer)"""
        # Find closest PAM4 level
        distances = np.abs(self.pam4_levels - sample)
        decision_idx = np.argmin(distances)
        return self.pam4_levels[decision_idx]
    
    def demodulate(self, rx_signal, return_diagnostics=False):
        """
        Main demodulation function with Mueller-Muller CDR
        
        Parameters:
        - rx_signal: Received PAM4 signal
        - return_diagnostics: If True, return timing error and sampling instants
        
        Returns:
        - decisions: PAM4 decisions
        - timing_errors (optional): Timing error signal
        - sampling_instants (optional): Sample timing information
        """
        # Apply matched filter
        filtered_signal = self.matched_filter_rx(rx_signal)
        
        # Initialize outputs
        decisions = []
        timing_errors = []
        sampling_instants = []
        
        # Initialize CDR state
        self.nco_phase = self.sps // 2  # Start at nominal sampling point
        sample_counter = 0
        previous_sample = 0
        
        while self.nco_phase < len(filtered_signal) - self.sps:
            # Compute fractional delay
            mu = self.nco_phase - int(self.nco_phase)
            
            # Interpolate to get current sample
            current_sample = self.interpolate(filtered_signal, mu)
            
            # Make PAM4 decision
            decision = self.pam4_slicer(current_sample)
            decisions.append(decision)
            
            # Timing error detection (Mueller-Muller)
            if sample_counter > 0:
                timing_error = self.mueller_muller_ted(
                    current_sample, previous_sample, decision
                )
                timing_errors.append(timing_error)
                
                # Loop filter
                control_signal = self.loop_filter(timing_error)
                
                # Update NCO phase
                # Nominal increment is sps, adjusted by control signal
                self.nco_phase += self.sps * (1 + control_signal)
            else:
                self.nco_phase += self.sps
                timing_errors.append(0)
            
            sampling_instants.append(int(self.nco_phase))
            previous_sample = current_sample
            sample_counter += 1
        
        decisions = np.array(decisions)
        
        if return_diagnostics:
            return decisions, np.array(timing_errors), np.array(sampling_instants)
        else:
            return decisions
    
    def calculate_ser(self, tx_symbols, rx_symbols):
        """Calculate symbol error rate"""
        # Align sequences (CDR may cause slight length mismatch)
        min_len = min(len(tx_symbols), len(rx_symbols))
        tx_aligned = tx_symbols[:min_len]
        rx_aligned = rx_symbols[:min_len]
        
        errors = np.sum(tx_aligned != rx_aligned)
        ser = errors / min_len
        
        return ser, errors

def apply_timing_offset(signal, frequency_offset_ppm):
    """
    Apply timing offset to simulate clock frequency mismatch
    Uses custom cubic interpolation instead of interp1d
    """
    freq_offset = frequency_offset_ppm * 1e-6
    timing_drift = 1 + freq_offset
    
    # Original sample positions
    original_indices = np.arange(len(signal))
    
    # New sample positions (with timing drift)
    new_indices = original_indices / timing_drift
    
    # Output signal
    output = np.zeros_like(signal)
    
    # Custom cubic interpolation
    for i, new_idx in enumerate(new_indices):
        if new_idx >= len(signal) - 2:
            break
            
        # Integer and fractional parts
        idx = int(new_idx)
        frac = new_idx - idx
        
        if idx == 0:
            # Linear interpolation at the boundary
            output[i] = signal[0] * (1 - frac) + signal[1] * frac
        elif idx >= len(signal) - 2:
            # Linear interpolation at the boundary
            output[i] = signal[-2] * (1 - frac) + signal[-1] * frac
        else:
            # Cubic interpolation using 4 points
            y0 = signal[idx - 1]
            y1 = signal[idx]
            y2 = signal[idx + 1]
            y3 = signal[idx + 2] if idx + 2 < len(signal) else signal[idx + 1]
            
            # Catmull-Rom cubic interpolation coefficients
            a0 = -0.5*y0 + 1.5*y1 - 1.5*y2 + 0.5*y3
            a1 = y0 - 2.5*y1 + 2*y2 - 0.5*y3
            a2 = -0.5*y0 + 0.5*y2
            a3 = y1
            
            output[i] = a0*frac**3 + a1*frac**2 + a2*frac + a3
    
    return output[:len(signal)]

# Example usage and testing
if __name__ == "__main__":
    # Simulation parameters
    num_symbols = 1000
    sps = 8
    snr_db = 20
    frequency_offset_ppm = 100  # Clock frequency offset in ppm
    
    # Generate random PAM4 symbols
    tx_bits = np.random.randint(0, 4, num_symbols)
    tx_symbols = np.array([-3, -1, 1, 3])[tx_bits]
    
    # Create PAM4 transmitter signal (with RRC pulse shaping)
    tx_filter = PAM4Receiver(sps).matched_filter
    tx_upsampled = np.zeros(num_symbols * sps)
    tx_upsampled[::sps] = tx_symbols
    tx_signal = signal.convolve(tx_upsampled, tx_filter, mode='same')
    
    # Add timing offset (simulating clock frequency offset)
    rx_signal = apply_timing_offset(tx_signal, frequency_offset_ppm)
    
    # Add AWGN noise
    signal_power = np.mean(rx_signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(rx_signal))
    rx_signal_noisy = rx_signal + noise
    
    # Create receiver and demodulate
    receiver = PAM4Receiver(samples_per_symbol=sps)
    decisions, timing_errors, sampling_instants = receiver.demodulate(
        rx_signal_noisy, return_diagnostics=True
    )
    
    # Calculate performance
    ser, num_errors = receiver.calculate_ser(tx_symbols, decisions)
    
    print(f"PAM4 Receiver Performance:")
    print(f"SNR: {snr_db} dB")
    print(f"Frequency offset: {frequency_offset_ppm} ppm")
    print(f"Symbol Error Rate: {ser:.2e}")
    print(f"Number of errors: {num_errors}/{len(decisions)}")
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Eye diagram
    ax1 = axes[0]
    eye_samples = 2 * sps
    num_traces = 100
    filtered = receiver.matched_filter_rx(rx_signal_noisy)
    
    for i in range(num_traces):
        start_idx = i * sps + int(sps/2)
        if start_idx + eye_samples < len(filtered):
            ax1.plot(np.arange(eye_samples), 
                    filtered[start_idx:start_idx+eye_samples], 
                    'b', alpha=0.1)
    
    ax1.set_title('Eye Diagram')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Timing error
    ax2 = axes[1]
    ax2.plot(timing_errors)
    ax2.set_title('Mueller-Muller Timing Error')
    ax2.set_xlabel('Symbol Index')
    ax2.set_ylabel('Timing Error')
    ax2.grid(True)
    
    # Constellation
    ax3 = axes[2]
    ax3.scatter(decisions[100:], np.zeros_like(decisions[100:]), 
                alpha=0.5, s=10)
    ax3.scatter([-3, -1, 1, 3], [0, 0, 0, 0], c='red', s=100, 
                marker='x', label='Ideal levels')
    ax3.set_title('PAM4 Constellation')
    ax3.set_xlabel('Amplitude')
    ax3.set_ylim([-0.5, 0.5])
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()