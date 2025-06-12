import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

class PAM4_MM_CDR:
    def __init__(self, samples_per_symbol=8, loop_bw=0.01, damping_factor=1.0, isi_taps=5):
        """
        PAM4 Mueller-Muller CDR with Digital Loop Filter
        
        Parameters:
        - samples_per_symbol: Oversampling ratio (default: 8)
        - loop_bw: Normalized loop bandwidth (default: 0.01)
        - damping_factor: PLL damping factor (default: 1.0)
        - isi_taps: Number of taps for adaptive equalizer (default: 5)
        """
        self.sps = samples_per_symbol
        self.phase = 0
        
        # Digital Loop Filter (Proportional + Integral)
        self.set_loop_parameters(loop_bw, damping_factor)
        self.integrator = 0
        self.last_error = 0
        
        # Adaptive equalizer
        self.isi_taps = isi_taps
        self.equalizer = np.zeros(isi_taps)
        self.equalizer[isi_taps//2] = 1.0  # Center tap initialization
        self.mu = 0.001  # LMS step size
        
        # PAM4 levels
        self.levels = np.array([-3, -1, 1, 3])
        
        # Buffers
        self.sample_buffer = np.zeros(max(4, isi_taps))  # For TED and equalizer
        self.symbol_buffer = np.zeros(2)  # For symbol decisions
        
        # Monitoring
        self.ted_errors = []
        self.phases = []
        self.freq_offsets = []
        self.sliced_symbols = []
    
    def set_loop_parameters(self, loop_bw, damping_factor):
        """Calculate digital loop filter coefficients"""
        # Natural frequency (from Gardner)
        wn = loop_bw / (damping_factor + 1/(4*damping_factor))
        
        # Proportional and integral gains
        self.Kp = 4 * damping_factor * wn
        self.Ki = (4 * wn**2) / (damping_factor + 1/(4*damping_factor))
    
    def digital_loop_filter(self, error):
        """Digital proportional-integral loop filter"""
        proportional = self.Kp * error
        self.integrator += self.Ki * error
        return proportional + self.integrator
    
    def pam4_slicer(self, sample):
        """Slice to nearest PAM4 level"""
        return self.levels[np.argmin(np.abs(self.levels - sample))]
    
    def mueller_muller_ted(self):
        """
        PAM4 Modified Mueller-Muller Timing Error Detector
        TED = y[n-1]*(x[n] - x[n-2]) - y[n]*(x[n-1] - x[n-3])
        """
        if len(self.sample_buffer) < 4:
            return 0
            
        x = self.sample_buffer
        y = [self.pam4_slicer(v) for v in x]
        
        # Main TED equation
        error = y[-2]*(x[0] - x[-4]) - y[0]*(x[-1] - x[-3])
        
        # Normalization
        error /= (y[0]**2 + 1e-6)  # Avoid division by zero
        
        return error
    
    def update_equalizer(self, error):
        """LMS adaptive equalizer update"""
        self.equalizer -= self.mu * error * self.sample_buffer[:self.isi_taps]
        # Normalize to maintain DC gain
        self.equalizer /= np.sum(np.abs(self.equalizer))
    
    def process_sample(self, sample):
        """Process one input sample"""
        # Update sample buffer
        self.sample_buffer = np.roll(self.sample_buffer, -1)
        self.sample_buffer[-1] = sample
        
        # Apply equalizer
        equalized = np.dot(self.sample_buffer[:self.isi_taps], self.equalizer)
        
        # Symbol decision at optimal sampling instant
        if self.phase == 0:
            # Slice the symbol
            sliced = self.pam4_slicer(equalized)
            self.sliced_symbols.append(sliced)
            
            # Update symbol buffer
            self.symbol_buffer = np.roll(self.symbol_buffer, -1)
            self.symbol_buffer[-1] = sliced
            
            # Calculate timing error
            ted_error = self.mueller_muller_ted()
            self.ted_errors.append(ted_error)
            
            # Update equalizer
            self.update_equalizer(equalized - sliced)
            
            # Update loop filter
            loop_output = self.digital_loop_filter(ted_error)
            
            # Update phase and frequency
            self.phase = (self.phase - loop_output) % self.sps
            self.phases.append(self.phase)
            
            # Frequency offset estimation (for monitoring)
            freq_offset = (self.last_error - ted_error) / self.sps
            self.freq_offsets.append(freq_offset)
            self.last_error = ted_error
            
            return sliced, ted_error
        
        # Increment phase
        self.phase = (self.phase + 1) % self.sps
        return None, None
    
    def plot_results(self):
        """Plot CDR performance metrics"""
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.ted_errors)
        plt.title('Timing Error Detector Output')
        plt.xlabel('Symbol')
        plt.ylabel('TED Error')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(self.phases)
        plt.title('Phase Trajectory')
        plt.xlabel('Symbol')
        plt.ylabel('Phase (samples)')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(self.freq_offsets)
        plt.title('Estimated Frequency Offset')
        plt.xlabel('Symbol')
        plt.ylabel('Frequency Error (1/symbols)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Channel simulation with frequency offset
def simulate_pam4_channel(num_symbols=1000, samples_per_symbol=8, snr=20, freq_offset=0.01):
    """Generate PAM4 signal with noise and frequency offset"""
    symbols = np.random.choice([-3, -1, 1, 3], num_symbols)
    samples = np.zeros(num_symbols * samples_per_symbol)
    
    # Apply frequency offset
    phase = 0
    for i in range(num_symbols):
        pos = int(i*samples_per_symbol + phase)
        if pos < len(samples):
            samples[pos] = symbols[i]
        phase += freq_offset * samples_per_symbol
    
    # Apply pulse shaping (rectangular)
    pulse = np.ones(samples_per_symbol)
    samples = np.convolve(samples, pulse, mode='same')
    
    # Add noise
    noise_power = 10**(-snr/10) * np.var([-3, -1, 1, 3])
    samples += np.random.normal(0, np.sqrt(noise_power), len(samples))
    
    return symbols, samples

# Run simulation
tx_symbols, rx_samples = simulate_pam4_channel(num_symbols=2000, snr=15, freq_offset=0.005)

# Initialize CDR
cdr = PAM4_MM_CDR(samples_per_symbol=8, loop_bw=0.02, damping_factor=1.0, isi_taps=5)

# Process samples
recovered_symbols = []
for i, sample in enumerate(rx_samples):
    symbol, error = cdr.process_sample(sample)
    if symbol is not None:
        recovered_symbols.append(symbol)

# Plot results
cdr.plot_results()

# Calculate BER
min_len = min(len(tx_symbols), len(recovered_symbols))
ber = np.mean(np.array(tx_symbols[:min_len]) != np.array(recovered_symbols[:min_len]))
print(f"Bit Error Rate: {ber:.2%}")

# Print equalizer coefficients
print("\nFinal Equalizer Coefficients:")
print(cdr.equalizer)