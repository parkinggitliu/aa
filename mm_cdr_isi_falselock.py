import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class PAM4_MM_CDR:
    def __init__(self, samples_per_symbol=8, loop_gain=0.01, beta=0.5, isi_taps=5):
        """
        Initialize the PAM4 Mueller-Muller CDR
        
        Parameters:
        - samples_per_symbol: Oversampling ratio
        - loop_gain: CDR loop gain
        - beta: Proportional path gain (0 < beta < 1)
        - isi_taps: Number of taps for ISI filter
        """
        self.sps = samples_per_symbol
        self.loop_gain = loop_gain
        self.beta = beta
        self.phase = 0
        self.prev_sample = 0
        self.prev_sliced = 0
        self.isi_taps = isi_taps
        self.isi_history = np.zeros(isi_taps)
        self.lock_detector = 0
        self.lock_threshold = 0.9  # Threshold for lock detection
        self.false_lock_count = 0
        self.max_false_lock = 5  # Maximum false locks before correction
        
        # Initialize timing error detector history
        self.ted_history = np.zeros(10)
        
        # Initialize ISI filter coefficients (adaptive)
        self.isi_coeff = np.zeros(isi_taps)
        self.isi_coeff[isi_taps//2] = 1.0  # Start with center-tap as 1
        
    def slice_pam4(self, sample):
        """Slice the sample to nearest PAM4 level (-3, -1, 1, 3)"""
        if sample > 2:
            return 3
        elif sample > 0:
            return 1
        elif sample > -2:
            return -1
        else:
            return -3
    
    def mueller_muller_ted(self, sample, sliced):
        """
        PAM4 Mueller-Muller timing error detector
        Modified for PAM4: TED = y[n-1]*(x[n] - x[n-2]) - y[n]*(x[n-1] - x[n-3])
        """
        if len(self.isi_history) < 4:
            return 0
            
        # Standard MM TED for PAM4
        error = (self.isi_history[-2] * (sliced - self.prev_sliced) - 
                 sliced * (self.isi_history[-1] - self.isi_history[-3]))
        
        # Normalize by signal power
        error /= (sliced**2 + 1e-6)
        
        return error
    
    def update_isi_filter(self, error, sliced):
        """Adaptive ISI filter update using LMS algorithm"""
        mu = 0.01  # Learning rate
        x = np.array(self.isi_history)
        e = error
        
        # Update coefficients
        self.isi_coeff -= mu * e * x
        
        # Normalize to maintain DC gain
        self.isi_coeff /= np.sum(np.abs(self.isi_coeff))
    
    def detect_false_lock(self, ted_error):
        """Detect false lock condition"""
        # Update TED history
        self.ted_history = np.roll(self.ted_history, -1)
        self.ted_history[-1] = ted_error
        
        # Check for oscillating TED values (sign of false lock)
        sign_changes = np.sum(np.diff(np.sign(self.ted_history)) != 0)
        if sign_changes > len(self.ted_history) * 0.8:  # Too many sign changes
            self.false_lock_count += 1
        else:
            self.false_lock_count = max(0, self.false_lock_count - 1)
        
        return self.false_lock_count > self.max_false_lock
    
    def correct_false_lock(self):
        """Corrective measures for false lock"""
        print("False lock detected! Applying correction...")
        
        # 1. Phase bump (try to jump out of false lock)
        self.phase += self.sps // 2
        self.phase %= self.sps
        
        # 2. Reset ISI filter
        self.isi_coeff = np.zeros(self.isi_taps)
        self.isi_coeff[self.isi_taps//2] = 1.0
        
        # 3. Reset false lock counter
        self.false_lock_count = 0
        
        # 4. Increase loop gain temporarily
        self.loop_gain *= 2
    
    def process(self, sample):
        """Process one input sample"""
        # Update ISI history
        self.isi_history = np.roll(self.isi_history, -1)
        self.isi_history[-1] = sample
        
        # Apply ISI filter
        isi_corrected = np.dot(self.isi_history, self.isi_coeff)
        
        # Slice the ISI-corrected sample
        sliced = self.slice_pam4(isi_corrected)
        
        # Only process at symbol instances
        if self.phase == 0:
            # Calculate timing error
            ted_error = self.mueller_muller_ted(sample, sliced)
            
            # Check for false lock
            if self.detect_false_lock(ted_error):
                self.correct_false_lock()
                return None, None, True
            
            # Update ISI filter
            self.update_isi_filter(isi_corrected - sliced, sliced)
            
            # Update loop filter (proportional + integral)
            self.lock_detector = 0.9 * self.lock_detector + 0.1 * np.abs(ted_error)
            
            # Store previous values for next symbol
            self.prev_sample = sample
            self.prev_sliced = sliced
            
            return sliced, ted_error, False
        
        # Update phase
        self.phase = (self.phase + 1) % self.sps
        return None, None, False
    
    def is_locked(self):
        """Check if CDR is locked"""
        return self.lock_detector < (1 - self.lock_threshold)

# Example usage
def simulate_channel(symbols, snr=20, samples_per_symbol=8, isi=[0.2, -0.1, 0.05]):
    """Simulate a PAM4 signal with noise and ISI"""
    # Upsample symbols
    samples = np.zeros(len(symbols) * samples_per_symbol)
    samples[::samples_per_symbol] = symbols
    
    # Apply pulse shaping (rectangular for simplicity)
    pulse = np.ones(samples_per_symbol)
    samples = np.convolve(samples, pulse, mode='same')
    
    # Add ISI
    if len(isi) > 0:
        isi_filter = np.array(isi)
        samples = np.convolve(samples, isi_filter, mode='same')
    
    # Add noise
    noise_power = 10**(-snr/10) * np.var(symbols)
    noise = np.random.normal(0, np.sqrt(noise_power), len(samples))
    return samples + noise

# Generate random PAM4 symbols
num_symbols = 1000
pam4_levels = [-3, -1, 1, 3]
tx_symbols = np.random.choice(pam4_levels, num_symbols)

# Simulate channel
sps = 8
rx_samples = simulate_channel(tx_symbols, snr=15, samples_per_symbol=sps)

# Process with CDR
cdr = PAM4_MM_CDR(samples_per_symbol=sps, loop_gain=0.05, isi_taps=5)

rx_symbols = []
recovered_symbols = []
timing_errors = []

for i, sample in enumerate(rx_samples):
    symbol, ted_error, false_lock = cdr.process(sample)
    
    if false_lock:
        print(f"False lock corrected at sample {i}")
    
    if symbol is not None:
        recovered_symbols.append(symbol)
        timing_errors.append(ted_error)
        
        # Compare with transmitted symbols (for BER calculation)
        if len(recovered_symbols) < len(tx_symbols):
            rx_symbols.append(tx_symbols[len(recovered_symbols)-1])

# Calculate BER
if len(recovered_symbols) > 0 and len(rx_symbols) > 0:
    min_len = min(len(recovered_symbols), len(rx_symbols))
    ber = np.sum(np.array(recovered_symbols[:min_len]) != np.array(rx_symbols[:min_len])) / min_len
    print(f"Bit Error Rate: {ber:.2%}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(timing_errors)
plt.title('Timing Error Detector Output')
plt.xlabel('Symbol')
plt.ylabel('TED Error')

plt.subplot(3, 1, 2)
plt.stem(cdr.isi_coeff)
plt.title('Learned ISI Filter Coefficients')
plt.xlabel('Tap')
plt.ylabel('Coefficient Value')

plt.subplot(3, 1, 3)
plt.plot(np.convolve(np.abs(timing_errors), np.ones(20)/20, mode='valid'))
plt.title('Moving Average of Absolute Timing Error (Lock Detector)')
plt.xlabel('Symbol')
plt.ylabel('Lock Detector Value')

plt.tight_layout()
plt.show()