import numpy as np
import matplotlib.pyplot as plt

class NRZ_Equalizer:
    def __init__(self, ffe_taps=5, dfe_taps=3, mu_ffe=0.01, mu_dfe=0.01):
        """
        Initialize NRZ equalizer with FFE and DFE
        
        Parameters:
        ffe_taps: Number of FFE taps
        dfe_taps: Number of DFE taps
        mu_ffe: FFE adaptation step size
        mu_dfe: DFE adaptation step size
        """
        self.ffe_taps = ffe_taps
        self.dfe_taps = dfe_taps
        self.mu_ffe = mu_ffe
        self.mu_dfe = mu_dfe
        
        # Initialize coefficients
        self.ffe_coeffs = np.zeros(ffe_taps)
        self.ffe_coeffs[ffe_taps//2] = 1.0  # Center tap initialization
        self.dfe_coeffs = np.zeros(dfe_taps)
        
        # Buffers
        self.ffe_buffer = np.zeros(ffe_taps)
        self.dfe_buffer = np.zeros(dfe_taps)
        
    def autocorrelation_matrix(self, signal, taps):
        """
        Compute autocorrelation matrix for signal
        """
        R = np.zeros((taps, taps))
        for i in range(taps):
            for j in range(taps):
                if i >= j:
                    lag = i - j
                    if lag < len(signal):
                        R[i, j] = np.correlate(signal[lag:], signal[:len(signal)-lag])[0] / len(signal)
                    R[j, i] = R[i, j]
        return R
    
    def cross_correlation_vector(self, signal, desired, taps):
        """
        Compute cross-correlation vector between signal and desired output
        """
        p = np.zeros(taps)
        for i in range(taps):
            if i < len(signal) and i < len(desired):
                p[i] = np.correlate(signal[i:], desired[:len(desired)-i])[0] / len(signal)
        return p
    
    def adapt_wiener(self, input_signal, desired_signal):
        """
        Adapt coefficients using Wiener solution (autocorrelation method)
        """
        # FFE adaptation
        if len(input_signal) >= self.ffe_taps:
            R_ffe = self.autocorrelation_matrix(input_signal, self.ffe_taps)
            p_ffe = self.cross_correlation_vector(input_signal, desired_signal, self.ffe_taps)
            
            # Add small regularization to avoid singular matrix
            R_ffe += np.eye(self.ffe_taps) * 1e-6
            
            try:
                self.ffe_coeffs = np.linalg.solve(R_ffe, p_ffe)
            except np.linalg.LinAlgError:
                print("FFE: Singular matrix, using pseudo-inverse")
                self.ffe_coeffs = np.linalg.pinv(R_ffe) @ p_ffe
    
    def adapt_lms(self, input_sample, error):
        """
        Adapt coefficients using LMS algorithm
        """
        # FFE LMS update
        self.ffe_coeffs += self.mu_ffe * error * self.ffe_buffer
        
        # DFE LMS update
        self.dfe_coeffs += self.mu_dfe * error * self.dfe_buffer
    
    def equalize_sample(self, input_sample, use_decision=True):
        """
        Process single sample through equalizer
        """
        # Update FFE buffer
        self.ffe_buffer = np.roll(self.ffe_buffer, 1)
        self.ffe_buffer[0] = input_sample
        
        # FFE output
        ffe_out = np.dot(self.ffe_coeffs, self.ffe_buffer)
        
        # DFE output
        dfe_out = np.dot(self.dfe_coeffs, self.dfe_buffer)
        
        # Combined output
        eq_out = ffe_out - dfe_out
        
        # Decision (slicer)
        if use_decision:
            decision = 1.0 if eq_out > 0 else -1.0
        else:
            decision = eq_out
        
        # Update DFE buffer with decision
        self.dfe_buffer = np.roll(self.dfe_buffer, 1)
        self.dfe_buffer[0] = decision
        
        return eq_out, decision
    
    def batch_adapt_autocorrelation(self, channel_output, training_symbols):
        """
        Batch adaptation using autocorrelation method
        """
        # Create shifted versions for FFE
        X_ffe = np.zeros((len(channel_output) - self.ffe_taps + 1, self.ffe_taps))
        for i in range(self.ffe_taps):
            X_ffe[:, i] = channel_output[i:len(channel_output) - self.ffe_taps + i + 1]
        
        # Compute autocorrelation matrix and cross-correlation vector
        R_ffe = X_ffe.T @ X_ffe / X_ffe.shape[0]
        p_ffe = X_ffe.T @ training_symbols[:X_ffe.shape[0]] / X_ffe.shape[0]
        
        # Solve for optimal coefficients
        R_ffe += np.eye(self.ffe_taps) * 1e-6  # Regularization
        self.ffe_coeffs = np.linalg.solve(R_ffe, p_ffe)
        
        # For DFE, we need to run through the data once with FFE
        # and collect decisions for DFE adaptation
        decisions = []
        errors = []
        
        for i in range(len(channel_output)):
            eq_out, decision = self.equalize_sample(channel_output[i], use_decision=True)
            decisions.append(decision)
            if i < len(training_symbols):
                errors.append(training_symbols[i] - eq_out)
        
        # Now adapt DFE coefficients if we have enough data
        if len(decisions) >= self.dfe_taps and len(errors) >= self.dfe_taps:
            X_dfe = np.zeros((len(errors) - self.dfe_taps, self.dfe_taps))
            for i in range(self.dfe_taps):
                X_dfe[:, i] = decisions[i:len(errors) - self.dfe_taps + i]
            
            y_dfe = np.array(errors[self.dfe_taps:])
            
            # Compute DFE coefficients
            R_dfe = X_dfe.T @ X_dfe / X_dfe.shape[0]
            p_dfe = X_dfe.T @ y_dfe / X_dfe.shape[0]
            
            R_dfe += np.eye(self.dfe_taps) * 1e-6
            self.dfe_coeffs = np.linalg.solve(R_dfe, p_dfe)

# Example usage and testing
def create_channel_response(channel_type='mild_isi'):
    """Create different channel responses"""
    if channel_type == 'mild_isi':
        h = np.array([0.2, 0.9, 0.3])
    elif channel_type == 'severe_isi':
        h = np.array([0.1, 0.2, 0.5, 0.9, 0.4, 0.2, 0.1])
    else:
        h = np.array([1.0])  # No ISI
    return h / np.linalg.norm(h)

def generate_nrz_data(n_symbols):
    """Generate random NRZ data"""
    return 2 * np.random.randint(0, 2, n_symbols) - 1

def add_noise(signal, snr_db):
    """Add AWGN noise to signal"""
    signal_power = np.mean(signal**2)
    noise_power = signal_power / (10**(snr_db/10))
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# Demonstration
if __name__ == "__main__":
    # Parameters
    n_symbols = 1000
    n_training = 500
    snr_db = 20
    
    # Generate data
    tx_symbols = generate_nrz_data(n_symbols)
    training_symbols = tx_symbols[:n_training]
    
    # Channel
    h = create_channel_response('severe_isi')
    rx_signal = np.convolve(tx_symbols, h, mode='same')
    rx_signal = add_noise(rx_signal, snr_db)
    
    # Initialize equalizer
    eq = NRZ_Equalizer(ffe_taps=15, dfe_taps=5, mu_ffe=0.01, mu_dfe=0.005)
    
    # Method 1: Batch autocorrelation adaptation
    eq.batch_adapt_autocorrelation(rx_signal[:n_training], training_symbols)
    
    # Equalize test data
    eq_output = []
    decisions = []
    eq.dfe_buffer.fill(0)  # Reset DFE buffer
    
    for i in range(n_training, n_symbols):
        eq_out, decision = eq.equalize_sample(rx_signal[i])
        eq_output.append(eq_out)
        decisions.append(decision)
    
    # Calculate BER
    test_symbols = tx_symbols[n_training:]
    ber = np.mean(np.array(decisions) != test_symbols)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot channel response
    plt.subplot(2, 3, 1)
    plt.stem(h)
    plt.title('Channel Response')
    plt.xlabel('Tap')
    plt.ylabel('Amplitude')
    
    # Plot FFE coefficients
    plt.subplot(2, 3, 2)
    plt.stem(eq.ffe_coeffs)
    plt.title('FFE Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Amplitude')
    
    # Plot DFE coefficients
    plt.subplot(2, 3, 3)
    plt.stem(eq.dfe_coeffs)
    plt.title('DFE Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Amplitude')
    
    # Eye diagram before equalization
    plt.subplot(2, 3, 4)
    eye_length = 100
    for i in range(0, min(len(rx_signal)-2, 200), 2):
        plt.plot(rx_signal[i:i+3], 'b', alpha=0.1)
    plt.title('Eye Diagram - Before EQ')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # Eye diagram after equalization
    plt.subplot(2, 3, 5)
    for i in range(0, min(len(eq_output)-2, 200), 2):
        plt.plot(eq_output[i:i+3], 'r', alpha=0.1)
    plt.title('Eye Diagram - After EQ')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    # BER text
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.5, f'BER: {ber:.4f}', fontsize=16)
    plt.text(0.1, 0.3, f'SNR: {snr_db} dB', fontsize=16)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Alternative: LMS adaptation example
    print("\nLMS Adaptation Example:")
    eq_lms = NRZ_Equalizer(ffe_taps=15, dfe_taps=5, mu_ffe=0.01, mu_dfe=0.005)
    
    # Reset buffers
    eq_lms.ffe_buffer.fill(0)
    eq_lms.dfe_buffer.fill(0)
    
    # Training with LMS
    for i in range(n_training):
        eq_out, decision = eq_lms.equalize_sample(rx_signal[i])
        error = training_symbols[i] - eq_out
        eq_lms.adapt_lms(rx_signal[i], error)
    
    # Test with LMS-adapted coefficients
    decisions_lms = []
    for i in range(n_training, n_symbols):
        _, decision = eq_lms.equalize_sample(rx_signal[i])
        decisions_lms.append(decision)
    
    ber_lms = np.mean(np.array(decisions_lms) != test_symbols)
    print(f"Autocorrelation method BER: {ber:.4f}")
    print(f"LMS method BER: {ber_lms:.4f}")