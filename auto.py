import numpy as np
import scipy.signal as signal
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

class SelfInterferenceCanceller:
    def __init__(self, filter_length=64, mu=0.001, regularization=1e-6):
        """
        Initialize the self-interference canceller
        
        Args:
            filter_length: Length of the adaptive filter
            mu: Step size for LMS algorithm
            regularization: Regularization parameter for RLS
        """
        self.filter_length = filter_length
        self.mu = mu
        self.reg = regularization
        self.weights = np.zeros(filter_length, dtype=complex)
        self.tx_buffer = np.zeros(filter_length, dtype=complex)
        
    def lms_cancel(self, tx_signal, rx_signal):
        """
        LMS-based self-interference cancellation
        
        Args:
            tx_signal: Transmitted signal samples
            rx_signal: Received signal samples (contains self-interference)
            
        Returns:
            cancelled_signal: Signal after self-interference cancellation
            error_signal: Error/residual signal
        """
        N = len(rx_signal)
        cancelled_signal = np.zeros(N, dtype=complex)
        error_signal = np.zeros(N, dtype=complex)
        
        for n in range(N):
            # Update transmit signal buffer
            self.tx_buffer = np.roll(self.tx_buffer, 1)
            self.tx_buffer[0] = tx_signal[n] if n < len(tx_signal) else 0
            
            # Estimate self-interference
            si_estimate = np.dot(self.weights.conj(), self.tx_buffer)
            
            # Cancel self-interference
            cancelled_signal[n] = rx_signal[n] - si_estimate
            error_signal[n] = cancelled_signal[n]
            
            # Update filter weights using LMS
            self.weights += self.mu * error_signal[n].conj() * self.tx_buffer
            
        return cancelled_signal, error_signal
    
    def rls_cancel(self, tx_signal, rx_signal, forgetting_factor=0.99):
        """
        RLS-based self-interference cancellation
        
        Args:
            tx_signal: Transmitted signal samples
            rx_signal: Received signal samples
            forgetting_factor: RLS forgetting factor
            
        Returns:
            cancelled_signal: Signal after cancellation
            error_signal: Error signal
        """
        N = len(rx_signal)
        cancelled_signal = np.zeros(N, dtype=complex)
        error_signal = np.zeros(N, dtype=complex)
        
        # Initialize RLS parameters
        P = np.eye(self.filter_length) / self.reg  # Inverse correlation matrix
        weights = np.zeros(self.filter_length, dtype=complex)
        
        for n in range(N):
            # Update transmit signal buffer
            self.tx_buffer = np.roll(self.tx_buffer, 1)
            self.tx_buffer[0] = tx_signal[n] if n < len(tx_signal) else 0
            
            # Estimate self-interference
            si_estimate = np.dot(weights.conj(), self.tx_buffer)
            
            # Calculate error
            error_signal[n] = rx_signal[n] - si_estimate
            cancelled_signal[n] = error_signal[n]
            
            # RLS update
            k = (P @ self.tx_buffer.conj()) / (forgetting_factor + 
                                               self.tx_buffer.T @ P @ self.tx_buffer.conj())
            P = (P - np.outer(k, self.tx_buffer.T @ P)) / forgetting_factor
            weights += k * error_signal[n]
            
        self.weights = weights
        return cancelled_signal, error_signal
    
    def autocorr_based_cancel(self, tx_signal, rx_signal, delay_taps=10):
        """
        Auto-correlation based cancellation using Wiener filtering
        
        Args:
            tx_signal: Transmitted signal
            rx_signal: Received signal
            delay_taps: Number of delay taps to consider
            
        Returns:
            cancelled_signal: Cancelled signal
            wiener_filter: Computed Wiener filter coefficients
        """
        # Pad signals to same length
        max_len = max(len(tx_signal), len(rx_signal))
        tx_padded = np.pad(tx_signal, (0, max_len - len(tx_signal)), 'constant')
        rx_padded = np.pad(rx_signal, (0, max_len - len(rx_signal)), 'constant')
        
        # Compute auto-correlation of TX signal
        R_xx = np.correlate(tx_padded, tx_padded, mode='full')
        R_xx = R_xx[len(R_xx)//2:][:self.filter_length]
        
        # Compute cross-correlation between TX and RX
        R_xy = np.correlate(rx_padded, tx_padded, mode='full')
        R_xy = R_xy[len(R_xy)//2:][:self.filter_length]
        
        # Form Toeplitz matrix for Wiener filter
        R_matrix = toeplitz(R_xx)
        
        # Solve Wiener-Hopf equation: R * h = r
        try:
            wiener_filter = np.linalg.solve(R_matrix + self.reg * np.eye(len(R_matrix)), R_xy)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if matrix is singular
            wiener_filter = np.linalg.pinv(R_matrix) @ R_xy
        
        # Apply filter to cancel self-interference
        si_estimate = np.convolve(tx_padded, wiener_filter, mode='same')
        cancelled_signal = rx_padded - si_estimate[:len(rx_padded)]
        
        return cancelled_signal, wiener_filter

def generate_test_signals(N=1000, snr_db=20):
    """Generate test signals for simulation"""
    # Generate random TX signal (QPSK-like)
    tx_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N)
    
    # Simulate channel response (multipath)
    channel_taps = np.array([1.0, 0.3*np.exp(1j*np.pi/4), 0.1*np.exp(1j*np.pi/2)])
    si_signal = np.convolve(tx_symbols, channel_taps, mode='same')
    
    # Add desired signal (from remote transmitter)
    desired_signal = 0.1 * np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], N)
    
    # Add noise
    noise_power = 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j*np.random.randn(N))
    
    # Received signal = self-interference + desired + noise
    rx_signal = si_signal + desired_signal + noise
    
    return tx_symbols, rx_signal, desired_signal, si_signal

def simulate_cancellation():
    """Simulate and compare different cancellation methods"""
    # Generate test signals
    tx_signal, rx_signal, desired_signal, si_signal = generate_test_signals(N=2000, snr_db=20)
    
    # Initialize canceller
    canceller = SelfInterferenceCanceller(filter_length=32, mu=0.01)
    
    # Test different methods
    print("Testing Self-Interference Cancellation Methods...")
    
    # LMS cancellation
    cancelled_lms, error_lms = canceller.lms_cancel(tx_signal, rx_signal)
    si_suppression_lms = 10 * np.log10(np.mean(np.abs(si_signal)**2) / 
                                       np.mean(np.abs(error_lms[500:])**2))
    
    # Reset weights for fair comparison
    canceller.weights = np.zeros(canceller.filter_length, dtype=complex)
    
    # RLS cancellation
    cancelled_rls, error_rls = canceller.rls_cancel(tx_signal, rx_signal)
    si_suppression_rls = 10 * np.log10(np.mean(np.abs(si_signal)**2) / 
                                       np.mean(np.abs(error_rls[500:])**2))
    
    # Auto-correlation based cancellation
    cancelled_autocorr, wiener_filter = canceller.autocorr_based_cancel(tx_signal, rx_signal)
    si_suppression_autocorr = 10 * np.log10(np.mean(np.abs(si_signal)**2) / 
                                            np.mean(np.abs(cancelled_autocorr[500:])**2))
    
    # Print results
    print(f"LMS SI Suppression: {si_suppression_lms:.2f} dB")
    print(f"RLS SI Suppression: {si_suppression_rls:.2f} dB")
    print(f"Auto-correlation SI Suppression: {si_suppression_autocorr:.2f} dB")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(np.real(rx_signal[:500]))
    plt.title('Original RX Signal (Real Part)')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 3, 2)
    plt.plot(np.real(cancelled_lms[:500]))
    plt.title('LMS Cancelled Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 3, 3)
    plt.plot(np.real(cancelled_rls[:500]))
    plt.title('RLS Cancelled Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 3, 4)
    plt.plot(np.real(cancelled_autocorr[:500]))
    plt.title('Auto-correlation Cancelled Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 3, 5)
    plt.semilogy(np.abs(error_lms)**2)
    plt.semilogy(np.abs(error_rls)**2)
    plt.legend(['LMS', 'RLS'])
    plt.title('Error Power Convergence')
    plt.xlabel('Sample')
    plt.ylabel('Error Power')
    
    plt.subplot(2, 3, 6)
    plt.plot(np.abs(wiener_filter))
    plt.title('Wiener Filter Coefficients')
    plt.xlabel('Tap')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'lms_suppression': si_suppression_lms,
        'rls_suppression': si_suppression_rls,
        'autocorr_suppression': si_suppression_autocorr,
        'cancelled_signals': {
            'lms': cancelled_lms,
            'rls': cancelled_rls,
            'autocorr': cancelled_autocorr
        }
    }

# Example usage
if __name__ == "__main__":
    results = simulate_cancellation()
    
    # Additional real-time processing example
    print("\nReal-time processing example:")
    canceller = SelfInterferenceCanceller(filter_length=16, mu=0.005)
    
    # Simulate real-time sample-by-sample processing
    tx_sample = 1 + 1j
    rx_sample = 2 + 0.5j  # Contains self-interference
    
    # Process single sample
    cancelled_sample, _ = canceller.lms_cancel([tx_sample], [rx_sample])
    print(f"TX: {tx_sample}, RX: {rx_sample}, Cancelled: {cancelled_sample[0]}")