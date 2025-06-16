import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.linalg import toeplitz

class EqualizerAdaptation:
    """
    Compare Autocorrelation and LMS-based adaptation for DFE/FFE equalizers.
    """
    
    def __init__(self, ffe_taps=15, dfe_taps=5, modulation='pam4'):
        """
        Initialize equalizer parameters.
        
        Parameters:
        -----------
        ffe_taps : int
            Number of FFE (feedforward) taps
        dfe_taps : int
            Number of DFE (feedback) taps
        modulation : str
            Modulation type ('pam2', 'pam4', etc.)
        """
        self.ffe_taps = ffe_taps
        self.dfe_taps = dfe_taps
        self.modulation = modulation
        
        # Initialize tap coefficients
        self.ffe_weights_lms = np.zeros(ffe_taps)
        self.ffe_weights_lms[ffe_taps//2] = 1  # Center spike initialization
        
        self.dfe_weights_lms = np.zeros(dfe_taps)
        
        self.ffe_weights_auto = np.zeros(ffe_taps)
        self.ffe_weights_auto[ffe_taps//2] = 1
        
        self.dfe_weights_auto = np.zeros(dfe_taps)
        
        # Modulation parameters
        if modulation == 'pam2':
            self.levels = np.array([-1, 1])
        elif modulation == 'pam4':
            self.levels = np.array([-3, -1, 1, 3]) / 3
        
    def generate_channel(self, channel_type='multipath'):
        """
        Generate channel impulse response.
        
        Parameters:
        -----------
        channel_type : str
            Type of channel ('multipath', 'optical', 'backplane')
        
        Returns:
        --------
        h : array
            Channel impulse response
        """
        if channel_type == 'multipath':
            # Multipath channel with ISI
            h = np.array([0.1, 0.3, 0.7, 1.0, 0.8, 0.4, 0.2, 0.1])
        elif channel_type == 'optical':
            # Optical channel with chromatic dispersion
            t = np.arange(0, 10, 0.1)
            h = np.exp(-t/3) * np.sin(2*np.pi*0.3*t)
            h = h[::10]  # Downsample
        elif channel_type == 'backplane':
            # High-speed backplane with severe ISI
            h = np.array([0.05, 0.1, 0.3, 0.8, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05])
        
        return h / np.max(np.abs(h))  # Normalize
    
    def lms_adaptation(self, received_signal, training_symbols, mu_ffe=0.01, mu_dfe=0.005):
        """
        LMS-based adaptation for FFE/DFE.
        
        Parameters:
        -----------
        received_signal : array
            Received signal samples
        training_symbols : array
            Known training symbols
        mu_ffe : float
            LMS step size for FFE
        mu_dfe : float
            LMS step size for DFE
        
        Returns:
        --------
        error_history : array
            MSE over iterations
        """
        N = len(received_signal)
        error_history = []
        
        # Initialize buffers
        ffe_buffer = np.zeros(self.ffe_taps)
        dfe_buffer = np.zeros(self.dfe_taps)
        
        # LMS adaptation loop
        for n in range(self.ffe_taps, N):
            # FFE input vector (received samples)
            ffe_buffer = received_signal[n-self.ffe_taps+1:n+1][::-1]
            
            # FFE output
            ffe_out = np.dot(self.ffe_weights_lms, ffe_buffer)
            
            # DFE output (using previous decisions)
            dfe_out = np.dot(self.dfe_weights_lms, dfe_buffer)
            
            # Equalizer output
            eq_output = ffe_out - dfe_out
            
            # Make decision (slicer)
            if self.modulation == 'pam2':
                decision = np.sign(eq_output)
            else:  # PAM4
                decision = self.levels[np.argmin(np.abs(eq_output - self.levels))]
            
            # Training mode: use known symbols
            if n < len(training_symbols):
                decision = training_symbols[n]
            
            # Error calculation
            error = decision - eq_output
            
            # LMS weight update
            self.ffe_weights_lms += mu_ffe * error * ffe_buffer
            self.dfe_weights_lms += mu_dfe * error * dfe_buffer
            
            # Update DFE buffer
            dfe_buffer = np.roll(dfe_buffer, 1)
            dfe_buffer[0] = decision
            
            # Record MSE
            error_history.append(error**2)
        
        return np.array(error_history)
    
    def autocorrelation_adaptation(self, received_signal, training_symbols):
        """
        Autocorrelation-based (Wiener) adaptation for FFE/DFE.
        
        Parameters:
        -----------
        received_signal : array
            Received signal samples
        training_symbols : array
            Known training symbols
        
        Returns:
        --------
        mse : float
            Mean squared error after adaptation
        """
        N_train = len(training_symbols)
        
        # Ensure we have enough samples
        if N_train < self.ffe_taps + self.dfe_taps:
            raise ValueError("Not enough training samples for autocorrelation method")
        
        # Form data matrices for FFE
        X_ffe = np.zeros((N_train - self.ffe_taps + 1, self.ffe_taps))
        for i in range(N_train - self.ffe_taps + 1):
            X_ffe[i, :] = received_signal[i:i+self.ffe_taps][::-1]
        
        # Form data matrices for DFE (using training symbols as "decisions")
        X_dfe = np.zeros((N_train - self.ffe_taps + 1, self.dfe_taps))
        for i in range(N_train - self.ffe_taps + 1):
            start_idx = i + self.ffe_taps - self.dfe_taps
            if start_idx >= 0:
                X_dfe[i, :] = training_symbols[start_idx:start_idx+self.dfe_taps][::-1]
        
        # Combined matrix
        X = np.hstack([X_ffe, -X_dfe])
        
        # Target vector (desired output)
        d = training_symbols[self.ffe_taps-1:N_train]
        
        # Autocorrelation matrix
        R = X.T @ X / X.shape[0]
        
        # Cross-correlation vector
        p = X.T @ d / X.shape[0]
        
        # Solve Wiener-Hopf equation: R * w = p
        try:
            # Add small regularization for numerical stability
            R_reg = R + 1e-6 * np.eye(R.shape[0])
            w_opt = np.linalg.solve(R_reg, p)
            
            # Extract FFE and DFE weights
            self.ffe_weights_auto = w_opt[:self.ffe_taps]
            self.dfe_weights_auto = w_opt[self.ffe_taps:]
            
            # Calculate MSE
            y_pred = X @ w_opt
            mse = np.mean((d - y_pred)**2)
            
        except np.linalg.LinAlgError:
            print("Matrix inversion failed, using pseudo-inverse")
            w_opt = np.linalg.pinv(R) @ p
            self.ffe_weights_auto = w_opt[:self.ffe_taps]
            self.dfe_weights_auto = w_opt[self.ffe_taps:]
            y_pred = X @ w_opt
            mse = np.mean((d - y_pred)**2)
        
        return mse
    
    def compare_methods(self, num_symbols=5000, snr_db=20, channel_type='multipath'):
        """
        Compare LMS and autocorrelation methods.
        
        Parameters:
        -----------
        num_symbols : int
            Number of symbols to simulate
        snr_db : float
            Signal-to-noise ratio in dB
        channel_type : str
            Type of channel to simulate
        
        Returns:
        --------
        results : dict
            Comparison results
        """
        # Generate random symbols
        if self.modulation == 'pam2':
            symbols = np.random.choice(self.levels, num_symbols)
        else:
            symbols = np.random.choice(self.levels, num_symbols)
        
        # Generate channel
        h = self.generate_channel(channel_type)
        
        # Pass through channel
        received = np.convolve(symbols, h, mode='same')
        
        # Add noise
        signal_power = np.mean(received**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.sqrt(noise_power) * np.random.randn(len(received))
        received_noisy = received + noise
        
        # Training sequence (10% of data)
        num_training = int(0.1 * num_symbols)
        training_symbols = symbols[:num_training]
        
        # Reset weights
        self.ffe_weights_lms = np.zeros(self.ffe_taps)
        self.ffe_weights_lms[self.ffe_taps//2] = 1
        self.dfe_weights_lms = np.zeros(self.dfe_taps)
        
        # LMS adaptation
        print("Running LMS adaptation...")
        lms_errors = self.lms_adaptation(received_noisy, training_symbols)
        
        # Autocorrelation adaptation
        print("Running autocorrelation adaptation...")
        auto_mse = self.autocorrelation_adaptation(received_noisy, training_symbols)
        
        # Test both equalizers on remaining data
        test_start = num_training
        test_symbols = symbols[test_start:]
        
        # Test LMS equalizer
        ber_lms, decisions_lms = self.test_equalizer(
            received_noisy[test_start:], test_symbols, 
            self.ffe_weights_lms, self.dfe_weights_lms
        )
        
        # Test autocorrelation equalizer
        ber_auto, decisions_auto = self.test_equalizer(
            received_noisy[test_start:], test_symbols,
            self.ffe_weights_auto, self.dfe_weights_auto
        )
        
        results = {
            'lms_errors': lms_errors,
            'lms_final_mse': np.mean(lms_errors[-100:]) if len(lms_errors) > 100 else np.mean(lms_errors),
            'auto_mse': auto_mse,
            'ber_lms': ber_lms,
            'ber_auto': ber_auto,
            'ffe_weights_lms': self.ffe_weights_lms,
            'ffe_weights_auto': self.ffe_weights_auto,
            'dfe_weights_lms': self.dfe_weights_lms,
            'dfe_weights_auto': self.dfe_weights_auto,
            'channel': h
        }
        
        return results
    
    def test_equalizer(self, received_signal, true_symbols, ffe_weights, dfe_weights):
        """
        Test equalizer performance.
        """
        N = len(received_signal)
        decisions = np.zeros(N)
        errors = 0
        
        ffe_buffer = np.zeros(self.ffe_taps)
        dfe_buffer = np.zeros(self.dfe_taps)
        
        for n in range(self.ffe_taps, N):
            # FFE processing
            ffe_buffer = received_signal[n-self.ffe_taps+1:n+1][::-1]
            ffe_out = np.dot(ffe_weights, ffe_buffer)
            
            # DFE processing
            dfe_out = np.dot(dfe_weights, dfe_buffer)
            
            # Equalizer output
            eq_output = ffe_out - dfe_out
            
            # Decision
            if self.modulation == 'pam2':
                decision = np.sign(eq_output)
                if decision == 0:
                    decision = 1
            else:  # PAM4
                decision = self.levels[np.argmin(np.abs(eq_output - self.levels))]
            
            decisions[n] = decision
            
            # Update DFE buffer
            dfe_buffer = np.roll(dfe_buffer, 1)
            dfe_buffer[0] = decision
            
            # Count errors
            if n < len(true_symbols) and decision != true_symbols[n]:
                errors += 1
        
        ber = errors / (N - self.ffe_taps)
        return ber, decisions
    
    def plot_comparison(self, results):
        """
        Plot comparison results.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. LMS convergence curve
        ax = axes[0, 0]
        ax.semilogy(results['lms_errors'][:1000], 'b-', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Squared Error')
        ax.set_title('LMS Convergence')
        ax.grid(True, alpha=0.3)
        
        # 2. Channel and equalizer frequency response
        ax = axes[0, 1]
        freq = np.fft.fftfreq(1024)
        
        # Channel response
        H_channel = np.fft.fft(results['channel'], 1024)
        ax.plot(freq[:512], 20*np.log10(np.abs(H_channel[:512])), 'r-', 
                label='Channel', linewidth=2)
        
        # FFE responses
        H_ffe_lms = np.fft.fft(results['ffe_weights_lms'], 1024)
        H_ffe_auto = np.fft.fft(results['ffe_weights_auto'], 1024)
        ax.plot(freq[:512], 20*np.log10(np.abs(H_ffe_lms[:512])), 'b--', 
                label='FFE (LMS)', linewidth=2)
        ax.plot(freq[:512], 20*np.log10(np.abs(H_ffe_auto[:512])), 'g--', 
                label='FFE (Auto)', linewidth=2)
        
        ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Frequency Responses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 0.5])
        
        # 3. Tap coefficients comparison
        ax = axes[0, 2]
        tap_indices = np.arange(self.ffe_taps)
        width = 0.35
        ax.bar(tap_indices - width/2, results['ffe_weights_lms'], width, 
               label='LMS', alpha=0.7)
        ax.bar(tap_indices + width/2, results['ffe_weights_auto'], width, 
               label='Autocorr', alpha=0.7)
        ax.set_xlabel('Tap Index')
        ax.set_ylabel('Tap Weight')
        ax.set_title('FFE Tap Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. DFE tap coefficients
        ax = axes[1, 0]
        if self.dfe_taps > 0:
            dfe_indices = np.arange(self.dfe_taps)
            ax.bar(dfe_indices - width/2, results['dfe_weights_lms'], width, 
                   label='LMS', alpha=0.7)
            ax.bar(dfe_indices + width/2, results['dfe_weights_auto'], width, 
                   label='Autocorr', alpha=0.7)
        ax.set_xlabel('Tap Index')
        ax.set_ylabel('Tap Weight')
        ax.set_title('DFE Tap Coefficients')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Performance metrics
        ax = axes[1, 1]
        metrics = ['Final MSE', 'BER']
        lms_values = [results['lms_final_mse'], results['ber_lms']]
        auto_values = [results['auto_mse'], results['ber_auto']]
        
        x = np.arange(len(metrics))
        ax.bar(x - width/2, lms_values, width, label='LMS', alpha=0.7)
        ax.bar(x + width/2, auto_values, width, label='Autocorr', alpha=0.7)
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # 6. Complexity comparison
        ax = axes[1, 2]
        categories = ['Computation\nper Symbol', 'Memory\nRequirement', 'Convergence\nTime']
        
        # Normalized complexity scores (arbitrary units)
        lms_complexity = [2, 1, 3]  # Low computation, low memory, slow convergence
        auto_complexity = [4, 3, 1]  # High computation, high memory, fast convergence
        
        x = np.arange(len(categories))
        ax.bar(x - width/2, lms_complexity, width, label='LMS', alpha=0.7)
        ax.bar(x + width/2, auto_complexity, width, label='Autocorr', alpha=0.7)
        ax.set_ylabel('Relative Complexity')
        ax.set_title('Complexity Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Demonstration and comparison
if __name__ == "__main__":
    # Create equalizer adaptation instance
    equalizer = EqualizerAdaptation(ffe_taps=21, dfe_taps=5, modulation='pam4')
    
    # Run comparison
    print("Comparing LMS and Autocorrelation-based adaptation...")
    results = equalizer.compare_methods(
        num_symbols=10000,
        snr_db=25,
        channel_type='multipath'
    )
    
    # Print numerical results
    print("\n=== Adaptation Results ===")
    print(f"LMS Final MSE: {results['lms_final_mse']:.6f}")
    print(f"Autocorrelation MSE: {results['auto_mse']:.6f}")
    print(f"LMS BER: {results['ber_lms']:.6f}")
    print(f"Autocorrelation BER: {results['ber_auto']:.6f}")
    
    # Plot comparison
    equalizer.plot_comparison(results)
    
    # Additional analysis: SNR sweep
    print("\n=== SNR Sweep Analysis ===")
    snr_range = np.arange(10, 35, 5)
    ber_lms_list = []
    ber_auto_list = []
    
    for snr in snr_range:
        results_snr = equalizer.compare_methods(
            num_symbols=5000,
            snr_db=snr,
            channel_type='multipath'
        )
        ber_lms_list.append(results_snr['ber_lms'])
        ber_auto_list.append(results_snr['ber_auto'])
        print(f"SNR {snr} dB: LMS BER = {results_snr['ber_lms']:.6f}, "
              f"Auto BER = {results_snr['ber_auto']:.6f}")
    
    # Plot SNR sweep results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_lms_list, 'bo-', label='LMS', linewidth=2, markersize=8)
    plt.semilogy(snr_range, ber_auto_list, 'rs-', label='Autocorrelation', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER Performance: LMS vs Autocorrelation Adaptation', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n=== Method Comparison Summary ===")
    print("┌─────────────────────┬──────────────────────┬──────────────────────┐")
    print("│ Characteristic      │ LMS                  │ Autocorrelation      │")
    print("├─────────────────────┼──────────────────────┼──────────────────────┤")
    print("│ Complexity          │ O(N) per sample      │ O(N²) matrix inv.    │")
    print("│ Memory              │ Tap weights only     │ Correlation matrix   │")
    print("│ Convergence         │ Slow (iterative)     │ Immediate (batch)    │")
    print("│ Tracking            │ Excellent            │ Poor                 │")
    print("│ Stability           │ Depends on μ         │ Guaranteed           │")
    print("│ Training Required   │ Minimal              │ Substantial          │")
    print("│ Noise Sensitivity   │ Moderate             │ High                 │")
    print("│ Implementation      │ Simple               │ Complex              │")
    print("└─────────────────────┴──────────────────────┴──────────────────────┘")